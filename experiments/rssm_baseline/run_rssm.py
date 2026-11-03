"""阶段 5：RSSM 基线（重新实现，严格检查 KL 与先验/后验）

相对此前版本的修正：
  1. **KL 正确应用 free bits**：per-dimension 而非整体
     `KL_reg = max(KL_per_dim, free_bits).sum()` 而非 clip(总KL)
  2. **训练用后验，推理用先验**：严格分离
  3. **可靠的状态重置**：每条窗口独立初态，不跨窗口污染
  4. 参数量与 GRU 可比（0.5–2 倍）
  5. 完整记录 KL / rec / multi 曲线

参考：DreamerV3 (Hafner 2023) 的 RSSM 设计
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import (UnifiedData, UnifiedEval, rollout_far, check_physics,
                      WINDOW, HORIZON, LONG_TARGETS)

OUT = Path(r"C:\pclwm\pclwm_experiments")
CKPT = OUT / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)

print("=" * 82)
print("阶段 5：RSSM 基线（DreamerV3 风格，重新实现）")
print("=" * 82)
print(f"  设备: {DEV}")
print(f"  窗口: train={len(tr_W)} val={len(va_W)} "
      f"test={len(te_W)} ood={len(ood_W)}")


def build_batch(ws, bs=64, shuffle=True):
    n = len(ws)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i+bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c]))
               .float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut"]}


# ══════════════════════════════════════════════════════════════
# RSSM 组件
# ══════════════════════════════════════════════════════════════
class RSSM(nn.Module):
    """离散潜变量 RSSM

    状态 = (h_t deterministic, z_t stochastic one-hot)
    z 由 32 个类别分布（32 类）组成 => 32×32 维 one-hot
    """

    def __init__(self, c_obs, c_act, deter=192, stoch=32, classes=32,
                 hidden=192, free_bits=1.0, dyn_scale=0.5,
                 rep_scale=0.1, unimix=0.01):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.free_bits = free_bits
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.unimix = unimix

        # Encoder: obs -> embed
        self.enc = nn.Sequential(
            nn.Linear(c_obs, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())

        # 后验 q(z|h,x)
        self.post = nn.Sequential(
            nn.Linear(deter + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, stoch * classes))
        # 先验 p(z|h)
        self.prior = nn.Sequential(
            nn.Linear(deter, hidden), nn.SiLU(),
            nn.Linear(hidden, stoch * classes))
        # 状态转移 h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.cell = nn.GRUCell(stoch * classes + c_act, deter)

        # Decoder: feat -> obs
        self.dec = nn.Sequential(
            nn.Linear(deter + stoch * classes, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, c_obs))

        self._feat_dim = deter + stoch * classes

    def initial(self, B):
        return {"h": torch.zeros(B, self.deter, device=DEV),
                "z": torch.zeros(B, self.stoch, self.classes, device=DEV)}

    def feat(self, state):
        z = state["z"].reshape(state["z"].shape[0], -1)
        return torch.cat([state["h"], z], -1)

    def observe(self, obs, act):
        """训练路径：用后验（见到 obs）"""
        B, T, _ = obs.shape
        state = self.initial(B)
        posts, priors, recons = [], [], []
        for t in range(T):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act[:, t]], -1), state["h"])
            prior_logits = self.prior(h).reshape(B, self.stoch, self.classes)
            embed = self.enc(obs[:, t])
            post_logits = self.post(torch.cat([h, embed], -1)) \
                .reshape(B, self.stoch, self.classes)
            z = self._sample(post_logits)
            state = {"h": h, "z": z}
            posts.append(post_logits)
            priors.append(prior_logits)
            recons.append(self.dec(self.feat(state)))
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1))

    def imagine(self, obs_hist, act_hist, act_future):
        """推理路径：**只用先验**（不看未来 obs）"""
        B, T, _ = obs_hist.shape
        H = act_future.shape[1]
        state = self.initial(B)
        for t in range(T):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act_hist[:, t]], -1),
                          state["h"])
            embed = self.enc(obs_hist[:, t])
            post_logits = self.post(torch.cat([h, embed], -1)) \
                .reshape(B, self.stoch, self.classes)
            z = self._sample(post_logits)
            state = {"h": h, "z": z}
        outs = []
        for k in range(H):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act_future[:, k]], -1),
                          state["h"])
            prior_logits = self.prior(h).reshape(B, self.stoch,
                                                 self.classes)
            z = self._sample(prior_logits)     # 先验采样
            state = {"h": h, "z": z}
            outs.append(self.dec(self.feat(state)))
        return torch.stack(outs, 1)

    def _sample(self, logits):
        """带 unimix 的 Gumbel-softmax 采样 + straight-through"""
        if self.unimix > 0:
            probs = F.softmax(logits, -1)
            probs = probs * (1 - self.unimix) + self.unimix / self.classes
            logits = torch.log(probs.clamp_min(1e-9))
        if self.training:
            y = F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        else:
            idx = logits.argmax(-1, keepdim=True)
            y = torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        return y

    def kl_divergence(self, post_logits, prior_logits):
        """per-dimension KL（关键修正）"""
        p = F.softmax(post_logits, -1)
        q = F.softmax(prior_logits, -1)
        logp = F.log_softmax(post_logits, -1)
        logq = F.log_softmax(prior_logits, -1)
        # KL(p || q) 对每个 (stoch, classes) 求和
        kl_classes = (p * (logp - logq)).sum(-1)     # [B,T,stoch]
        # per-dimension free bits（核心修正）
        kl_free = torch.clamp(kl_classes, min=self.free_bits)
        return kl_free.mean(), kl_classes.mean()


class WorldModel(nn.Module):
    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.rssm = RSSM(c_obs, c_act, **kw)

    def forward(self, obs, act, act_fut):
        """返回 (loss_dict, pred)"""
        B, T, _ = obs.shape
        H = act_fut.shape[1]
        post, prior, recon = self.rssm.observe(obs, act)

        # 重建损失：burn-in 前 1/4 不计（让隐状态先稳定）
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - obs[:, burn:])**2).mean()
        # KL（per-dim free bits）
        kl, kl_raw = self.rssm.kl_divergence(post, prior)
        # 多步预测
        pred = self.rssm.imagine(obs, act, act_fut)
        return {"rec": rec, "kl": kl, "kl_raw": kl_raw,
                "post": post, "prior": prior, "pred": pred}

    def imagine(self, obs, act, act_fut):
        return self.rssm.imagine(obs, act, act_fut)


# ══════════════════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════════════════
def train_rssm(name, deter=192, stoch=32, classes=32, hidden=192,
               free_bits=1.0, lr=4e-4, max_ep=60, patience=15, bs=64):
    torch.manual_seed(0); np.random.seed(0)
    model = WorldModel(C_O, C_A, deter=deter, stoch=stoch,
                       classes=classes, hidden=hidden,
                       free_bits=free_bits).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n  [{name}] deter={deter} stoch={stoch}x{classes} "
          f"free={free_bits} 参数={npar:,}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": [], "rec": [], "kl": [], "multi": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()

    for ep in range(1, max_ep+1):
        model.train()
        acc = {"tot": 0.0, "rec": 0.0, "kl": 0.0, "multi": 0.0}
        n = 0
        for b in build_batch(tr_W, bs, True):
            opt.zero_grad(set_to_none=True)
            out = model(b["obs"], b["act"], b["act_fut"])
            rec, kl = out["rec"], out["kl"]
            multi = ((out["pred"] - b["fut"])**2).mean()
            loss = rec + 0.1*kl + multi
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = len(b["obs"])
            acc["tot"] += float(loss.detach())*nb
            acc["rec"] += float(rec.detach())*nb
            acc["kl"] += float(kl.detach())*nb
            acc["multi"] += float(multi.detach())*nb
            n += nb
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in build_batch(va_W, 64, False):
                out = model(b["obs"], b["act"], b["act_fut"])
                multi = ((out["pred"] - b["fut"])**2).mean()
                l = out["rec"] + 0.1*out["kl"] + multi
                vs += float(l)*len(b["obs"]); vn += len(b["obs"])
        v = vs/vn
        hist["train"].append(acc["tot"]/n)
        hist["rec"].append(acc["rec"]/n)
        hist["kl"].append(acc["kl"]/n)
        hist["multi"].append(acc["multi"]/n)
        hist["val"].append(v)

        if v < best - 1e-7:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), CKPT/f"{name}.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"    ep {ep:>3} loss={acc['tot']/n:.5f} val={v:.5f} "
                  f"rec={acc['rec']/n:.5f} kl={acc['kl']/n:.4f} "
                  f"multi={acc['multi']/n:.5f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= patience:
            print(f"    早停 ep {ep}")
            break

    model.load_state_dict(torch.load(CKPT/f"{name}.pt",
                                    map_location=DEV, weights_only=True))
    model.eval()
    return model, hist, best, bep, time.time()-t0, npar


@torch.no_grad()
def torch_predict(model, ws, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c])).float().to(DEV)
             for k in ["obs", "act", "act_fut"]}
        out.append(model.imagine(b["obs"], b["act"], b["act_fut"])
                   .cpu().numpy())
    return np.concatenate(out, 0)


# ══════════════════════════════════════════════════════════════
# 跑 3 个配置
# ══════════════════════════════════════════════════════════════
RES = {}
CFGS = [
    ("RSSM-d128", dict(deter=128, hidden=128)),
    ("RSSM-d192", dict(deter=192, hidden=192)),
    ("RSSM-d256", dict(deter=256, hidden=256)),
]

for name, kw in CFGS:
    model, hist, bestv, bep, el, npar = train_rssm(name, **kw)
    p_te = torch_predict(model, te_W)
    p_ood = torch_predict(model, ood_W)
    r_te = ev.evaluate(p_te, te_W)
    r_ood = ev.evaluate(p_ood, ood_W)

    def mfn(obs, act, af):
        with torch.no_grad():
            return model.imagine(
                torch.from_numpy(obs).float().to(DEV),
                torch.from_numpy(act).float().to(DEV),
                torch.from_numpy(af).float().to(DEV)).cpu().numpy()

    lng = {}
    for tgt in LONG_TARGETS:
        rs = []
        for r in data.splits["test_id"]:
            x = rollout_far(mfn, data, r, tgt)
            if x:
                rs.append(x)
        if rs:
            lng[f"{tgt}s"] = {
                "mean_rel": float(np.mean([x["mean_rel"] for x in rs])),
                "final_rel": float(np.mean([x["final_rel"] for x in rs])),
            }

    # KL 诊断
    kl_final = hist["kl"][-1]
    rec_final = hist["rec"][-1]
    multi_final = hist["multi"][-1]

    RES[name] = {
        "params": int(npar), "best_val": float(bestv),
        "best_epoch": int(bep), "train_time_s": el,
        "hist": {k: [float(x) for x in v] for k, v in hist.items()},
        "test_id": r_te, "test_ood": r_ood, "long": lng,
        "kl_final": float(kl_final), "rec_final": float(rec_final),
        "multi_final": float(multi_final),
        "config": kw,
    }
    print(f"    test_id : rel={r_te['overall_rel']:.4f} "
          f"k1={r_te['k1_rel']:.4f} kH={r_te['kH_rel']:.4f}")
    print(f"    test_ood: rel={r_ood['overall_rel']:.4f}")
    for k, v in lng.items():
        print(f"    长时 {k}: mean={v['mean_rel']:.4f} "
              f"final={v['final_rel']:.4f}")
    print(f"    KL={kl_final:.4f} rec={rec_final:.5f} "
          f"multi={multi_final:.5f}")

(OUT / "results_rssm.json").write_text(
    json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  RSSM 结果 -> {OUT/'results_rssm.json'}")


