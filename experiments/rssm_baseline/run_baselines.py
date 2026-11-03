"""阶段 3-5：三组基线（ARX / GRU / RSSM）

关键修正（相对此前所有实验）：
  1. 归一化只用 train 拟合（修复 ARX 泄漏）
  2. 所有模型共用同一个 UnifiedData / UnifiedEval
  3. RSSM 重新检查：从零实现，KL free bits 正确应用，
     训练用后验、推理用先验（严格区分）
  4. 保存完整 loss 曲线 + 检查点 + 配置
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import (UnifiedData, UnifiedEval, rollout_far,
                      rollout_teacher_oracle, check_physics,
                      WINDOW, HORIZON, LONG_TARGETS)

OUT = Path(r"C:\pclwm\pclwm_experiments")
OUT.mkdir(parents=True, exist_ok=True)
CKPT = OUT / "checkpoints"
CKPT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt

print("=" * 82)
print("PC-LWM 基线实验")
print("=" * 82)
print(f"  设备: {DEV}")
print(f"  观测: {C_O} 通道 {data.obs_cols}")
print(f"  动作: {C_A} 通道 {data.act_cols}")
print(f"  dt={DT:.3f}s  WINDOW={WINDOW}  HORIZON={HORIZON}")

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)
print(f"  窗口: train={len(tr_W)} val={len(va_W)} "
      f"test_id={len(te_W)} test_ood={len(ood_W)}")


def to_batch(ws, idx=None):
    if idx is None:
        idx = range(len(ws))
    c = [ws[i] for i in idx]
    return {k: torch.from_numpy(np.stack([w[k] for w in c])).float().to(DEV)
            for k in ["obs", "act", "fut", "act_fut"]}


def iterate(ws, bs, shuffle=True):
    n = len(ws)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        yield to_batch(ws, idx[i:i+bs])


RESULTS = {}

# ══════════════════════════════════════════════════════════════
# 基线 1：ARX（线性，闭式解）
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 82)
print("[1/3] ARX 线性基线")
print("=" * 82)

# ARX(na=2, nb=1)：只用 train 轨迹拟合（修复泄漏）
NA, NB = 2, 1
Xtr, Ytr = [], []
for r in data.splits["train"]:
    o = data.norm_obs(r).astype(np.float64)
    a = data.norm_act(r).astype(np.float64)
    for k in range(NA, len(o) - 1):
        row = [o[k-i] for i in range(1, NA+1)]
        row += [a[k-j] for j in range(1, NB+1)]
        Xtr.append(np.concatenate(row))
        Ytr.append(o[k])
Xtr = np.array(Xtr); Ytr = np.array(Ytr)
print(f"  训练样本: {len(Xtr):,}（仅 train 17 条）")

W_arx = np.linalg.lstsq(
    np.c_[Xtr, np.ones(len(Xtr))], Ytr, rcond=1e-10)[0]
n_arx = W_arx.size
print(f"  参数量: {n_arx}")


def arx_predict(obs, act, af):
    """返回 [B,H,C]"""
    B, H = af.shape[0], af.shape[1]
    o = obs.copy()          # [B,W,C]
    out = np.zeros((B, H, C_O), np.float64)
    for h in range(H):
        row = [o[:, -i, :] for i in range(1, NA+1)]
        row += [af[:, h, :] for _ in range(1, NB+1)]
        x = np.concatenate(row, axis=1)
        y = np.c_[x, np.ones(B)] @ W_arx
        out[:, h] = y
        o = np.concatenate([o[:, 1:], y[:, None, :]], 1)
    return out


# 评估
def eval_fn(fn, ws):
    """分块推理避免内存溢出"""
    preds = []
    bs = 64
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        obs = np.stack([w["obs"] for w in c], 0).astype(np.float64)
        act = np.stack([w["act"] for w in c], 0).astype(np.float64)
        af = np.stack([w["act_fut"] for w in c], 0).astype(np.float64)
        preds.append(fn(obs, act, af))
    return np.concatenate(preds, 0)


t0 = time.time()
p_te = eval_fn(arx_predict, te_W)
p_ood = eval_fn(arx_predict, ood_W)
arx_te = ev.evaluate(p_te, te_W)
arx_ood = ev.evaluate(p_ood, ood_W)
arx_time = time.time() - t0

# 长时递推
def arx_model_fn(obs, act, af):
    return arx_predict(obs, act, af)

arx_long = {}
for tgt in LONG_TARGETS:
    rs = []
    for r in data.splits["test_id"]:
        x = rollout_far(arx_model_fn, data, r, tgt)
        if x:
            rs.append(x)
    if rs:
        arx_long[f"{tgt}s"] = {
            "mean_rel": float(np.mean([x["mean_rel"] for x in rs])),
            "final_rel": float(np.mean([x["final_rel"] for x in rs])),
        }

RESULTS["ARX"] = {
    "params": int(n_arx),
    "test_id": arx_te, "test_ood": arx_ood,
    "long": arx_long, "eval_time_s": arx_time,
    "config": {"na": NA, "nb": NB, "fit_on": "train only"},
}
print(f"  test_id : rel={arx_te['overall_rel']:.4f}  "
      f"k1={arx_te['k1_rel']:.4f}  kH={arx_te['kH_rel']:.4f}")
print(f"  test_ood: rel={arx_ood['overall_rel']:.4f}")
for k, v in arx_long.items():
    print(f"  长时 {k}: mean={v['mean_rel']:.4f} final={v['final_rel']:.4f}")

# ══════════════════════════════════════════════════════════════
# 基线 2：GRU
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 82)
print("[2/3] GRU 确定性神经基线")
print("=" * 82)


class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=HORIZON,
                 mlp=256, layers=2):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Sequential(
            nn.Linear(c_obs + c_act, hidden), nn.LayerNorm(hidden),
            nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        mods = [nn.Linear(hidden, mlp), nn.SiLU()]
        for _ in range(layers-1):
            mods += [nn.Linear(mlp, mlp), nn.SiLU()]
        mods += [nn.Linear(mlp, c_obs)]
        self.dec = nn.Sequential(*mods)
        self.o2h = nn.Linear(c_obs + c_act, hidden)

    def forward(self, obs, act, af):
        B, T, _ = obs.shape
        H = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


def train_model(model, name, max_ep=120, patience=25, lr=1e-3,
                bs=64, wd=1e-4):
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()
    for ep in range(1, max_ep+1):
        model.train()
        tl, n = 0.0, 0
        for b in iterate(tr_W, bs, True):
            opt.zero_grad(set_to_none=True)
            p = model(b["obs"], b["act"], b["act_fut"])
            loss = ((p - b["fut"])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            tl += float(loss.detach())*len(b["obs"]); n += len(b["obs"])
        sched.step()
        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in iterate(va_W, 128, False):
                p = model(b["obs"], b["act"], b["act_fut"])
                vs += float(((p - b["fut"])**2).mean())*len(b["obs"])
                vn += len(b["obs"])
        v = vs/vn
        hist["train"].append(tl/n); hist["val"].append(v)
        if v < best - 1e-7:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), CKPT / f"{name}.pt")
        else:
            bad += 1
        if ep % 20 == 0 or ep == 1:
            print(f"    ep {ep:>3} train={tl/n:.6f} val={v:.6f} "
                  f"best={best:.6f}@{bep} ({time.time()-t0:.0f}s)",
                  flush=True)
        if bad >= patience:
            print(f"    早停 ep {ep}")
            break
    model.load_state_dict(torch.load(CKPT / f"{name}.pt",
                                    map_location=DEV, weights_only=True))
    model.eval()
    return hist, best, bep, time.time()-t0


@torch.no_grad()
def torch_predict(model, ws, bs=128):
    out = []
    for i in range(0, len(ws), bs):
        b = to_batch(ws, range(i, min(i+bs, len(ws))))
        out.append(model(b["obs"], b["act"], b["act_fut"]).cpu().numpy())
    return np.concatenate(out, 0)


gru_cfgs = [("GRU-h96", 96, 192), ("GRU-h128", 128, 256),
            ("GRU-h192", 192, 384)]
for name, hid, mlp in gru_cfgs:
    print(f"\n  [{name}] hidden={hid} mlp={mlp}")
    torch.manual_seed(SEED); np.random.seed(SEED)
    m = GRUModel(C_O, C_A, hidden=hid, mlp=mlp).to(DEV)
    npar = sum(p.numel() for p in m.parameters())
    hist, bestv, bep, el = train_model(m, name)
    p_te = torch_predict(m, te_W)
    p_ood = torch_predict(m, ood_W)
    r_te = ev.evaluate(p_te, te_W)
    r_ood = ev.evaluate(p_ood, ood_W)

    # 长时
    def mfn(obs, act, af):
        with torch.no_grad():
            return m(torch.from_numpy(obs).float().to(DEV),
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

    RESULTS[name] = {
        "params": int(npar), "best_val": float(bestv),
        "best_epoch": int(bep), "train_time_s": el,
        "hist": {k: [float(x) for x in v] for k, v in hist.items()},
        "test_id": r_te, "test_ood": r_ood, "long": lng,
        "config": {"hidden": hid, "mlp": mlp},
    }
    print(f"    test_id : rel={r_te['overall_rel']:.4f}  "
          f"k1={r_te['k1_rel']:.4f}  kH={r_te['kH_rel']:.4f}")
    print(f"    test_ood: rel={r_ood['overall_rel']:.4f}")
    for k, v in lng.items():
        print(f"    长时 {k}: mean={v['mean_rel']:.4f} "
              f"final={v['final_rel']:.4f}")

(OUT / "results_baselines.json").write_text(
    json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  ARX + GRU 结果已保存")
print(f"  -> {OUT/'results_baselines.json'}")
