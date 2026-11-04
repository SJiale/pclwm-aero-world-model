"""DT-RSSM 第二轮：参数量匹配对照 + 完整评价

按用户第 2/4/5 条要求：
  2. 增加参数量匹配实验（RSSM-wide: deter=136, hidden=256 -> 1,486,141）
  4. 每个 epoch 输出关键指标（不再等 ep%10）
  5. 给出 K=1/4/8 的 val / test_id / test_ood / 长时递推，
     与原始 RSSM 相同评价定义

实验矩阵：
  RSSM-d128        939,909   基线
  RSSM-wide      1,486,141   容量对照（与 DT 同容量）
  DT-K1  (96+96) 1,484,773
  DT-K4  (96+96) 1,484,773
  DT-K8  (96+96) 1,484,773
  DT-K4-matched (48+48) 1,069,045  低容量对照
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_dt")))

from evaluate import (UnifiedData, UnifiedEval, rollout_far, WINDOW,
                      HORIZON, LONG_TARGETS)
from dt_rssm import DTWorldModel
from rssm_impl import WorldModel as BaseRSSM

OUT = Path(r"C:\pclwm\pclwm_dt")
CKPT = OUT / "checkpoints"
CKPT.mkdir(exist_ok=True)
RES = OUT / "results_v2"
RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MODE = "gumbel"
N_REP = 3
MAX_EP = 60
PATIENCE = 15
BS = 64
LR = 4e-4

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)

print("=" * 92)
print("DT-RSSM 第二轮：参数量匹配对照")
print("=" * 92)
print(f"  设备={DEV}  dt={DT:.3f}s  WINDOW={WINDOW}  HORIZON={HORIZON}")
print(f"  推理={MODE}（显式）  重复={N_REP}  EPOCH={MAX_EP}")
print(f"  窗口: train={len(tr_W)} val={len(va_W)} "
      f"te={len(te_W)} ood={len(ood_W)}")
print()


def batch_iter(ws, bs, shuffle=True):
    n = len(ws)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i+bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c]))
               .float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut"]}


# ══════════════════════════════════════════════════
# 统一训练函数（支持两种模型）
# ══════════════════════════════════════════════════
def train(name, model, max_ep=MAX_EP, patience=PATIENCE, bs=BS, lr=LR,
          resume=False):
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*92}")
    print(f"[{name}] 参数={npar:,}")
    print(f"{'='*92}")

    ck = CKPT / f"{name}.pt"
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": [], "rec": [], "kl": [], "multi": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()

    for ep in range(1, max_ep+1):
        model.train()
        acc = {"tot": 0.0, "rec": 0.0, "kl": 0.0, "multi": 0.0}
        n = 0
        for b in batch_iter(tr_W, bs, True):
            opt.zero_grad(set_to_none=True)
            if isinstance(model, DTWorldModel):
                out = model(b["obs"], b["act"], b["act_fut"], b["fut"],
                            mode=MODE)
            else:
                # 基线 RSSM：手动组装（与原训练口径一致）
                rssm = model.rssm
                post, prior, recon, _ = rssm.observe(b["obs"], b["act"],
                                                     MODE)
                T = b["obs"].shape[1]
                burn = max(1, T//4)
                rec = ((recon[:, burn:] - b["obs"][:, burn:])**2).mean()
                kl_dyn, kl_rep = rssm.kl_terms(post, prior)
                kl_all = kl_dyn + kl_rep
                kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
                pred = rssm.imagine(b["obs"], b["act"], b["act_fut"], MODE)
                multi = ((pred - b["fut"])**2).mean()
                out = {"rec": rec, "kl": kl_clip, "multi": multi}

            loss = out["rec"] + 0.1*out["kl"] + out["multi"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = len(b["obs"])
            acc["tot"] += float(loss.detach())*nb
            acc["rec"] += float(out["rec"].detach())*nb
            acc["kl"] += float(out["kl"].detach())*nb
            acc["multi"] += float(out["multi"].detach())*nb
            n += nb
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in batch_iter(va_W, 128, False):
                if isinstance(model, DTWorldModel):
                    out = model(b["obs"], b["act"], b["act_fut"],
                                b["fut"], mode=MODE)
                    l = out["rec"] + 0.1*out["kl"] + out["multi"]
                else:
                    rssm = model.rssm
                    post, prior, recon, _ = rssm.observe(
                        b["obs"], b["act"], MODE)
                    T = b["obs"].shape[1]
                    burn = max(1, T//4)
                    rec = ((recon[:, burn:] - b["obs"][:, burn:])**2).mean()
                    kl_dyn, kl_rep = rssm.kl_terms(post, prior)
                    kl_clip = torch.clamp(kl_dyn+kl_rep,
                                          min=rssm.free_bits).mean()
                    pred = rssm.imagine(b["obs"], b["act"], b["act_fut"],
                                        MODE)
                    multi = ((pred - b["fut"])**2).mean()
                    l = rec + 0.1*kl_clip + multi
                vs += float(l)*len(b["obs"]); vn += len(b["obs"])
        v = vs/vn
        hist["train"].append(acc["tot"]/n)
        hist["rec"].append(acc["rec"]/n)
        hist["kl"].append(acc["kl"]/n)
        hist["multi"].append(acc["multi"]/n)
        hist["val"].append(v)

        improved = v < best - 1e-7
        if improved:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), ck)
        else:
            bad += 1

        # ★ 第 4 条：每个 epoch 都打印
        print(f"  ep {ep:>3} train={acc['tot']/n:.5f} val={v:.5f} "
              f"rec={acc['rec']/n:.5f} kl={acc['kl']/n:.4f} "
              f"multi={acc['multi']/n:.5f} "
              f"{'*' if improved else ' '} best={best:.5f}@{bep} "
              f"({time.time()-t0:.0f}s)", flush=True)

        if bad >= patience:
            print(f"  早停 ep {ep}（{patience} 轮无改善）")
            break

    model.load_state_dict(torch.load(ck, map_location=DEV,
                                    weights_only=True))
    model.eval()
    return model, hist, best, bep, time.time()-t0, npar


# ══════════════════════════════════════════════════
# 评价
# ══════════════════════════════════════════════════
@torch.no_grad()
def predict(model, ws, seed=None, bs=64):
    out = []
    is_dt = isinstance(model, DTWorldModel)
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        if seed is not None:
            torch.manual_seed(seed + i)
        if is_dt:
            p = model.imagine(b["obs"], b["act"], b["act_fut"], mode=MODE)
        else:
            p = model.imagine(b["obs"], b["act"], b["act_fut"], mode=MODE)
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


def eval_split(model, ws, n_rep=N_REP):
    vals, ch_acc = [], []
    for r in range(n_rep):
        p = predict(model, ws, seed=1000+r*777)
        e = ev.evaluate(p, ws)
        vals.append(e["overall_rel"])
        ch_acc.append(e["per_channel"])
    ch = {}
    for c in data.obs_cols:
        arr = [x[c]["rmse_rel"] for x in ch_acc]
        ch[c] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "per_channel": ch, "n_rep": n_rep}


def long_segmented(model, split, tgt, n_rep=N_REP):
    """分段重编码（现行协议）"""
    is_dt = isinstance(model, DTWorldModel)
    finals, means = [], []
    for r in range(n_rep):
        def mfn(obs, act, af, _sd=1000+r*777):
            with torch.no_grad():
                torch.manual_seed(_sd)
                if is_dt:
                    return model.imagine(
                        torch.from_numpy(obs).float().to(DEV),
                        torch.from_numpy(act).float().to(DEV),
                        torch.from_numpy(af).float().to(DEV),
                        mode=MODE).cpu().numpy()
                return model.imagine(
                    torch.from_numpy(obs).float().to(DEV),
                    torch.from_numpy(act).float().to(DEV),
                    torch.from_numpy(af).float().to(DEV),
                    mode=MODE).cpu().numpy()
        rs = []
        for rid in data.splits[split]:
            x = rollout_far(mfn, data, rid, tgt)
            if x:
                rs.append(x)
        if rs:
            finals.append(np.mean([x["final_rel"] for x in rs]))
            means.append(np.mean([x["mean_rel"] for x in rs]))
    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)), "n_rep": n_rep}


def long_continuous(model, split, tgt, seed=42):
    """★ 一次历史初始化后连续潜状态递推（不看任何未来真值）"""
    is_dt = isinstance(model, DTWorldModel)
    rssm = model.rssm
    finals, means = [], []
    for rid in data.splits[split]:
        o = data.norm_obs(rid).astype(np.float32)
        a = data.norm_act(rid).astype(np.float32)
        n_steps = int(round(tgt / DT))
        start = WINDOW + 1
        if len(o) <= start + n_steps + 2:
            n_steps = len(o) - start - 2
            if n_steps < 10:
                continue
        torch.manual_seed(seed)
        with torch.no_grad():
            obs_h = torch.from_numpy(o[start-WINDOW:start][None]) \
                .float().to(DEV)
            act_h = torch.from_numpy(a[start-WINDOW-1:start-1][None]) \
                .float().to(DEV)
            if is_dt:
                state = rssm.initial(1, DEV)
                for t in range(WINDOW):
                    need = (t % rssm.k_slow == 0) or (t == 0)
                    s = rssm.step(state, act_h[:, t], need)
                    post = rssm.posterior(s["prior_in"], obs_h[:, t])
                    z = rssm.sample(post, MODE)
                    state = {"hf": s["hf"], "hs": s["hs"], "z": z, "step": t}
                errs = []
                for k in range(n_steps):
                    t_abs = WINDOW + k
                    need = (t_abs % rssm.k_slow == 0) or (t_abs == 0)
                    a_t = torch.from_numpy(
                        a[start-1+k:start+k][None]).float().to(DEV)
                    a_t = a_t.reshape(1, -1)      # [1,A]
                    s = rssm.step(state, a_t, need)
                    z = rssm.sample(s["prior_logits"], MODE)
                    state = {"hf": s["hf"], "hs": s["hs"], "z": z,
                             "step": t_abs}
                    pred = rssm.dec(rssm.feat(state))          # [1,C]
                    gt = torch.from_numpy(
                        o[start+k:start+k+1][None]).float().to(DEV)  # [1,1,C]
                    errs.append((pred - gt[:, 0]).cpu().numpy()[0])  # [C]
            else:
                state = rssm.initial(1, DEV)
                for t in range(WINDOW):
                    pz = state["z"].reshape(1, -1)
                    h = rssm.cell(torch.cat([pz, act_h[:, t]], -1),
                                  state["h"])
                    embed = rssm.enc(obs_h[:, t])
                    pl = rssm.post(torch.cat([h, embed], -1)).reshape(
                        1, rssm.stoch, rssm.classes)
                    z = rssm.sample(pl, MODE)
                    state = {"h": h, "z": z}
                errs = []
                for k in range(n_steps):
                    pz = state["z"].reshape(1, -1)
                    a_t = torch.from_numpy(
                        a[start-1+k:start+k][None]).float().to(DEV)
                    a_t = a_t.reshape(1, -1)      # [1,A]
                    h = rssm.cell(torch.cat([pz, a_t], -1), state["h"])
                    pl = rssm.prior(h).reshape(1, rssm.stoch,
                                               rssm.classes)
                    z = rssm.sample(pl, MODE)
                    state = {"h": h, "z": z}
                    pred = rssm.dec(rssm.feat(state))          # [1,C]
                    gt = torch.from_numpy(
                        o[start+k:start+k+1][None]).float().to(DEV)  # [1,1,C]
                    errs.append((pred - gt[:, 0]).cpu().numpy()[0])  # [C]
        # errs 每项是 [C] 的归一化误差
        E = np.stack([np.ravel(e) for e in errs], 0).astype(np.float64)
        assert E.shape == (len(errs), C_O), f"E.shape={E.shape}"
        E = E * np.asarray(data.obs_sd, dtype=np.float64).ravel()[None, :]
        ps = np.sqrt((E ** 2).mean(axis=1))    # [n_steps] 标量误差
        ref = float(np.asarray(data.obs_sd).mean())
        finals.append(float(ps[-1]) / ref)
        means.append(float(ps.mean()) / ref)

    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)), "n_traj": len(finals)}


def full_eval(model, name):
    print(f"  评价 {name}...", flush=True)
    t0 = time.time()
    r = {
        "test_id": eval_split(model, te_W),
        "test_ood": eval_split(model, ood_W),
        "long_segmented": {}, "long_continuous": {},
    }
    for split in ["test_id", "test_ood"]:
        for tgt in LONG_TARGETS:
            x = long_segmented(model, split, tgt)
            if x:
                r["long_segmented"][f"{split}_{tgt}s"] = x
            y = long_continuous(model, split, tgt)
            if y:
                r["long_continuous"][f"{split}_{tgt}s"] = y
    print(f"    完成 ({time.time()-t0:.0f}s)")
    ti = r["test_id"]; od = r["test_ood"]
    print(f"    test_id={ti['mean']:.4f}±{ti['std']:.4f}  "
          f"ood={od['mean']:.4f}±{od['std']:.4f}")
    for kk, vv in r["long_segmented"].items():
        if "test_id" in kk:
            print(f"    分段 {kk}: {vv['final_mean']:.4f}")
    for kk, vv in r["long_continuous"].items():
        if "test_id" in kk:
            print(f"    连续 {kk}: {vv['final_mean']:.4f}")
    return r


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--runs", default="all")
_args, _ = _p.parse_known_args()

MATRIX = {
    "RSSM-d128": ("base", dict(deter=128, hidden=128)),
    "RSSM-wide": ("base", dict(deter=136, hidden=256)),
    "DT-K1": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=1)),
    "DT-K4": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=4)),
    "DT-K8": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=8)),
    "DT-K4-matched": ("dt", dict(deter_fast=48, deter_slow=48, k_slow=4)),
}

to_run = list(MATRIX.keys()) if _args.runs == "all" else _args.runs.split(",")

ALL = {}
resfile = RES / "results_v2.json"
if resfile.exists():
    ALL = json.loads(resfile.read_text(encoding="utf-8"))

for name in to_run:
    if name not in MATRIX:
        print(f"  跳过未知配置 {name}")
        continue
    kind, kw = MATRIX[name]
    torch.manual_seed(SEED); np.random.seed(SEED)
    if kind == "base":
        model = BaseRSSM(C_O, C_A, **kw).to(DEV)
    else:
        model = DTWorldModel(C_O, C_A, **kw).to(DEV)

    model, hist, bv, bep, el, npar = train(name, model)
    ev_res = full_eval(model, name)
    ALL[name] = {
        "kind": kind, "config": kw, "params": int(npar),
        "best_val": float(bv), "best_epoch": int(bep),
        "train_time_s": el, **ev_res,
        "hist": {k: [float(x) for x in v] for k, v in hist.items()},
    }
    resfile.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                      encoding="utf-8")
    print(f"  -> 已保存 {name}")

print(f"\n{'='*92}")
print("全部完成")
print(f"{'='*92}")
print(f"  {'模型':<18}{'参数':>12}{'test_id':>12}{'test_ood':>12}"
      f"{'分段60s':>10}{'连续60s':>10}")
print("  " + "-" * 76)
for nm, r in ALL.items():
    ti = r["test_id"]["mean"]
    od = r["test_ood"]["mean"]
    s60 = r["long_segmented"].get("test_id_60s", {}).get("final_mean")
    c60 = r["long_continuous"].get("test_id_60s", {}).get("final_mean")
    print(f"  {nm:<18}{r['params']:>12,}{ti:>12.4f}{od:>12.4f}"
          f"{s60 if s60 else 0:>10.4f}{c60 if c60 else 0:>10.4f}")
print(f"\n  -> {resfile}")




