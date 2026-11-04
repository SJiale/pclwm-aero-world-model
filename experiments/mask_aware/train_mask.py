"""Mask-aware 传感器失效重构实验（正式版）"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))

from evaluate import UnifiedData, WINDOW, DT
from mask_impl import (MaskGRU, MaskWorldModel, apply_mask,
                       augment_mask_batch)

OUT = Path(r"C:\pclwm\pclwm_mask")
CKPT = OUT / "checkpoints"
CKPT.mkdir(exist_ok=True)
RES = OUT / "results"
RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MODE = "gumbel"

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
REF = float(data.obs_sd.mean())
SD_T4 = float(data.obs_sd[I_T4])
MU_T4 = float(data.obs_mu[I_T4])

print("=" * 92)
print("Mask-aware 传感器失效重构实验")
print("=" * 92)
print(f"  设备={DEV}  dt={DT:.3f}s  WINDOW={WINDOW}")
print(f"  T4: idx={I_T4}, mu={MU_T4:.2f} R, sd={SD_T4:.2f} R")
print(f"  通道: {data.obs_cols}")

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)
print(f"  窗口: train={len(tr_W)} val={len(va_W)} "
      f"te={len(te_W)} ood={len(ood_W)}")

AUG = {
    "p_none": 0.20, "p_fixed": 0.25, "p_random": 0.25, "p_burst": 0.30,
    "fixed_channels": [I_T4],
    "burst_channels": [I_T4],
    "random_ratio": 0.30,
    "burst_min": 8, "burst_max": 32,
}


def train(name, kind, max_ep=60, patience=15, bs=64, lr=4e-4):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if kind == "gru":
        model = MaskGRU(C_O, C_A, hidden=128, mlp=256).to(DEV)
    else:
        model = MaskWorldModel(C_O, C_A, deter=128, hidden=128,
                               use_dt=False).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*92}")
    print(f"[{name}] kind={kind} 参数={npar:,}")
    print(f"{'='*92}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()

    def get_batch(ws, idx):
        c = [ws[j] for j in idx]
        obs = torch.from_numpy(np.stack([w["obs"] for w in c])).float().to(DEV)
        act = torch.from_numpy(np.stack([w["act"] for w in c])).float().to(DEV)
        fut = torch.from_numpy(np.stack([w["fut"] for w in c])).float().to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).float().to(DEV)
        return obs, act, fut, af

    for ep in range(1, max_ep+1):
        model.train()
        tl, n = 0.0, 0
        idx = np.random.permutation(len(tr_W))
        for i in range(0, len(tr_W), bs):
            obs, act, fut, af = get_batch(tr_W, idx[i:i+bs])
            obs_m, m_hist, fut_m, m_fut = augment_mask_batch(
                obs, act, fut, af, AUG, DEV)
            opt.zero_grad(set_to_none=True)
            if kind == "gru":
                p = model(obs_m, m_hist, act, af)
                loss = ((p - fut)**2).mean()
            else:
                out = model(obs_m, m_hist, act, af, fut, mode=MODE)
                loss = out["rec"] + 0.1*out["kl"] + out["multi"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = obs.shape[0]
            tl += float(loss.detach())*nb; n += nb
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_W), 128):
                obs, act, fut, af = get_batch(
                    va_W, list(range(i, min(i+128, len(va_W)))))
                m = torch.ones(obs.shape[0], obs.shape[1], C_O, device=DEV)
                m[:, :, I_T4] = 0.0
                if kind == "gru":
                    p = model(apply_mask(obs, m), m, act, af)
                    v = ((p - fut)**2).mean()
                else:
                    out = model(apply_mask(obs, m), m, act, af, fut, mode=MODE)
                    v = out["rec"] + 0.1*out["kl"] + out["multi"]
                vs += float(v)*obs.shape[0]; vn += obs.shape[0]
        v = vs/vn
        hist["train"].append(tl/n); hist["val"].append(v)
        improved = v < best - 1e-7
        if improved:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), CKPT/f"{name}.pt")
        else:
            bad += 1
        print(f"  ep {ep:>3} train={tl/n:.5f} val={v:.5f} "
              f"{'*' if improved else ' '} best={best:.5f}@{bep} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if bad >= patience:
            print(f"  早停 ep {ep}")
            break

    model.load_state_dict(torch.load(CKPT/f"{name}.pt",
                                    map_location=DEV, weights_only=True))
    model.eval()
    return model, hist, best, bep, time.time()-t0, npar


def reconstruct(model, kind, rid, t_fail, dur=None, horizon=32,
                online=True, seed=42):
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_rec = n if dur is None else min(n, int(round((t_fail + dur) / DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    cur_obs = o[i_fail-WINDOW:i_fail].copy()
    cur_mask = np.ones((WINDOW, C_O), np.float32)

    errs, preds, gts, t_idx = [], [], [], i_fail
    torch.manual_seed(seed)
    with torch.no_grad():
        while t_idx < i_rec:
            H = min(horizon, i_rec - t_idx)
            if H <= 0:
                break
            obs_t = torch.from_numpy(cur_obs[None]).float().to(DEV)
            mask_t = torch.from_numpy(cur_mask[None]).float().to(DEV)
            act_t = torch.from_numpy(a[t_idx-WINDOW:t_idx][None]).float().to(DEV)
            af_t = torch.from_numpy(a[t_idx-1:t_idx-1+H][None]).float().to(DEV)

            if kind == "gru":
                p = model(obs_t, mask_t, act_t, af_t)
            else:
                p = model.imagine(obs_t, mask_t, act_t, af_t, mode=MODE)
            p = p[0].cpu().numpy()

            gt = o[t_idx:t_idx+H]
            e = (p[:, I_T4] - gt[:, I_T4]) * SD_T4
            errs.extend(e.tolist())
            preds.extend((p[:, I_T4]*SD_T4 + MU_T4).tolist())
            gts.extend((gt[:, I_T4]*SD_T4 + MU_T4).tolist())

            if online:
                nxt = gt.copy()
            else:
                nxt = p.copy()
            nxt[:, I_T4] = p[:, I_T4]
            cur_obs = np.concatenate([cur_obs, nxt], 0)[-WINDOW:]
            nm = np.ones((H, C_O), np.float32)
            nm[:, I_T4] = 0.0
            cur_mask = np.concatenate([cur_mask, nm], 0)[-WINDOW:]
            t_idx += H

    E = np.array(errs)
    return {"err": E, "pred": np.array(preds), "gt": np.array(gts),
            "n_steps": len(E)}


def open_loop_baseline(rid, t_fail, dur=None):
    o = data.norm_obs(rid)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_rec = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail >= i_rec:
        return None
    last = o[i_fail-1, I_T4]
    gt = o[i_fail:i_rec, I_T4]
    E = (last - gt) * SD_T4
    return {"err": E, "pred": np.full_like(E, last*SD_T4 + MU_T4),
            "gt": gt*SD_T4 + MU_T4, "n_steps": len(E)}


SCENARIOS = [
    ("perm20s", 20.0, None, "T4 永久失效 @20s"),
    ("burst5s", 20.0, 5.0, "T4 缺失 5s @20s"),
    ("burst10s", 20.0, 10.0, "T4 缺失 10s @20s"),
    ("burst30s", 20.0, 30.0, "T4 缺失 30s @20s"),
]


def summarize(results_list):
    valid = [r for r in results_list if r and len(r["err"]) > 0]
    if not valid:
        return None
    all_err = np.concatenate([np.abs(r["err"]) for r in valid])
    per_traj = [float(np.sqrt((r["err"]**2).mean())) for r in valid]
    seg = {}
    for lo, hi in [(0, 5), (5, 15), (15, 30), (30, 1000)]:
        chunk = []
        for r in valid:
            e = r["err"]
            i0 = int(round(lo / DT))
            i1 = min(len(e), int(round(hi / DT)))
            if i1 > i0:
                chunk.append(e[i0:i1])
        if chunk:
            c = np.concatenate(chunk)
            seg[f"{lo}-{hi}s"] = {
                "rmse": float(np.sqrt((c**2).mean())),
                "mae": float(np.abs(c).mean()), "n": len(c)}
    return {
        "rmse": float(np.sqrt((all_err**2).mean())),
        "mae": float(all_err.mean()),
        "rmse_std": float(np.sqrt((all_err**2).mean()) / SD_T4),
        "per_traj_rmse": per_traj,
        "worst_rmse": float(np.max(per_traj)),
        "best_rmse": float(np.min(per_traj)),
        "n_traj": len(valid),
        "interval": seg,
    }


import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--only", default="all", choices=["all", "gru", "rssm"])
_args, _ = _p.parse_known_args()

MODELS = [("MaskGRU", "gru"), ("MaskRSSM", "rssm")]
if _args.only != "all":
    MODELS = [m for m in MODELS if m[1] == _args.only]

ALL = {}
resfile = RES / "mask_results.json"
if resfile.exists():
    try:
        ALL = json.loads(resfile.read_text(encoding="utf-8"))
    except Exception:
        ALL = {}

for name, kind in MODELS:
    model, hist, bv, bep, el, npar = train(name, kind)

    b = {k: torch.from_numpy(np.stack([w[k] for w in te_W[:8]])).float().to(DEV)
         for k in ["obs", "act", "act_fut"]}
    b["mask"] = torch.ones(8, WINDOW, C_O, device=DEV)
    with torch.no_grad():
        for _ in range(3):
            if kind == "gru":
                model(b["obs"], b["mask"], b["act"], b["act_fut"])
            else:
                model.imagine(b["obs"], b["mask"], b["act"], b["act_fut"], mode=MODE)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            if kind == "gru":
                model(b["obs"], b["mask"], b["act"], b["act_fut"])
            else:
                model.imagine(b["obs"], b["mask"], b["act"], b["act_fut"], mode=MODE)
        if DEV == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time()-t0)/10*1000

    rec = {"kind": kind, "params": int(npar), "best_val": float(bv),
           "best_epoch": int(bep), "train_time_s": el,
           "infer_ms": float(infer_ms),
           "hist": {k: [float(x) for x in v] for k, v in hist.items()},
           "scenarios": {}}

    for scen, t_fail, dur, desc in SCENARIOS:
        print(f"\n  场景 {desc}")
        rec["scenarios"][scen] = {"desc": desc, "t_fail": t_fail,
                                  "dur": dur, "splits": {}}
        for split in ["test_id", "test_ood"]:
            r_on, r_off, r_pers = [], [], []
            for rid in data.splits[split]:
                x = reconstruct(model, kind, rid, t_fail, dur, online=True)
                if x: r_on.append(x)
                y = reconstruct(model, kind, rid, t_fail, dur, online=False)
                if y: r_off.append(y)
                z = open_loop_baseline(rid, t_fail, dur)
                if z: r_pers.append(z)

            rec["scenarios"][scen]["splits"][split] = {
                "online": summarize(r_on),
                "open_loop_model": summarize(r_off),
                "persistence": summarize(r_pers),
                "curves": {
                    "pred": (np.concatenate([r["pred"] for r in r_on]).tolist()
                             if r_on else []),
                    "gt": (np.concatenate([r["gt"] for r in r_on]).tolist()
                           if r_on else []),
                },
            }
            s = rec["scenarios"][scen]["splits"][split]
            o = s["online"]; p_ = s["persistence"]
            if o and p_:
                gain = (1 - o["rmse"]/max(p_["rmse"], 1e-9))*100
                print(f"    {split:<10} 在线={o['rmse']:8.2f} R  "
                      f"persist={p_['rmse']:8.2f} R  改善={gain:+.1f}%")

    ALL[name] = rec
    resfile.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\n  -> 已保存 {name}")

print(f"\n{'='*92}")
print("汇总")
print(f"{'='*92}")
for scen, _, _, desc in SCENARIOS:
    print(f"\n[{desc}] test_id T4")
    print(f"  {'模型':<14}{'在线RMSE':>12}{'开环模型':>12}"
          f"{'persist':>12}{'改善':>10}{'最差':>10}")
    print("  " + "-" * 72)
    for nm, r in ALL.items():
        s = r["scenarios"].get(scen, {}).get("splits", {}).get("test_id")
        if not s: continue
        o = s["online"]; ol = s["open_loop_model"]; p_ = s["persistence"]
        if not o: continue
        gain = (1 - o["rmse"]/max(p_["rmse"], 1e-9))*100 if p_ else 0
        print(f"  {nm:<14}{o['rmse']:>12.2f}"
              f"{ol['rmse'] if ol else 0:>12.2f}"
              f"{p_['rmse'] if p_ else 0:>12.2f}"
              f"{gain:>9.1f}%{o['worst_rmse']:>10.2f}")
print(f"\n  -> {resfile}")