"""Mask-aware 修复版训练 + 逐步在线估计评价

修复：
  1. 重建损失用完整真实观测作监督（不是遮蔽值 0）
  2. 逐步在线估计：每 0.105 s 用正常传感器更新，不是每 32 步
  3. 保留每 32 步方案作为对照
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))

from evaluate import UnifiedData, WINDOW, DT
from mask_impl_fixed import (MaskGRU, MaskWorldModel, apply_mask,
                             augment_mask_batch)

OUT = Path(r"C:\pclwm\pclwm_mask")
CKPT = OUT / "checkpoints_fixed"
CKPT.mkdir(exist_ok=True)
RES = OUT / "results_fixed"
RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MODE = "gumbel"

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
SD_T4 = float(data.obs_sd[I_T4])
MU_T4 = float(data.obs_mu[I_T4])

print("=" * 92)
print("Mask-aware 修复版训练")
print("=" * 92)
print(f"  设备={DEV}  dt={DT:.3f}s  WINDOW={WINDOW}")
print(f"  T4: idx={I_T4}, mu={MU_T4:.2f}, sd={SD_T4:.2f}")

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)
print(f"  窗口: train={len(tr_W)} val={len(va_W)} te={len(te_W)} ood={len(ood_W)}")

AUG = {"p_none": 0.20, "p_fixed": 0.25, "p_random": 0.25, "p_burst": 0.30,
       "fixed_channels": [I_T4], "burst_channels": [I_T4],
       "random_ratio": 0.30, "burst_min": 8, "burst_max": 32}


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
        return (torch.from_numpy(np.stack([w["obs"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["act"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["fut"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["act_fut"] for w in c])).float().to(DEV))

    for ep in range(1, max_ep+1):
        model.train()
        tl, n = 0.0, 0
        idx = np.random.permutation(len(tr_W))
        for i in range(0, len(tr_W), bs):
            obs_true, act, fut_true, af = get_batch(tr_W, idx[i:i+bs])
            # ★ 返回 (obs_masked, mask, obs_true, fut_true)
            obs_m, m_hist, obs_t, fut_t = augment_mask_batch(
                obs_true, act, fut_true, af, AUG, DEV)
            opt.zero_grad(set_to_none=True)
            if kind == "gru":
                p = model(obs_m, m_hist, act, af)
                # ★ GRU 的监督也是真实 fut（本来就对）
                loss = ((p - fut_t)**2).mean()
            else:
                # ★ 关键修复：传 obs_true 作为重建监督
                out = model(obs_m, m_hist, act, af, fut_t,
                           obs_true=obs_t, mode=MODE)
                loss = out["rec"] + 0.1*out["kl"] + out["multi"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = obs_true.shape[0]
            tl += float(loss.detach())*nb; n += nb
        sched.step()

        # 验证：T4 全遮蔽，监督用真实值
        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_W), 128):
                obs_t, act, fut_t, af = get_batch(
                    va_W, list(range(i, min(i+128, len(va_W)))))
                m = torch.ones(obs_t.shape[0], obs_t.shape[1], C_O, device=DEV)
                m[:, :, I_T4] = 0.0
                obs_m = apply_mask(obs_t, m)
                if kind == "gru":
                    p = model(obs_m, m, act, af)
                    v = ((p - fut_t)**2).mean()
                else:
                    out = model(obs_m, m, act, af, fut_t,
                               obs_true=obs_t, mode=MODE)
                    v = out["rec"] + 0.1*out["kl"] + out["multi"]
                vs += float(v)*obs_t.shape[0]; vn += obs_t.shape[0]
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


# ══════════════════════════════════════════════════
# 两种在线估计方式
# ══════════════════════════════════════════════════
def reconstruct_stepwise(model, kind, rid, t_fail, dur=None, mode=MODE,
                         seed=42):
    """★ 逐步在线：每 0.105 s 用正常传感器更新隐状态"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    torch.manual_seed(seed)
    errs, preds, gts = [], [], []
    with torch.no_grad():
        if kind == "rssm":
            rssm = model.rssm
            state = rssm.initial(1, DEV)
            # 故障前：全部传感器正常，逐步更新
            for t in range(i_fail):
                ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
                at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
                mt = torch.ones(1, C_O, device=DEV)
                _, state = rssm.step(ot[:, 0], mt, at, state, t, mode=mode)
            # 故障后：T4 mask=0，其他正常，逐步更新
            for t in range(i_fail, i_end):
                ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
                at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
                mt = torch.ones(1, C_O, device=DEV)
                mt[:, I_T4] = 0.0
                o_hat, state = rssm.step(ot[:, 0], mt, at, state, t, mode=mode)
                e = float(o_hat[0, I_T4]) - float(o[t, I_T4])
                errs.append(e * SD_T4)
                preds.append(float(o_hat[0, I_T4])*SD_T4 + MU_T4)
                gts.append(float(o[t, I_T4])*SD_T4 + MU_T4)
        else:
            h = model.init_hidden(1, DEV)
            for t in range(i_fail):
                ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
                at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
                mt = torch.ones(1, C_O, device=DEV)
                _, h = model.step(ot[:, 0], mt, at, h)
            for t in range(i_fail, i_end):
                ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
                at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
                mt = torch.ones(1, C_O, device=DEV)
                mt[:, I_T4] = 0.0
                o_hat, h = model.step(ot[:, 0], mt, at, h)
                e = float(o_hat[0, I_T4]) - float(o[t, I_T4])
                errs.append(e * SD_T4)
                preds.append(float(o_hat[0, I_T4])*SD_T4 + MU_T4)
                gts.append(float(o[t, I_T4])*SD_T4 + MU_T4)

    return {"err": np.array(errs), "pred": np.array(preds),
            "gt": np.array(gts), "n_steps": len(errs),
            "method": "stepwise"}


def reconstruct_segmented(model, kind, rid, t_fail, dur=None, horizon=32,
                          mode=MODE, seed=42):
    """原方案（每 32 步更新）作为对照"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    cur_obs = o[i_fail-WINDOW:i_fail].copy()
    cur_mask = np.ones((WINDOW, C_O), np.float32)
    errs, preds, gts, t_idx = [], [], [], i_fail
    torch.manual_seed(seed)
    with torch.no_grad():
        while t_idx < i_end:
            H = min(horizon, i_end - t_idx)
            if H <= 0: break
            obs_t = torch.from_numpy(cur_obs[None]).float().to(DEV)
            mask_t = torch.from_numpy(cur_mask[None]).float().to(DEV)
            act_t = torch.from_numpy(a[t_idx-WINDOW:t_idx][None]).float().to(DEV)
            af_t = torch.from_numpy(a[t_idx-1:t_idx-1+H][None]).float().to(DEV)
            p = (model(obs_t, mask_t, act_t, af_t) if kind == "gru"
                 else model.imagine(obs_t, mask_t, act_t, af_t, mode=mode))
            p = p[0].cpu().numpy()
            gt = o[t_idx:t_idx+H]
            e = (p[:, I_T4] - gt[:, I_T4]) * SD_T4
            errs.extend(e.tolist())
            preds.extend((p[:, I_T4]*SD_T4 + MU_T4).tolist())
            gts.extend((gt[:, I_T4]*SD_T4 + MU_T4).tolist())
            nxt = gt.copy(); nxt[:, I_T4] = p[:, I_T4]
            cur_obs = np.concatenate([cur_obs, nxt], 0)[-WINDOW:]
            nm = np.ones((H, C_O), np.float32); nm[:, I_T4] = 0.0
            cur_mask = np.concatenate([cur_mask, nm], 0)[-WINDOW:]
            t_idx += H
    return {"err": np.array(errs), "pred": np.array(preds),
            "gt": np.array(gts), "n_steps": len(errs),
            "method": "segmented"}


def open_loop_baseline(rid, t_fail, dur=None):
    o = data.norm_obs(rid)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail >= i_end: return None
    last = o[i_fail-1, I_T4]
    gt = o[i_fail:i_end, I_T4]
    E = (last - gt) * SD_T4
    return {"err": E, "pred": np.full_like(E, last*SD_T4+MU_T4),
            "gt": gt*SD_T4+MU_T4, "n_steps": len(E), "method": "persistence"}


def summarize(lst):
    v = [r for r in lst if r and len(r["err"]) > 0]
    if not v: return None
    allerr = np.concatenate([np.abs(r["err"]) for r in v])
    per = [float(np.sqrt((r["err"]**2).mean())) for r in v]
    seg = {}
    for lo, hi in [(0,5),(5,15),(15,30),(30,1000)]:
        ch = []
        for r in v:
            i0 = int(round(lo/DT)); i1 = min(len(r["err"]), int(round(hi/DT)))
            if i1 > i0: ch.append(r["err"][i0:i1])
        if ch:
            c = np.concatenate(ch)
            seg[f"{lo}-{hi}s"] = {"rmse": float(np.sqrt((c**2).mean())),
                                  "n": len(c)}
    return {"rmse": float(np.sqrt((allerr**2).mean())),
            "mae": float(allerr.mean()),
            "rmse_std": float(np.sqrt((allerr**2).mean())/SD_T4),
            "per_traj_rmse": per, "worst_rmse": float(np.max(per)),
            "n_traj": len(v), "interval": seg}


SCENARIOS = [("perm20s", 20.0, None, "T4 永久失效 @20s"),
             ("burst5s", 20.0, 5.0, "T4 缺失 5s"),
             ("burst10s", 20.0, 10.0, "T4 缺失 10s"),
             ("burst30s", 20.0, 30.0, "T4 缺失 30s")]

import argparse
p = argparse.ArgumentParser()
p.add_argument("--only", default="all", choices=["all","gru","rssm"])
a, _ = p.parse_known_args()
MODELS = [("MaskGRU","gru"), ("MaskRSSM","rssm")]
if a.only != "all":
    MODELS = [m for m in MODELS if m[1]==a.only]

ALL = {}
resfile = RES / "mask_results_fixed.json"

for name, kind in MODELS:
    model, hist, bv, bep, el, npar = train(name, kind)

    # 推理耗时（单步）
    o1 = torch.randn(1, C_O).to(DEV); m1 = torch.ones(1, C_O, device=DEV)
    a1 = torch.randn(1, C_A).to(DEV)
    with torch.no_grad():
        if kind == "rssm":
            st = model.rssm.initial(1, DEV)
            for _ in range(5): _, st = model.rssm.step(o1, m1, a1, st, 0, mode=MODE)
            if DEV=="cuda": torch.cuda.synchronize()
            t0=time.time()
            for _ in range(100): _, st = model.rssm.step(o1, m1, a1, st, 0, mode=MODE)
            if DEV=="cuda": torch.cuda.synchronize()
            step_ms = (time.time()-t0)/100*1000
        else:
            h = model.init_hidden(1, DEV)
            for _ in range(5): _, h = model.step(o1, m1, a1, h)
            if DEV=="cuda": torch.cuda.synchronize()
            t0=time.time()
            for _ in range(100): _, h = model.step(o1, m1, a1, h)
            if DEV=="cuda": torch.cuda.synchronize()
            step_ms = (time.time()-t0)/100*1000

    rec = {"kind": kind, "params": int(npar), "best_val": float(bv),
           "best_epoch": int(bep), "train_time_s": el,
           "step_ms": float(step_ms),
           "hist": {k: [float(x) for x in v] for k,v in hist.items()},
           "scenarios": {}}

    for scen, tf, dur, desc in SCENARIOS:
        print(f"\n  [{desc}]")
        rec["scenarios"][scen] = {"desc": desc, "t_fail": tf, "dur": dur,
                                  "splits": {}}
        for split in ["test_id", "test_ood"]:
            r_step, r_seg, r_pers = [], [], []
            for rid in data.splits[split]:
                x = reconstruct_stepwise(model, kind, rid, tf, dur)
                if x: r_step.append(x)
                y = reconstruct_segmented(model, kind, rid, tf, dur)
                if y: r_seg.append(y)
                z = open_loop_baseline(rid, tf, dur)
                if z: r_pers.append(z)
            rec["scenarios"][scen]["splits"][split] = {
                "stepwise": summarize(r_step),
                "segmented": summarize(r_seg),
                "persistence": summarize(r_pers),
                "curves": {"pred": np.concatenate([r["pred"] for r in r_step]).tolist() if r_step else [],
                           "gt": np.concatenate([r["gt"] for r in r_step]).tolist() if r_step else []},
            }
            s = rec["scenarios"][scen]["splits"][split]
            sw, sg, ps = s["stepwise"], s["segmented"], s["persistence"]
            if sw and ps:
                print(f"    {split:<10} 逐步={sw['rmse']:7.2f}  分段={sg['rmse'] if sg else 0:7.2f}  "
                      f"persist={ps['rmse']:7.2f}  "
                      f"逐步vs外推={(1-sw['rmse']/sg['rmse'])*100 if sg else 0:+.1f}%")
    ALL[name] = rec
    resfile.write_text(json.dumps(ALL, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> 已保存 {name}")

print(f"\n{'='*92}")
print("汇总（逐步在线估计）")
print(f"{'='*92}")
for scen, tf, dur, desc in SCENARIOS:
    print(f"\n[{desc}] test_id")
    print(f"  {'模型':<12}{'逐步':>10}{'分段':>10}{'persist':>10}{'逐步vs分段':>12}{'最差':>9}")
    print("  " + "-" * 66)
    for nm, r in ALL.items():
        s = r["scenarios"][scen]["splits"]["test_id"]
        sw, sg, ps = s["stepwise"], s["segmented"], s["persistence"]
        if not sw: continue
        gain = (1-sw["rmse"]/sg["rmse"])*100 if sg else 0
        print(f"  {nm:<12}{sw['rmse']:>10.2f}{sg['rmse'] if sg else 0:>10.2f}"
              f"{ps['rmse'] if ps else 0:>10.2f}{gain:>11.1f}%{sw['worst_rmse']:>9.2f}")
print(f"\n  -> {resfile}")