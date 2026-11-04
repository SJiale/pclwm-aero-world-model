"""Mask-aware v3 训练 + 三种推理模式 + 逐通道移除实验

三种推理模式（严格定义）：
  1. open_loop     严格开环：故障时初始化，之后只给控制输入，不接收任何观测
  2. segmented     分段反馈：每 32 步用正常传感器更新一次
  3. stepwise      逐步在线：每 0.105 s 用正常传感器更新

共同点：
  - 相同的故障前 32 步真实历史初始化
  - 相同的控制输入序列
  - 相同的故障时刻与评价区间
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
from evaluate import UnifiedData, WINDOW, DT
from mask_impl_v3 import (MaskGRU, MaskRSSM, MaskWorldModel,
                          MaskGRUModel, apply_mask, augment_mask_batch)

OUT = Path(r"C:\pclwm\pclwm_mask")
CKPT = OUT / "checkpoints_v3"
CKPT.mkdir(exist_ok=True)
RES = OUT / "results_v3"
RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MODE = "gumbel"
N_SEED = 3          # RSSM 多种子推理
MAX_EP = 60
PATIENCE = 15
BS = 64
LR = 4e-4

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
SD_T4 = float(data.obs_sd[I_T4])
MU_T4 = float(data.obs_mu[I_T4])

print("=" * 92)
print("Mask-aware v3：统一训练 + 三种推理模式")
print("=" * 92)
print(f"  设备={DEV}  WINDOW={WINDOW}  dt={DT:.3f}s")
print(f"  T4: sd={SD_T4:.2f} °R")
print(f"  训练: {MAX_EP} epoch, patience={PATIENCE}, bs={BS}, lr={LR}")
print(f"  RSSM 推理种子: {N_SEED}")

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)
print(f"  窗口: train={len(tr_W)} val={len(va_W)} te={len(te_W)} ood={len(ood_W)}")

AUG = {"p_none": 0.20, "p_fixed": 0.25, "p_random": 0.25, "p_burst": 0.30,
       "fixed_channels": [I_T4], "burst_channels": [I_T4],
       "random_ratio": 0.30, "burst_min": 8, "burst_max": 32}


RESUME = "--resume" in sys.argv

def train(name, kind):
    torch.manual_seed(SEED); np.random.seed(SEED)
    if kind == "gru":
        model = MaskGRUModel(C_O, C_A, hidden=128, mlp=256).to(DEV)
    else:
        model = MaskWorldModel(C_O, C_A, deter=128, hidden=128).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    ck = CKPT / f"{name}.pt"
    if RESUME and ck.exists():
        model.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
        model.eval()
        print(f"\n[{'='*90}]")
        print(f"[{name}] 从已有权重加载（跳过训练）参数={npar:,}")
        return model, {"train": [], "val": []}, 0.0, 0, 0.0, npar
    print(f"\n{'='*92}")
    print(f"[{name}] kind={kind} 参数={npar:,}")
    print(f"{'='*92}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP)
    hist = {"train": [], "val": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()

    def get_batch(ws, idx):
        c = [ws[j] for j in idx]
        return (torch.from_numpy(np.stack([w["obs"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["act"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["fut"] for w in c])).float().to(DEV),
                torch.from_numpy(np.stack([w["act_fut"] for w in c])).float().to(DEV))

    for ep in range(1, MAX_EP+1):
        model.train()
        tl, n = 0.0, 0
        idx = np.random.permutation(len(tr_W))
        for i in range(0, len(tr_W), BS):
            obs_t, act, fut_t, af = get_batch(tr_W, idx[i:i+BS])
            obs_m, m_hist, obs_t, fut_t = augment_mask_batch(
                obs_t, act, fut_t, af, AUG, DEV)
            opt.zero_grad(set_to_none=True)
            out = model(obs_m, m_hist, act, af, fut_t, obs_true=obs_t,
                        mode=MODE)
            loss = out["multi"] if kind == "gru" else \
                out["rec"] + 0.1*out["kl"] + out["multi"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = obs_t.shape[0]
            tl += float(loss.detach())*nb; n += nb
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(va_W), 128):
                obs_t, act, fut_t, af = get_batch(
                    va_W, list(range(i, min(i+128, len(va_W)))))
                m = torch.ones(obs_t.shape[0], obs_t.shape[1], C_O, device=DEV)
                m[:, :, I_T4] = 0.0
                obs_m = apply_mask(obs_t, m)
                out = model(obs_m, m, act, af, fut_t, obs_true=obs_t,
                            mode=MODE)
                v = out["multi"] if kind == "gru" else \
                    out["rec"] + 0.1*out["kl"] + out["multi"]
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
        if bad >= PATIENCE:
            print(f"  早停 ep {ep}")
            break

    model.load_state_dict(torch.load(CKPT/f"{name}.pt",
                                    map_location=DEV, weights_only=True))
    model.eval()
    return model, hist, best, bep, time.time()-t0, npar


# ══════════════════════════════════════════════════
# 三种推理模式（统一初始化与评价区间）
# ══════════════════════════════════════════════════
def run_mode(model, kind, rid, t_fail, dur=None, mode_name="stepwise",
             extra_mask=None, seed=42):
    """统一接口

    mode_name: open_loop | segmented | stepwise
    extra_mask: 额外遮蔽的通道（用于逐通道移除实验）
    """
    o = data.norm_obs(rid).astype(np.float32).copy()
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    # 构造 mask：故障后 T4=0，外加 extra_mask
    mask = np.ones((n, C_O), np.float32)
    mask[i_fail:, I_T4] = 0.0
    if extra_mask:
        for c in extra_mask:
            mask[i_fail:, c] = 0.0

    torch.manual_seed(seed)
    rssm = model.rssm if kind == "rssm" else None
    errs, preds, gts = [], [], []

    with torch.no_grad():
        # ---- 统一初始化：故障前 WINDOW 步真实历史 ----
        h_obs = torch.from_numpy(o[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        h_mask = torch.from_numpy(mask[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        h_act = torch.from_numpy(a[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        if kind == "gru":
            st = model.gru.online_init(h_obs, h_mask, h_act)
        else:
            st = rssm.online_init(h_obs, h_mask, h_act, MODE)
        m_last = torch.from_numpy(mask[i_fail-1][None]).float().to(DEV)

        # ---- 模式 1：严格开环 ----
        if mode_name == "open_loop":
            af_all = torch.from_numpy(
                a[i_fail-1:i_end-1][None]).float().to(DEV)
            H = i_end - i_fail
            if kind == "gru":
                pred = model.gru.rollout_open(st, af_all, m_last)
            else:
                pred = rssm.rollout_open(st, af_all, MODE)
            pred = pred[0].cpu().numpy()
            gt = o[i_fail:i_end]
            e = (pred[:, I_T4] - gt[:, I_T4]) * SD_T4
            errs = e.tolist()
            preds = (pred[:, I_T4]*SD_T4 + MU_T4).tolist()
            gts = (gt[:, I_T4]*SD_T4 + MU_T4).tolist()

        # ---- 模式 2：分段反馈（每 32 步） ----
        elif mode_name == "segmented":
            H_SEG = 32
            t = i_fail
            while t < i_end:
                h = min(H_SEG, i_end - t)
                af_seg = torch.from_numpy(
                    a[t-1:t-1+h][None]).float().to(DEV)
                if kind == "gru":
                    pred = model.gru.rollout_open(st, af_seg, m_last)
                else:
                    pred = rssm.rollout_open(st, af_seg, MODE)
                pred = pred[0].cpu().numpy()
                gt = o[t:t+h]
                e = (pred[:, I_T4] - gt[:, I_T4]) * SD_T4
                errs.extend(e.tolist())
                preds.extend((pred[:, I_T4]*SD_T4 + MU_T4).tolist())
                gts.extend((gt[:, I_T4]*SD_T4 + MU_T4).tolist())
                t += h
                if t >= i_end:
                    break
                # 用真实历史重编码（含其他传感器）
                h_obs = torch.from_numpy(
                    o[t-WINDOW:t][None]).float().to(DEV)
                h_mask = torch.from_numpy(
                    mask[t-WINDOW:t][None]).float().to(DEV)
                h_act = torch.from_numpy(
                    a[t-WINDOW:t][None]).float().to(DEV)
                if kind == "gru":
                    st = model.gru.online_init(h_obs, h_mask, h_act)
                else:
                    st = rssm.online_init(h_obs, h_mask, h_act, MODE)
                m_last = torch.from_numpy(mask[t-1][None]).float().to(DEV)

        # ---- 模式 3：逐步在线 ----
        else:
            for t in range(i_fail, i_end):
                ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
                mt = torch.from_numpy(mask[t:t+1][None]).float().to(DEV)
                at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
                if kind == "gru":
                    o_hat, st = model.gru.online_step(st, ot[:, 0], mt[:, 0], at)
                else:
                    o_hat, st = rssm.online_step(st, ot[:, 0], mt[:, 0], at, MODE)
                e = float(o_hat[0, I_T4]) - float(o[t, I_T4])
                errs.append(e * SD_T4)
                preds.append(float(o_hat[0, I_T4])*SD_T4 + MU_T4)
                gts.append(float(o[t, I_T4])*SD_T4 + MU_T4)

    return {"err": np.array(errs), "pred": np.array(preds),
            "gt": np.array(gts), "n_steps": len(errs),
            "method": mode_name}


def summarize(lst, t_fail=20.0):
    v = [r for r in lst if r and len(r["err"]) > 0]
    if not v:
        return None
    allerr = np.concatenate([np.abs(r["err"]) for r in v])
    per = [float(np.sqrt((r["err"]**2).mean())) for r in v]
    seg = {}
    for lo, hi in [(0,5),(5,10),(10,30),(30,1000)]:
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
            "per_traj_rmse": per,
            "worst_rmse": float(np.max(per)),
            "best_rmse": float(np.min(per)),
            "n_traj": len(v), "interval": seg}


# ══════════════════════════════════════════════════
SCENARIOS = [("perm20s", 20.0, None, "T4 永久失效 @20s"),
             ("burst5s", 20.0, 5.0, "T4 缺失 5s"),
             ("burst10s", 20.0, 10.0, "T4 缺失 10s"),
             ("burst30s", 20.0, 30.0, "T4 缺失 30s")]
MODES = ["open_loop", "segmented", "stepwise"]

ALL = {}
for name, kind in [("MaskGRU", "gru"), ("MaskRSSM", "rssm")]:
    model, hist, bv, bep, el, npar = train(name, kind)

    # 单步推理耗时
    o1 = torch.randn(1, C_O).to(DEV); m1 = torch.ones(1, C_O, device=DEV)
    a1 = torch.randn(1, C_A).to(DEV)
    with torch.no_grad():
        if kind == "gru":
            h = model.gru.init_state(1, DEV)
            for _ in range(5): _, h = model.gru.online_step(h, o1, m1, a1)
            if DEV=="cuda": torch.cuda.synchronize()
            t0=time.time()
            for _ in range(100): _, h = model.gru.online_step(h, o1, m1, a1)
            if DEV=="cuda": torch.cuda.synchronize()
            step_ms = (time.time()-t0)/100*1000
        else:
            st = model.rssm.init_state(1, DEV)
            for _ in range(5): _, st = model.rssm.online_step(st, o1, m1, a1, MODE)
            if DEV=="cuda": torch.cuda.synchronize()
            t0=time.time()
            for _ in range(100): _, st = model.rssm.online_step(st, o1, m1, a1, MODE)
            if DEV=="cuda": torch.cuda.synchronize()
            step_ms = (time.time()-t0)/100*1000

    rec = {"kind": kind, "params": int(npar), "best_val": float(bv),
           "best_epoch": int(bep), "train_time_s": el,
           "step_ms": float(step_ms),
           "hist": {k: [float(x) for x in v] for k,v in hist.items()},
           "scenarios": {}, "channel_removal": {}}

    # ---- 三种模式 × 四场景 ----
    for scen, tf, dur, desc in SCENARIOS:
        print(f"\n  [{name}] {desc}")
        rec["scenarios"][scen] = {"desc": desc, "t_fail": tf, "dur": dur,
                                  "modes": {}}
        for md in MODES:
            for split in ["test_id", "test_ood"]:
                rs = [run_mode(model, kind, rid, tf, dur, md) for rid in data.splits[split]]
                if split not in rec["scenarios"][scen]["modes"]:
                    rec["scenarios"][scen]["modes"][split] = {}
                rec["scenarios"][scen]["modes"][split][md] = summarize(rs, tf)
        s = rec["scenarios"][scen]["modes"]["test_id"]
        print(f"    {name}: 开环={s['open_loop']['rmse']:6.2f}  "
              f"分段={s['segmented']['rmse']:6.2f}  "
              f"逐步={s['stepwise']['rmse']:6.2f}")

    # ---- 逐通道移除（永久失效场景，逐步模式） ----
    print(f"\n  [{name}] 逐通道移除实验（永久失效，逐步在线）")
    CH = {"Wfact": 0, "N2int": 1, "T45": 3, "P36": 4}
    base = rec["scenarios"]["perm20s"]["modes"]["test_id"]["stepwise"]
    rec["channel_removal"]["baseline"] = base["rmse"]
    for cn, ci in CH.items():
        rs = [run_mode(model, kind, rid, 20.0, None, "stepwise",
                       extra_mask=[ci]) for rid in data.splits["test_id"]]
        sm = summarize(rs)
        rec["channel_removal"][cn] = {
            "rmse": sm["rmse"],
            "delta": sm["rmse"] - base["rmse"],
            "delta_pct": (sm["rmse"]/base["rmse"] - 1)*100,
        }
        print(f"    移除 {cn:<8} RMSE={sm['rmse']:7.2f}  "
              f"Δ={sm['rmse']-base['rmse']:+7.2f} ({(sm['rmse']/base['rmse']-1)*100:+.1f}%)")

    ALL[name] = rec
    (RES / "mask_results_v3.json").write_text(
        json.dumps(ALL, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  -> 已保存 {name}")

print(f"\n{'='*92}")
print("汇总")
print(f"{'='*92}")
for scen, tf, dur, desc in SCENARIOS:
    print(f"\n[{desc}] test_id")
    print(f"  {'模型':<12}{'开环':>9}{'分段':>9}{'逐步':>9}"
          f"{'逐步vs开环':>12}{'最差':>8}")
    print("  " + "-" * 60)
    for nm, r in ALL.items():
        s = r["scenarios"][scen]["modes"]["test_id"]
        o_, sg, sw = s["open_loop"], s["segmented"], s["stepwise"]
        gain = (1-sw["rmse"]/o_["rmse"])*100
        print(f"  {nm:<12}{o_['rmse']:>9.2f}{sg['rmse']:>9.2f}{sw['rmse']:>9.2f}"
              f"{gain:>11.1f}%{sw['worst_rmse']:>8.2f}")
print(f"\n  -> {RES/'mask_results_v3.json'}")

