"""诊断 GRU 长时递推在 k≈13s 的跳跃

现象：GRU 曲线在 t≈13 s 从 0.001 突然跳到 0.017（17 倍），之后缓慢下降。
这不符合物理，说明递推逻辑有问题。

13 s ≈ 124 步 ≈ 3.875 段（HORIZON=32）。
第一段结束在 k=32（3.36s），第 4 段结束在 k=128（13.4 s）。
跳跃发生在第 4 段边界 -> 滚动逻辑在某一处出错。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(r"C:\pclwm\handoff\PC-LWM_handoff_20261102")
sys.path.insert(0, str(PKG / "src"))
from aero_world_model.models.baselines import BaselineGRU

DATA = PKG / "data" / "prepared"
CKPT = PKG / "checkpoints"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEP, WINDOW, HORIZON = 7, 64, 32
DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]

d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = [str(c) for c in d["obs_cols"]]
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
runs = {str(r): {"obs": d[f"{r}_obs"][::STEP].astype(np.float32),
                 "act": d[f"{r}_act"][::STEP].astype(np.float32)}
        for r in d["run_ids"]}
KEEP = [c for c in OBS_ALL if c not in DROP]
KIDX = [OBS_ALL.index(c) for c in KEEP]
C_O = len(OBS_ALL)

tr_obs = np.concatenate([runs[r]["obs"] for r in splits["train"]], 0)
tr_act = np.concatenate([runs[r]["act"] for r in splits["train"]], 0)
o_mu, o_sd = tr_obs.mean(0), tr_obs.std(0) + 1e-8
a_mu, a_sd = tr_act.mean(0), tr_act.std(0) + 1e-8

gru = BaselineGRU(C_O, 1, hidden=192, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
gru.load_state_dict(torch.load(CKPT / "v2_gru.pt", map_location=DEV,
                              weights_only=True))
gru.eval()

rid = splits["test"][0]
o = ((runs[rid]["obs"] - o_mu) / o_sd).astype(np.float32)
a = ((runs[rid]["act"] - a_mu) / a_sd).astype(np.float32)
print(f"轨迹: {rid}  长度 {len(o)}")

start = WINDOW + 1
NSEG = 12

# ---------- 方法A：当前实现（滚动时 obs 用自己的预测）----------
@torch.no_grad()
def rollout_A():
    cur = torch.from_numpy(o[start - WINDOW:start][None]).to(DEV)
    ain = torch.from_numpy(a[start - WINDOW - 1:start - 1][None]).to(DEV)
    preds = []
    for s in range(NSEG):
        p0 = start + s * HORIZON
        af = torch.from_numpy(a[p0 - 1:p0 - 1 + HORIZON][None]).to(DEV)
        p = gru(cur, ain, af)
        preds.append(p)
        if s < NSEG - 1:
            cur = torch.cat([cur[:, HORIZON:], p], 1)
            ain = torch.cat([ain[:, HORIZON:], af], 1)
    return torch.cat(preds, 1)[0].cpu().numpy()


# ---------- 方法B：每段都用真值历史（teacher forcing，作对照）----------
@torch.no_grad()
def rollout_B():
    preds = []
    for s in range(NSEG):
        p0 = start + s * HORIZON
        cur = torch.from_numpy(o[p0 - WINDOW:p0][None]).to(DEV)
        ain = torch.from_numpy(a[p0 - WINDOW - 1:p0 - 1][None]).to(DEV)
        af = torch.from_numpy(a[p0 - 1:p0 - 1 + HORIZON][None]).to(DEV)
        preds.append(gru(cur, ain, af))
    return torch.cat(preds, 1)[0].cpu().numpy()


# ---------- 方法C：单段一次预测 32 步（不滚动，只测第一段）----------
@torch.no_grad()
def rollout_C(nseg):
    preds = []
    for s in range(nseg):
        p0 = start + s * HORIZON
        cur = torch.from_numpy(o[p0 - WINDOW:p0][None]).to(DEV)
        ain = torch.from_numpy(a[p0 - WINDOW - 1:p0 - 1][None]).to(DEV)
        af = torch.from_numpy(a[p0 - 1:p0 - 1 + HORIZON][None]).to(DEV)
        preds.append(gru(cur, ain, af))
    return torch.cat(preds, 1)[0].cpu().numpy()


tgt = o[start:start + NSEG * HORIZON]
T = len(tgt)


def rmse_curve(pred):
    e = (pred[:T] - tgt)[:, KIDX]
    return np.sqrt((e ** 2).mean(1))


for name, fn in [("A 滚动(当前)", rollout_A),
                 ("B 真值历史", rollout_B)]:
    p = fn()
    r = rmse_curve(p)
    print(f"\n[{name}] 均值={r.mean():.4f}")
    for k in [0, 15, 31, 32, 63, 95, 96, 127, 128, 255, 383]:
        if k < len(r):
            print(f"    k={k+1:>3}  t={k*STEP*0.015:>6.2f}s  rmse={r[k]:.5f}")

# 逐步分析跳跃位置
pA = rollout_A()
rA = rmse_curve(pA)
print("\n=== 逐段边界分析（A 方法）===")
print(f"{'段':>3} {'k范围':>12} {'段内均值':>10} {'段首':>10} {'段末':>10}")
for s in range(NSEG):
    a0, a1 = s * HORIZON, (s + 1) * HORIZON
    if a1 > len(rA):
        break
    seg = rA[a0:a1]
    print(f"{s+1:>3} {a0+1:>5}-{a1:<6} {seg.mean():>10.5f} "
          f"{seg[0]:>10.5f} {seg[-1]:>10.5f}")

# 检查第 4 段（跳跃处）
print("\n=== 第 3-5 段详细（跳跃区）===")
for k in range(80, 145):
    if k < len(rA):
        mark = " <<<" if abs(rA[k] - rA[k-1]) / max(rA[k-1], 1e-9) > 2 else ""
        if k < 90 or k > 135 or mark:
            print(f"  k={k+1:>3} t={k*STEP*0.015:>6.2f}s  {rA[k]:.5f}{mark}")

