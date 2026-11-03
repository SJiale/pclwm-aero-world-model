"""包内可直接运行的评估脚本（使用相对路径）

这是 eval_v2_ckpt.py 的移植版，所有路径改为相对于本包的路径。
相比原版：
  1. 路径相对化
  2. 检查点文件名改为 v2_gru.pt / v2_rssm.pt
  3. 增加了 400 步长时自回归评估
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

# 包根目录 = 本文件的上上级
PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG / "src"))

from aero_world_model.models.rssm import RSSMConfig
from aero_world_model.models.baselines import BaselineGRU
from aero_world_model.training.offline import WorldModel

DATA = PKG / "data" / "prepared"
CKPT = PKG / "checkpoints"
RES = PKG / "results"
RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"

STEP, WINDOW, HORIZON = 7, 64, 32
DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]
LONG_STEPS = 400          # 长时自回归步数 (42 s)

print(f"包根: {PKG}")
print(f"设备: {DEV}")

if not (DATA / "data.npz").exists():
    print(f"错误: 缺 {DATA/'data.npz'}")
    sys.exit(1)
if not (CKPT / "v2_gru.pt").exists():
    print(f"错误: 缺 {CKPT/'v2_gru.pt'}")
    sys.exit(1)

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
o_mu = tr_obs.mean(0).astype(np.float64)
o_sd = tr_obs.std(0).astype(np.float64) + 1e-8
a_mu = tr_act.mean(0).astype(np.float64)
a_sd = tr_act.std(0).astype(np.float64) + 1e-8


def mkw(ids, stride):
    W = []
    for r in ids:
        o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float32)
        a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float32)
        for st in range(1, len(o) - WINDOW - HORIZON, stride):
            W.append({"obs": o[st:st + WINDOW],
                      "act": a[st - 1:st - 1 + WINDOW],
                      "fut": o[st + WINDOW:st + WINDOW + HORIZON],
                      "act_fut": a[st - 1 + WINDOW:
                                    st - 1 + WINDOW + HORIZON],
                      "rid": r})
    return W


tr_W = mkw(splits["train"], 8)
va_W = mkw(splits["val"], 8)
te_W = mkw(splits["test"], 8)


def rel(preds, ws):
    e = np.stack([p - w["fut"] for p, w in zip(preds, ws)], 0)
    return np.sqrt((e ** 2).mean(0))[:, KIDX].mean(1)


@torch.no_grad()
def nn_predict(model, ws, kind, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i + bs]
        o = torch.from_numpy(np.stack([w["obs"] for w in c])).to(DEV)
        a = torch.from_numpy(np.stack([w["act"] for w in c])).to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).to(DEV)
        p = model(o, a, af) if kind == "gru" else model.imagine(o, a, af)
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


# ---------------- 基础评估 ----------------
print("\n" + "=" * 80)
print("v2 检查点评估（包内路径）")
print("=" * 80)

gru = BaselineGRU(C_O, 1, hidden=192, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
gru.load_state_dict(torch.load(CKPT / "v2_gru.pt", map_location=DEV,
                               weights_only=True))
gru.eval()

rssm = WorldModel(C_O, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                     hidden=256)).to(DEV)
rssm.load_state_dict(torch.load(CKPT / "v2_rssm.pt", map_location=DEV,
                                weights_only=True))
rssm.eval()


def baselines(ws):
    out = {}
    out["Persistence"] = rel(
        [np.repeat(w["obs"][-1:], HORIZON, 0) for w in ws], ws)
    Xa, Ya = [], []
    for r in splits["train"]:
        o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float64)
        a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float64)
        Xa.append(np.c_[o[:-1], a[:-1]])
        Ya.append(o[1:])
    X = np.concatenate(Xa, 0)
    Y = np.concatenate(Ya, 0)
    Wm = np.linalg.lstsq(np.c_[X, np.ones((len(X), 1))], Y, rcond=1e-6)[0]
    A, B, c0 = Wm[:C_O].T, Wm[C_O:C_O + 1].T, Wm[-1]
    preds = []
    for w in ws:
        y = w["obs"][-1].astype(np.float64).copy()
        oo = []
        for k in range(HORIZON):
            y = y @ A.T + w["act_fut"][k].astype(np.float64) @ B.T + c0
            oo.append(y.copy())
        preds.append(np.stack(oo, 0))
    out["ARX"] = rel(preds, ws)
    return out


res = {}
for name, ws in [("train", tr_W), ("val", va_W), ("test", te_W)]:
    b = baselines(ws)
    b["GRU"] = rel(nn_predict(gru, ws, "gru"), ws)
    b["RSSM"] = rel(nn_predict(rssm, ws, "rssm"), ws)
    res[name] = {k: v.tolist() for k, v in b.items()}

print(f"\n{'模型':<14}{'训练':>10}{'验证':>10}{'测试':>10}{'泛化比':>9}")
print("-" * 56)
for nm in ["Persistence", "ARX", "GRU", "RSSM"]:
    tr = np.mean(res["train"][nm])
    va = np.mean(res["val"][nm])
    te = np.mean(res["test"][nm])
    print(f"{nm:<14}{tr:>10.4f}{va:>10.4f}{te:>10.4f}{te/tr:>9.1f}x")

# ---------------- 长时自回归 ----------------
print("\n" + "=" * 80)
print(f"长时自由递推（{LONG_STEPS} 步 = {LONG_STEPS*STEP*0.015:.1f} s）")
print("=" * 80)


@torch.no_grad()
def long_rollout(model, kind, rid_list, steps=LONG_STEPS):
    """从 test 轨迹起点开始，完全用模型自身预测递推（不使用未来真值）"""
    all_err = []
    nseg = steps // HORIZON          # 只用完整 HORIZON 段
    steps = nseg * HORIZON
    for rid in rid_list:
        o_full = ((runs[rid]["obs"] - o_mu) / o_sd).astype(np.float32)
        a_full = ((runs[rid]["act"] - a_mu) / a_sd).astype(np.float32)
        # 从轨迹中段开始，保证有足够的递推长度
        max_start = len(o_full) - steps - 1
        if max_start < WINDOW:
            # 轨迹不够长，用能支持的最大段数
            avail = (len(o_full) - WINDOW - 1) // HORIZON
            if avail < 1:
                continue
            nseg_i, steps_i = avail, avail * HORIZON
        else:
            nseg_i, steps_i = nseg, steps
        start = WINDOW if max_start < WINDOW else max(WINDOW, max_start)
        start = min(start, len(o_full) - steps_i - 1)
        if start < WINDOW:
            start = WINDOW
        if start + steps_i + 1 > len(o_full):
            continue
        cur = torch.from_numpy(o_full[start - WINDOW:start][None]).to(DEV)
        ain = torch.from_numpy(a_full[start - WINDOW:start][None]).to(DEV)
        preds = []
        for s in range(nseg_i):
            p0 = start + s * HORIZON
            af = torch.from_numpy(
                a_full[p0:p0 + HORIZON][None]).to(DEV)
            if kind == "gru":
                p = model(cur, ain, af)
            else:
                p = model.imagine(cur, ain, af)
            preds.append(p)
            # 滚动：用预测替代历史，保持长度 = WINDOW
            cur = torch.cat([cur[:, HORIZON:], p], 1)[:, -WINDOW:]
            ain = torch.cat([ain[:, HORIZON:], af], 1)[:, -WINDOW:]
        pred = torch.cat(preds, 1)[0].cpu().numpy()
        tgt = o_full[start:start + steps_i]
        n = min(len(pred), len(tgt))
        all_err.append(pred[:n] - tgt[:n])
    if not all_err:
        return None
    # all_err: list of [T_i, C] -> 每条轨迹按时间算跨通道 RMSE，再平均
    per_traj = []
    for e in all_err:
        ek = e[:, KIDX]                       # [T_i, 11]
        per_traj.append(np.sqrt((ek ** 2).mean(1)))   # [T_i]
    L = min(len(x) for x in per_traj)
    M = np.stack([x[:L] for x in per_traj], 0)        # [n_traj, L]
    return M.mean(0)                                   # [L] 每步平均误差

# Persistence 长时
pers_long = []
for rid in splits["test"]:
    o_full = ((runs[rid]["obs"] - o_mu) / o_sd).astype(np.float32)
    start = WINDOW + 1
    if len(o_full) < start + LONG_STEPS + 1:
        continue
    y = np.repeat(o_full[start - 1:start], LONG_STEPS, 0)
    tgt = o_full[start:start + LONG_STEPS]
    pers_long.append(y - tgt)
if pers_long:
    E = np.concatenate(pers_long, 0)[:, KIDX]
    r_pers_long = np.sqrt((E ** 2).mean(0))
    print(f"  Persistence: 终值误差 {r_pers_long[-1]:.4f}")

for nm, model, kind in [("GRU", gru, "gru"), ("RSSM", rssm, "rssm")]:
    try:
        r = long_rollout(model, kind, splits["test"])
        if r is not None:
            idx = [0, 9, 49, 99, 199, 399]
            idx = [i for i in idx if i < len(r)]
            print(f"  {nm:<6} 步数 {len(r)}: " +
                  "  ".join(f"k={i+1}:{r[i]:.4f}" for i in idx))
            res.setdefault("long", {})[nm] = r.tolist()
    except Exception as e:
        print(f"  {nm:<6} 长时评估失败: {type(e).__name__}: {e}")
        res.setdefault("long", {})[nm + "_error"] = f"{type(e).__name__}: {e}"

if pers_long:
    res.setdefault("long", {})["Persistence"] = r_pers_long.tolist()

res["meta"] = {"data": str(DATA), "ckpt": str(CKPT),
               "dt_s": STEP * 0.015, "window": WINDOW,
               "horizon": HORIZON, "long_steps": LONG_STEPS,
               "keep_channels": KEEP, "all_channels": OBS_ALL}

out_f = RES / "eval_package.json"
out_f.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                 encoding="utf-8")
print(f"\n保存 -> {out_f}")






