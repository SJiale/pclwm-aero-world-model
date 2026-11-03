"""从已保存检查点评估 v2，不重新训练"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(r"D:\涡轴WM\_backup\workspace_20261031_182906")
sys.path.insert(0, str(ROOT / "src"))
from aero_world_model.models.rssm import RSSMConfig
from aero_world_model.models.baselines import BaselineGRU
from aero_world_model.training.offline import WorldModel

DATA = Path(r"C:\pclwm\agtf30\prepared2")
T2 = Path(r"C:\pclwm\agtf30\train2")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

STEP, WINDOW, HORIZON = 7, 64, 32
STRIDE_TR, STRIDE_EV = 4, 8

d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = list(d["obs_cols"])
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
runs = {str(r): {"obs": d[f"{r}_obs"][::STEP].astype(np.float32),
                 "act": d[f"{r}_act"][::STEP].astype(np.float32)}
        for r in d["run_ids"]}

DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]
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
                      "act_fut": a[st - 1 + WINDOW:st - 1 + WINDOW + HORIZON],
                      "rid": r})
    return W


tr_W = mkw(splits["train"], STRIDE_TR)
va_W = mkw(splits["val"], STRIDE_EV)
te_W = mkw(splits["test"], STRIDE_EV)


def rel_norm(preds, ws):
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


# ---- 加载检查点 ----
print("=" * 84)
print("从检查点评估 v2（未重新训练）")
print("=" * 84)

gru = BaselineGRU(C_O, 1, hidden=192, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
gru_sd = torch.load(T2 / "gru.pt", map_location=DEV, weights_only=True)
gru.load_state_dict(gru_sd)
gru.eval()
n_gru = sum(p.numel() for p in gru.parameters())
print(f"GRU 参数: {n_gru:,}  (hidden=192)")

rssm = WorldModel(C_O, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                     hidden=256)).to(DEV)
rssm_sd = torch.load(T2 / "rssm.pt", map_location=DEV, weights_only=True)
rssm.load_state_dict(rssm_sd)
rssm.eval()
n_rssm = sum(p.numel() for p in rssm.parameters())
print(f"RSSM 参数: {n_rssm:,}")


# ---- 基线 ----
def baselines(ws):
    out = {}
    out["Persistence"] = rel_norm(
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
    out["ARX"] = rel_norm(preds, ws)
    return out


res = {}
for name, ws in [("train", tr_W), ("val", va_W), ("test", te_W)]:
    b = baselines(ws)
    b["GRU"] = rel_norm(nn_predict(gru, ws, "gru"), ws)
    b["RSSM"] = rel_norm(nn_predict(rssm, ws, "rssm"), ws)
    res[name] = {k: v.tolist() for k, v in b.items()}

print()
print("=" * 84)
print(f"{'模型':<14}{'训练':>10}{'验证':>10}{'测试':>10}{'泛化比':>9}"
      f"{'k=1':>9}{'k=H':>9}")
print("-" * 72)
for nm in ["Persistence", "ARX", "GRU", "RSSM"]:
    tr = np.mean(res["train"][nm])
    va = np.mean(res["val"][nm])
    te = np.mean(res["test"][nm])
    k1 = res["test"][nm][0]
    kH = res["test"][nm][-1]
    print(f"{nm:<14}{tr:>10.4f}{va:>10.4f}{te:>10.4f}"
          f"{te/tr:>9.1f}x{k1:>9.4f}{kH:>9.4f}")

print()
print("=" * 84)
print("检查点元信息")
print("=" * 84)
import os
from datetime import datetime
for f in ["gru.pt", "rssm.pt"]:
    p = T2 / f
    st = os.stat(p)
    print(f"  {f}: {st.st_size/1024/1024:.1f} MB  "
          f"修改 {datetime.fromtimestamp(st.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

# 检查点里的 epoch 信息（如果有）
print()
print("  gru.pt 键数:", len(gru_sd))
print("  rssm.pt 键数:", len(rssm_sd))

json.dump({"results": res,
           "config": {"data": str(DATA), "step": STEP, "window": WINDOW,
                      "horizon": HORIZON, "stride_train": STRIDE_TR,
                      "stride_eval": STRIDE_EV,
                      "dt_effective_s": STEP * 0.015,
                      "keep_channels": [str(k) for k in KEEP],
                      "all_channels": [str(c) for c in OBS_ALL]},
           "params": {"gru": n_gru, "rssm": n_rssm},
           "note": "从检查点评估，训练过程因日志缓冲被中止，未记录逐epoch曲线"},
          open(T2 / "eval_from_ckpt.json", "w"), ensure_ascii=False, indent=1)
print(f"\n保存 -> {T2/'eval_from_ckpt.json'}")
