"""正确的统一评估：全程在归一化空间，不混用物理单位"""
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

DATA = Path(r"C:\pclwm\agtf30\prepared")
T = Path(r"C:\pclwm\agtf30\train")
STEP, WINDOW, HORIZON, STRIDE = 7, 64, 16, 8
DEV = "cuda" if torch.cuda.is_available() else "cpu"

d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = list(d["obs_cols"])
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
runs = {str(r): {"obs": d[f"{r}_obs"][::STEP].astype(np.float32),
                 "act": d[f"{r}_act"][::STEP].astype(np.float32)}
        for r in d["run_ids"]}

DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]
KEEP = [c for c in OBS_ALL if c not in DROP]
KIDX = [OBS_ALL.index(c) for c in KEEP]
C16 = len(OBS_ALL)
CK = len(KEEP)

tr_obs = np.concatenate([runs[r]["obs"] for r in splits["train"]], 0)
tr_act = np.concatenate([runs[r]["act"] for r in splits["train"]], 0)
o_mu, o_sd = tr_obs.mean(0).astype(np.float64), tr_obs.std(0).astype(np.float64) + 1e-8
a_mu, a_sd = tr_act.mean(0).astype(np.float64), tr_act.std(0).astype(np.float64) + 1e-8

norm = lambda r: (((runs[r]["obs"] - o_mu) / o_sd).astype(np.float32),
                  ((runs[r]["act"] - a_mu) / a_sd).astype(np.float32))


def mkw(ids):
    W = []
    for r in ids:
        o, a = norm(r)
        for st in range(1, len(o) - WINDOW - HORIZON, STRIDE):
            W.append({"obs": o[st:st + WINDOW],
                      "act": a[st - 1:st - 1 + WINDOW],
                      "fut": o[st + WINDOW:st + WINDOW + HORIZON],
                      "act_fut": a[st - 1 + WINDOW:st - 1 + WINDOW + HORIZON]})
    return W


def rel_norm(preds, ws):
    """归一化空间 RMSE（直接就是相对误差，因为 sd=1）"""
    e = np.stack([p - w["fut"] for p, w in zip(preds, ws)], 0)
    return np.sqrt((e ** 2).mean(0)).mean(1)   # [H]


# ---------- 基线 ----------
def baselines(ws):
    out = {}
    out["Zero"] = rel_norm([np.zeros_like(w["fut"]) for w in ws], ws)
    out["Persistence"] = rel_norm(
        [np.repeat(w["obs"][-1:], HORIZON, 0) for w in ws], ws)
    Xa, Ya = [], []
    for r in splits["train"]:
        o, a = norm(r)
        Xa.append(np.c_[o[:-1], a[:-1]])
        Ya.append(o[1:])
    X = np.concatenate(Xa, 0).astype(np.float64)
    Y = np.concatenate(Ya, 0).astype(np.float64)
    Wm = np.linalg.lstsq(np.c_[X, np.ones((len(X), 1))], Y, rcond=1e-6)[0]
    A, B, c0 = Wm[:C16].T, Wm[C16:C16 + 1].T, Wm[-1]
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


# ---------- 神经网络 ----------
gru = BaselineGRU(C16, 1, hidden=256, horizon=HORIZON, mlp_units=512,
                  mlp_layers=2).to(DEV)
gru.load_state_dict(torch.load(T / "gru.pt", map_location=DEV, weights_only=True))
gru.eval()
rssm = WorldModel(C16, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                    hidden=256)).to(DEV)
rssm.load_state_dict(torch.load(T / "rssm.pt", map_location=DEV, weights_only=True))
rssm.eval()


@torch.no_grad()
def nn_pred(model, ws, kind):
    out = []
    for i in range(0, len(ws), 64):
        c = ws[i:i + 64]
        o = torch.from_numpy(np.stack([w["obs"] for w in c])).to(DEV)
        a = torch.from_numpy(np.stack([w["act"] for w in c])).to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).to(DEV)
        p = model(o, a, af) if kind == "gru" else model.imagine(o, a, af)
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


print("=" * 84)
print("正确评估（归一化空间，sd=1，无单位混用）")
print("=" * 84)

results = {}
for split in ["train", "test"]:
    ws = mkw(splits[split])
    b = baselines(ws)
    b["GRU"] = rel_norm(nn_pred(gru, ws, "gru"), ws)
    b["RSSM"] = rel_norm(nn_pred(rssm, ws, "rssm"), ws)
    results[split] = {k: v.tolist() for k, v in b.items()}

    print(f"\n[{split}]  {len(ws)} 窗口")
    print(f"  {'模型':<14}{'mean':>10}{'k=1':>10}{'k=8':>10}{'k=16':>10}"
          f"{'增长':>9}")
    print("  " + "-" * 60)
    for nm in ["Zero", "Persistence", "ARX", "GRU", "RSSM"]:
        a = b[nm]
        print(f"  {nm:<14}{a.mean():>10.4f}{a[0]:>10.4f}{a[7]:>10.4f}"
              f"{a[15]:>10.4f}{a[15]/max(a[0],1e-12):>9.2f}x")

print()
print("=" * 84)
print("核心结论")
print("=" * 84)
tr, te = results["train"], results["test"]
for nm in ["Persistence", "ARX", "GRU", "RSSM"]:
    trm = np.mean(tr[nm])
    tem = np.mean(te[nm])
    print(f"  {nm:<14} 训练 {trm:.4f}  测试 {tem:.4f}  "
          f"泛化比 {tem/trm:>6.2f}x")
print()
best_tr = min(np.mean(tr[nm]) for nm in ["ARX", "GRU", "RSSM"])
print(f"  训练集最优: {best_tr:.4f}")
for nm in ["ARX", "GRU", "RSSM"]:
    if abs(np.mean(tr[nm]) - best_tr) < 1e-9:
        print(f"    -> {nm}")
print(f"  测试集最优: {min((np.mean(te[nm]), nm) for nm in ['ARX','GRU','RSSM'])}")

json.dump(results, open(T / "eval_correct.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"\n保存 -> {T/'eval_correct.json'}")
