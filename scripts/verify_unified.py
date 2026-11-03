"""统一核验：v1 与 v2 在同一通道集、同一指标下重新评估

发现问题：
  - v1 (eval_fixed.py) 用全部 16 通道，其中 S4_FAR 的 std 极小(0.0056)
    导致相对误差被放大到 39.7，ARX 的 346 主要来自这一个通道
  - v2 (eval_v2_ckpt.py) 用剔除后的 11 通道
  => 两者不可直接比较

本脚本对两版检查点都在 11 通道上重算，给出可比数字。
"""
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]


def evaluate_version(data_dir, ckpt_dir, step, window, horizon,
                     gru_kw, tag):
    print("=" * 88)
    print(f"{tag}")
    print("=" * 88)
    d = np.load(data_dir / "data.npz", allow_pickle=True)
    OBS_ALL = [str(c) for c in d["obs_cols"]]
    splits = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))
    runs = {str(r): {"obs": d[f"{r}_obs"][::step].astype(np.float32),
                     "act": d[f"{r}_act"][::step].astype(np.float32)}
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

    print(f"  数据: {data_dir}")
    print(f"  轨迹: train={len(splits['train'])} val={len(splits['val'])} "
          f"test={len(splits['test'])}")
    print(f"  采样: step={step}  dt={step*0.015:.3f}s  "
          f"window={window}  horizon={horizon} ({horizon*step*0.015:.2f}s)")
    print(f"  通道: 全部 {C_O}，评估用 {len(KEEP)}")

    def mkw(ids, stride):
        W = []
        for r in ids:
            o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float32)
            a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float32)
            for st in range(1, len(o) - window - horizon, stride):
                W.append({"obs": o[st:st + window],
                          "act": a[st - 1:st - 1 + window],
                          "fut": o[st + window:st + window + horizon],
                          "act_fut": a[st - 1 + window:
                                        st - 1 + window + horizon]})
        return W

    tr_W = mkw(splits["train"], 8)
    te_W = mkw(splits["test"], 8)
    print(f"  窗口: train={len(tr_W)} test={len(te_W)}")

    def rel(preds, ws):
        e = np.stack([p - w["fut"] for p, w in zip(preds, ws)], 0)
        return np.sqrt((e ** 2).mean(0))[:, KIDX].mean(1)

    # 基线
    pers_tr = rel([np.repeat(w["obs"][-1:], horizon, 0) for w in tr_W], tr_W)
    pers_te = rel([np.repeat(w["obs"][-1:], horizon, 0) for w in te_W], te_W)

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

    def arx(ws):
        out = []
        for w in ws:
            y = w["obs"][-1].astype(np.float64).copy()
            oo = []
            for k in range(horizon):
                y = y @ A.T + w["act_fut"][k].astype(np.float64) @ B.T + c0
                oo.append(y.copy())
            out.append(np.stack(oo, 0))
        return out

    # 模型
    gru = BaselineGRU(C_O, 1, horizon=horizon, **gru_kw).to(DEV)
    gru.load_state_dict(torch.load(ckpt_dir / "gru.pt", map_location=DEV,
                                   weights_only=True))
    gru.eval()
    rssm = WorldModel(C_O, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                         hidden=256)).to(DEV)
    rssm.load_state_dict(torch.load(ckpt_dir / "rssm.pt", map_location=DEV,
                                    weights_only=True))
    rssm.eval()

    @torch.no_grad()
    def pred(model, ws, kind):
        out = []
        for i in range(0, len(ws), 64):
            c = ws[i:i + 64]
            o = torch.from_numpy(np.stack([w["obs"] for w in c])).to(DEV)
            a = torch.from_numpy(np.stack([w["act"] for w in c])).to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).to(DEV)
            p = model(o, a, af) if kind == "gru" else model.imagine(o, a, af)
            out.append(p.cpu().numpy())
        return np.concatenate(out, 0)

    n_gru = sum(p.numel() for p in gru.parameters())
    n_rssm = sum(p.numel() for p in rssm.parameters())

    R = {
        "Persistence": (pers_tr, pers_te),
        "ARX": (rel(arx(tr_W), tr_W), rel(arx(te_W), te_W)),
        "GRU": (rel(pred(gru, tr_W, "gru"), tr_W),
                rel(pred(gru, te_W, "gru"), te_W)),
        "RSSM": (rel(pred(rssm, tr_W, "rssm"), tr_W),
                 rel(pred(rssm, te_W, "rssm"), te_W)),
    }

    print()
    print(f"  {'模型':<14}{'训练':>10}{'测试':>10}{'泛化比':>9}"
          f"{'k=1':>9}{'k=H':>9}{'参数量':>12}")
    print("  " + "-" * 72)
    out = {"meta": {"data": str(data_dir), "ckpt": str(ckpt_dir),
                    "step": step, "window": window, "horizon": horizon,
                    "dt_s": step * 0.015, "keep_channels": KEEP,
                    "n_train_traj": len(splits["train"]),
                    "n_test_traj": len(splits["test"]),
                    "n_train_win": len(tr_W), "n_test_win": len(te_W),
                    "params": {"gru": n_gru, "rssm": n_rssm}}}
    for nm, (a, b) in R.items():
        np_ = n_gru if nm == "GRU" else (n_rssm if nm == "RSSM" else
                                         (C_O * (C_O + 2) if nm == "ARX" else 0))
        print(f"  {nm:<14}{a.mean():>10.4f}{b.mean():>10.4f}"
              f"{b.mean()/a.mean():>9.1f}x{b[0]:>9.4f}{b[-1]:>9.4f}"
              f"{np_:>12,}")
        out[nm] = {"train_mean": float(a.mean()),
                   "test_mean": float(b.mean()),
                   "test_curve": b.tolist(),
                   "train_curve": a.tolist()}
    return out


v1 = evaluate_version(
    Path(r"C:\pclwm\agtf30\prepared"), Path(r"C:\pclwm\agtf30\train"),
    7, 64, 16,
    {"hidden": 256, "mlp_units": 512, "mlp_layers": 2},
    "v1：dataset1（10 条轨迹），H=16")

print()
v2 = evaluate_version(
    Path(r"C:\pclwm\agtf30\prepared2"), Path(r"C:\pclwm\agtf30\train2"),
    7, 64, 32,
    {"hidden": 192, "mlp_units": 512, "mlp_layers": 2},
    "v2：dataset2（30 条轨迹），H=32")

print()
print("=" * 88)
print("核验结论")
print("=" * 88)
print(f"  {'版本':<8}{'数据':<16}{'GRU':>10}{'RSSM':>10}{'ARX':>10}"
      f"{'Persist':>10}")
print("  " + "-" * 64)
for nm, r in [("v1", v1), ("v2", v2)]:
    print(f"  {nm:<8}{'10/30 条':<16}"
          f"{r['GRU']['test_mean']:>10.4f}{r['RSSM']['test_mean']:>10.4f}"
          f"{r['ARX']['test_mean']:>10.4f}"
          f"{r['Persistence']['test_mean']:>10.4f}")

json.dump({"v1": v1, "v2": v2},
          open(Path(r"C:\pclwm\agtf30\verify_unified.json"), "w"),
          ensure_ascii=False, indent=1)
print(f"\n保存 -> C:\\pclwm\\agtf30\\verify_unified.json")
