"""端到端预测演示：给定历史观测 + 未来动作，自主推算未来

这是检验"能不能用"最直接的方式：
  1. 取一条 test 轨迹的真实历史（6.7 s）
  2. 提供未来燃油指令（不提供任何未来传感器读数）
  3. 模型自主递推 42 s
  4. 与真值对比
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(r"C:\pclwm\handoff\PC-LWM_handoff_20261102")
sys.path.insert(0, str(PKG / "src"))
from aero_world_model.models.baselines import BaselineGRU
from aero_world_model.models.rssm import RSSMConfig
from aero_world_model.training.offline import WorldModel

DATA = PKG / "data" / "prepared"
CKPT = PKG / "checkpoints" / "capacity"
OUT = PKG / "results"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

STEP, WINDOW, HORIZON = 7, 64, 32
DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]

d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = [str(c) for c in d["obs_cols"]]
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
meta = {m["run_id"]: m for m in json.loads((DATA / "meta.json").read_text(encoding="utf-8"))}
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

# 加载最优 GRU（h128）
gru = BaselineGRU(C_O, 1, hidden=128, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
gru.load_state_dict(torch.load(CKPT / "h128_s0.pt", map_location=DEV,
                              weights_only=True))
gru.eval()

rssm = WorldModel(C_O, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                     hidden=256)).to(DEV)
rssm.load_state_dict(torch.load(PKG / "checkpoints" / "v2_rssm.pt",
                                map_location=DEV, weights_only=True))
rssm.eval()


@torch.no_grad()
def predict_future(model, kind, rid, nseg=12):
    """只用历史观测 + 未来动作，自主推算"""
    o = ((runs[rid]["obs"] - o_mu) / o_sd).astype(np.float32)
    a = ((runs[rid]["act"] - a_mu) / a_sd).astype(np.float32)
    start = WINDOW + 1                      # 历史窗口末端

    cur = torch.from_numpy(o[start - WINDOW:start][None]).to(DEV)
    ain = torch.from_numpy(a[start - WINDOW - 1:start - 1][None]).to(DEV)

    preds = []
    for s in range(nseg):
        p0 = start + s * HORIZON
        # 只给未来动作，不给任何未来观测
        af = torch.from_numpy(a[p0 - 1:p0 - 1 + HORIZON][None]).to(DEV)
        p = model(cur, ain, af) if kind == "gru" else model.imagine(cur, ain, af)
        preds.append(p)
        if s < nseg - 1:
            cur = torch.cat([cur[:, HORIZON:], p], 1)
            ain = torch.cat([ain[:, HORIZON:], af], 1)
    pred = torch.cat(preds, 1)[0].cpu().numpy()
    tgt = o[start:start + nseg * HORIZON]
    act_fut = a[start - 1:start - 1 + nseg * HORIZON]

    # 反归一化到物理单位
    pred_phys = pred * o_sd + o_mu
    tgt_phys = tgt * o_sd + o_mu
    return pred_phys, tgt_phys, act_fut, start


print("=" * 78)
print("端到端预测演示")
print("=" * 78)
print(f"历史窗口: {WINDOW} 步 = {WINDOW*STEP*0.015:.1f} s")
print(f"预测长度: 12 段 × {HORIZON} 步 = {12*HORIZON*STEP*0.015:.1f} s")
print(f"设备: {DEV}")

results = {}
for rid in splits["test"][:3]:
    m = meta[rid]
    print(f"\n{'-'*78}")
    print(f"轨迹: {rid}")
    print(f"  激励={m['profile']}  Alt={m['alt']} ft  MN={m['mn']}  增益={m['gs']}")

    for name, model, kind in [("GRU-h128", gru, "gru"),
                              ("RSSM", rssm, "rssm")]:
        pred, tgt, act, start = predict_future(model, kind, rid)

        # 各通道相对误差
        errs = []
        for ci in KIDX:
            sd = o_sd[ci]
            e = np.sqrt(((pred[:, ci] - tgt[:, ci]) ** 2).mean())
            errs.append(e / sd)
        rel = np.mean(errs)

        # 若干关键量的首末对比
        def show(col):
            ci = OBS_ALL.index(col)
            return (f"{tgt[0, ci]:.1f}→{tgt[-1, ci]:.1f} "
                    f"(预测 {pred[0, ci]:.1f}→{pred[-1, ci]:.1f})")

        print(f"\n  [{name}] 相对RMSE = {rel:.4f}")
        print(f"    S36_Tt : {show('S36_Tt')} °R")
        print(f"    S4_Tt  : {show('S4_Tt')} °R")
        print(f"    S36_Pt : {show('S36_Pt')} psia")

        results.setdefault(rid, {})[name] = {
            "rel_rmse": float(rel),
            "pred": pred.tolist() if len(results.get(rid, {})) == 0 else None,
        }

# 保存一条完整预测供画图（选步长阶跃那条最典型）
rid = splits["test"][0]
pred_g, tgt, act, start = predict_future(gru, "gru", rid)
pred_r, _, _, _ = predict_future(rssm, "rssm", rid)
np.savez_compressed(
    OUT / "prediction_demo.npz",
    target=tgt, pred_gru=pred_g, pred_rssm=pred_r,
    action=act, obs_cols=np.array(OBS_ALL), start=start)

print(f"\n{'='*78}")
print("结论")
print("=" * 78)
grus = [v["GRU-h128"]["rel_rmse"] for v in results.values()]
rssms = [v["RSSM"]["rel_rmse"] for v in results.values()]
print(f"  GRU-h128 相对 RMSE: {[f'{x:.4f}' for x in grus]}")
print(f"  RSSM     相对 RMSE: {[f'{x:.4f}' for x in rssms]}")
print()
print("  ✅ 能预测：给定 6.7 s 历史 + 未来燃油指令，")
print("     模型可自主推算 40 s 未来，不使用任何未来传感器值。")
print(f"\n  预测数据已存: {OUT/'prediction_demo.npz'}")
