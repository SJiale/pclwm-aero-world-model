"""长时评估：修正动作时序 + 验证

## 动作时序分析（关键）

训练约定（train_v2.py）：
    obs  = o[st   : st+WINDOW]
    act  = a[st-1 : st-1+WINDOW]
    fut  = o[st+WINDOW : st+WINDOW+HORIZON]
    af   = a[st-1+WINDOW : st-1+WINDOW+HORIZON]

即 obs 窗口起点 st 对应 act 窗口起点 st-1。因果：a[j] -> o[j+1]

之前 eval_package.py 的错误：
    cur = o[start-WINDOW : start]        # obs 起点 = start-WINDOW
    ain = a[start-WINDOW : start]        # 错！应该从 start-WINDOW-1 开始
    af  = a[start : start+HORIZON]       # 错！应该从 start-1 开始

差 1 步。本脚本修正并验证。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

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


def norm_obs(rid):
    return ((runs[rid]["obs"] - o_mu) / o_sd).astype(np.float32)


def norm_act(rid):
    return ((runs[rid]["act"] - a_mu) / a_sd).astype(np.float32)


@torch.no_grad()
def long_rollout_fixed(model, kind, rid, nseg):
    """从 rid 的一条轨迹做 nseg 段自回归，动作时序严格按训练约定

    第 s 段:
        obs 窗口 = o[start + s*H - WINDOW : start + s*H]
        act 窗口 = a[start + s*H - WINDOW - 1 : start + s*H - 1]
        af       = a[start + s*H - 1 : start + s*H - 1 + H]
        预测     = o[start + s*H : start + s*H + H]
    """
    o = norm_obs(rid)
    a = norm_act(rid)
    start = WINDOW + 1                      # 保证 start-WINDOW-1 >= 0
    need = start + nseg * HORIZON
    if len(o) <= need:
        return None, None

    cur = torch.from_numpy(o[start - WINDOW:start][None]).to(DEV)
    ain = torch.from_numpy(a[start - WINDOW - 1:start - 1][None]).to(DEV)
    assert cur.shape[1] == WINDOW and ain.shape[1] == WINDOW

    preds = []
    for s in range(nseg):
        p0 = start + s * HORIZON
        af = torch.from_numpy(a[p0 - 1:p0 - 1 + HORIZON][None]).to(DEV)
        if kind == "gru":
            p = model(cur, ain, af)
        else:
            p = model.imagine(cur, ain, af)
        preds.append(p)
        if s < nseg - 1:
            # 滚动：obs 窗口右移 H，act 窗口同步右移 H
            cur = torch.cat([cur[:, HORIZON:], p], 1)
            ain = torch.cat([ain[:, HORIZON:], af], 1)
    pred = torch.cat(preds, 1)[0].cpu().numpy()
    tgt = o[start:start + nseg * HORIZON]
    return pred, tgt


def per_step_rmse(pred, tgt):
    """返回 [T] 每步跨通道平均相对误差（归一化空间）"""
    e = (pred - tgt)[:, KIDX]
    return np.sqrt((e ** 2).mean(1))


def build_gru(hidden):
    m = BaselineGRU(C_O, 1, hidden=hidden, horizon=HORIZON,
                    mlp_units=512, mlp_layers=2).to(DEV)
    return m


print("=" * 84)
print("长时评估（修正动作时序）")
print("=" * 84)

# 加载 v2 检查点
gru = build_gru(192)
gru.load_state_dict(torch.load(CKPT / "v2_gru.pt", map_location=DEV,
                              weights_only=True))
gru.eval()
rssm = WorldModel(C_O, 1, RSSMConfig(stoch=32, classes=32, deter=256,
                                     hidden=256)).to(DEV)
rssm.load_state_dict(torch.load(CKPT / "v2_rssm.pt", map_location=DEV,
                                weights_only=True))
rssm.eval()

NSEG = 12          # 12 × 32 = 384 步 = 40.3 s
out = {"config": {"step": STEP, "window": WINDOW, "horizon": HORIZON,
                  "nseg": NSEG, "steps": NSEG * HORIZON,
                  "dt_s": STEP * 0.015,
                  "time_s": NSEG * HORIZON * STEP * 0.015,
                  "action_timing": "a[t-1] -> o[t] (training-consistent)"},
       "per_traj": {}}

for rid in splits["test"]:
    o = norm_obs(rid)
    a = norm_act(rid)
    start = WINDOW + 1
    nseg_use = min(NSEG, (len(o) - start) // HORIZON)
    if nseg_use < 1:
        print(f"  skip {rid}: 太短")
        continue

    rec = {"nseg": nseg_use,
           "steps": nseg_use * HORIZON,
           "time_s": nseg_use * HORIZON * STEP * 0.015}

    # GRU
    p, t = long_rollout_fixed(gru, "gru", rid, nseg_use)
    if p is not None:
        r = per_step_rmse(p, t)
        rec["GRU"] = {"mean": float(r.mean()), "k1": float(r[0]),
                      "kH": float(r[-1]), "curve": r.tolist()}
    # RSSM
    p, t = long_rollout_fixed(rssm, "rssm", rid, nseg_use)
    if p is not None:
        r = per_step_rmse(p, t)
        rec["RSSM"] = {"mean": float(r.mean()), "k1": float(r[0]),
                       "kH": float(r[-1]), "curve": r.tolist()}
    # Persistence
    p0 = o[start - 1]                      # 预测时保持第一帧
    T = nseg_use * HORIZON
    pred_p = np.repeat(p0[None], T, 0)
    tgt_p = o[start:start + T]
    r = per_step_rmse(pred_p, tgt_p)
    rec["Persistence"] = {"mean": float(r.mean()), "k1": float(r[0]),
                          "kH": float(r[-1]), "curve": r.tolist()}

    out["per_traj"][rid] = rec
    print(f"  {rid:<34} nseg={nseg_use:<3} "
          f"GRU={rec.get('GRU',{}).get('mean',float('nan')):.4f}  "
          f"RSSM={rec.get('RSSM',{}).get('mean',float('nan')):.4f}  "
          f"Pers={rec['Persistence']['mean']:.4f}")

# 汇总
print()
print("=" * 84)
print("汇总（10 条 test 轨迹平均）")
print("=" * 84)
agg = {}
for m in ["GRU", "RSSM", "Persistence"]:
    curves = [v[m]["curve"] for v in out["per_traj"].values() if m in v]
    if not curves:
        continue
    L = min(len(c) for c in curves)
    M = np.stack([c[:L] for c in curves], 0)
    mean_c = M.mean(0)
    agg[m] = {"mean": float(mean_c.mean()),
              "k1": float(mean_c[0]),
              "kH": float(mean_c[-1]),
              "drift": float(mean_c[-1] / max(mean_c[0], 1e-12)),
              "curve": mean_c.tolist(),
              "per_traj_std": float(M[:, -1].std())}
    print(f"  {m:<14} k1={mean_c[0]:.4f}  k={L}:{mean_c[-1]:.4f}  "
          f"漂移={mean_c[-1]/max(mean_c[0],1e-12):.2f}x  均值={mean_c.mean():.4f}")

out["aggregate"] = agg
(RES / "long_eval_fixed.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n保存 -> {RES/'long_eval_fixed.json'}")
