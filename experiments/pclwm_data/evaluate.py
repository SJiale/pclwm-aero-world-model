"""阶段 2：统一评价框架

关键修正（相对旧版）：
  1. 修复 ARX 数据泄漏：旧版用整条轨迹算归一化参数并拟合，
     现在只用 train 轨迹拟合
  2. 统一历史长度 / 预测时域 / 递推方式
  3. 提供三类评估：
     - 短期多步（H=32 步 = 3.36 s）
     - 长时自由递推（10/30/60 s）
     - 逐轨迹与逐通道分解
  4. 物理一致性检查（温度序）
"""
from pathlib import Path
import json
import numpy as np

DATA = Path(r"C:\pclwm\pclwm_data")
DT = 0.105
WINDOW = 32        # 历史 3.36 s（优化：从 64 降低，速度×2.4）
HORIZON = 32       # 短期 3.36 s

# 长时递推目标（秒）
LONG_TARGETS = [10, 30, 60]


class UnifiedData:
    """统一数据接口"""

    def __init__(self, data_dir=DATA):
        d = np.load(data_dir / "prepared.npz", allow_pickle=True)
        n = np.load(data_dir / "norm.npz", allow_pickle=True)
        sp = json.loads((data_dir / "splits.json").read_text(encoding="utf-8"))

        self.obs_cols = [str(c) for c in d["obs_cols"]]
        self.act_cols = [str(c) for c in d["act_cols"]]
        self.aux_cols = [str(c) for c in d["aux_cols"]]
        self.splits = sp["splits"]
        self.step = sp["step"]
        self.dt = sp["dt"]
        self.fs = sp["fs"]

        self.run_ids = [str(r) for r in d["run_ids"]]
        self.runs = {}
        for r in self.run_ids:
            self.runs[r] = {
                "obs": d[f"{r}_obs"].astype(np.float32),
                "act": d[f"{r}_act"].astype(np.float32),
                "aux": d[f"{r}_aux"].astype(np.float32),
                "time": d[f"{r}_time"].astype(np.float64),
            }

        self.obs_mu = n["obs_mu"]
        self.obs_sd = n["obs_sd"]
        self.act_mu = n["act_mu"]
        self.act_sd = n["act_sd"]

    @property
    def n_obs(self):
        return len(self.obs_cols)

    @property
    def n_act(self):
        return len(self.act_cols)

    def norm_obs(self, rid):
        return (self.runs[rid]["obs"] - self.obs_mu) / self.obs_sd

    def norm_act(self, rid):
        return (self.runs[rid]["act"] - self.act_mu) / self.act_sd

    def denorm_obs(self, x):
        return x * self.obs_sd + self.obs_mu

    def windows(self, split, stride=4, horizon=HORIZON):
        """构造滑窗（严格按轨迹，不跨轨迹）"""
        ids = self.splits[split] if isinstance(split, str) else split
        W = []
        for r in ids:
            o = self.norm_obs(r).astype(np.float32)
            a = self.norm_act(r).astype(np.float32)
            t = self.runs[r]["time"]
            n = len(o)
            for st in range(1, n - WINDOW - horizon, stride):
                W.append({
                    "obs": o[st:st+WINDOW],
                    "act": a[st-1:st-1+WINDOW],
                    "fut": o[st+WINDOW:st+WINDOW+horizon],
                    "act_fut": a[st-1+WINDOW:st-1+WINDOW+horizon],
                    "rid": r,
                    "t0": float(t[st+WINDOW]),
                })
        return W


class UnifiedEval:
    """统一评价器

    指标定义（明确记录，避免歧义）：
      - 每 (窗口, 时间步)：对通道求 RMSE（物理单位）
      - 每窗口：对时间步平均
      - 总体：对所有窗口平均
      - 相对化：除以各通道物理 std 的平均
    """

    def __init__(self, data: UnifiedData):
        self.d = data
        self.ref_std = float(data.obs_sd.mean())

    def evaluate(self, preds, windows, horizon=None):
        """preds: [N, H, C] 归一化空间"""
        P = np.asarray(preds, np.float64)
        F = np.stack([w["fut"] for w in windows], 0).astype(np.float64)
        assert P.shape == F.shape, f"{P.shape} vs {F.shape}"

        # 反归一化到物理单位
        E = (P - F) * self.d.obs_sd[None, None, :]

        per_step = np.sqrt((E ** 2).mean(2))        # [N, H]
        per_win = per_step.mean(1)                   # [N]

        res = {
            "overall_abs": float(per_win.mean()),
            "overall_rel": float(per_win.mean() / self.ref_std),
            "k1_rel": float(per_step[:, 0].mean() / self.ref_std),
            "kH_rel": float(per_step[:, -1].mean() / self.ref_std),
            "curve_abs": per_step.mean(0).tolist(),
            "curve_rel": (per_step.mean(0) / self.ref_std).tolist(),
            "n_windows": len(windows),
            "ref_std": self.ref_std,
        }

        # 逐通道
        ch = {}
        for i, c in enumerate(self.d.obs_cols):
            ec = np.sqrt((E[:, :, i] ** 2).mean())
            ch[c] = {"rmse_abs": float(ec),
                     "rmse_rel": float(ec / self.d.obs_sd[i])}
        res["per_channel"] = ch

        # 逐轨迹
        per = {}
        rids = np.array([w["rid"] for w in windows])
        for r in sorted(set(rids)):
            m = rids == r
            per[r] = {"n_win": int(m.sum()),
                      "rel": float(per_win[m].mean() / self.ref_std)}
        res["per_traj"] = per

        return res


def rollout_far(model_fn, data: UnifiedData, rid, t_target_s,
                horizon=HORIZON):
    """长时自由递推：从 rid 起点开始，用模型自己的预测滚动

    model_fn(obs_win, act_win, act_fut) -> [H, C]（归一化空间）
    返回每步相对误差曲线
    """
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    nseg = int(round(t_target_s / (horizon * data.dt)))
    if nseg < 1:
        return None
    start = WINDOW + 1
    need = start + nseg * horizon
    if len(o) <= need:
        nseg = (len(o) - start - 1) // horizon
        if nseg < 1:
            return None
    need = start + nseg * horizon

    cur = o[start-WINDOW:start][None].copy()
    ain = a[start-WINDOW-1:start-1][None].copy()

    preds, tgts = [], []
    for s in range(nseg):
        p0 = start + s * horizon
        af = a[p0-1:p0-1+horizon][None]
        p = model_fn(cur, ain, af)                    # [1,H,C]
        preds.append(p[0])
        tgts.append(o[p0:p0+horizon])
        if s < nseg - 1:
            cur = np.concatenate([cur[:, horizon:], p], 1)
            ain = np.concatenate([ain[:, horizon:], af], 1)

    P = np.stack(preds, 0)     # [nseg,H,C]
    T = np.stack(tgts, 0)
    E = (P - T) * data.obs_sd[None, None, :]
    per_step = np.sqrt((E**2).mean(2))     # [nseg,H]
    mode = "free_rollout"
    return {
        "mode": mode,
        "nseg": nseg,
        "steps": nseg * horizon,
        "seconds": nseg * horizon * data.dt,
        "curve_rel": (per_step.reshape(-1) / data.obs_sd.mean()).tolist(),
        "final_rel": float(per_step[-1].mean() / data.obs_sd.mean()),
        "mean_rel": float(per_step.mean() / data.obs_sd.mean()),
    }


def rollout_teacher_oracle(model_fn, data: UnifiedData, rid, t_target_s,
                           horizon=HORIZON):
    """对照：每段都用真值历史（teacher forcing，非自由递推）

    用于区分"模型本身不准" vs "误差累积"。
    """
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    nseg = int(round(t_target_s / (horizon * data.dt)))
    start = WINDOW + 1
    if len(o) <= start + nseg * horizon + 1:
        nseg = (len(o) - start - 1) // horizon
        if nseg < 1:
            return None

    preds, tgts = [], []
    for s in range(nseg):
        p0 = start + s * horizon
        cur = o[p0-WINDOW:p0][None]
        ain = a[p0-WINDOW-1:p0-1][None]
        af = a[p0-1:p0-1+horizon][None]
        p = model_fn(cur, ain, af)
        preds.append(p[0])
        tgts.append(o[p0:p0+horizon])

    P = np.stack(preds, 0); T = np.stack(tgts, 0)
    E = (P - T) * data.obs_sd[None, None, :]
    per_step = np.sqrt((E**2).mean(2))
    return {
        "mode": "teacher_forcing",
        "nseg": nseg,
        "curve_rel": (per_step.reshape(-1) / data.obs_sd.mean()).tolist(),
        "final_rel": float(per_step[-1].mean() / data.obs_sd.mean()),
        "mean_rel": float(per_step.mean() / data.obs_sd.mean()),
    }


def check_physics(data: UnifiedData, pred_phys):
    """物理一致性：T4 > T45（索引 2 > 3）"""
    i4 = data.obs_cols.index("S4_Tt")
    i45 = data.obs_cols.index("S45_Tt")
    viol = int((pred_phys[:, :, i4] <= pred_phys[:, :, i45]).sum())
    total = pred_phys.shape[0] * pred_phys.shape[1]
    return {"violations": viol, "total": total,
            "rate": float(viol / max(total, 1))}


if __name__ == "__main__":
    print("=" * 78)
    print("阶段 2：统一评价框架自检")
    print("=" * 78)
    d = UnifiedData()
    print(f"  轨迹: {len(d.runs)}, 观测 {d.n_obs} 通道, 动作 {d.n_act} 通道")
    print(f"  dt={d.dt:.3f}s, fs={d.fs:.2f}Hz")
    print(f"  obs cols: {d.obs_cols}")
    print(f"  act cols: {d.act_cols}")

    ev = UnifiedEval(d)
    te = d.windows("test_id", stride=8)
    te_ood = d.windows("test_ood", stride=8)
    print(f"\n  test_id 窗口: {len(te)}, test_ood 窗口: {len(te_ood)}")

    # 基线：Zero / Persistence
    for nm, fn in [
        ("Zero", lambda ws: np.zeros((len(ws), HORIZON, d.n_obs))),
        ("Persistence", lambda ws: np.stack(
            [np.repeat(w["obs"][-1:], HORIZON, 0) for w in ws], 0)),
    ]:
        p = fn(te)
        r = ev.evaluate(p, te)
        print(f"\n  [{nm}] overall_rel={r['overall_rel']:.4f}  "
              f"k1={r['k1_rel']:.4f}  kH={r['kH_rel']:.4f}")

    # 长时递推自检（用 persistence）
    def pers_fn(obs, act, af):
        B, H = act.shape[0], af.shape[1]
        return np.repeat(obs[:, -1:], H, 1)

    print(f"\n  长时递推自检（Persistence, test_id 首条轨迹）:")
    rid = d.splits["test_id"][0]
    for tgt in LONG_TARGETS:
        r = rollout_far(pers_fn, d, rid, tgt)
        if r:
            print(f"    {tgt:>2}s: {r['steps']} 步, "
                  f"mean_rel={r['mean_rel']:.4f}, "
                  f"final_rel={r['final_rel']:.4f}")
        else:
            print(f"    {tgt:>2}s: 数据不足")

    print(f"\n  统一评价框架就绪")

