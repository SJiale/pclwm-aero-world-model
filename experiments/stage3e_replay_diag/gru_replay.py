"""任务2：GRU 动作回放预测

由于 B2 缺少 S4_Tt / S36_Pt 两通道，无法直接回放。
改用 test_ctrl V4/V6（完整 5 通道 + PLA 阶跃）做 GRU 预测能力诊断。

严格协议：
  - 历史窗口：只用起始时刻之前的真实观测
  - 未来动作：用真实的 PLA 执行序列（离线诊断，非在线决策）
  - 预测 1/10/32 步
  - PyTorch 加载 strict=True
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, WINDOW, HORIZON, DT

OUT = Path(r"C:\pclwm\pclwm_stage3a\replay")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"
SEED = 777

data = UnifiedData()
obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()
act_mu = np.asarray(data.act_mu).ravel()
act_sd = np.asarray(data.act_sd).ravel()
OBS_COLS = list(data.obs_cols)

print("=" * 92)
print("任务 2：GRU 动作回放预测")
print("=" * 92)
print(f"  WINDOW={WINDOW} HORIZON={HORIZON} DT={DT}")
print(f"  obs_cols={OBS_COLS}")


class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=HORIZON,
                 mlp=256, layers=2):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Sequential(
            nn.Linear(c_obs + c_act, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        mods = [nn.Linear(hidden, mlp), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(mlp, mlp), nn.SiLU()]
        mods += [nn.Linear(mlp, c_obs)]
        self.dec = nn.Sequential(*mods)
        self.o2h = nn.Linear(c_obs + c_act, hidden)

    def forward(self, obs, act, af):
        B, T, _ = obs.shape
        Hh = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(Hh):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


sd = torch.load(r"C:\pclwm\pclwm_experiments\checkpoints\GRU-h128.pt",
                map_location=DEV, weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
GRU = GRUModel(5, 3).to(DEV)
GRU.load_state_dict(sd, strict=True)
GRU.eval()
print(f"  ✅ GRU 加载 strict=True, {sum(p.numel() for p in GRU.parameters()):,} 参数")

# ══════════════════════════════════════════════════
# 载入 test_ctrl（完整 5 通道）
# ══════════════════════════════════════════════════
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7

TC = {}
for f in sorted(CTRL.glob("V4*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    TC[f.stem] = {
        "obs_raw": d[OBS_COLS].to_numpy(np.float64),
        "act_raw": d[["PLA_exec_deg", "Alt_ft", "MN"]].to_numpy(np.float64),
        "t": d["time_s"].to_numpy(np.float64),
        "pla": d["PLA_exec_deg"].to_numpy(np.float64),
    }

for f in sorted(CTRL.glob("V6*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    TC[f.stem] = {
        "obs_raw": d[OBS_COLS].to_numpy(np.float64),
        "act_raw": d[["PLA_exec_deg", "Alt_ft", "MN"]].to_numpy(np.float64),
        "t": d["time_s"].to_numpy(np.float64),
        "pla": d["PLA_exec_deg"].to_numpy(np.float64),
    }

print(f"\n  载入 {len(TC)} 条轨迹")
for k, v in TC.items():
    print(f"    {k}: {len(v['t'])} 步")

# ══════════════════════════════════════════════════
# 找 PLA 阶跃时刻，作为回放起点
# ══════════════════════════════════════════════════
i_n2 = OBS_COLS.index("N2int")

print("\n" + "=" * 92)
print("[1] 阶跃事件检测与回放")
print("=" * 92)

RESULTS = []
for name, tc in TC.items():
    pla = tc["pla"]
    dp = np.diff(pla)
    jumps = np.where(np.abs(dp) > 0.5)[0]
    if len(jumps) == 0:
        continue

    print(f"\n  ── {name} ──")
    for j in jumps[:3]:
        i_ev = j + 1          # 阶跃后的第一步
        if i_ev < WINDOW or i_ev + HORIZON >= len(tc["t"]):
            continue

        # 历史窗口（只用起始之前的真实观测）
        obs_h = tc["obs_raw"][i_ev - WINDOW:i_ev]
        act_h = tc["act_raw"][i_ev - WINDOW:i_ev]
        # 未来动作（真实执行序列）
        act_f = tc["act_raw"][i_ev:i_ev + HORIZON]
        # 真值
        obs_true = tc["obs_raw"][i_ev:i_ev + HORIZON]

        # 归一化
        obs_hz = (obs_h - obs_mu) / obs_sd
        act_hz = (act_h - act_mu) / act_sd
        act_fz = (act_f - act_mu) / act_sd

        o = torch.from_numpy(obs_hz[None]).float().to(DEV)
        a = torch.from_numpy(act_hz[None]).float().to(DEV)
        af = torch.from_numpy(act_fz[None]).float().to(DEV)

        with torch.no_grad():
            torch.manual_seed(SEED)
            p = GRU(o, a, af)
        pred = p[0].cpu().numpy() * obs_sd + obs_mu   # 物理单位

        # N2 对比
        n2_pred = pred[:, i_n2]
        n2_true = obs_true[:, i_n2]
        n2_0 = obs_h[-1, i_n2]

        e = n2_pred - n2_true
        print(f"    阶跃 t={tc['t'][i_ev]:.2f}s  PLA {pla[j]:.2f}→{pla[i_ev]:.2f} "
              f"(Δ={dp[j]:+.2f})")
        print(f"      N2 起点 = {n2_0:.2f}")
        print(f"      {'步':>4}{'预测':>10}{'真值':>10}{'误差':>10}"
              f"{'Δpred':>9}{'Δtrue':>9}")
        for kk in [0, 9, 31]:
            print(f"      {kk+1:>4}{n2_pred[kk]:>10.2f}{n2_true[kk]:>10.2f}"
                  f"{e[kk]:>+10.2f}{n2_pred[kk]-n2_0:>+9.2f}"
                  f"{n2_true[kk]-n2_0:>+9.2f}")

        RESULTS.append({
            "traj": name, "t_event": float(tc["t"][i_ev]),
            "dpla": float(dp[j]), "n2_start": float(n2_0),
            "pred_1": float(n2_pred[0]), "true_1": float(n2_true[0]),
            "pred_10": float(n2_pred[9]), "true_10": float(n2_true[9]),
            "pred_32": float(n2_pred[31]), "true_32": float(n2_true[31]),
            "err_1": float(e[0]), "err_10": float(e[9]), "err_32": float(e[31]),
            "dn2_pred_32": float(n2_pred[31]-n2_0),
            "dn2_true_32": float(n2_true[31]-n2_0),
            "pred_curve": n2_pred.tolist(),
            "true_curve": n2_true.tolist(),
            "pla_curve": tc["pla"][i_ev:i_ev+HORIZON].tolist(),
        })

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[2] 汇总：预测误差 vs 预测时域")
print("=" * 92)

if RESULTS:
    print(f"\n  {'轨迹':<22}{'ΔPLA':>8}{'步1误差':>11}{'步10误差':>11}{'步32误差':>11}")
    print("  " + "-" * 64)
    for r in RESULTS:
        print(f"  {r['traj']:<22}{r['dpla']:>+8.2f}{r['err_1']:>+11.2f}"
              f"{r['err_10']:>+11.2f}{r['err_32']:>+11.2f}")

    e1 = np.array([r["err_1"] for r in RESULTS])
    e10 = np.array([r["err_10"] for r in RESULTS])
    e32 = np.array([r["err_32"] for r in RESULTS])

    print(f"\n  {'时域':<10}{'MAE(rpm)':>12}{'有符号均值':>13}{'最大|误差|':>13}")
    print("  " + "-" * 50)
    for nm, e in [("1 步", e1), ("10 步", e10), ("32 步", e32)]:
        print(f"  {nm:<10}{np.mean(np.abs(e)):>12.2f}"
              f"{np.mean(e):>+13.2f}{np.max(np.abs(e)):>13.2f}")

    print(f"\n  ΔN2 对比（起点 → 第 32 步）:")
    print(f"  {'轨迹':<22}{'Δpred':>10}{'Δtrue':>10}{'低估量':>10}")
    print("  " + "-" * 54)
    for r in RESULTS:
        under = r["dn2_true_32"] - r["dn2_pred_32"]
        print(f"  {r['traj']:<22}{r['dn2_pred_32']:>+10.2f}"
              f"{r['dn2_true_32']:>+10.2f}{under:>+10.2f}")

    import json
    summary = {
        "n_events": len(RESULTS),
        "mae_1": float(np.mean(np.abs(e1))),
        "mae_10": float(np.mean(np.abs(e10))),
        "mae_32": float(np.mean(np.abs(e32))),
        "bias_1": float(np.mean(e1)),
        "bias_10": float(np.mean(e10)),
        "bias_32": float(np.mean(e32)),
        "results": RESULTS,
    }
    (OUT / "gru_replay.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    # 保存 npz 供绘图
    np.savez(OUT / "gru_replay.npz",
             pred_curves=[r["pred_curve"] for r in RESULTS],
             true_curves=[r["true_curve"] for r in RESULTS],
             pla_curves=[r["pla_curve"] for r in RESULTS],
             labels=[r["traj"] for r in RESULTS],
             t_events=[r["t_event"] for r in RESULTS],
             dpla=[r["dpla"] for r in RESULTS])

    print(f"\n  -> {OUT/'gru_replay.json'}")
else:
    print("  ❌ 无有效回放事件")

# ══════════════════════════════════════════════════
# 额外：用稳态观测测试 GRU 对 PLA 阶跃的预测
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[3] 补充：稳态起点下不同 PLA 阶跃的 GRU 预测")
print("=" * 92)

obs_steady = np.array([0.6180, 4349.29, 2465.17, 1715.78, 303.61])
obs_h = np.tile(obs_steady, (WINDOW, 1))
act_h = np.tile([55.0, 0.0, 0.0], (WINDOW, 1))

print(f"\n  起点: N2={obs_steady[1]:.2f}, PLA=55")
print(f"  {'PLA阶跃':<12}{'ΔPLA':>8}{'N2@1':>10}{'N2@10':>10}{'N2@32':>10}"
      f"{'ΔN2@32':>10}{'经验值':>10}")
print("  " + "-" * 76)

# 经验增益：96.6 rpm/deg（来自可行性扫描）
K_EMP = 96.6

steady_preds = []
for pla_new in [56, 58, 60, 63.64, 66, 70]:
    act_f = np.tile([pla_new, 0.0, 0.0], (HORIZON, 1))
    obs_hz = (obs_h - obs_mu) / obs_sd
    act_hz = (act_h - act_mu) / act_sd
    act_fz = (act_f - act_mu) / act_sd
    o = torch.from_numpy(obs_hz[None]).float().to(DEV)
    a = torch.from_numpy(act_hz[None]).float().to(DEV)
    af = torch.from_numpy(act_fz[None]).float().to(DEV)
    with torch.no_grad():
        torch.manual_seed(SEED)
        p = GRU(o, a, af)
    pred = p[0].cpu().numpy() * obs_sd + obs_mu
    n2p = pred[:, i_n2]
    d_n2 = n2p[31] - obs_steady[1]
    emp = (pla_new - 55.0) * K_EMP
    print(f"  {pla_new:<12.2f}{pla_new-55:>+8.2f}{n2p[0]:>10.2f}{n2p[9]:>10.2f}"
          f"{n2p[31]:>10.2f}{d_n2:>+10.2f}{emp:>+10.1f}")
    steady_preds.append({"pla": float(pla_new), "dn2_pred": float(d_n2),
                         "dn2_emp": float(emp)})

(OUT / "steady_step_preds.json").write_text(
    __import__("json").dumps(steady_preds, ensure_ascii=False, indent=1),
    encoding="utf-8")
print(f"\n  经验值 = ΔPLA × 96.6 rpm/deg（可行性扫描静态增益）")
print(f"  -> {OUT/'steady_step_preds.json'}")
