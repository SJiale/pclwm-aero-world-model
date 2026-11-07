"""任务2b：GRU 长时程预测 vs 稳态真值

关键假设检验：
  MPC 用 horizon=32（3.36s）优化
  若 GRU 在 3.36s 内预测不到目标 -> MPC 会持续加油门 -> 超调

验证：
  1. GRU 预测 1/10/32/100/200 步
  2. 对比可行性扫描的稳态真值
  3. 判断 GRU 是否存在系统性低估
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, WINDOW, HORIZON, DT

OUT = Path(r"C:\pclwm\pclwm_stage3a\replay")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"

data = UnifiedData()
obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()
act_mu = np.asarray(data.act_mu).ravel()
act_sd = np.asarray(data.act_sd).ravel()
OBS_COLS = list(data.obs_cols)
i_n2 = OBS_COLS.index("N2int")


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
print("=" * 92)
print("任务 2b：GRU 长时程预测 vs 稳态真值")
print("=" * 92)

# ══ 可行性扫描真值（来自 feasibility.txt）══
FEAS = {
    40: 2755.70, 45: 3289.68, 50: 3830.12, 52: 4041.73,
    55: 4349.29, 58: 4621.21, 60: 4795.20, 62: 4967.66,
    65: 5208.97, 70: 5575.92, 75: 5927.54, 80: 6243.05,
}
print(f"\n[真值来源] feasibility.txt (12 个工作点 × 60s 稳态)")
print(f"  PLA=55 -> N2={FEAS[55]:.2f}")
print(f"  PLA=65 -> N2={FEAS[65]:.2f}")
print(f"  PLA=75 -> N2={FEAS[75]:.2f}")

# ══ 稳态起点 ══
obs_steady = np.array([0.6180, 4349.29, 2465.17, 1715.78, 303.61])
obs_h = np.tile(obs_steady, (WINDOW, 1))
act_h = np.tile([55.0, 0.0, 0.0], (WINDOW, 1))
obs_hz = (obs_h - obs_mu) / obs_sd
act_hz = (act_h - act_mu) / act_sd

# ══ 长时程预测 ══
STEPS = [1, 10, 32, 64, 100, 200, 400]
print("\n" + "=" * 92)
print("[1] GRU 长时程自主预测（PLA 阶跃后保持）")
print("=" * 92)

print(f"\n  起点 N2 = {obs_steady[1]:.2f}, PLA = 55")
print(f"\n  {'PLA':>7}{'步1':>10}{'步10':>10}{'步32':>10}{'步100':>10}"
      f"{'步400':>10}{'稳态真值':>11}{'误差@400':>11}")
print("  " + "-" * 82)

HMAX = max(STEPS)
LONG = []
for pla_new in [58, 60, 62, 65, 68, 70, 75]:
    act_f = np.tile([pla_new, 0.0, 0.0], (HMAX, 1))
    act_fz = (act_f - act_mu) / act_sd
    o = torch.from_numpy(obs_hz[None]).float().to(DEV)
    a = torch.from_numpy(act_hz[None]).float().to(DEV)
    af = torch.from_numpy(act_fz[None]).float().to(DEV)
    with torch.no_grad():
        p = GRU(o, a, af)
    pred = p[0].cpu().numpy() * obs_sd + obs_mu
    n2p = pred[:, i_n2]

    # 插值真值
    feas_keys = sorted(FEAS.keys())
    true_ss = np.interp(pla_new, feas_keys,
                        [FEAS[k] for k in feas_keys])

    vals = {s: n2p[s-1] for s in STEPS}
    err400 = n2p[399] - true_ss

    print(f"  {pla_new:>7.1f}{vals[1]:>10.2f}{vals[10]:>10.2f}"
          f"{vals[32]:>10.2f}{vals[100]:>10.2f}{vals[400]:>10.2f}"
          f"{true_ss:>11.2f}{err400:>+11.2f}")

    LONG.append({
        "pla": float(pla_new),
        "pred": {str(s): float(vals[s]) for s in STEPS},
        "true_steady": float(true_ss),
        "err_400": float(err400),
        "dn2_pred_400": float(n2p[399] - obs_steady[1]),
        "dn2_true": float(true_ss - obs_steady[1]),
    })

# ══ 增益对比 ══
print("\n" + "=" * 92)
print("[2] 静态增益对比（ΔN2/ΔPLA）")
print("=" * 92)
print(f"\n  {'PLA':>7}{'ΔPLA':>8}{'GRU增益':>12}{'真值增益':>12}{'低估%':>10}")
print("  " + "-" * 52)

for r in LONG:
    dpla = r["pla"] - 55.0
    k_gru = r["dn2_pred_400"] / dpla
    k_true = r["dn2_true"] / dpla
    under = (k_true - k_gru) / k_true * 100
    print(f"  {r['pla']:>7.1f}{dpla:>+8.1f}{k_gru:>12.2f}{k_true:>12.2f}"
          f"{under:>+10.1f}%")

# ══ 关键：3.36s 内 GRU 认为到不了稳态 ══
print("\n" + "=" * 92)
print("[3] ★ 关键：MPC horizon(32步=3.36s) 内 GRU 能预测到多少？")
print("=" * 92)

print(f"\n  {'PLA':>7}{'稳态真值':>11}{'GRU@32步':>11}{'差距':>10}{'达成率':>9}")
print("  " + "-" * 50)
for r in LONG:
    gap = r["true_steady"] - r["pred"]["32"]
    rate = (r["pred"]["32"] - obs_steady[1]) / (r["true_steady"] - obs_steady[1]) * 100
    print(f"  {r['pla']:>7.1f}{r['true_steady']:>11.2f}"
          f"{r['pred']['32']:>11.2f}{gap:>+10.2f}{rate:>8.1f}%")

print(f"""
  解读：
    如果 GRU@32步 明显低于稳态真值，
    MPC（horizon=32）会认为"还差很多"，
    于是继续加大 PLA -> 超调
""")

# ══ 保存 ══
out = {"steps": STEPS, "start_n2": float(obs_steady[1]),
       "start_pla": 55.0, "results": LONG}
(OUT / "long_horizon.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT/'long_horizon.json'}")
