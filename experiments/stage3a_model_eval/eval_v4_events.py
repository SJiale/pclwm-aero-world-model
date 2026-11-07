"""P0-1：V4 事件对齐的阶跃响应评价

设计（严格无泄漏）：
  对每个阶跃时刻 t_ev：
    - 历史窗口 = [t_ev - WINDOW*DT, t_ev)   只用真实观测
    - 预测窗口 = [t_ev, t_ev + HORIZON*DT)  自主推演
    - 真值 = 该区间的真实观测（仅用于评价，不输入模型）

  三个事件：t=15.02, 30.03, 45.05
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, DT

OUT = Path(r"C:\pclwm\pclwm_stage3a\results")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 777

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()
act_mu = np.asarray(data.act_mu).ravel()
act_sd = np.asarray(data.act_sd).ravel()
REF = float(ev.ref_std)
OBS_COLS = list(data.obs_cols)
i_n2 = OBS_COLS.index("N2int")

print("=" * 100)
print("P0-1：V4 事件对齐的阶跃响应评价")
print("=" * 100)


# ── 模型 ──
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
        H = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


def strip(sd, pref="rssm."):
    return {k[len(pref):] if k.startswith(pref) else k: v for k, v in sd.items()}


def load_sd(p):
    sd = torch.load(p, map_location=DEV, weights_only=True)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


from rssm_impl import RSSM
from pc_rssm import PCWorldModel

MODELS = {}
m = GRUModel(C_O, C_A).to(DEV)
m.load_state_dict(load_sd(r"C:\pclwm\pclwm_experiments\checkpoints\GRU-h128.pt"),
                  strict=True)
m.eval(); MODELS["GRU"] = m

m = RSSM(C_O, C_A, deter=128, stoch=32, classes=32, hidden=128).to(DEV)
m.load_state_dict(strip(load_sd(
    r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt")), strict=True)
m.eval(); MODELS["RSSM"] = m

m = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                 lam_static=0.01, lam_dynamic=0.003).to(DEV)
sd = load_sd(r"C:\pclwm\pclwm_pcrssm\checkpoints\seed0_E3_both.pt")
if any(k.startswith("model.") for k in sd):
    sd = {k[len("model."):]: v for k, v in sd.items()}
m.load_state_dict(sd, strict=True)
m.eval(); MODELS["PC-LWM"] = m


def predict(nm, obs, act, af):
    m = MODELS[nm]
    with torch.no_grad():
        torch.manual_seed(SEED)
        if nm == "GRU":
            return m(obs, act, af)
        elif nm == "RSSM":
            return m.imagine(obs, act, af, mode="gumbel")
        else:
            return m.imagine(obs, torch.ones_like(obs), act, af, mode="gumbel")


# ── 载入 V4 ──
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7
df = pd.read_csv(CTRL / "V4_dstep_alt5k.csv")
d = df.iloc[::STEP].reset_index(drop=True)
obs = ((d[OBS_COLS].to_numpy(np.float64) - obs_mu) / obs_sd).astype(np.float32)
act = ((d[["PLA_cmd_deg", "Alt_ft", "MN"]].to_numpy(np.float64)
        - act_mu) / act_sd).astype(np.float32)
pla = d["PLA_cmd_deg"].to_numpy(np.float64)
n2_raw = d["N2int"].to_numpy(np.float64)
t = d["time_s"].to_numpy(np.float64)

# 找阶跃
dp = np.diff(pla)
jumps = np.where(np.abs(dp) > 0.5)[0]
EVENTS = [(t[j + 1], pla[j], pla[j + 1], dp[j]) for j in jumps]

print(f"\n  V4 阶跃事件（共 {len(EVENTS)} 个）:")
for te, p0, p1, dd in EVENTS:
    print(f"    t={te:>6.2f}s:  PLA {p0:.2f} -> {p1:.2f}  (Δ={dd:+.2f})")

# ══════════════════════════════════════════════════
# 事件对齐的自主推演
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("事件对齐推演（历史=真实观测，未来=自主推演，无泄漏）")
print("=" * 100)

EVAL = {}
H = HORIZON
for te, p0, p1, dd in EVENTS:
    # 事件在索引中的位置
    ie = int(np.argmin(np.abs(t - te)))
    i0 = ie - WINDOW          # 历史起点
    i1 = ie                   # 历史终点（不含事件时刻之后）
    i2 = min(ie + H, len(obs))  # 预测终点

    if i0 < 0 or i2 - ie < 10:
        print(f"  t={te:.2f}s 跳过（数据不足）")
        continue

    oh = torch.from_numpy(obs[i0:i1][None]).float().to(DEV)   # 历史观测
    ah = torch.from_numpy(act[i0:i1][None]).float().to(DEV)   # 历史动作
    af = torch.from_numpy(act[i1:i2][None]).float().to(DEV)   # 未来动作（已知）
    tf = torch.from_numpy(obs[i1:i2][None]).float().to(DEV)   # 真值（仅评价）

    n_fut = af.shape[1]
    print(f"\n  ── 事件 t={te:.2f}s (PLA {p0:.1f}→{p1:.1f}), "
          f"历史=[{t[i0]:.2f},{t[i1]:.2f}]s, 预测 {n_fut} 步 ──")
    print(f"    {'模型':<10}{'推演时长':>10}{'N2初值':>10}{'N2末值':>10}"
          f"{'N2变化':>10}{'真值变化':>10}{'变化误差':>10}{'rel':>9}")

    # 真值 N2 变化
    n2_true = tf[0, :, i_n2].cpu().numpy() * obs_sd[i_n2] + obs_mu[i_n2]
    dn2_true = n2_true[-1] - n2_true[0]

    rec = {"event_t": float(te), "pla_from": float(p0), "pla_to": float(p1),
           "dpla": float(dd), "n_steps": int(n_fut),
           "t_start": float(t[i1]), "n2_true": n2_true.tolist(),
           "dn2_true": float(dn2_true), "models": {}}

    for nm in MODELS:
        p = predict(nm, oh, ah, af)
        Hh = min(p.shape[1], tf.shape[1])
        n2_p = p[0, :Hh, i_n2].cpu().numpy() * obs_sd[i_n2] + obs_mu[i_n2]
        dn2_p = n2_p[-1] - n2_p[0]
        d2 = (p[:, :Hh] - tf[:, :Hh]) * torch.from_numpy(obs_sd).float().to(DEV)
        rel = float((d2 ** 2).mean(-1).sqrt().mean() / REF)
        chg_err = abs(dn2_p - dn2_true)

        print(f"    {nm:<10}{Hh*DT:>9.2f}s{n2_p[0]:>10.1f}{n2_p[-1]:>10.1f}"
              f"{dn2_p:>+10.1f}{dn2_true:>+10.1f}{chg_err:>10.1f}{rel:>9.4f}")

        rec["models"][nm] = {
            "n2": n2_p.tolist(), "dn2": float(dn2_p),
            "change_err": float(chg_err), "rel": rel,
        }
    EVAL[f"t{te:.0f}"] = rec

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[汇总] 阶跃响应能力")
print("=" * 100)
print(f"\n  {'事件':<10}{'dPLA':>8}{'真值ΔN2':>12}"
      + "".join(f"{n+'ΔN2':>12}" for n in MODELS))
print("  " + "-" * 66)
for k, r in EVAL.items():
    row = f"  {k:<10}{r['dpla']:>+8.2f}{r['dn2_true']:>+12.1f}"
    for nm in MODELS:
        row += f"{r['models'][nm]['dn2']:>+12.1f}"
    print(row)

print(f"\n  {'事件':<10}" + "".join(f"{n:>12}" for n in MODELS) + f"{'最优':>10}")
print("  " + "-" * 56)
for k, r in EVAL.items():
    row = f"  {k:<10}"
    best, bv = None, 1e9
    for nm in MODELS:
        v = r["models"][nm]["rel"]
        row += f"{v:>12.4f}"
        if v < bv:
            bv, best = v, nm
    row += f"{best:>10}"
    print(row)

(OUT / "v4_event_aligned.json").write_text(
    json.dumps(EVAL, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'v4_event_aligned.json'}")
