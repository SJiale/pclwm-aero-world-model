"""阶段 III-A 图表

图1: 六条 test_ctrl 的真实值 vs 三模型预测（连续推演 10s，N2 通道）
图2: 配对误差差值（GRU / RSSM / PC-LWM）
图3: 误差随推演时间增长曲线
图4: 分通道误差对比
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, DT

RES = Path(r"C:\pclwm\pclwm_stage3a\results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 777

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.4, "savefig.dpi": 300, "savefig.bbox": "tight"})

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()
act_mu = np.asarray(data.act_mu).ravel()
act_sd = np.asarray(data.act_sd).ravel()
REF = float(ev.ref_std)
OBS_COLS = list(data.obs_cols)


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
m.eval()
MODELS["GRU"] = m

m = RSSM(C_O, C_A, deter=128, stoch=32, classes=32, hidden=128).to(DEV)
m.load_state_dict(strip(load_sd(
    r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt")), strict=True)
m.eval()
MODELS["RSSM"] = m

m = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                 lam_static=0.01, lam_dynamic=0.003).to(DEV)
sd = load_sd(r"C:\pclwm\pclwm_pcrssm\checkpoints\seed0_E3_both.pt")
if any(k.startswith("model.") for k in sd):
    sd = {k[len("model."):]: v for k, v in sd.items()}
m.load_state_dict(sd, strict=True)
m.eval()
MODELS["PC-LWM"] = m

COLORS = {"GRU": "#1f77b4", "RSSM": "#ff7f0e", "PC-LWM": "#2ca02c"}


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


def rel_e(p, t):
    d = (p - t) * torch.from_numpy(obs_sd).float().to(p.device)
    return ((d ** 2).mean(-1).sqrt() / REF).cpu().numpy()


# ── 载入 test_ctrl ──
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7
TC = {}
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    obs = (d[OBS_COLS].to_numpy(np.float64) - obs_mu) / obs_sd
    act = (d[["PLA_cmd_deg", "Alt_ft", "MN"]].to_numpy(np.float64)
           - act_mu) / act_sd
    TC[f.stem] = {"obs": obs.astype(np.float32), "act": act.astype(np.float32),
                  "pla": d["PLA_cmd_deg"].to_numpy(np.float64),
                  "t": d["time_s"].to_numpy(np.float64)}


def rollout(tc, nm, tgt_s=10):
    obs_all, act_all = tc["obs"], tc["act"]
    n = len(obs_all)
    ns = min(int(round(tgt_s / DT)), n - WINDOW)
    oh = torch.from_numpy(obs_all[WINDOW - WINDOW:WINDOW][None]).float().to(DEV)
    ah = torch.from_numpy(act_all[0:WINDOW][None]).float().to(DEV)
    af = torch.from_numpy(act_all[WINDOW:WINDOW + ns][None]).float().to(DEV)
    tf = torch.from_numpy(obs_all[WINDOW:WINDOW + ns][None]).float().to(DEV)
    p = predict(nm, oh, ah, af)
    H = min(p.shape[1], tf.shape[1])
    return p[:, :H].cpu().numpy()[0], tf[:, :H].cpu().numpy()[0]


print("=" * 90)
print("生成图表")
print("=" * 90)

# ══════════════════════════════════════════════════
# 图1：六条轨迹的 N2 预测曲线
# ══════════════════════════════════════════════════
i_n2 = OBS_COLS.index("N2int")

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (k, tc) in zip(axes, TC.items()):
    t_axis = np.arange(int(10 / DT)) * DT
    # 真值
    tf_n2 = tc["obs"][WINDOW:WINDOW + len(t_axis), i_n2] * obs_sd[i_n2] + obs_mu[i_n2]
    ax.plot(t_axis, tf_n2, "k-", lw=1.8, label="真实值", zorder=5)
    for nm in MODELS:
        p, _ = rollout(tc, nm, 10)
        pn2 = p[:len(t_axis), i_n2] * obs_sd[i_n2] + obs_mu[i_n2]
        ax.plot(t_axis[:len(pn2)], pn2, color=COLORS[nm], lw=1.2,
                alpha=0.85, label=nm)
    ax.set_title(f"{k.split('_')[0]}  Alt={tc['t'][0]:.0f}s  "
                 f"PLA∈[{tc['pla'].min():.1f},{tc['pla'].max():.1f}]°",
                 fontsize=9)
    ax.set_xlabel("连续推演时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    if ax is axes[0]:
        ax.legend(fontsize=8, loc="best")

fig.suptitle("图 1  test_ctrl 六条轨迹：N2 连续自主推演（10 s，仅用历史观测）",
             fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_ctrl_n2_rollout.{e}")
plt.close(fig)
print("  图1")

# ══════════════════════════════════════════════════
# 图2：配对误差差值（RSSM/PC-LWM 相对 GRU）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

for ax, (ds_nm, ws) in zip(
        axes, [("test_ood", data.windows("test_ood", stride=8)),
               ("test_ctrl", None)]):
    if ws is None:
        # test_ctrl：用所有窗口
        all_w = []
        for k, tc in TC.items():
            n = len(tc["obs"])
            for t0 in range(0, n - WINDOW - HORIZON, 4):
                all_w.append({
                    "obs": tc["obs"][t0:t0 + WINDOW],
                    "act": tc["act"][t0:t0 + WINDOW],
                    "fut": tc["obs"][t0 + WINDOW:t0 + WINDOW + HORIZON],
                    "act_fut": tc["act"][t0 + WINDOW:t0 + WINDOW + HORIZON],
                })
        ws = all_w

    # 逐窗口误差曲线
    curves = {}
    bs = 64
    for nm in MODELS:
        acc = None
        cnt = 0
        for i in range(0, len(ws), bs):
            b = ws[i:i + bs]
            o = torch.from_numpy(np.stack([w["obs"] for w in b])).float().to(DEV)
            a = torch.from_numpy(np.stack([w["act"] for w in b])).float().to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in b])).float().to(DEV)
            f = torch.from_numpy(np.stack([w["fut"] for w in b])).float().to(DEV)
            p = predict(nm, o, a, af)
            H = min(p.shape[1], f.shape[1])
            E = rel_e(p[:, :H], f[:, :H])
            acc = E if acc is None else np.concatenate([acc, E], 0)
        curves[nm] = acc.mean(0)

    t_axis = np.arange(len(curves["GRU"])) * DT
    base = curves["GRU"]
    for nm in ["RSSM", "PC-LWM"]:
        ax.plot(t_axis, curves[nm] - base, color=COLORS[nm], lw=1.5,
                label=f"{nm} − GRU")
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xlabel("预测步长时间 (s)")
    ax.set_ylabel("相对误差差值（正 = 优于 GRU）")
    ax.set_title(f"{'test_ood' if ds_nm=='test_ood' else 'test_ctrl'}",
                 fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("图 2  配对误差差值：RSSM / PC-LWM 相对 GRU（正=更优）", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_paired_diff.{e}")
plt.close(fig)
print("  图2")

# ══════════════════════════════════════════════════
# 图3：误差随推演时间增长
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

for ax, tgt in zip(axes, [10, 60]):
    for nm in MODELS:
        curves = []
        for k, tc in TC.items():
            p, t = rollout(tc, nm, tgt)
            curves.append(rel_e(
                torch.from_numpy(p[None]), torch.from_numpy(t[None]))[0])
        L = min(len(c) for c in curves)
        C = np.array([c[:L] for c in curves])
        t_axis = np.arange(L) * DT
        ax.plot(t_axis, C.mean(0), color=COLORS[nm], lw=1.4, label=nm)
        ax.fill_between(t_axis, C.mean(0) - C.std(0), C.mean(0) + C.std(0),
                        color=COLORS[nm], alpha=0.15)
    ax.set_xlabel("连续推演时间 (s)")
    ax.set_ylabel("相对误差")
    ax.set_title(f"连续推演 {tgt} s（mean ± std 跨 6 条轨迹）", fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("图 3  test_ctrl 连续自主推演误差增长", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_error_growth.{e}")
plt.close(fig)
print("  图3")

# ══════════════════════════════════════════════════
# 图4：分通道误差
# ══════════════════════════════════════════════════
R = json.loads((RES / "eval_final.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, ds in zip(axes, ["test_id", "test_ood", "test_ctrl"]):
    x = np.arange(len(OBS_COLS))
    w = 0.26
    for i, nm in enumerate(["GRU-h128", "RSSM-d128", "PC-LWM-E3"]):
        ch = R["results"][nm][ds]["per_channel"]
        vals = [ch[c] for c in OBS_COLS]
        ax.bar(x + (i - 1) * w, vals, w,
               label=nm.split("-")[0], color=list(COLORS.values())[i],
               alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split("_")[0] for c in OBS_COLS], fontsize=8)
    ax.set_ylabel("相对误差 (rmse/sd)")
    ax.set_title(ds, fontsize=10)
    ax.legend(fontsize=7)

fig.suptitle("图 4  分通道 32-step 预测误差", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_per_channel.{e}")
plt.close(fig)
print("  图4")

print(f"\n  -> {FIG}")
