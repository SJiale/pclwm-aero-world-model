"""P0-2：修正图表（5 处错误）

修正项：
  1. 图2 标题/纵轴：model_error − GRU_error，正值 = GRU 更优（我原来写反了）
  2. 标注真实推演时长（60s 请求实际约 56.7s）
  3. 32-step 补最后一步误差（k=32）
  4. 早期/晚期误差绝对值都列出
  5. 图1 展示真实 Alt/MN（原来误写了 t[0]）
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
FIG = RES / "figures_v2"
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


CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7
TC = {}
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    TC[f.stem] = {
        "obs": ((d[OBS_COLS].to_numpy(np.float64) - obs_mu) / obs_sd).astype(np.float32),
        "act": ((d[["PLA_cmd_deg", "Alt_ft", "MN"]].to_numpy(np.float64)
                 - act_mu) / act_sd).astype(np.float32),
        "pla": d["PLA_cmd_deg"].to_numpy(np.float64),
        "alt": float(d["Alt_ft"].iloc[0]),
        "mn": float(d["MN"].iloc[0]),
        "t": d["time_s"].to_numpy(np.float64),
    }

print("=" * 95)
print("生成修正版图表")
print("=" * 95)

# 先算真实推演时长
print("\n  [修正2] 真实推演时长:")
ROLL = {}
for k, tc in TC.items():
    n = len(tc["obs"])
    avail = n - WINDOW
    for tgt in [10, 60]:
        ns = min(int(round(tgt / DT)), avail)
        ROLL[(k, tgt)] = ns * DT
    print(f"    {k:<22} 请求60s -> 实际 {ROLL[(k,60)]:.2f}s "
          f"(可用 {avail} 步)")


def rollout(tc, nm, tgt_s):
    obs_all, act_all = tc["obs"], tc["act"]
    ns = min(int(round(tgt_s / DT)), len(obs_all) - WINDOW)
    oh = torch.from_numpy(obs_all[0:WINDOW][None]).float().to(DEV)
    ah = torch.from_numpy(act_all[0:WINDOW][None]).float().to(DEV)
    af = torch.from_numpy(act_all[WINDOW:WINDOW + ns][None]).float().to(DEV)
    tf = torch.from_numpy(obs_all[WINDOW:WINDOW + ns][None]).float().to(DEV)
    p = predict(nm, oh, ah, af)
    H = min(p.shape[1], tf.shape[1], ns)
    return p[:, :H], tf[:, :H], H


# ══════════════════════════════════════════════════
# 图 1（修正5）：真实 Alt/MN
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, (k, tc) in zip(axes, TC.items()):
    p_ref, tf_ref, H = rollout(tc, "GRU", 10)
    t_axis = np.arange(H) * DT
    n2_true = tf_ref[0, :, i_n2].cpu().numpy() * obs_sd[i_n2] + obs_mu[i_n2]
    ax.plot(t_axis, n2_true, "k-", lw=1.9, label="真实值", zorder=5)
    for nm in MODELS:
        p, tf, Hh = rollout(tc, nm, 10)
        n2p = p[0, :, i_n2].cpu().numpy() * obs_sd[i_n2] + obs_mu[i_n2]
        ax.plot(np.arange(len(n2p)) * DT, n2p, color=COLORS[nm], lw=1.1,
                alpha=0.85, label=nm)
    # ★ 修正：真实 Alt / MN
    ax.set_title(f"{k.split('_')[0]}   Alt={tc['alt']:.0f} ft, MN={tc['mn']:.2f}\n"
                 f"PLA∈[{tc['pla'].min():.1f},{tc['pla'].max():.1f}]°",
                 fontsize=8.5)
    ax.set_xlabel("连续推演时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    if ax is axes[0]:
        ax.legend(fontsize=7.5, loc="best")

fig.suptitle("图 1  test_ctrl 六条轨迹：N2 连续自主推演 10 s"
             "（历史观测驱动，未来纯先验，无泄漏）", fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_ctrl_n2_rollout_corrected.{e}")
plt.close(fig)
print("\n  图1（修正 Alt/MN）")

# ══════════════════════════════════════════════════
# 图 2（修正1）：正确的差值方向
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

for ax, ds_nm in zip(axes, ["test_ood", "test_ctrl"]):
    if ds_nm == "test_ood":
        ws = data.windows("test_ood", stride=8)
    else:
        ws = []
        for k, tc in TC.items():
            n = len(tc["obs"])
            for t0 in range(0, n - WINDOW - HORIZON, 4):
                ws.append({
                    "obs": tc["obs"][t0:t0 + WINDOW],
                    "act": tc["act"][t0:t0 + WINDOW],
                    "fut": tc["obs"][t0 + WINDOW:t0 + WINDOW + HORIZON],
                    "act_fut": tc["act"][t0 + WINDOW:t0 + WINDOW + HORIZON],
                })

    curves = {}
    for nm in MODELS:
        acc = []
        bs = 64
        for i in range(0, len(ws), bs):
            b = ws[i:i + bs]
            o = torch.from_numpy(np.stack([w["obs"] for w in b])).float().to(DEV)
            a = torch.from_numpy(np.stack([w["act"] for w in b])).float().to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in b])).float().to(DEV)
            f = torch.from_numpy(np.stack([w["fut"] for w in b])).float().to(DEV)
            p = predict(nm, o, a, af)
            Hh = min(p.shape[1], f.shape[1])
            acc.append(rel_e(p[:, :Hh], f[:, :Hh]))
        curves[nm] = np.concatenate(acc, 0).mean(0)

    t_axis = np.arange(len(curves["GRU"])) * DT
    for nm in ["RSSM", "PC-LWM"]:
        # ★ 修正：曲线值 = model_err - GRU_err
        #   正值 -> model 误差更大 -> GRU 更优
        d = curves[nm] - curves["GRU"]
        ax.plot(t_axis, d, color=COLORS[nm], lw=1.5,
                label=f"{nm} − GRU")
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xlabel("预测步长时间 (s)")
    # ★ 修正纵轴标签
    ax.set_ylabel("$\\epsilon_{\\mathrm{model}} - \\epsilon_{\\mathrm{GRU}}$")
    ax.set_title(f"{ds_nm}   （>0 = GRU 更优；<0 = 该模型更优）", fontsize=10)
    ax.legend(fontsize=8)
    # 标注两侧含义
    ax.text(0.02, 0.96, "GRU 更优 ↑", transform=ax.transAxes,
            fontsize=7.5, va="top", color="0.4")
    ax.text(0.02, 0.04, "该模型更优 ↓", transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="0.4")

fig.suptitle("图 2  配对误差差值（model − GRU）。正值表示 GRU 误差更小，即 GRU 更优。",
             fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_paired_diff_corrected.{e}")
plt.close(fig)
print("  图2（修正标题/纵轴/方向说明）")

# ══════════════════════════════════════════════════
# 图 3（修正2+3）：真实时长 + 最后一步误差
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

for ax, tgt in zip(axes, [10, 60]):
    curves = {}
    for nm in MODELS:
        cs = []
        for k, tc in TC.items():
            p, tf, H = rollout(tc, nm, tgt)
            cs.append(rel_e(p, tf)[0])
        L = min(len(c) for c in cs)
        curves[nm] = np.array([c[:L] for c in cs])
    L = min(c.shape[1] for c in curves.values())
    t_axis = np.arange(L) * DT
    for nm in MODELS:
        C = curves[nm][:, :L]
        ax.plot(t_axis, C.mean(0), color=COLORS[nm], lw=1.4, label=nm)
        ax.fill_between(t_axis, C.mean(0) - C.std(0), C.mean(0) + C.std(0),
                        color=COLORS[nm], alpha=0.15)
    actual = L * DT
    ax.set_xlabel("连续推演时间 (s)")
    ax.set_ylabel("相对误差")
    # ★ 修正2：标明真实时长
    ax.set_title(f"请求 {tgt} s → 实际 {actual:.2f} s（数据长度限制）",
                 fontsize=9.5)
    ax.legend(fontsize=8)

fig.suptitle("图 3  test_ctrl 连续自主推演误差（mean ± std 跨 6 条轨迹）",
             fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_error_growth_corrected.{e}")
plt.close(fig)
print(f"  图3（标注真实时长 {L*DT:.2f}s）")

# ══════════════════════════════════════════════════
# 图 4（修正4）：早期/晚期绝对值
# ══════════════════════════════════════════════════
print("\n  [修正4] 早期/晚期误差绝对值（10s 推演）:")
print(f"  {'轨迹':<22}{'模型':<10}{'早期(前10%)':>13}{'晚期(后10%)':>13}"
      f"{'极值':>10}")
early_late = {}
for k, tc in TC.items():
    for nm in MODELS:
        p, tf, H = rollout(tc, nm, 10)
        E = rel_e(p, tf)[0]
        n10 = max(3, len(E) // 10)
        e_early = float(E[:n10].mean())
        e_late = float(E[-n10:].mean())
        early_late.setdefault(k, {})[nm] = {"early": e_early, "late": e_late,
                                            "max": float(E.max())}
        print(f"  {k:<22}{nm:<10}{e_early:>13.4f}{e_late:>13.4f}"
              f"{E.max():>10.4f}")
    print()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for ax, (which, lbl) in zip(axes, [("early", "前 10%"), ("late", "后 10%")]):
    x = np.arange(len(TC))
    w = 0.26
    for i, nm in enumerate(MODELS):
        vals = [early_late[k][nm][which] for k in TC]
        ax.bar(x + (i - 1) * w, vals, w, label=nm,
               color=COLORS[nm], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("_")[0] for k in TC], fontsize=8)
    ax.set_ylabel("相对误差（绝对值）")
    ax.set_title(f"{lbl} 的误差绝对值（10 s 推演）", fontsize=10)
    ax.legend(fontsize=8)

fig.suptitle("图 4  早期 vs 晚期误差绝对量（不是增长倍数）", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_early_late_abs.{e}")
plt.close(fig)
print("  图4（早期/晚期绝对值）")

# ══════════════════════════════════════════════════
# 图 5（修正3）：32-step 平均 + 最后一步
# ══════════════════════════════════════════════════
print("\n  [修正3] 32-step 平均误差 vs 最后一步（k=32）:")
win_res = {}
for ds_nm in ["test_id", "test_ood"]:
    ws = data.windows(ds_nm, stride=8)
    win_res[ds_nm] = {}
    for nm in MODELS:
        acc = []
        for i in range(0, len(ws), 64):
            b = ws[i:i + 64]
            o = torch.from_numpy(np.stack([w["obs"] for w in b])).float().to(DEV)
            a = torch.from_numpy(np.stack([w["act"] for w in b])).float().to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in b])).float().to(DEV)
            f = torch.from_numpy(np.stack([w["fut"] for w in b])).float().to(DEV)
            p = predict(nm, o, a, af)
            Hh = min(p.shape[1], f.shape[1])
            acc.append(rel_e(p[:, :Hh], f[:, :Hh]))
        E = np.concatenate(acc, 0)
        win_res[ds_nm][nm] = {"avg32": float(E.mean()),
                              "k32": float(E[:, -1].mean())}
        print(f"    {ds_nm:<10}{nm:<10} 平均={E.mean():.5f}  "
              f"最后一步={E[:, -1].mean():.5f}")

fig, ax = plt.subplots(figsize=(9, 4.2))
x = np.arange(2 * len(MODELS))
labels = []
vals = []
cols = []
for j, ds_nm in enumerate(["test_id", "test_ood"]):
    for i, nm in enumerate(MODELS):
        vals.append(win_res[ds_nm][nm]["avg32"])
        labels.append(f"{ds_nm}\n{nm}\n(平均)")
        cols.append(COLORS[nm])
        vals.append(win_res[ds_nm][nm]["k32"])
        labels.append(f"{ds_nm}\n{nm}\n(k=32)")
        cols.append(COLORS[nm])
b = ax.bar(np.arange(len(vals)), vals, color=cols, alpha=0.75)
for i, v in enumerate(vals):
    ax.text(i, v + 0.001, f"{v:.4f}", ha="center", fontsize=6.5, rotation=90)
ax.set_xticks(np.arange(len(vals)))
ax.set_xticklabels(labels, fontsize=6)
ax.set_ylabel("相对误差")
ax.set_title("图 5  32-step 平均误差 vs 第 32 步误差（浅色=平均，深色=k32）",
             fontsize=10)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig5_avg_vs_last_step.{e}")
plt.close(fig)
print("  图5（32-step 平均 vs 最后一步）")

# 保存
(OUT / "corrections.json").write_text(
    json.dumps({"rollout_actual_s": {f"{k[0]}_{k[1]}": v for k, v in ROLL.items()},
                "early_late": early_late, "window_metrics": win_res},
               ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'corrections.json'}")
