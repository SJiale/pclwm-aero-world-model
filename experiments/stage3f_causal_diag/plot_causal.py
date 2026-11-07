"""因果诊断两张核心图

图1: 各关键控制时刻的 MPC 预测未来 N2 vs 后续真实 N2
     （明确区分"计划动作"和"实际后续重规划动作"）
图2: PLA、N2、N2_ref、MPC 预测超调量的时序关系
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CD = Path(r"C:\pclwm\pclwm_stage3a\causal")
FIG = CD / "figures"
FIG.mkdir(parents=True, exist_ok=True)
DT = 0.105

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.85,
    "lines.linewidth": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight"})

S = pd.read_csv(CD / "steps.csv")
P = pd.read_csv(CD / "pred_n2_matrix.csv", header=None).to_numpy()
PS = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()

cfg = {}
for ln in (CD / "cfg.txt").read_text().split("\n"):
    if "=" in ln:
        k, v = ln.split("=")
        cfg[k] = float(v)

N2_REF = cfg["n2_ref"]
H = int(cfg["H"])
N = len(S)

# 真实 N2 轨迹（用 n2_after 作为第 k 步之后的值）
t_real = np.append(S["t"].to_numpy(), S["t"].iloc[-1] + DT)
n2_real = np.append(S["n2_before"].to_numpy(), S["n2_after"].iloc[-1])
pla_real = np.append(S["pla_apply"].to_numpy(), S["pla_apply"].iloc[-1])

print("=" * 90)
print("生成因果诊断图")
print("=" * 90)
print(f"  N={N}, H={H}, N2_ref={N2_REF:.2f}")

# ══════════════════════════════════════════════════
# 图 1：关键时刻的预测轨迹 vs 后续真实
# ══════════════════════════════════════════════════
KEY = [9, 13, 14, 17, 20, 25]
KEY_LBL = {
    9:  "k=9  (0.84s) 预测峰值 4844，未预见超调",
    13: "k=13 (1.26s) 预测峰值 5082，首次接近目标",
    14: "k=14 (1.36s) 预测峰值 5132，★首次预见超调 +33",
    17: "k=17 (1.68s) PLA 峰值 71.5，预测超调 +79",
    20: "k=20 (2.00s) N2 刚达标，预测超调 +99",
    25: "k=25 (2.52s) N2 真实峰值 5238，预测超调 +128",
}

fig, axes = plt.subplots(3, 2, figsize=(15, 11))
axes = axes.flatten()

for ax, k in zip(axes, KEY):
    i = k - 1
    t_now = S["t"].iloc[i]

    # MPC 预测轨迹（用当时的计划 PLA 序列）
    t_pred = t_now + np.arange(1, H+1) * DT
    pred = P[i]

    # 真实后续 N2（实际执行的是后续重规划动作）
    mask = t_real >= t_now
    t_fut = t_real[mask]
    n2_fut = n2_real[mask]

    ax.plot(t_fut, n2_fut, "k-", lw=2.4, label="真实 N2（后续重规划）", zorder=5)
    ax.plot(t_pred, pred, "-", color="#2ca02c", lw=1.9, alpha=0.9,
            label="MPC 对该步计划序列的预测")
    ax.axhline(N2_REF, color="#d62728", ls="--", lw=1.4, label=f"目标 {N2_REF:.0f}")

    # 标注当前点
    ax.plot([t_now], [S["n2_before"].iloc[i]], "o", color="#1f77b4",
            ms=9, zorder=6, label="当前 N2")

    # 标注预测峰值
    pk = S["pred_peak"].iloc[i]
    pk_t = t_now + S["pred_peak_k"].iloc[i] * DT
    ax.plot([pk_t], [pk], "^", color="#ff7f0e", ms=10, zorder=6)
    ax.annotate(f"预测峰值 {pk:.0f}\n(+{S['pred_overshoot'].iloc[i]:.0f})",
                xy=(pk_t, pk), xytext=(pk_t + 1.2, pk + 30),
                fontsize=8, color="#ff7f0e",
                arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.2))

    ax.set_title(KEY_LBL[k], fontsize=9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    ax.set_xlim(t_now - 0.3, min(t_now + H*DT + 1.0, t_real[-1] + 0.3))
    if k == 9:
        ax.legend(fontsize=7, loc="lower right")

fig.suptitle("图 1  各关键控制时刻：MPC 预测的未来 N2 vs 随后真实 N2\n"
             "橙三角 = MPC 当时预测的峰值 ｜ 绿线 = 基于当时计划 PLA 序列的预测 ｜ "
             "黑线 = 实际执行（含后续重规划）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_pred_vs_real.{e}")
plt.close(fig)
print("  图1: 预测 vs 真实")

# ══════════════════════════════════════════════════
# 图 2：PLA / N2 / 预测超调量 时序
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1.5, 1.3]})

# (a) N2 + ref + PLA（双轴）
ax = axes[0]
ax.plot(t_real, n2_real, "-", color="#2ca02c", lw=2.4, label="真实 N2")
ax.axhline(N2_REF, color="#d62728", ls="--", lw=1.5,
           label=f"N2_ref {N2_REF:.0f}")
ax.fill_between(t_real, N2_REF, n2_real,
                where=(n2_real > N2_REF), color="#d62728", alpha=0.15,
                label="超调区")
ax.set_ylabel("N2 (rpm)")
ax.legend(fontsize=8.5, loc="lower right")

ax2 = ax.twinx()
ax2.step(t_real, pla_real, "-", color="#1f77b4", lw=1.9, where="post",
         label="PLA")
ax2.set_ylabel("PLA (deg)", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")

# 标注关键点
i_pk = int(np.argmax(S["pla_apply"].to_numpy()))
ax2.annotate(f"PLA 峰值 {S['pla_apply'].iloc[i_pk]:.2f}°\nt={S['t'].iloc[i_pk]:.2f}s",
             xy=(S["t"].iloc[i_pk], S["pla_apply"].iloc[i_pk]),
             xytext=(S["t"].iloc[i_pk] + 0.6, S["pla_apply"].iloc[i_pk] - 3),
             fontsize=8.5, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))

i_n2pk = int(np.argmax(n2_real))
ax.annotate(f"N2 峰值 {n2_real[i_n2pk]:.0f}\nt={t_real[i_n2pk]:.2f}s",
            xy=(t_real[i_n2pk], n2_real[i_n2pk]),
            xytext=(t_real[i_n2pk] + 0.7, n2_real[i_n2pk] + 25),
            fontsize=8.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))

ax.set_title("(a) PLA 与 N2 时序", fontsize=10.5)

# (b) MPC 预测超调量
ax = axes[1]
ov = S["pred_overshoot"].to_numpy()
t_s = S["t"].to_numpy()
colors = ["#d62728" if v > 0 else "#2ca02c" for v in ov]
ax.bar(t_s, ov, width=DT*0.85, color=colors, alpha=0.8)
ax.axhline(0, color="0.3", lw=1.2)
ax.set_ylabel("MPC 预测超调 (rpm)")
ax.set_title("(b) MPC 当时预测的超调量（正=已预见会超目标）", fontsize=10.5)

# 标注首次预见
i_first = int(np.where(ov > 0)[0][0])
ax.axvline(t_s[i_first], color="#ff7f0e", ls=":", lw=2.0)
ax.annotate(f"首次预见超调\nk={int(S['k'].iloc[i_first])}, "
            f"t={t_s[i_first]:.2f}s\n峰值预测 +{ov[i_first]:.0f}",
            xy=(t_s[i_first], ov[i_first]),
            xytext=(t_s[i_first] + 0.8, ov[i_first] + 60),
            fontsize=8.5, color="#ff7f0e",
            arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=1.5))

# (c) 代价分解
ax = axes[2]
ax.plot(t_s, S["j_track"].to_numpy(), "o-", color="#1f77b4", ms=4,
        lw=1.8, label="j_track（跟踪）")
ax.plot(t_s, S["j_du"].to_numpy(), "s--", color="#ff7f0e", ms=4,
        lw=1.5, label="j_du（动作）")
ax.set_yscale("log")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("代价值")
ax.set_title("(c) 代价分解（对数轴）", fontsize=10.5)
ax.legend(fontsize=8.5)

# 标注权重比
ax.text(0.02, 0.95, f"前 8 步权重 = 后段 × {3.0:.0f}",
        transform=ax.transAxes, fontsize=8.5, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

fig.suptitle("图 2  PLA / N2 / MPC 预测超调量 时序关系", fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_timeline.{e}")
plt.close(fig)
print("  图2: 时序关系")

print(f"\n  -> {FIG}")

