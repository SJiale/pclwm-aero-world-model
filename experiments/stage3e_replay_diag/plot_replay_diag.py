"""任务5 图表：GRU 预测诊断 + B2 时序"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = Path(r"C:\pclwm\pclwm_stage3a\replay")
FIG = REP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.85,
    "lines.linewidth": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight"})

DT = 0.105
N2_0 = 4349.29
N2_TARGET = 5099.29

b2 = json.loads((REP / "b2_timing.json").read_text(encoding="utf-8"))
lh = json.loads((REP / "long_horizon.json").read_text(encoding="utf-8"))
rp = json.loads((REP / "gru_replay.json").read_text(encoding="utf-8"))

D = b2["data"]
t = np.array(D["t"])
n2 = np.array(D["n2"])
pla = np.array(D["pla"])
ref = np.array(D["ref"])

print("=" * 88)
print("生成诊断图表")
print("=" * 88)

# ══════════════════════════════════════════════════
# 图 1：GRU 多步预测 vs 真实（test_ctrl 回放）
# ══════════════════════════════════════════════════
Z = np.load(REP / "gru_replay.npz", allow_pickle=True)
pred_curves = Z["pred_curves"]
true_curves = Z["true_curves"]
pla_curves = Z["pla_curves"]
labels = Z["labels"]
t_events = Z["t_events"]
dpla = Z["dpla"]

n_ev = len(pred_curves)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for i in range(min(n_ev, 6)):
    ax = axes[i]
    t_axis = np.arange(32) * DT
    ax.plot(t_axis, true_curves[i], "k-", lw=2.0, label="真实值", zorder=5)
    ax.plot(t_axis, pred_curves[i], "-", color="#2ca02c", lw=1.8,
            label="GRU 预测")

    # 标注 1.05s 和 3.36s
    for kk, nm in [(10, "1.05s"), (32, "3.36s")]:
        if kk <= 32:
            ax.axvline(t_axis[kk-1] if kk < 32 else t_axis[-1],
                       color="gray", ls=":", lw=0.8, alpha=0.6)

    e1 = pred_curves[i][0] - true_curves[i][0]
    e32 = pred_curves[i][31] - true_curves[i][31]
    ax.set_title(f"{labels[i].split('_')[0]}  t={t_events[i]:.1f}s  "
                 f"ΔPLA={dpla[i]:+.2f}°\n"
                 f"err(1步)={e1:+.1f}, err(32步)={e32:+.1f} rpm",
                 fontsize=8.5)
    ax.set_xlabel("阶跃后时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    if i == 0:
        ax.legend(fontsize=7.5)

for j in range(n_ev, 6):
    axes[j].axis("off")

fig.suptitle("图 1  GRU 多步预测 vs 真实（test_ctrl 阶跃事件，历史=真实观测，未来PLA=真实执行）",
             fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_gru_multistep.{e}")
plt.close(fig)
print("  图1: GRU 多步预测")

# ══════════════════════════════════════════════════
# 图 2：预测误差 vs 时域
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

# (a) 误差随时域
ax = axes[0]
steps = [1, 10, 32]
maes = [rp["mae_1"], rp["mae_10"], rp["mae_32"]]
biases = [rp["bias_1"], rp["bias_10"], rp["bias_32"]]
t_axis = np.array(steps) * DT

ax.plot(t_axis, maes, "o-", color="#d62728", lw=2.0, ms=8, label="MAE")
ax.plot(t_axis, [-b for b in biases], "s--", color="#1f77b4", lw=1.8,
        ms=7, label="有符号均值（取反）")
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xlabel("预测时域 (s)")
ax.set_ylabel("N2 误差 (rpm)")
ax.set_title("(a) 预测误差随时域增长", fontsize=10)
ax.legend(fontsize=8.5)

# (b) 稳态增益对比（关键是这里准）
ax = axes[1]
pla_list = [r["pla"] for r in lh["results"]]
k_gru = [(r["dn2_pred_400"]) / (r["pla"]-55) for r in lh["results"]]
k_true = [(r["dn2_true"]) / (r["pla"]-55) for r in lh["results"]]
ax.plot(pla_list, k_gru, "o-", color="#2ca02c", lw=2.0, ms=8,
        label="GRU 增益")
ax.plot(pla_list, k_true, "s--", color="#ff7f0e", lw=1.8, ms=7,
        label="真值增益（可行性扫描）")
ax.set_xlabel("PLA (deg)")
ax.set_ylabel("静态增益 ΔN2/ΔPLA (rpm/deg)")
ax.set_title("(b) 静态增益：GRU 误差仅 0.5-1.9%", fontsize=10)
ax.legend(fontsize=8.5)

fig.suptitle("图 2  GRU 预测精度：时域越长误差越大，但稳态增益很准", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_error_vs_horizon.{e}")
plt.close(fig)
print("  图2: 误差 vs 时域 + 增益对比")

# ══════════════════════════════════════════════════
# 图 3：B2 时序（PLA 峰值 vs N2 峰值）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# (a) N2 + PLA 双轴
ax = axes[0]
ax.plot(t, n2, "-", color="#2ca02c", lw=2.2, label="真实 N2")
ax.plot(t, ref, "k--", lw=1.3, label="N2_ref")
ax.axhline(N2_TARGET, color="gray", ls=":", lw=1.0)
ax.set_ylabel("N2 (rpm)")
ax.legend(fontsize=8.5, loc="lower right")

ax2 = ax.twinx()
ax2.plot(t, pla, "-", color="#1f77b4", lw=1.8, alpha=0.85, label="PLA")
ax2.set_ylabel("PLA (deg)", color="#1f77b4")
ax2.tick_params(axis="y", labelcolor="#1f77b4")

# 标注峰值
i_pm = b2["pla_peak_k"] - 1
i_nm = b2["n2_peak_k"] - 1
ax2.annotate(f"PLA 峰值 {b2['pla_peak']:.1f}°\nt={b2['pla_peak_t']:.2f}s",
             xy=(b2["pla_peak_t"], b2["pla_peak"]),
             xytext=(b2["pla_peak_t"]+0.5, b2["pla_peak"]-2),
             fontsize=8.5, color="#1f77b4",
             arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.annotate(f"N2 峰值 {b2['n2_peak']:.1f}\nt={b2['n2_peak_t']:.2f}s",
            xy=(b2["n2_peak_t"], b2["n2_peak"]),
            xytext=(b2["n2_peak_t"]+0.6, b2["n2_peak"]+18),
            fontsize=8.5, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728"))

ax.set_title("(a) N2 与 PLA 时序（双轴）", fontsize=10.5)

# (b) 跟踪误差
ax = axes[1]
ax.plot(t, n2 - N2_TARGET, "-", color="#d62728", lw=2.0)
ax.axhline(0, color="0.3", lw=1.0)
ax.axvspan(b2["n2_cross_target_t"] or 2.0, t[-1], alpha=0.08, color="#d62728")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 跟踪误差（阴影 = N2 越过目标后）", fontsize=10.5)

fig.suptitle(f"图 3  B2 第一阶段时序：PLA 峰值 71.4°（对应稳态 N2≈5672）→ "
             f"超调 +144 rpm\n"
             f"PLA 峰值早于 N2 峰值 {b2['delay_pla_to_n2_peak_s']:.2f}s",
             fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_b2_timing.{e}")
plt.close(fig)
print("  图3: B2 时序")

# ══════════════════════════════════════════════════
# 图 4：GRU@32步 vs 稳态真值（达成率）
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.8))

pla_v = [r["pla"] for r in lh["results"]]
true_ss = [r["true_steady"] for r in lh["results"]]
pred32 = [r["pred"]["32"] for r in lh["results"]]

x = np.arange(len(pla_v))
w = 0.35
ax.bar(x - w/2, [v - N2_0 for v in true_ss], w, label="稳态真值 ΔN2",
       color="#ff7f0e", alpha=0.85)
ax.bar(x + w/2, [v - N2_0 for v in pred32], w, label="GRU@32步 ΔN2",
       color="#2ca02c", alpha=0.85)
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"{p:.0f}°" for p in pla_v])
ax.set_xlabel("PLA 阶跃后的常值")
ax.set_ylabel("ΔN2 (rpm) 相对起点 4349")
ax.set_title("图 4  GRU 在 32 步（3.36s，即 MPC horizon）内已达稳态的 93-97%\n"
             "→ 不存在「预测不到」的问题", fontsize=10.5)
ax.legend(fontsize=9)

# 标注达成率
for i, (ts, p32) in enumerate(zip(true_ss, pred32)):
    rate = (p32 - N2_0) / (ts - N2_0) * 100
    ax.text(i, max(ts, p32) - N2_0 + 25, f"{rate:.0f}%",
            ha="center", fontsize=8)

fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_h32_achieve.{e}")
plt.close(fig)
print("  图4: GRU@32步达成率")

print(f"\n  -> {FIG}")

