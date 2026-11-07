"""阶段 III-C 图表（用解析后的 npz）"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IIIC = Path(r"C:\pclwm\pclwm_stage3a\iiic")
FIG = IIIC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight"})

D = np.load(IIIC / "logs.npz")
M = D["mpc"]     # [k, n2, ref, err, pla, opt, step]
P = D["pi"]      # [k, n2, ref, err, pla, step]
DW = 0.105
MET = json.loads((IIIC / "metrics.json").read_text(encoding="utf-8"))

# MPC 列: 0=k,1=n2,2=ref,3=err,4=pla,5=opt,6=step
# PI  列: 0=k,1=n2,2=ref,3=err,4=pla,5=step
t_m = M[:, 0] * DW
t_p = P[:, 0] * DW

print("=" * 85)
print("生成 III-C 图表")
print(f"  MPC: {M.shape}  PI: {P.shape}")
print("=" * 85)

# ══════════════════════════════════════════════════
# 图 1：闭环对比（3 联）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

ax = axes[0]
ax.plot(t_m, M[:, 2], "k--", lw=1.2, label="参考", zorder=1)
ax.plot(t_m, M[:, 1], "-", color="#2ca02c", lw=1.9, label="GRU-MPC")
ax.plot(t_p, P[:, 1], "-", color="#ff7f0e", lw=1.9, label="PI (整定)")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) N2 参考跟踪", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t_m, M[:, 3], "-", color="#2ca02c", lw=1.9, label="GRU-MPC")
ax.plot(t_p, P[:, 3], "-", color="#ff7f0e", lw=1.9, label="PI (整定)")
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 跟踪误差", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
ax.step(t_m, M[:, 4], "-", color="#2ca02c", lw=1.4, label="GRU-MPC", where="post")
ax.step(t_p, P[:, 4], "-", color="#ff7f0e", lw=1.4, label="PI (整定)", where="post")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("PLA 指令 (deg)")
ax.set_title("(c) PLA 动作", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle(f"图 1  真实 Simulink 闭环：GRU-MPC（{len(M)} 步）vs PI 整定（{len(P)} 步）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_closedloop.{e}")
plt.close(fig)
print("  图1")

# ══════════════════════════════════════════════════
# 图 2：MPC 耗时
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))

ax = axes[0]
t_opt = M[:, 5]
ax.plot(t_m, t_opt, "o-", color="#2ca02c", ms=4, lw=1.2)
ax.axhline(105, color="#d62728", ls="--", lw=1.8, label="控制周期 105 ms")
ax.axhline(np.median(t_opt), color="#1f77b4", ls=":", lw=1.5,
           label=f"中位 {np.median(t_opt):.1f} ms")
ax.set_xlabel("闭环步")
ax.set_ylabel("MPC 优化耗时 (ms)")
ax.set_title("(a) 逐次 MPC 优化耗时", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
vals = [np.min(t_opt), np.median(t_opt), np.percentile(t_opt, 95), np.max(t_opt)]
labs = ["最小", "中位", "P95", "最大"]
cols = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"]
bars = ax.bar(labs, vals, color=cols, alpha=0.85)
ax.axhline(105, color="k", ls="--", lw=1.5, label="105 ms")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+2, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("耗时 (ms)")
n_ok = MET["timing"]["n_ok"]
n_tot = MET["timing"]["n_total"]
ax.set_title(f"(b) 统计（{n_ok}/{n_tot} 步满足预算）", fontsize=10)
ax.legend(fontsize=8)
ax.set_ylim(0, max(vals)*1.2)

fig.suptitle("图 2  MPC 优化耗时（Simulink 环境 10.2 s/步不计入此图）", fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_timing.{e}")
plt.close(fig)
print("  图2")

# ══════════════════════════════════════════════════
# 图 3：消融 Pareto
# ══════════════════════════════════════════════════
abl = {
    "baseline":   (29.57, 19.394),
    "w_du=1":     (30.23, 16.123),
    "w_du=10":    (32.45, 12.120),
    "w_du=100":   (36.81,  5.671),
    "warm_start": (17.66,  7.899),
    "dpla=0.3":   (34.65,  4.886),
    "no_short_w": (34.70, 12.783),
    "iter=6":     (30.70, 17.486),
}

fig, ax = plt.subplots(figsize=(8.5, 5))
for name, (mae, sd) in abl.items():
    c = "#2ca02c" if name == "warm_start" else "#1f77b4"
    mk = "s" if name == "warm_start" else "o"
    sz = 160 if name == "warm_start" else 100
    ax.scatter(sd, mae, s=sz, color=c, marker=mk, zorder=5, alpha=0.85,
               edgecolors="k", linewidths=0.6)
    ax.annotate(name, (sd, mae), textcoords="offset points",
                xytext=(7, 6), fontsize=8)

pts = sorted(abl.values())
front, best = [], 1e9
for mae, sd in pts:
    if mae < best:
        front.append((sd, mae)); best = mae
if front:
    fx, fy = zip(*front)
    ax.plot(fx, fy, "r--", lw=1.4, alpha=0.6, label="Pareto 前沿")

ax.set_xlabel("PLA 总变化量 $\\Sigma|\\Delta$PLA$|$ (deg)　←　越小越平滑")
ax.set_ylabel("MAE (rpm)　←　越小越准")
ax.set_title("图 3  MPC 消融：跟踪精度 vs 动作平滑（左下最优）", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_ablation.{e}")
plt.close(fig)
print("  图3")

# ══════════════════════════════════════════════════
# 图 4：指标对比柱状
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
items = [
    ("MAE (rpm)", "mae"),
    ("超调 (rpm)", "overshoot"),
    ("稳态误差 (rpm)", "final"),
    ("$\\Sigma|\\Delta$PLA$|$ (deg)", "sdpla"),
]
for ax, (lab, key) in zip(axes, items):
    a = MET["mpc"][key] if key in MET["mpc"] else 0
    b = MET["pi"][key] if key in MET["pi"] else 0
    bars = ax.bar(["MPC", "PI"], [a, b], color=["#2ca02c", "#ff7f0e"], alpha=0.85)
    ax.set_ylabel(lab, fontsize=8)
    ax.set_title(lab, fontsize=9)
    for bb, v in zip(bars, [a, b]):
        ax.text(bb.get_x()+bb.get_width()/2, v*1.02 if v > 0 else 0.1,
                f"{v:.1f}", ha="center", fontsize=8)

fig.suptitle("图 4  关键指标对比（越低越好）", fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_metrics.{e}")
plt.close(fig)
print("  图4")

print(f"\n  -> {FIG}")
