"""阶段 II 控制环境准备：图表"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(r"C:\pclwm\ctrl\probe\figures")
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight",
})

# ══════════════════════════════════════════════════
# 图 1：PLA -> Wfdmd 静态增益（线性度）
# ══════════════════════════════════════════════════
pla = np.array([51, 53, 54, 55, 56, 57, 59], dtype=float)
wfd = np.array([0.5004, 0.5583, 0.5881, 0.6180, 0.6498, 0.6836, 0.7498])
n2 = np.array([3936.39, 4146.36, 4249.88, 4349.53, 4442.53, 4533.89, 4708.60])
n3 = np.array([18931.63, 19181.64, 19301.03, 19411.55, 19515.14, 19619.53, 19818.85])

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

ax = axes[0]
ax.plot(pla, wfd, "o-", color="#1f77b4", ms=6)
# 线性拟合
p = np.polyfit(pla, wfd, 1)
xx = np.linspace(pla.min(), pla.max(), 50)
ax.plot(xx, np.polyval(p, xx), "--", color="gray", lw=1.2,
        label=f"线性拟合 k={p[0]:.4f}")
ax.set_xlabel("PLA (deg)")
ax.set_ylabel("稳态 Wfdmd (lb/s)")
ax.set_title("(a) PLA → 燃油指令", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(pla, n2, "o-", color="#d62728", ms=6)
p2 = np.polyfit(pla, n2, 1)
ax.plot(xx, np.polyval(p2, xx), "--", color="gray", lw=1.2,
        label=f"线性拟合 k={p2[0]:.1f}")
ax.set_xlabel("PLA (deg)")
ax.set_ylabel("稳态 N2 (rpm)")
ax.set_title("(b) PLA → 低压转速", fontsize=10)
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(pla, n3, "o-", color="#2ca02c", ms=6)
p3 = np.polyfit(pla, n3, 1)
ax.plot(xx, np.polyval(p3, xx), "--", color="gray", lw=1.2,
        label=f"线性拟合 k={p3[0]:.1f}")
ax.set_xlabel("PLA (deg)")
ax.set_ylabel("稳态 N3 (rpm)")
ax.set_title("(c) PLA → 高压转速", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("图 1  上层监督控制（S）的静态映射：PLA → 稳态输出", fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_pla_static_gain.{e}")
plt.close(fig)
print("  图1")

# ══════════════════════════════════════════════════
# 图 2：局部增益（非线性程度）
# ══════════════════════════════════════════════════
mid = (pla[:-1] + pla[1:]) / 2
kw = np.diff(wfd) / np.diff(pla)
kn2 = np.diff(n2) / np.diff(pla)
kn3 = np.diff(n3) / np.diff(pla)

fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(mid, kw / kw.mean(), "o-", label="dWfdmd/dPLA", color="#1f77b4")
ax.plot(mid, kn2 / kn2.mean(), "s-", label="dN2/dPLA", color="#d62728")
ax.plot(mid, kn3 / kn3.mean(), "^-", label="dN3/dPLA", color="#2ca02c")
ax.axhline(1.0, color="gray", ls="--", lw=1)
ax.set_xlabel("PLA (deg)")
ax.set_ylabel("归一化局部增益（相对均值）")
ax.set_title("图 2  局部增益随工作点的变化（非线性程度）", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_local_gain.{e}")
plt.close(fig)
print("  图2")

# ══════════════════════════════════════════════════
# 图 3：阶跃响应（PLA 55->57）
# ══════════════════════════════════════════════════
# 用已知的 63%/90% 响应时间重建示意曲线
t = np.linspace(0, 20, 400)
wf0, wf1 = 0.6180, 0.6836
# 一阶近似（tau 从 63% 响应时间 0.575s 反推）
tau = -0.575 / np.log(1 - 0.632)
wf_curve = wf0 + (wf1 - wf0) * (1 - np.exp(-t / tau))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
ax.plot(t, wf_curve, color="#1f77b4", lw=1.8)
ax.axhline(wf0, color="gray", ls=":", lw=1, label=f"阶跃前 {wf0:.4f}")
ax.axhline(wf1, color="gray", ls=":", lw=1, label=f"稳态 {wf1:.4f}")
ax.axvline(0.575, color="#d62728", ls="--", lw=1.2, label="63% 时间 0.575 s")
ax.axvline(1.355, color="#ff7f0e", ls="--", lw=1.2, label="90% 时间 1.355 s")
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("Wfdmd (lb/s)")
ax.set_title("(a) 燃油指令响应（一阶近似）", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0, 5)

ax = axes[1]
n2_0, n2_1 = 4349.30, 4533.80
tau_n2 = 1.5   # 转速慢于燃油
n2_curve = n2_0 + (n2_1 - n2_0) * (1 - np.exp(-t / tau_n2))
ax.plot(t, n2_curve, color="#d62728", lw=1.8)
ax.axhline(n2_0, color="gray", ls=":", lw=1)
ax.axhline(n2_1, color="gray", ls=":", lw=1)
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(b) 低压转速响应（慢于燃油）", fontsize=10)
ax.set_xlim(0, 10)

fig.suptitle("图 3  PLA 55→57 阶跃响应（+2 deg）", fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_step_response.{e}")
plt.close(fig)
print("  图3")

# ══════════════════════════════════════════════════
# 图 4：动作覆盖
# ══════════════════════════════════════════════════
cov = json.loads((Path(r"C:\pclwm\ctrl\probe") / "action_coverage.json")
                 .read_text(encoding="utf-8"))
rows = cov["rows"]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

ax = axes[0]
for grp, col in [("A", "#1f77b4"), ("B", "#ff7f0e"), ("C", "#2ca02c")]:
    sub = [r for r in rows if r["group"] == grp]
    if not sub:
        continue
    idx = np.arange(len(sub))
    ax.scatter(idx, [r["pla_delta"] for r in sub], color=col, s=50,
               label=f"组 {grp}（{len(sub)} 条）", alpha=0.8)
ax.axhline(0.5, color="#d62728", ls="--", lw=1.2,
           label="MPC 典型步长 ±0.5 deg")
ax.set_xlabel("轨迹索引")
ax.set_ylabel("PLA 变化幅度 (deg)")
ax.set_title("(a) 各轨迹的 PLA 激励幅度", fontsize=10)
ax.legend(fontsize=8)
ax.set_yscale("symlog", linthresh=1)

ax = axes[1]
pla_min = [min(r["pla_min"] for r in rows if r["group"] == g) if
           any(r["group"] == g for r in rows) else 0 for g in "ABC"]
pla_max = [max(r["pla_max"] for r in rows if r["group"] == g) if
           any(r["group"] == g for r in rows) else 0 for g in "ABC"]
y = np.arange(3)
for i, g in enumerate("ABC"):
    ax.barh(i, pla_max[i] - pla_min[i], left=pla_min[i],
            height=0.5, alpha=0.7, color=["#1f77b4", "#ff7f0e", "#2ca02c"][i])
ax.set_yticks(y)
ax.set_yticklabels([f"组 {g}" for g in "ABC"])
ax.set_xlabel("PLA (deg)")
ax.set_title("(b) 各组 PLA 覆盖范围", fontsize=10)
ax.set_xlim(35, 85)

fig.suptitle("图 4  现有 26 条训练数据的动作覆盖", fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_coverage.{e}")
plt.close(fig)
print("  图4")

print(f"\n  -> {FIG}")
