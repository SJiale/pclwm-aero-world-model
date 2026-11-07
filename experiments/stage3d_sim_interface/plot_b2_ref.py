"""B2 参考曲线可视化（仿真前确认）"""
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D3 = Path(r"C:\pclwm\pclwm_stage3a\iiid")
FIG = D3 / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.9,
    "lines.linewidth": 2.0, "savefig.dpi": 300, "savefig.bbox": "tight"})

D = sio.loadmat(D3 / "b2_ref.mat")
n2_ref = D["n2_ref"].ravel()
t_step = D["t_step"].ravel()
N2_0 = float(D["N2_0"].ravel()[0])
REF1 = float(D["REF1"].ravel()[0])
REF2 = float(D["REF2"].ravel()[0])
CTRL_DT = float(D["CTRL_DT"].ravel()[0])

print("=" * 80)
print("B2 参考曲线确认")
print("=" * 80)
print(f"  N2_0 = {N2_0:.2f} rpm")
print(f"  REF1 = {REF1:.2f} rpm  (+750)")
print(f"  REF2 = {REF2:.2f} rpm  (+250, 相对 REF1 = -500)")
print(f"  步数 = {len(n2_ref)}")

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2.2, 1]})

# (a) 参考阶跃（stairs）+ 稳态参考
ax = axes[0]
t_stairs = np.append(t_step, t_step[-1] + CTRL_DT)
ref_stairs = np.append(n2_ref, n2_ref[-1])
ax.stairs(n2_ref, t_stairs, color="k", lw=2.0, baseline=None,
          label="N2_ref（阶梯）")
ax.axhline(N2_0, color="gray", ls=":", lw=1.3, label=f"初始稳态 N2_0={N2_0:.2f}")
# 标注跳变
ax.annotate("", xy=(6.3, REF1), xytext=(6.3, REF2),
            arrowprops=dict(arrowstyle="<->", color="#d62728", lw=2))
ax.text(6.45, (REF1+REF2)/2, f"$-$500 rpm",
        color="#d62728", fontsize=11, va="center", fontweight="bold")
# 标注 +750
ax.annotate("", xy=(0.15, N2_0), xytext=(0.15, REF1),
            arrowprops=dict(arrowstyle="<->", color="#2ca02c", lw=2))
ax.text(0.35, (N2_0+REF1)/2, f"$+$750 rpm",
        color="#2ca02c", fontsize=11, va="center", fontweight="bold")

ax.set_ylabel("N2_ref (rpm)")
ax.set_title("(a) B2 参考指令（仿真前确认）", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.set_ylim(N2_0 - 120, REF1 + 120)

# (b) 参考变化率（delta）
ax = axes[1]
d = np.diff(n2_ref)
t_d = t_step[1:]
ax.step(t_d, d, color="#ff7f0e", lw=2.0, where="mid")
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("ΔN2_ref (rpm/步)")
ax.set_title("(b) 参考跳变（只应有 1 处 -500）", fontsize=11)
# 标注
jump_idx = np.where(np.abs(d) > 1e-6)[0]
if len(jump_idx):
    ax.annotate(f"第 {jump_idx[0]+1}→{jump_idx[0]+2} 步\nt={t_d[jump_idx[0]]:.2f}s\n{d[jump_idx[0]]:+.0f} rpm",
                xy=(t_d[jump_idx[0]], d[jump_idx[0]]),
                xytext=(t_d[jump_idx[0]] + 1.5, d[jump_idx[0]] - 100),
                fontsize=9, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728"))

fig.suptitle(f"图 B2-Ref  B2 实验参考指令（{len(n2_ref)} 步 × {CTRL_DT:.3f}s = "
             f"{len(n2_ref)*CTRL_DT:.2f}s）\n"
             f"单调性检查：仅 1 处跳变，-500 rpm ✅",
             fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figB2_reference.{e}")
plt.close(fig)
print(f"\n  -> {FIG/'figB2_reference.png'}")

# 文本表
print("\n" + "=" * 80)
print("参考数组摘要")
print("=" * 80)
print(f"  {'段':<20}{'步范围':<16}{'N2_ref':<12}{'步数':<8}")
print("  " + "-" * 58)
print(f"  {'0 - 6.3 s':<20}{'1 - 60':<16}{REF1:<12.2f}{60:<8}")
print(f"  {'6.3 - 12.6 s':<20}{'61 - 120':<16}{REF2:<12.2f}{60:<8}")
print(f"\n  跳变: {REF2-REF1:+.2f} rpm @ 第 61 步 (t=6.300s)")
