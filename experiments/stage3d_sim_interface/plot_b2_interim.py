"""B2 中间结果绘图（81/120 步，第二阶段参考值有误）

如实标注：
  - 0-6.3s 阶段（+750）参考值正确，数据有效
  - 6.3s 之后参考值设错（3849 应为 4599），该段不能作为跟踪结果
"""
import re
from pathlib import Path
import numpy as np
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
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.7, "savefig.dpi": 300, "savefig.bbox": "tight"})

N2_0 = 4349.29
T_REF_BUG = 6.30     # 参考值出错时刻

txt = (D3 / "b2.txt").read_text(encoding="utf-8")
rows = []
for ln in txt.split("\n"):
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+"
                 r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        rows.append([float(x) for x in m.groups()])

L = np.array(rows)
print("=" * 80)
print("B2 中间结果")
print("=" * 80)
print(f"  已解析 {len(L)} 步")

t = L[:, 1]
n2 = L[:, 2]
ref = L[:, 3]
err = L[:, 4]
pla = L[:, 5]
wf = L[:, 6]
T45 = L[:, 7]
wfmax = L[:, 8]
opt = L[:, 9]

print(f"  时间范围: {t[0]:.2f} - {t[-1]:.2f} s")
print(f"  N2 范围: [{n2.min():.1f}, {n2.max():.1f}]")
print(f"  PLA 范围: [{pla.min():.2f}, {pla.max():.2f}]")

# 分段
m1 = t < T_REF_BUG       # 阶段1（参考正确）
m2 = t >= T_REF_BUG      # 阶段2（参考错误）

print(f"\n  阶段1（0-6.3s，参考正确）: {m1.sum()} 步")
print(f"  阶段2（6.3s 后，参考错误）: {m2.sum()} 步")

# ══════════════════════════════════════════════════
# 图 1：三联图
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# (a) N2 跟踪
ax = axes[0]
ax.plot(t, ref, "k--", lw=1.5, label="N2_ref", zorder=1)
ax.plot(t, n2, "-", color="#2ca02c", lw=2.0, label="真实 N2")
ax.axvline(T_REF_BUG, color="#d62728", ls=":", lw=1.5, alpha=0.8)
ax.axvspan(T_REF_BUG, t[-1], alpha=0.10, color="#d62728")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) N2 参考与真实跟踪", fontsize=11)
ax.legend(fontsize=9, loc="lower right")
ax.text(T_REF_BUG + 0.15, n2.min() + 30,
        "⚠ 此段参考值设错\n(3849 应为 4599)",
        fontsize=8.5, color="#d62728", va="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85,
                  edgecolor="#d62728"))

# (b) 跟踪误差
ax = axes[1]
ax.plot(t, err, "-", color="#d62728", lw=1.8)
ax.axhline(0, color="0.3", lw=1.0)
ax.axvline(T_REF_BUG, color="#d62728", ls=":", lw=1.5, alpha=0.8)
ax.axvspan(T_REF_BUG, t[-1], alpha=0.10, color="#d62728")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 转速跟踪误差（N2 − N2_ref）", fontsize=11)

# (c) PLA
ax = axes[2]
ax.step(t, pla, "-", color="#1f77b4", lw=1.5, where="post")
ax.axhline(48, color="gray", ls="--", lw=1.0, label="PLA 下限 48")
ax.axhline(78, color="gray", ls="--", lw=1.0, label="PLA 上限 78")
ax.axvline(T_REF_BUG, color="#d62728", ls=":", lw=1.5, alpha=0.8)
ax.axvspan(T_REF_BUG, t[-1], alpha=0.10, color="#d62728")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("PLA 指令 (deg)")
ax.set_title("(c) PLA 控制指令", fontsize=11)
ax.legend(fontsize=8, loc="upper left")

fig.suptitle(f"图 1  B2 中间结果：{len(L)}/120 步（0–{t[-1]:.2f} s）\n"
             f"阶段1（+750 rpm）数据有效；阶段2 参考值有误，仅供参考",
             fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figB2_interim.{e}")
plt.close(fig)
print("\n  图1: 三联图")

# ══════════════════════════════════════════════════
# 图 2：辅助变量
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

ax = axes[0]
ax.plot(t, wf, "-", color="#9467bd", lw=1.6)
ax.axhline(1.5489, color="#d62728", ls="--", lw=1.3, label="WfMax=1.549")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("Wfact (lb/s)")
ax.set_title("(a) 燃油流量", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t, T45, "-", color="#ff7f0e", lw=1.6)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("T45 (°R)")
ax.set_title("(b) 高压涡轮出口温度", fontsize=10)

ax = axes[2]
ax.plot(t, opt, "o-", color="#2ca02c", ms=3.5, lw=1.2)
ax.axhline(105, color="#d62728", ls="--", lw=1.5, label="105 ms")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("MPC 优化耗时 (ms)")
ax.set_title(f"(c) 计算耗时（中位 {np.median(opt):.0f} ms）", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("图 2  辅助变量与计算开销", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figB2_aux.{e}")
plt.close(fig)
print("  图2: 辅助变量")

# ══════════════════════════════════════════════════
# 阶段1 详细分析（参考正确，结论有效）
# ══════════════════════════════════════════════════
print("\n" + "=" * 80)
print("阶段 1 分析（+750 rpm，参考正确，结论有效）")
print("=" * 80)

t1, n21 = t[m1], n2[m1]
ref1 = 5099.29
e1 = n21 - ref1

# 上升时间（10%→90%）
n_start = n21[0]
n_target = ref1
n10 = n_start + 0.1 * (n_target - n_start)
n90 = n_start + 0.9 * (n_target - n_start)
i10 = np.where(n21 >= n10)[0]
i90 = np.where(n21 >= n90)[0]
tr = (t1[i90[0]] - t1[i10[0]]) if len(i90) and len(i10) else -1

print(f"  初始 N2: {n_start:.2f}")
print(f"  目标: {n_target:.2f} (+750)")
print(f"  10% 时刻: {t1[i10[0]]:.2f}s" if len(i10) else "  10%: 未达到")
print(f"  90% 时刻: {t1[i90[0]]:.2f}s" if len(i90) else "  90%: 未达到")
print(f"  10%-90% 上升时间: {tr:.2f} s")
print(f"  最大 N2: {n21.max():.2f}")
print(f"  超调: {n21.max() - n_target:.2f} rpm ({(n21.max()-n_target)/750*100:.1f}%)")
print(f"  末步 N2: {n21[-1]:.2f}")
print(f"  末步误差: {n21[-1] - n_target:+.2f} rpm")
print(f"  阶段内 MAE: {np.mean(np.abs(e1)):.2f} rpm")
print(f"  PLA 范围: [{pla[m1].min():.2f}, {pla[m1].max():.2f}]")
print(f"  Wf 范围: [{wf[m1].min():.4f}, {wf[m1].max():.4f}]")
print(f"  T45 范围: [{T45[m1].min():.1f}, {T45[m1].max():.1f}] R")
print(f"  Wf 饱和: {np.sum(wf[m1] >= wfmax[m1]*0.999)} / {m1.sum()} 点")

# 是否达到目标
target_band = 15.0
reached = abs(n21[-1] - n_target) < target_band
print(f"\n  末步误差 {abs(n21[-1]-n_target):.1f} rpm < {target_band} -> "
      f"{'✅ 已稳定' if reached else '❌ 未收敛'}")

# 保存
summary = {
    "n_steps_done": int(len(L)),
    "n_steps_total": 120,
    "t_end": float(t[-1]),
    "ref_bug_at_s": T_REF_BUG,
    "stage1": {
        "ref": ref1, "delta": 750.0,
        "n_start": float(n_start),
        "n_max": float(n21.max()),
        "overshoot": float(n21.max() - n_target),
        "overshoot_pct": float((n21.max()-n_target)/750*100),
        "rise_time_10_90_s": float(tr),
        "final_n2": float(n21[-1]),
        "final_err": float(n21[-1] - n_target),
        "mae": float(np.mean(np.abs(e1))),
        "pla_range": [float(pla[m1].min()), float(pla[m1].max())],
        "wf_range": [float(wf[m1].min()), float(wf[m1].max())],
        "t45_range": [float(T45[m1].min()), float(T45[m1].max())],
        "wf_saturated_pts": int(np.sum(wf[m1] >= wfmax[m1]*0.999)),
        "reached": bool(reached),
    },
    "stage2": {
        "note": "参考值设错：实现为 3849（N2_0-500），应为 4599（峰值-500）",
        "ref_implemented": 3849.29,
        "ref_expected": 4599.29,
    },
    "timing": {
        "opt_median_ms": float(np.median(opt)),
        "opt_p95_ms": float(np.percentile(opt, 95)),
        "opt_max_ms": float(np.max(opt)),
    },
}
(D3 / "b2_interim_summary.json").write_text(
    __import__("json").dumps(summary, ensure_ascii=False, indent=1),
    encoding="utf-8")
print(f"\n  -> {FIG}")
print(f"  -> {D3/'b2_interim_summary.json'}")
