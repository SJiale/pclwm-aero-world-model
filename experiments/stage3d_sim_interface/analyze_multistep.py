"""多级阶跃结果分析与绘图"""
import re
import json
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
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight"})

N2_0 = 4349.29
SEGS = [(0, 8, 0), (8, 16, 750), (16, 24, 1500),
        (24, 32, 1000), (32, 40, 500), (40, 48, 0)]

txt = (D3 / "multistep.txt").read_text(encoding="utf-8")
rows = []
for ln in txt.split("\n"):
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+"
                 r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        rows.append([float(x) for x in m.groups()])

L = np.array(rows)
print("=" * 85)
print("多级阶跃结果分析")
print("=" * 85)
print(f"  已解析 {len(L)} 步")

if len(L) < 10:
    print("  数据不足，退出")
    raise SystemExit

t = L[:, 1]
n2 = L[:, 2]
ref = L[:, 3]
err = L[:, 4]
pla = L[:, 5]
wf = L[:, 6]
T45 = L[:, 7]
opt = L[:, 8]

print(f"  时间: {t[0]:.1f} - {t[-1]:.1f} s")
print(f"  N2: [{n2.min():.1f}, {n2.max():.1f}]")
print(f"  PLA: [{pla.min():.2f}, {pla.max():.2f}]")
print(f"  Wf: [{wf.min():.4f}, {wf.max():.4f}]")
print(f"  T45: [{T45.min():.1f}, {T45.max():.1f}] R")

# ══════════════════════════════════════════════════
# 逐段分析
# ══════════════════════════════════════════════════
print("\n" + "=" * 85)
print("逐段性能")
print("=" * 85)
print(f"  {'段':<20}{'目标N2':>10}{'末值':>10}{'末期误差':>10}"
      f"{'MAE':>9}{'最大误差':>10}{'PLA范围':>16}{'达标':>6}")
print("  " + "-" * 93)

seg_stats = []
for (t0, t1, d) in SEGS:
    m = (t >= t0) & (t < t1)
    if m.sum() < 3:
        continue
    tn, tr, te = t[m], n2[m], ref[m]
    pe = pla[m]
    # 末 20%
    n_tail = max(2, int(len(tn) * 0.2))
    final_err = abs(te[-n_tail:].mean())
    mae = np.mean(np.abs(te))
    maxe = np.max(np.abs(te))
    reached = final_err < 15.0
    name = f"{t0}-{t1}s ({d:+d})"
    print(f"  {name:<20}{tr[-1]:>10.1f}{tn[-1]:>10.1f}{te[-1]:>+10.1f}"
          f"{mae:>9.1f}{maxe:>10.1f}"
          f"[{pe.min():>5.1f},{pe.max():>5.1f}]{'✓' if reached else '✗':>6}")
    seg_stats.append({
        "seg": name, "t0": t0, "t1": t1, "delta": d,
        "target": float(tr[-1]), "final": float(tn[-1]),
        "final_err": float(te[-1]), "mae": float(mae),
        "max_err": float(maxe), "reached": bool(reached),
        "pla_min": float(pe.min()), "pla_max": float(pe.max()),
    })

# ══════════════════════════════════════════════════
# 阶跃响应特征
# ══════════════════════════════════════════════════
print("\n" + "=" * 85)
print("阶跃响应特征")
print("=" * 85)
print(f"  {'阶跃时刻':>10}{'Δref':>8}{'上升时间':>10}{'超调':>9}"
      f"{'稳态误差':>10}{'方向':>6}")
print("  " + "-" * 55)

step_metrics = []
for i in range(1, len(SEGS)):
    t_ev = SEGS[i][0]
    dtarget = SEGS[i][2] - SEGS[i - 1][2]
    if dtarget == 0:
        continue
    # 事件后窗口
    m = (t >= t_ev) & (t < SEGS[i][1])
    if m.sum() < 5:
        continue
    te, ne = t[m], n2[m]
    ref_target = N2_0 + SEGS[i][2]
    # 上升时间：达到 90% 变化量
    n_start = n2[(t >= t_ev) & (t < t_ev + 0.2)]
    n0 = n_start[0] if len(n_start) else ne[0]
    n90 = n0 + 0.9 * (ref_target - n0)
    if dtarget > 0:
        idx90 = np.where(ne >= n90)[0]
    else:
        idx90 = np.where(ne <= n90)[0]
    tr = float(te[idx90[0]] - t_ev) if len(idx90) else -1.0
    # 超调
    if dtarget > 0:
        ovs = float(max(0, ne.max() - ref_target))
    else:
        ovs = float(max(0, ref_target - ne.min()))
    se = float(abs(ne[-1] - ref_target))
    print(f"  {t_ev:>10.1f}{dtarget:>+8d}{tr:>10.2f}{ovs:>9.1f}"
          f"{se:>10.1f}{'↑' if dtarget > 0 else '↓':>6}")
    step_metrics.append({
        "t": t_ev, "dref": dtarget, "rise_time": tr,
        "overshoot": ovs, "steady_err": se,
        "dir": "up" if dtarget > 0 else "down",
    })

# ══════════════════════════════════════════════════
# 图 1：三联图（任务要求的主图）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax = axes[0]
ax.plot(t, ref, "k--", lw=1.5, label="N2_ref", zorder=1)
ax.plot(t, n2, "-", color="#2ca02c", lw=1.8, label="真实 N2")
for (t0, t1, d) in SEGS[1:]:
    ax.axvline(t0, color="gray", ls=":", lw=0.8, alpha=0.6)
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) N2 参考与真实跟踪", fontsize=10)
ax.legend(fontsize=9, loc="lower right")

ax = axes[1]
ax.plot(t, err, "-", color="#d62728", lw=1.5)
ax.axhline(0, color="0.3", lw=1.0)
ax.axhline(15, color="gray", ls=":", lw=0.9)
ax.axhline(-15, color="gray", ls=":", lw=0.9)
for (t0, t1, d) in SEGS[1:]:
    ax.axvline(t0, color="gray", ls=":", lw=0.8, alpha=0.6)
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 转速跟踪误差（虚线 = ±15 rpm）", fontsize=10)

ax = axes[2]
ax.step(t, pla, "-", color="#1f77b4", lw=1.4, where="post")
for (t0, t1, d) in SEGS[1:]:
    ax.axvline(t0, color="gray", ls=":", lw=0.8, alpha=0.6)
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("PLA 指令 (deg)")
ax.set_title("(c) PLA 控制指令", fontsize=10)

fig.suptitle("图 1  多级阶跃跟踪：GRU-MPC（0→+750→+1500→+1000→+500→0 rpm）",
             fontsize=12)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_multistep.{e}")
plt.close(fig)
print("\n  图1: 三联图")

# ══════════════════════════════════════════════════
# 图 2：辅助变量
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(t, wf, "-", color="#9467bd", lw=1.5)
ax.axhline(1.549, color="#d62728", ls="--", lw=1.3, label="WfMax=1.549")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("Wfact (lb/s)")
ax.set_title("(a) 燃油流量", fontsize=10)
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t, T45, "-", color="#ff7f0e", lw=1.5)
ax.set_xlabel("时间 (s)")
ax.set_ylabel("T45 (°R)")
ax.set_title("(b) 高压涡轮出口温度", fontsize=10)

ax = axes[2]
ax.plot(t, opt, "-", color="#2ca02c", lw=1.2)
ax.axhline(105, color="#d62728", ls="--", lw=1.5, label="105 ms")
ax.set_xlabel("时间 (s)")
ax.set_ylabel("MPC 优化耗时 (ms)")
ax.set_title(f"(c) 计算耗时（中位 {np.median(opt):.0f} ms）", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("图 2  辅助变量与计算开销", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_aux.{e}")
plt.close(fig)
print("  图2: 辅助变量")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
summary = {
    "n_steps": int(len(L)),
    "duration_s": float(t[-1]),
    "n2_range": [float(n2.min()), float(n2.max())],
    "pla_range": [float(pla.min()), float(pla.max())],
    "wf_range": [float(wf.min()), float(wf.max())],
    "t45_range": [float(T45.min()), float(T45.max())],
    "opt_ms": {"median": float(np.median(opt)),
               "p95": float(np.percentile(opt, 95)),
               "max": float(np.max(opt))},
    "segments": seg_stats,
    "steps": step_metrics,
    "overall": {
        "mae": float(np.mean(np.abs(err))),
        "max_abs_err": float(np.max(np.abs(err))),
        "iae": float(np.sum(np.abs(err)) * 0.105),
    },
}
(D3 / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
np.savez(D3 / "logs.npz", L=L)
print(f"\n  -> {D3/'summary.json'}")
print(f"  -> {FIG}")

print("\n" + "=" * 85)
print("总体指标")
print("=" * 85)
print(f"  MAE = {summary['overall']['mae']:.1f} rpm")
print(f"  最大绝对误差 = {summary['overall']['max_abs_err']:.1f} rpm")
print(f"  IAE = {summary['overall']['iae']:.1f} rpm·s")
print(f"  MPC 优化: 中位 {summary['opt_ms']['median']:.0f} ms, "
      f"P95 {summary['opt_ms']['p95']:.0f} ms")
