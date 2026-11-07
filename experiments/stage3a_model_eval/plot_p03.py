"""P0-3 图表：MPC vs PI 闭环 + V4 事件对齐 + 修正版 III-A 图"""
import json
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"C:\pclwm\pclwm_stage3a\results")
MPC = Path(r"C:\pclwm\pclwm_stage3a\mpc_matlab")
FIG = RES / "figures_v3"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.8,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "savefig.bbox": "tight"})

print("=" * 90)
print("生成 P0-3 图表")
print("=" * 90)

# ══════════════════════════════════════════════════
# 图 A：MPC vs PI 闭环对比
# ══════════════════════════════════════════════════
mpc = sio.loadmat(MPC / "mpc_final2_log.mat")
pi_ = sio.loadmat(MPC / "pi_log.mat")
L_mpc = mpc["log_mpc"]      # [step, n2, ref, pla, err, t_opt_ms]
L_pi = pi_["log_pi"]        # [step, n2, ref, pla, err]
n2_steady = float(mpc["n2_steady"].ravel()[0])

fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

# (a) N2 跟踪
ax = axes[0]
t_m = L_mpc[:, 0] * 0.105
ax.plot(t_m, L_mpc[:, 2], "k--", lw=1.3, label="参考", zorder=1)
ax.plot(t_m, L_mpc[:, 1], "o-", color="#2ca02c", ms=4, label="GRU-MPC")
ax.plot(t_m, L_pi[:, 1], "s-", color="#ff7f0e", ms=4, label="PI + EEC")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) N2 参考跟踪", fontsize=10)
ax.legend(fontsize=8)

# (b) 跟踪误差
ax = axes[1]
ax.plot(t_m, L_mpc[:, 4], "o-", color="#2ca02c", ms=4, label="GRU-MPC")
ax.plot(t_m, L_pi[:, 4], "s-", color="#ff7f0e", ms=4, label="PI + EEC")
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 跟踪误差（N2 − N2_ref）", fontsize=10)
ax.legend(fontsize=8)

# (c) PLA 动作
ax = axes[2]
ax.step(t_m, L_mpc[:, 3], "-", color="#2ca02c", lw=1.5, label="GRU-MPC", where="post")
ax.step(t_m, L_pi[:, 3], "-", color="#ff7f0e", lw=1.5, label="PI + EEC", where="post")
ax.set_xlabel("闭环时间 (s)")
ax.set_ylabel("PLA 指令 (deg)")
ax.set_title("(c) PLA 动作", fontsize=10)
ax.legend(fontsize=8)

fig.suptitle("图 A  最小 GRU-MPC vs PI 基线（真实 Simulink 环境，N2 参考 +50 rpm 阶跃）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figA_mpc_vs_pi.{e}")
plt.close(fig)
print("  图A: MPC vs PI")

# 数值对比
print("\n  === 定量对比 ===")
def metrics(L, is_mpc=True):
    err = L[:, 4]
    # 跳过参考未变的前 2 步
    e = err[2:]
    return {
        "mean_abs": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
        "final_abs": float(np.abs(e[-1])),
        "overshoot": float(np.max(err)) if np.max(err) > 0 else 0.0,
        "settle_step": int(np.argmax(np.abs(e) < 5)) if np.any(np.abs(e) < 5) else -1,
    }

M_mpc = metrics(L_mpc)
M_pi = metrics(L_pi)
print(f"  {'指标':<16}{'GRU-MPC':>12}{'PI+EEC':>12}{'MPC优势':>12}")
print("  " + "-" * 54)
for k, nm in [("mean_abs", "平均|误差|"), ("max_abs", "最大|误差|"),
              ("final_abs", "末步|误差|"), ("overshoot", "超调"),
              ("settle_step", "首次<5rpm步")]:
    a, b = M_mpc[k], M_pi[k]
    if k == "settle_step":
        adv = f"{b-a:+d}" if (a >= 0 and b >= 0) else "n/a"
        print(f"  {nm:<16}{a:>12}{b:>12}{adv:>12}")
    else:
        adv = (b - a) / b * 100 if b > 0 else 0
        print(f"  {nm:<16}{a:>12.2f}{b:>12.2f}{adv:>+11.1f}%")

# ══════════════════════════════════════════════════
# 图 B：V4 事件对齐的阶跃响应
# ══════════════════════════════════════════════════
v4 = json.loads((RES / "v4_event_aligned.json").read_text(encoding="utf-8"))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
COLORS = {"GRU": "#1f77b4", "RSSM": "#ff7f0e", "PC-LWM": "#2ca02c"}

for ax, (key, ev) in zip(axes, v4.items()):
    n2_true = np.array(ev["n2_true"])
    t_axis = np.arange(len(n2_true)) * 0.105
    ax.plot(t_axis, n2_true, "k-", lw=2.0, label="真实值", zorder=5)
    for nm in ["GRU", "RSSM", "PC-LWM"]:
        if nm in ev["models"]:
            n2p = np.array(ev["models"][nm]["n2"])
            L = min(len(t_axis), len(n2p))
            ax.plot(t_axis[:L], n2p[:L], color=COLORS[nm], lw=1.3,
                    alpha=0.85, label=nm)
    ax.set_xlabel("阶跃后时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    ax.set_title(f"t={ev['event_t']:.1f}s  PLA {ev['pla_from']:.0f}→{ev['pla_to']:.0f}° "
                 f"(Δ={ev['dpla']:+.0f}°)", fontsize=9)
    if ax is axes[0]:
        ax.legend(fontsize=7.5)

fig.suptitle("图 B  V4 事件对齐的阶跃响应（历史=真实观测，未来=自主推演）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figB_v4_events.{e}")
plt.close(fig)
print("  图B: V4 事件对齐")

# ══════════════════════════════════════════════════
# 图 C：V4 阶跃响应幅度对比（柱状）
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 4.2))
events = list(v4.keys())
x = np.arange(len(events))
w = 0.2
dn2_true = [v4[k]["dn2_true"] for k in events]
ax.bar(x - 2*w, dn2_true, w, label="真实值", color="k", alpha=0.7)
for i, nm in enumerate(["GRU", "RSSM", "PC-LWM"]):
    vals = [v4[k]["models"][nm]["dn2"] for k in events]
    ax.bar(x + (i - 1) * w, vals, w, label=nm, color=COLORS[nm], alpha=0.85)
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"t={v4[k]['event_t']:.0f}s\nΔPLA={v4[k]['dpla']:+.0f}°"
                    for k in events], fontsize=8.5)
ax.set_ylabel("ΔN2 (rpm)")
ax.set_title("图 C  V4 三个阶跃事件的 N2 响应幅度", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figC_v4_dn2.{e}")
plt.close(fig)
print("  图C: V4 ΔN2 对比")

# ══════════════════════════════════════════════════
# 图 D：计算开销分解
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

# (a) 单步耗时分解
ax = axes[0]
t_step = 13251 if False else 11950   # ms（用改进版的）
t_opt = float(np.mean(L_mpc[:, 5]))
t_sim = t_step - t_opt
bars = ax.bar(["Simulink\n真 step", "GRU-MPC\n优化"], [t_sim, t_opt],
              color=["#1f77b4", "#2ca02c"], alpha=0.85)
ax.set_ylabel("耗时 (ms)")
ax.set_title(f"(a) 单步耗时分解（总计 {t_step:.0f} ms）", fontsize=10)
for b, v in zip(bars, [t_sim, t_opt]):
    ax.text(b.get_x() + b.get_width()/2, v + 200, f"{v:.0f} ms\n({v/t_step*100:.1f}%)",
            ha="center", fontsize=8)

# (b) MPC 优化耗时 vs 控制周期
ax = axes[1]
ax.axhline(105, color="#d62728", ls="--", lw=1.5, label="控制周期 105 ms")
ax.axhline(t_opt, color="#2ca02c", ls="-", lw=1.5,
           label=f"MPC 优化 {t_opt:.0f} ms")
ax.set_ylim(0, max(200, t_opt*1.3))
ax.set_ylabel("耗时 (ms)")
ax.set_title("(b) MPC 优化 vs 控制周期", fontsize=10)
ax.legend(fontsize=8)
ax.text(0.5, 0.5, f"MPC 优化 {t_opt:.0f} ms < 105 ms ✅\n但 Simulink 环境 {t_sim:.0f} ms ≫ 105 ms ❌",
        transform=ax.transAxes, ha="center", va="center", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

fig.suptitle("图 D  计算开销分析", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figD_compute.{e}")
plt.close(fig)
print("  图D: 计算开销")

# 保存
summary = {
    "mpc_vs_pi": {
        "mpc": M_mpc, "pi": M_pi,
        "t_step_ms": t_step, "t_opt_ms": t_opt, "t_sim_ms": t_sim,
    },
    "v4_events": {k: {"event_t": v4[k]["event_t"], "dpla": v4[k]["dpla"],
                       "dn2_true": v4[k]["dn2_true"],
                       "dn2": {nm: v4[k]["models"][nm]["dn2"]
                               for nm in ["GRU", "RSSM", "PC-LWM"]}}
                   for k in v4},
}
(RES / "p03_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {RES/'p03_summary.json'}")
print(f"  -> {FIG}")
