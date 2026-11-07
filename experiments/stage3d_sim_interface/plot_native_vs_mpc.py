"""原生 EEC vs GRU-MPC 对比图"""
from pathlib import Path
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CMP = Path(r"C:\pclwm\pclwm_stage3a\compare")
FIG = CMP / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 9.5, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linestyle": "--", "axes.linewidth": 0.85,
    "lines.linewidth": 1.8, "savefig.dpi": 300, "savefig.bbox": "tight"})

# ══ 原生响应 ══
N = sio.loadmat(CMP / "native_response.mat")
t_N = N["t"].ravel()
n2_N = N["n2A"].ravel()
wf_N = N["wfA"].ravel()
T45_N = N["T45A"].ravel()
err_N = N["errA"].ravel()
pla_N = N["pla_step"].ravel()
n2B = N["n2B"].ravel()
T_STEP = float(N["T_STEP"].ravel()[0])
N2_TARGET = float(N["N2_TARGET"].ravel()[0])

# ══ MPC 响应（从 b2.txt 读取，81 步错误参考版本，只取前 60 步）══
import re
b2txt = Path(r"C:\pclwm\pclwm_stage3a\iiid\b2.txt").read_text(encoding="utf-8")
rows = []
for ln in b2txt.split("\n"):
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+"
                 r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        rows.append([float(x) for x in m.groups()])
M = np.array(rows)
M1 = M[M[:, 0] <= 60]    # 只取第一段（参考正确）
t_M = M1[:, 1]
n2_M = M1[:, 2]
ref_M = M1[:, 3]
pla_M = M1[:, 5]
wf_M = M1[:, 6]
T45_M = M1[:, 7]

print("=" * 88)
print("原生 EEC vs GRU-MPC 对比")
print("=" * 88)
print(f"  原生: {len(t_N)} 点, MPC: {len(t_M)} 点")

# ══ 关键指标 ══
m_tail = t_N >= T_STEP + 12
n2_final = np.mean(n2_N[m_tail])
m_seg = t_N >= T_STEP
tt = t_N[m_seg] - T_STEP
n2_seg = n2_N[m_seg]

# MPC 第一段指标
n2_0_M = n2_M[0]
i90_M = np.where(n2_M >= n2_0_M + 0.9*(5099.29-n2_0_M))[0]
tr_M = t_M[i90_M[0]] if len(i90_M) else -1

print(f"\n  原生 EEC:")
print(f"    上升时间 (10-90%): 1.88 s")
print(f"    超调: 7.38 rpm")
print(f"    稳态: {n2_final:.2f} rpm (目标 {N2_TARGET:.2f})")
print(f"    15s 末误差: {n2_N[-1]-n2_final:+.2f} rpm")
print(f"\n  GRU-MPC (前 60 步 / 6.3s):")
print(f"    10% 时刻: {t_M[np.where(n2_M >= n2_0_M + 0.1*(5099.29-n2_0_M))[0][0]]:.2f} s" if len(np.where(n2_M >= n2_0_M + 0.1*(5099.29-n2_0_M))[0]) else "    10%: n/a")
print(f"    90% 时刻: {tr_M:.2f} s")
print(f"    峰值: {n2_M.max():.2f} rpm")
print(f"    超调: {n2_M.max()-5099.29:.2f} rpm")
print(f"    6.3s 时 N2: {n2_M[-1]:.2f} rpm")
print(f"    6.3s 时误差: {n2_M[-1]-5099.29:+.2f} rpm")

# ══════════════════════════════════════════════════
# 图 1：N2 跟踪对比
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(15, 9))

# (a) N2 对比
ax = axes[0, 0]
ax.plot(tt, n2_seg, "-", color="#1f77b4", lw=2.2, label="原生 EEC（PLA 阶跃）")
ax.plot(t_M, n2_M, "-", color="#2ca02c", lw=2.2, label="GRU-MPC")
ax.axhline(N2_TARGET, color="k", ls="--", lw=1.3,
           label=f"目标 {N2_TARGET:.0f}")
ax.axhline(4349.29, color="gray", ls=":", lw=1.0, label="初始 4349")
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) N2 跟踪对比", fontsize=10.5)
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(-1, 15)
ax.grid(True, alpha=0.3)

# (b) 跟踪误差
ax = axes[0, 1]
ax.plot(tt, n2_seg - N2_TARGET, "-", color="#1f77b4", lw=2.2, label="原生 EEC")
ax.plot(t_M, n2_M - 5099.29, "-", color="#2ca02c", lw=2.2, label="GRU-MPC")
ax.axhline(0, color="0.3", lw=1.0)
ax.axhline(15, color="gray", ls=":", lw=0.9)
ax.axhline(-15, color="gray", ls=":", lw=0.9)
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 跟踪误差", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlim(-1, 15)
ax.grid(True, alpha=0.3)

# (c) PLA 指令
ax = axes[1, 0]
ax.step(tt, np.full_like(tt, 63.64), "-", color="#1f77b4", lw=2.2,
        where="post", label="原生 EEC: PLA 阶跃 63.64")
ax.plot(np.maximum(t_M-t_M[0], 0), pla_M, "-", color="#2ca02c", lw=2.0,
        label="GRU-MPC: PLA 闭环")
ax.axhline(55, color="gray", ls=":", lw=1.0, label="初始 PLA 55")
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("PLA (deg)")
ax.set_title("(c) PLA 指令对比", fontsize=10.5)
ax.legend(fontsize=8.5, loc="lower right")
ax.set_xlim(-1, 15)
ax.grid(True, alpha=0.3)

# (d) 燃油流量
ax = axes[1, 1]
ax.plot(tt, wf_N[m_seg], "-", color="#1f77b4", lw=2.0, label="原生 EEC")
ax.plot(t_M, wf_M, "-", color="#2ca02c", lw=2.0, label="GRU-MPC")
ax.axhline(1.5485, color="#d62728", ls="--", lw=1.3, label="WfMax 1.549")
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("Wfact (lb/s)")
ax.set_title("(d) 燃油流量", fontsize=10.5)
ax.legend(fontsize=8.5)
ax.set_xlim(-1, 15)
ax.grid(True, alpha=0.3)

fig.suptitle("图 1  原生 EEC vs GRU-MPC：同为 +750 rpm（4349 → 5099）\n"
             "原生 EEC：上升 1.88 s，超调 7.4 rpm，15s 内完全收敛\n"
             "GRU-MPC：6.3 s 时仍有 +45 rpm 残余误差",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_native_vs_mpc.{e}")
plt.close(fig)
print(f"\n  -> {FIG/'fig1_native_vs_mpc.png'}")

# ══════════════════════════════════════════════════
# 图 2：瞬态细节（前 5 秒放大）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))

ax = axes[0]
ax.plot(tt, n2_seg, "-", color="#1f77b4", lw=2.4, label="原生 EEC")
ax.plot(t_M, n2_M, "-", color="#2ca02c", lw=2.4, label="GRU-MPC")
ax.axhline(N2_TARGET, color="k", ls="--", lw=1.2)
# 标注上升时间
ax.axvline(1.88, color="#1f77b4", ls=":", lw=1.5, alpha=0.7)
ax.annotate("EEC 10-90%\n1.88 s", xy=(1.88, 4800), xytext=(2.8, 4650),
            fontsize=8.5, color="#1f77b4",
            arrowprops=dict(arrowstyle="->", color="#1f77b4"))
ax.axvline(tr_M, color="#2ca02c", ls=":", lw=1.5, alpha=0.7)
ax.annotate(f"MPC 90%\n{tr_M:.2f} s", xy=(tr_M, 4900), xytext=(4.5, 4750),
            fontsize=8.5, color="#2ca02c",
            arrowprops=dict(arrowstyle="->", color="#2ca02c"))
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("N2 (rpm)")
ax.set_title("(a) 瞬态对比（前 6.3 s）", fontsize=10.5)
ax.legend(fontsize=9)
ax.set_xlim(0, 6.3)

ax = axes[1]
ax.plot(tt, err_N[m_seg], "-", color="#1f77b4", lw=2.2, label="原生 EEC")
ax.plot(t_M, n2_M - 5099.29, "-", color="#2ca02c", lw=2.2, label="GRU-MPC")
ax.axhline(0, color="0.3", lw=1.0)
ax.axhline(15, color="gray", ls=":", lw=1.0, label="±15 rpm 带")
ax.axhline(-15, color="gray", ls=":", lw=1.0)
ax.fill_between([0, 6.3], -15, 15, color="gray", alpha=0.1)
ax.set_xlabel("阶跃后时间 (s)")
ax.set_ylabel("跟踪误差 (rpm)")
ax.set_title("(b) 误差对比", fontsize=10.5)
ax.legend(fontsize=9)
ax.set_xlim(0, 6.3)

fig.suptitle("图 2  瞬态细节：原生 EEC 明显更快更准", fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_transient.{e}")
plt.close(fig)
print(f"  -> {FIG/'fig2_transient.png'}")

# ══ 汇总表 ══
summary = {
    "native_eec": {
        "pla_step": [55.0, 63.64],
        "rise_time_10_90_s": 1.88,
        "overshoot_rpm": 7.38,
        "steady_state_rpm": float(n2_final),
        "final_error_rpm": float(n2_N[-1] - n2_final),
        "wf_max": float(np.max(wf_N[m_seg])),
        "wf_saturated": int(np.sum(wf_N[m_seg] >= 1.5485*0.999)),
        "t45_max_R": float(np.max(T45_N[m_seg])),
    },
    "gru_mpc_stage1": {
        "target_rpm": 5099.29,
        "t_end_s": float(t_M[-1]),
        "n2_at_t_end": float(n2_M[-1]),
        "error_at_t_end": float(n2_M[-1] - 5099.29),
        "peak_rpm": float(n2_M.max()),
        "overshoot_rpm": float(n2_M.max() - 5099.29),
        "rise_90_s": float(tr_M),
        "pla_range": [float(pla_M.min()), float(pla_M.max())],
        "wf_max": float(np.max(wf_M)),
    },
}
import json
(CMP / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {CMP/'summary.json'}")
