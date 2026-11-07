"""III-H 图表：三配置消融 + 计划vs实际对齐"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

IIH = Path(r"C:\pclwm\pclwm_stage3a\iiih")
CD = Path(r"C:\pclwm\pclwm_stage3a\causal")
FIG = IIH / "figures"
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

A = pd.read_csv(IIH / "ablation3_clean.csv")
PS = pd.read_csv(IIH / "pla_seq3_clean.csv")
S = pd.read_csv(CD / "steps.csv")
CFG_A = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()

N2_REF = 5099.29
H = 32

CFG_COLORS = {
    "A": "#1f77b4",
    "B": "#ff7f0e",
    "C": "#2ca02c",
}
CFG_SHORT = {
    "A": "A 原始ws, w=3",
    "B": "B 移位ws, w=3",
    "C": "C 移位ws, w=1",
}

print("=" * 92)
print("生成 III-H 图表")
print("=" * 92)

# ══════════════════════════════════════════════════
# 图 1：三配置的完整 32 步 PLA 序列（k=10/14/17）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

for ax, k in zip(axes, [10, 14, 17]):
    sub = PS[PS["k"] == k]
    pla_prev = S[S["k"] == k]["pla_prev"].iloc[0]

    for cfg in CFG_COLORS:
        s2 = sub[sub["cfg"] == cfg]
        if len(s2) == 0:
            continue
        ax.plot(s2["step"], s2["pla"], "-", color=CFG_COLORS[cfg],
                lw=2.0, label=CFG_SHORT[cfg], alpha=0.9)

    ax.axhline(pla_prev, color="gray", ls=":", lw=1.3,
               label=f"PLA_prev={pla_prev:.2f}")
    ax.set_xlabel("MPC 规划步 (1..32)")
    ax.set_ylabel("PLA (deg)")
    ax.set_title(f"k={k} (t={S[S['k']==k]['t'].iloc[0]:.2f}s)", fontsize=10)
    if k == 10:
        ax.legend(fontsize=7.5, loc="best")

fig.suptitle("图 1  三配置的 32 步 PLA 规划序列（同历史、同 GRU、同种子）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figH1_pla_seqs.{e}")
plt.close(fig)
print("  图1: PLA 序列")

# ══════════════════════════════════════════════════
# 图 2：预测 N2 + 超调对比
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(16, 8))

for col, k in enumerate([10, 14, 17]):
    sub = PS[PS["k"] == k]
    n2_now = S[S["k"] == k]["n2_before"].iloc[0]

    # (上) 预测 N2
    ax = axes[0, col]
    for cfg in CFG_COLORS:
        s2 = sub[sub["cfg"] == cfg]
        if len(s2) == 0:
            continue
        ax.plot(s2["step"], s2["pred_n2"], "-", color=CFG_COLORS[cfg],
                lw=2.0, label=CFG_SHORT[cfg])
    ax.axhline(N2_REF, color="#d62728", ls="--", lw=1.4, label="目标")
    ax.axhline(n2_now, color="gray", ls=":", lw=1.2)
    ax.set_xlabel("MPC 规划步")
    ax.set_ylabel("预测 N2 (rpm)")
    ax.set_title(f"k={k} 预测 N2（当前 {n2_now:.0f}）", fontsize=9.5)
    if col == 0:
        ax.legend(fontsize=7)

    # (下) 超调柱状
    ax = axes[1, col]
    d = A[A["k"] == k]
    names = [CFG_SHORT[c] for c in d["cfg"]]
    ovs = d["overshoot"].to_numpy()
    cols = [CFG_COLORS[c] for c in d["cfg"]]
    bars = ax.bar(range(len(names)), ovs, color=cols, alpha=0.85)
    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.split()[0] for n in names], fontsize=9)
    ax.set_ylabel("预测超调 (rpm)")
    ax.set_title(f"k={k} 预测超调", fontsize=9.5)
    for b, v in zip(bars, ovs):
        ax.text(b.get_x()+b.get_width()/2, v + (5 if v>=0 else -12),
                f"{v:+.0f}", ha="center", fontsize=8.5)

fig.suptitle("图 2  三配置的预测 N2 轨迹与预测超调（统一评价：等权 MSE）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figH2_pred_overshoot.{e}")
plt.close(fig)
print("  图2: 预测与超调")

# ══════════════════════════════════════════════════
# 图 3：修正后的计划 vs 实际（绝对时间对齐）
# ══════════════════════════════════════════════════
t_s = S["t"].to_numpy()
pla_act = S["pla_apply"].to_numpy()

fig, axes = plt.subplots(3, 1, figsize=(13, 11))

for ax, k in zip(axes, [10, 14, 17]):
    i = k - 1
    # 计划：从 k+1 步开始的绝对时间
    t_plan = t_s[i] + np.arange(1, H+1) * DT
    valid = t_plan <= t_s[-1] + DT
    ax.plot(t_plan[valid], CFG_A[i][valid], "s--", color="#ff7f0e",
            lw=2.0, ms=6, label=f"k={k} 计划的未来 PLA（{valid.sum()} 步）")

    # 实际执行
    t_after = np.append(t_s[i], t_s[i+1:])
    p_after = np.append(pla_act[i], pla_act[i+1:])
    ax.step(t_after, p_after, "-", color="#1f77b4", lw=2.4, where="post",
            label="实际执行 PLA")

    ax.axvline(t_s[i], color="gray", ls=":", lw=1.3)
    ax.set_ylabel("PLA (deg)")
    ax.set_xlabel("绝对时间 (s)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(f"k={k} (t={t_s[i]:.2f}s) 计划 vs 实际", fontsize=10)

fig.suptitle("图 3  计划 PLA 与后续实际执行（按【绝对时间】对齐）\n"
             "橙色方点 = 该时刻规划的未来动作；蓝色实线 = 实际执行",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figH3_plan_actual_aligned.{e}")
plt.close(fig)
print("  图3: 计划 vs 实际（绝对时间对齐）")

# ══════════════════════════════════════════════════
# 图 4：三配置关键指标对比
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

metrics = [
    ("pla1", "首步 PLA (deg)", None),
    ("overshoot", "预测超调 (rpm)", None),
    ("mse_eq", "等权 MSE", None),
]

for ax, (mk, lab, _) in zip(axes, metrics):
    ks = [10, 14, 17]
    x = np.arange(len(ks))
    w = 0.26
    for j, cfg in enumerate(CFG_COLORS):
        vals = []
        for k in ks:
            d = A[(A["k"] == k) & (A["cfg"] == cfg)]
            vals.append(d[mk].iloc[0] if len(d) else np.nan)
        bars = ax.bar(x + (j-1)*w, vals, w, label=CFG_SHORT[cfg],
                      color=CFG_COLORS[cfg], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k}" for k in ks])
    ax.set_ylabel(lab)
    ax.set_title(lab, fontsize=10)
    if mk == "pla1":
        ax.legend(fontsize=7.5)

fig.suptitle("图 4  三配置关键指标对比", fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figH4_metrics.{e}")
plt.close(fig)
print("  图4: 指标对比")

print(f"\n  -> {FIG}")

