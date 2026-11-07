"""III-G 图表：计划 vs 实际 PLA + GRU 回放误差"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

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
PS = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()
P = pd.read_csv(CD / "pred_n2_matrix.csv", header=None).to_numpy()
OA = json.loads((CD / "offline_ablation.json").read_text(encoding="utf-8"))

cfg = {}
for ln in (CD / "cfg.txt").read_text().split("\n"):
    if "=" in ln:
        k, v = ln.split("=")
        cfg[k] = float(v)
N2_REF = cfg["n2_ref"]
H = int(cfg["H"])
N = len(S)

t_s = S["t"].to_numpy()
pla_act = S["pla_apply"].to_numpy()
n2_act = S["n2_after"].to_numpy()
t_real = np.append(t_s, t_s[-1] + DT)
n2_real = np.append(S["n2_before"].to_numpy()[0], n2_act)
t_real = np.append(0.0, t_s + DT)
t_real = np.insert(t_s, 0, -DT)[1:]
t_real = np.concatenate([[0.0], t_s[1:] + DT])
# 简化：用 t_s 与 n2_after 对齐（第 k 步执行后）
t_plot = np.concatenate([[0.0], t_s + DT])
n2_plot = np.concatenate([[S["n2_before"].iloc[0]], n2_act])

print("=" * 90)
print("生成 III-G 图表")
print("=" * 90)

# ══════════════════════════════════════════════════
# 图 1：计划 PLA vs 实际 PLA（k=10 的计划 vs 后续执行）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=False)

# (a) k=10 的计划 vs 实际
ax = axes[0]
i = 9
t_plan = t_s[i] + np.arange(1, H+1) * DT
valid = t_plan <= t_s[-1] + DT
ax.plot(t_plan[valid], PS[i][valid], "s--", color="#ff7f0e", lw=2.0, ms=5,
        label=f"k=10 时计划的未来 PLA（{int(valid.sum())} 步可见）")
t_after = t_s[i+1:]
ax.step(np.append(t_s[i], t_after), np.append(pla_act[i], pla_act[i+1:]),
        "-", color="#1f77b4", lw=2.4, where="post", label="实际执行 PLA")
ax.axvline(t_s[i], color="gray", ls=":", lw=1.2)
ax.annotate(f"k=10 决策点\nt={t_s[i]:.2f}s",
            xy=(t_s[i], pla_act[i]), xytext=(t_s[i]-0.5, pla_act[i]+4),
            fontsize=8.5, color="gray",
            arrowprops=dict(arrowstyle="->", color="gray"))
ax.set_ylabel("PLA (deg)")
ax.set_xlabel("时间 (s)")
ax.legend(fontsize=9, loc="upper left")
ax.set_title("(a) k=10 的计划 vs 实际执行：计划降 PLA 但实际继续上升", fontsize=10.5)

# 标注偏差
ax.fill_between(t_after, pla_act[i+1:], PS[i][1:len(t_after)+1],
                color="#d62728", alpha=0.15, step="post")

# (b) 逐步：计划末值 vs 实际
ax = axes[1]
plan_end = PS[:, -1]
ax.plot(t_s, plan_end, "s--", color="#ff7f0e", ms=5, lw=1.8,
        label="MPC 计划的 PLA[32]（未来末值）")
ax.step(t_s, pla_act, "-", color="#1f77b4", lw=2.4, where="post",
        label="实际执行 PLA")
ax.set_ylabel("PLA (deg)")
ax.set_xlabel("时间 (s)")
ax.legend(fontsize=9)
ax.set_title("(b) 每步计划末值 vs 实际执行：计划值始终低于实际，反复推迟下降",
             fontsize=10.5)

fig.suptitle("图 1  计划 PLA 与实际执行 PLA 的偏差（滚动优化的【推迟】现象）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figG1_plan_vs_actual.{e}")
plt.close(fig)
print("  图1: 计划 vs 实际")

# ══════════════════════════════════════════════════
# 图 2：GRU 真实动作回放 vs 真实 N2
# ══════════════════════════════════════════════════
resA = OA["A"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, r in zip(axes, resA):
    k = r["k"]
    n_fut = r["n_fut"]
    t_pred = r["t"] + np.arange(1, n_fut+1) * DT
    pred = np.array(r["pred_curve"])
    true = np.array(r["true_curve"])

    ax.plot(t_pred, true, "k-", lw=2.4, label="真实 N2", zorder=5)
    ax.plot(t_pred, pred, "-", color="#2ca02c", lw=2.0, label="GRU 回放预测")
    ax.axhline(N2_REF, color="#d62728", ls="--", lw=1.3)

    ax.set_title(f"k={k}, t={r['t']:.2f}s  MAE={r['mae']:.1f}\n"
                 f"ΔN2: 预测{r['dn2_pred']:+.0f} vs 真实{r['dn2_true']:+.0f}",
                 fontsize=9)
    ax.set_xlabel("时间 (s)")
    ax.set_ylabel("N2 (rpm)")
    if k == 9:
        ax.legend(fontsize=7.5)

fig.suptitle("图 2  GRU 用【真实执行动作序列】回放 vs 真实 N2\n"
             "k=9/10 明显高估（+130 rpm），k=17/20 明显低估（−128 rpm）",
             fontsize=11.5)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figG2_gru_replay.{e}")
plt.close(fig)
print("  图2: GRU 回放")

# ══════════════════════════════════════════════════
# 图 3：回放误差随时域
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

ax = axes[0]
ks = [r["k"] for r in resA]
maes = [r["mae"] for r in resA]
biases = [r["bias"] for r in resA]
ax.plot(ks, maes, "o-", color="#d62728", ms=8, lw=2.0, label="MAE")
ax.bar(ks, biases, alpha=0.4, color=["#2ca02c" if b > 0 else "#1f77b4" for b in biases],
       label="有符号均值")
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xlabel("回放起点 k")
ax.set_ylabel("误差 (rpm)")
ax.set_title("(a) GRU 回放误差（真实动作序列）", fontsize=10)
ax.legend(fontsize=8.5)

ax = axes[1]
dn2p = [r["dn2_pred"] for r in resA]
dn2t = [r["dn2_true"] for r in resA]
x = np.arange(len(ks))
w = 0.35
ax.bar(x - w/2, dn2t, w, label="真实 ΔN2", color="#ff7f0e", alpha=0.85)
ax.bar(x + w/2, dn2p, w, label="GRU 预测 ΔN2", color="#2ca02c", alpha=0.85)
ax.axhline(0, color="0.3", lw=1.0)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in ks])
ax.set_ylabel("ΔN2 (rpm)")
ax.set_title("(b) 转速变化量：ΔN2 系统性低估", fontsize=10)
ax.legend(fontsize=8.5)

fig.suptitle("图 3  GRU 回放误差：ΔN2 低估 42–128 rpm", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"figG3_replay_err.{e}")
plt.close(fig)
print("  图3: 回放误差")

print(f"\n  -> {FIG}")

