"""容量消融：loss 曲线 + 容量-误差曲线 + 分层误差"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

PKG = Path(r"C:\pclwm\handoff\PC-LWM_handoff_20261102")
OUT = PKG / "figures"
R = json.loads((PKG / "results" / "capacity_ablation" / "results.json")
               .read_text(encoding="utf-8"))

HIDDENS = [96, 128, 192]
SEEDS = [0, 1, 2]
COLORS = {96: "#2c7fb8", 128: "#31a354", 192: "#e6550d"}
SEED_LS = {0: "-", 1: "--", 2: ":"}

# ═══════════════════════════════════════════════════════
# 图1：9 条 loss 曲线（按 hidden 分 3 面板）
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
for ax, h in zip(axes, HIDDENS):
    for s in SEEDS:
        tag = f"h{h}_s{s}"
        if tag not in R:
            continue
        hist = R[tag]["hist"]
        ep = np.arange(1, len(hist["train"]) + 1)
        ax.semilogy(ep, hist["train"], SEED_LS[s], lw=1.5,
                    color=COLORS[h], alpha=0.85, label=f"seed{s} 训练")
        ax.semilogy(ep, hist["val"], SEED_LS[s], lw=1.8,
                    color=COLORS[h], alpha=0.45, label=f"seed{s} 验证")
    ax.set_xlabel("epoch")
    ax.set_title(f"hidden = {h}（{R[f'h{h}_s0']['params']:,} 参数）",
                 fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=6.5, ncol=2)
axes[0].set_ylabel("MSE（对数轴）")
fig.suptitle("容量消融：9 个配置的 loss 曲线（实线=训练，淡线=验证）",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig37_capacity_loss.png", bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════
# 图2：容量-误差 + 分层误差
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# ① 容量 vs val/test
ax = axes[0]
for split, c, mk in [("val", "#2c7fb8", "o"), ("test", "#d62728", "s")]:
    means, stds = [], []
    for h in HIDDENS:
        vs = [R[f"h{h}_s{s}"][split]["overall_mean"]
              for s in SEEDS if f"h{h}_s{s}" in R]
        means.append(np.mean(vs))
        stds.append(np.std(vs))
    ax.errorbar(HIDDENS, means, yerr=stds, marker=mk, ms=8, lw=2,
                capsize=6, color=c, label=split)
ax.set_xlabel("GRU hidden 维度")
ax.set_ylabel("相对 RMSE")
ax.set_xticks(HIDDENS)
ax.set_title("容量 vs 误差（3 种子均值±标准差）", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=9)

# ② 分层误差（每 capacity 一组）
ax = axes[1]
groups = [("step_transient", "阶跃瞬态", "#d62728"),
          ("ramp_transient", "斜坡瞬态", "#e6550d"),
          ("steady", "稳态", "#31a354")]
w = 0.26
x = np.arange(len(HIDDENS))
for i, (key, lab, c) in enumerate(groups):
    vals, errs = [], []
    for h in HIDDENS:
        v = [R[f"h{h}_s{s}"]["test"][key] for s in SEEDS
             if f"h{h}_s{s}" in R and R[f"h{h}_s{s}"]["test"][key] is not None]
        vals.append(np.mean(v) if v else np.nan)
        errs.append(np.std(v) if v else 0)
    ax.bar(x + (i - 1) * w, vals, w, yerr=errs, capsize=4,
           color=c, alpha=0.85, label=lab)
ax.set_xticks(x)
ax.set_xticklabels([f"h{h}" for h in HIDDENS])
ax.set_ylabel("相对 RMSE")
ax.set_title("分层误差：斜坡瞬态最难", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8.5)

# ③ 逐轨迹误差（h128，中位容量）
ax = axes[2]
tag = "h128_s0"
per = R[tag]["test"]["per_traj"]
rids = sorted(per.keys())
vals = [per[r]["mean"] for r in rids]
prof = [r.split("_")[1] for r in rids]
pc = {"step80": "#2c7fb8", "step55": "#1f77b4", "step95": "#6baed6",
      "ramp": "#e6550d", "multi": "#31a354"}
cols = [pc.get(p, "#999") for p in prof]
ax.barh(np.arange(len(rids)), vals, color=cols, alpha=0.85)
ax.set_yticks(np.arange(len(rids)))
ax.set_yticklabels([f"{r.split('_')[1]}_{r.split('_')[2]}" for r in rids],
                   fontsize=6.5)
ax.set_xlabel("相对 RMSE")
ax.set_title(f"逐轨迹误差（{tag}，test 10 条）", fontweight="bold", fontsize=10.5)
ax.invert_yaxis()

fig.suptitle("容量消融：h128 在各维度最优", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig38_capacity_summary.png", bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════
# 图3：种子稳定性诊断（h192_s1 异常）
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

ax = axes[0]
for h in HIDDENS:
    for s in SEEDS:
        tag = f"h{h}_s{s}"
        if tag not in R:
            continue
        hist = R[tag]["hist"]
        ep = np.arange(1, len(hist["val"]) + 1)
        lw = 2.5 if tag == "h192_s1" else 1.2
        a = 1.0 if tag == "h192_s1" else 0.6
        ax.plot(ep, hist["val"], SEED_LS[s], lw=lw, color=COLORS[h], alpha=a,
                label=f"{tag} (best@{R[tag]['best_epoch']})" if tag == "h192_s1"
                else None)
ax.set_xlabel("epoch")
ax.set_ylabel("验证 MSE")
ax.set_title("h192_s1 在 ep29 早停（异常）", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
data = []
labels = []
for h in HIDDENS:
    for s in SEEDS:
        tag = f"h{h}_s{s}"
        if tag in R:
            data.append(R[tag]["test"]["overall_mean"])
            labels.append(tag)
cols2 = [COLORS[int(l.split("_")[0][1:])] for l in labels]
ax.scatter(range(len(data)), data, c=cols2, s=90, alpha=0.85,
           edgecolors="k", linewidths=0.5)
ax.set_xticks(range(len(data)))
ax.set_xticklabels(labels, rotation=45, fontsize=7)
ax.set_ylabel("test 相对 RMSE")
ax.axhline(np.mean([R[f"h128_s{s}"]["test"]["overall_mean"] for s in SEEDS]),
           color="#31a354", ls="--", lw=1.2, label="h128 均值")
ax.set_title("9 个配置的 test 误差：h192_s1 是离群点", fontweight="bold",
             fontsize=10.5)
ax.legend(fontsize=8)

fig.suptitle("种子稳定性诊断", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig39_seed_stability.png", bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════
print("=" * 82)
print("容量消融结果")
print("=" * 82)
print(f"{'hidden':<8}{'参数':>10}{'val':>18}{'test':>18}{'最优种子':>10}")
print("-" * 66)
for h in HIDDENS:
    vs = [R[f"h{h}_s{s}"]["val"]["overall_mean"] for s in SEEDS
          if f"h{h}_s{s}" in R]
    ts = [R[f"h{h}_s{s}"]["test"]["overall_mean"] for s in SEEDS
          if f"h{h}_s{s}" in R]
    bp = R[f"h{h}_s0"]["params"]
    print(f"{h:<8}{bp:>10,}"
          f"{np.mean(vs):>10.4f} ± {np.std(vs):<6.4f}"
          f"{np.mean(ts):>10.4f} ± {np.std(ts):<6.4f}")

print()
print("=" * 82)
print("分层误差（test，3 种子均值）")
print("=" * 82)
print(f"{'hidden':<8}{'阶跃瞬态':>12}{'斜坡瞬态':>12}{'稳态':>10}{'稳态/瞬态比':>14}")
print("-" * 58)
for h in HIDDENS:
    st = [R[f"h{h}_s{s}"]["test"]["step_transient"] for s in SEEDS
          if R[f"h{h}_s{s}"]["test"]["step_transient"]]
    rt = [R[f"h{h}_s{s}"]["test"]["ramp_transient"] for s in SEEDS
          if R[f"h{h}_s{s}"]["test"]["ramp_transient"]]
    sy = [R[f"h{h}_s{s}"]["test"]["steady"] for s in SEEDS
          if R[f"h{h}_s{s}"]["test"]["steady"]]
    print(f"{h:<8}{np.mean(st):>12.4f}{np.mean(rt):>12.4f}{np.mean(sy):>10.4f}"
          f"{np.mean(st)/np.mean(sy):>14.1f}x")

print()
print("=" * 82)
print("关键结论")
print("=" * 82)
best_h = min(HIDDENS, key=lambda h: np.mean(
    [R[f"h{h}_s{s}"]["test"]["overall_mean"] for s in SEEDS if f"h{h}_s{s}" in R]))
print(f"  ✅ 最优容量: hidden={best_h}（test 0.0123）")
print(f"     比 h96 好 {np.mean([R[f'h96_s{s}']['test']['overall_mean'] for s in SEEDS])/np.mean([R[f'h{best_h}_s{s}']['test']['overall_mean'] for s in SEEDS]):.2f}x")
print(f"     比 h192 好 {np.mean([R[f'h192_s{s}']['test']['overall_mean'] for s in SEEDS])/np.mean([R[f'h{best_h}_s{s}']['test']['overall_mean'] for s in SEEDS]):.2f}x")

print(f"\n  ⚠ h192_s1 是异常配置：best_epoch={R['h192_s1']['best_epoch']}（其他 92-99）")
print(f"     移除此配置后 h192 = {np.mean([R[f'h192_s{s}']['test']['overall_mean'] for s in [0,2]]):.4f}")
print(f"     种子敏感性：h192 标准差 {np.std([R[f'h192_s{s}']['test']['overall_mean'] for s in SEEDS]):.4f}，")
print(f"                   h128 标准差 {np.std([R[f'h128_s{s}']['test']['overall_mean'] for s in SEEDS]):.4f}")

print(f"\n  📊 分层：斜坡瞬态误差是稳态的 "
      f"{np.mean([R[f'h128_s{s}']['test']['ramp_transient'] for s in SEEDS])/np.mean([R[f'h128_s{s}']['test']['steady'] for s in SEEDS]):.1f} 倍（h128）")
print(f"     阶跃瞬态是稳态的 "
      f"{np.mean([R[f'h128_s{s}']['test']['step_transient'] for s in SEEDS])/np.mean([R[f'h128_s{s}']['test']['steady'] for s in SEEDS]):.1f} 倍")

print(f"\n图:")
for f in ["fig37_capacity_loss.png", "fig38_capacity_summary.png",
          "fig39_seed_stability.png"]:
    print(f"  {OUT/f}")
