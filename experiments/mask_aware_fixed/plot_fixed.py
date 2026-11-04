"""修复版结果可视化 + 前后对比"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"C:\pclwm\pclwm_mask\results_fixed")
OLD = Path(r"C:\pclwm\pclwm_mask\results")
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight"})

new = json.loads((RES/"mask_results_fixed.json").read_text(encoding="utf-8"))
old = json.loads((OLD/"mask_results.json").read_text(encoding="utf-8")) \
      if (OLD/"mask_results.json").exists() else {}

MODELS = list(new.keys())
COL = {"MaskGRU": "#1f77b4", "MaskRSSM": "#d62728"}
SCEN = [("perm20s","永久失效 @20s"),("burst5s","缺失 5s"),
        ("burst10s","缺失 10s"),("burst30s","缺失 30s")]

# ── 图 A：修复后逐步 vs 分段 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, split, lab in zip(axes, ["test_id","test_ood"], ["(a) Test-ID","(b) Test-OOD"]):
    x = np.arange(len(SCEN)); w = 0.2
    for i, m in enumerate(MODELS):
        sw = [new[m]["scenarios"][s]["splits"][split]["stepwise"]["rmse"] for s,_ in SCEN]
        sg = [new[m]["scenarios"][s]["splits"][split]["segmented"]["rmse"] for s,_ in SCEN]
        ax.bar(x+(i*2-1.5)*w, sw, w, label=f"{m} 逐步", color=COL[m], alpha=0.9)
        ax.bar(x+(i*2-0.5)*w, sg, w, label=f"{m} 分段", color=COL[m], alpha=0.45,
               hatch="//", edgecolor=COL[m])
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in SCEN], fontsize=9)
    ax.set_ylabel("T4 重构 RMSE [°R]"); ax.set_title(lab, fontsize=11)
    ax.legend(fontsize=7.5, ncol=2)
fig.suptitle("图 A  修复后：逐步在线 vs 每 32 步分段", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"fixed_stepwise_vs_seg.{e}")
plt.close(fig)
print("  图A")

# ── 图 B：修复前后对比（MaskRSSM）──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, split, lab in zip(axes, ["test_id","test_ood"], ["(a) Test-ID","(b) Test-OOD"]):
    x = np.arange(len(SCEN)); w = 0.28
    # 修复前（旧实现只有单一方案）
    if "MaskRSSM" in old:
        before = [old["MaskRSSM"]["scenarios"][s]["splits"][split]["online"]["rmse"] for s,_ in SCEN]
        ax.bar(x-w, before, w, label="修复前（错误监督）", color="#999999", alpha=0.85)
    after = [new["MaskRSSM"]["scenarios"][s]["splits"][split]["stepwise"]["rmse"] for s,_ in SCEN]
    ax.bar(x, after, w, label="修复后（逐步）", color="#d62728", alpha=0.9)
    pers = [new["MaskRSSM"]["scenarios"][s]["splits"][split]["persistence"]["rmse"] for s,_ in SCEN]
    ax.plot(x+w, pers, "k--o", ms=5, lw=1.3, label="Persistence")
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in SCEN], fontsize=9)
    ax.set_ylabel("T4 重构 RMSE [°R]"); ax.set_title(lab, fontsize=11)
    ax.legend(fontsize=8)
fig.suptitle("图 B  Mask-RSSM 修复前后对比", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"before_after_rssm.{e}")
plt.close(fig)
print("  图B")

# ── 图 C：重构曲线 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (scen, desc) in zip(axes.ravel(), SCEN):
    for m in MODELS:
        c = new[m]["scenarios"][scen]["splits"]["test_id"]["curves"]
        gt = np.array(c["gt"]); pr = np.array(c["pred"])
        if len(gt)==0: continue
        t = np.arange(len(gt))*0.105
        if m == MODELS[0]:
            ax.plot(t, gt, "k-", lw=1.1, alpha=0.65, label="真实 T4")
        ax.plot(t, pr, "--", lw=1.3, color=COL[m], alpha=0.85, label=m)
    ax.set_xlabel("故障后时间 [s]"); ax.set_ylabel("T4 [°R]")
    ax.set_title(desc, fontsize=10); ax.legend(fontsize=8)
fig.suptitle("图 C  逐步在线重构曲线（test_id 拼接）", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"fixed_curves.{e}")
plt.close(fig)
print("  图C")

# ── 图 D：训练曲线 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for m in MODELS:
    h = new[m]["hist"]
    axes[0].plot(h["train"], label=m, color=COL[m])
    axes[1].plot(h["val"], label=m, color=COL[m])
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss"); axes[0].legend(fontsize=8.5)
axes[0].set_title("(a) 训练损失", fontsize=11)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val loss (T4 全遮蔽，真实监督)")
axes[1].set_title("(b) 验证损失", fontsize=11); axes[1].legend(fontsize=8.5)
axes[1].set_yscale("log")
fig.suptitle("图 D  修复版训练曲线", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"fixed_train_curves.{e}")
plt.close(fig)
print("  图D")