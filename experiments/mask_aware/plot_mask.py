"""Mask 重构结果可视化"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"C:\pclwm\pclwm_mask\results")
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight"})

d = json.loads((RES / "mask_results.json").read_text(encoding="utf-8"))
MODELS = list(d.keys())
COL = {"MaskGRU": "#1f77b4", "MaskRSSM": "#d62728"}
SCEN = [("perm20s","永久失效 @20s"),("burst5s","缺失 5s"),
        ("burst10s","缺失 10s"),("burst30s","缺失 30s")]

# ── 图 1：柱状对比 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, split, lab in zip(axes, ["test_id","test_ood"],
                          ["(a) Test-ID","(b) Test-OOD"]):
    x = np.arange(len(SCEN)); w = 0.28
    for i, m in enumerate(MODELS):
        vals = [d[m]["scenarios"][s]["splits"][split]["online"]["rmse"]
                for s,_ in SCEN]
        ax.bar(x+(i-0.5)*w, vals, w, label=m, color=COL.get(m,"#999"),
               alpha=0.85)
    pers = [d[MODELS[0]]["scenarios"][s]["splits"][split]["persistence"]["rmse"]
            for s,_ in SCEN]
    ax.plot(x, pers, "k--o", ms=6, lw=1.4, label="Persistence")
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in SCEN], fontsize=9)
    ax.set_ylabel("T4 重构 RMSE [°R]")
    ax.set_title(lab, fontsize=11)
    ax.legend(fontsize=8.5)
fig.suptitle("T4 传感器失效重构误差对比", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"mask_rmse_compare.{e}")
plt.close(fig)
print("  图1 ->", FIG/"mask_rmse_compare.png")

# ── 图 2：重构曲线 ──
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for ax, (scen, desc) in zip(axes.ravel(), SCEN):
    for m in MODELS:
        c = d[m]["scenarios"][scen]["splits"]["test_id"]["curves"]
        gt = np.array(c["gt"]); pr = np.array(c["pred"])
        if len(gt)==0: continue
        t = np.arange(len(gt))*0.105
        ax.plot(t, gt, "k-", lw=1.2, alpha=0.7, label="真实 T4" if m==MODELS[0] else None)
        ax.plot(t, pr, "--", lw=1.3, color=COL.get(m,"#999"),
                alpha=0.85, label=m)
    ax.set_xlabel("故障后时间 [s]")
    ax.set_ylabel("T4 [°R]")
    ax.set_title(desc, fontsize=10)
    ax.legend(fontsize=8)
fig.suptitle("T4 重构曲线（test_id 拼接）", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"mask_curves.{e}")
plt.close(fig)
print("  图2 ->", FIG/"mask_curves.png")

# ── 图 3：分区间误差 ──
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, split, lab in zip(axes, ["test_id","test_ood"],
                          ["(a) Test-ID","(b) Test-OOD"]):
    ivs = ["0-5s","5-15s","15-30s","30-1000s"]
    x = np.arange(len(ivs)); w = 0.35
    for i, m in enumerate(MODELS):
        vals = []
        for iv in ivs:
            v = d[m]["scenarios"]["perm20s"]["splits"][split]["online"]["interval"].get(iv)
            vals.append(v["rmse"] if v else np.nan)
        ax.bar(x+(i-0.5)*w, vals, w, label=m, color=COL.get(m,"#999"), alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(ivs, fontsize=9)
    ax.set_xlabel("故障后时间区间")
    ax.set_ylabel("RMSE [°R]")
    ax.set_title(lab + " 永久失效分区间", fontsize=11)
    ax.legend(fontsize=8.5)
fig.suptitle("重构误差随故障后时间的演变", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"mask_intervals.{e}")
plt.close(fig)
print("  图3 ->", FIG/"mask_intervals.png")

# ── 图 4：训练曲线 ──
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for m in MODELS:
    h = d[m]["hist"]
    axes[0].plot(h["train"], label=m, color=COL.get(m,"#999"))
    axes[1].plot(h["val"], label=m, color=COL.get(m,"#999"))
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss")
axes[0].set_title("(a) 训练损失", fontsize=11); axes[0].legend(fontsize=8.5)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val loss (T4 全缺失)")
axes[1].set_title("(b) 验证损失", fontsize=11); axes[1].legend(fontsize=8.5)
axes[1].set_yscale("log")
fig.suptitle("Mask 模型训练曲线", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"mask_train_curves.{e}")
plt.close(fig)
print("  图4 ->", FIG/"mask_train_curves.png")