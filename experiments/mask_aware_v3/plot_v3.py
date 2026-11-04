"""v3 图表"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"C:\pclwm\pclwm_mask\results_v3")
FIG = RES / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight"})

d = json.loads((RES/"mask_results_v3.json").read_text(encoding="utf-8"))
sup = json.loads((RES/"supplementary.json").read_text(encoding="utf-8"))
MODELS = list(d.keys())
COL = {"MaskGRU": "#1f77b4", "MaskRSSM": "#d62728"}
SCEN = [("perm20s","永久失效"),("burst5s","缺失5s"),
        ("burst10s","缺失10s"),("burst30s","缺失30s")]
MODES = [("open_loop","严格开环","#999999"),
         ("segmented","分段反馈","#ff7f0e"),
         ("stepwise","逐步在线","#2ca02c")]

# 图 1：三模式对比（2×2）
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
for ax, split in zip(axes.ravel(), ["test_id","test_ood"]):
    x = np.arange(len(SCEN)); w = 0.13
    for i, m in enumerate(MODELS):
        for j, (md, lab, col) in enumerate(MODES):
            vals = [d[m]["scenarios"][s]["modes"][split][md]["rmse"]
                    for s,_ in SCEN]
            ax.bar(x + (i*3+j-2.5)*w, vals, w, color=COL[m],
                   alpha=[0.35,0.65,0.95][j],
                   label=f"{m}-{lab}" if ax is axes.ravel()[0] else None)
    ax.set_xticks(x); ax.set_xticklabels([l for _,l in SCEN], fontsize=9)
    ax.set_ylabel("T4 RMSE [°R]")
    ax.set_title(f"{'(a) Test-ID' if split=='test_id' else '(b) Test-OOD'}",
                 fontsize=11)
    if ax is axes.ravel()[0]:
        ax.legend(fontsize=6.5, ncol=2)
fig.suptitle("图 1  三种推理模式 × 四场景", fontsize=13)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"v3_modes.{e}")
plt.close(fig)

# 图 2：逐通道移除
fig, ax = plt.subplots(figsize=(9, 5))
CH = ["Wfact","N2int","T45","P36"]
x = np.arange(len(CH)); w = 0.35
for i, m in enumerate(MODELS):
    cr = d[m].get("channel_removal", {})
    if not cr: continue
    base = cr["baseline"]
    vals = [cr[c]["rmse"] for c in CH]
    ax.bar(x+(i-0.5)*w, vals, w, label=f"{m} (基线 {base:.1f})",
           color=COL[m], alpha=0.85)
    ax.axhline(base, color=COL[m], ls="--", lw=1.2, alpha=0.7)
ax.set_xticks(x); ax.set_xticklabels([f"移除 {c}" for c in CH], fontsize=10)
ax.set_ylabel("T4 重构 RMSE [°R]")
ax.set_title("图 2  逐通道移除实验（永久失效，逐步在线，test_id）", fontsize=12)
ax.legend(fontsize=9)
ax.text(0.02, 0.98, "虚线=全传感器基线\n柱子高于虚线=该传感器有贡献",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"v3_channel_removal.{e}")
plt.close(fig)

# 图 3：多种子
fig, ax = plt.subplots(figsize=(8, 4.5))
ms = sup["multi_seed"]
splits = ["test_id","test_ood"]
x = np.arange(len(splits)); w = 0.35
for i, m in enumerate(MODELS):
    if m not in ms: continue
    means = [ms[m][s]["mean"] for s in splits]
    stds = [ms[m][s]["std"] for s in splits]
    ax.bar(x+(i-0.5)*w, means, w, yerr=stds, capsize=6,
           label=m, color=COL[m], alpha=0.85)
    for xi, mu, sd in zip(x+(i-0.5)*w, means, stds):
        ax.text(xi, mu+2, f"{mu:.1f}±{sd:.2f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(splits)
ax.set_ylabel("T4 RMSE [°R]")
ax.set_title("图 3  多种子重复（逐步在线，永久失效，5 seeds）", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"v3_multiseed.{e}")
plt.close(fig)

# 图 4：噪声
fig, ax = plt.subplots(figsize=(8, 4.5))
ns = sup["noise"]
sd_list = sorted([float(k) for k in ns.keys()])
for m in MODELS:
    vals = [ns[str(s)].get(m) for s in sd_list]
    ax.plot(sd_list, vals, "o-", label=m, color=COL[m], ms=6)
ax.set_xlabel("观测噪声标准差（归一化空间）")
ax.set_ylabel("T4 重构 RMSE [°R]")
ax.set_title("图 4  传感器噪声鲁棒性", fontsize=12)
ax.legend(fontsize=9)
fig.tight_layout()
for e in ["png","pdf"]: fig.savefig(FIG/f"v3_noise.{e}")
plt.close(fig)

print("  4 张图已生成")