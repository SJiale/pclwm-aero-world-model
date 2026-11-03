"""v2 结果：loss 曲线 + 精度对比"""
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

T2 = Path(r"C:\pclwm\agtf30\train2")
OUT = Path(r"D:\涡轴WM\_backup\workspace_20261031_182906\figures")

r2 = json.loads((T2 / "results2.json").read_text(encoding="utf-8"))
R = r2["results"]
hg, hr = r2["hist_gru"], r2["hist_rssm"]
HOR = r2["config"]["horizon"]
DT = r2["config"]["dt"]
P = r2["params"]

# ═══════════════════════════════════════════════════════
# 图1：loss 曲线（改进后）
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))

ax = axes[0]
ep = np.arange(1, len(hg["train"]) + 1)
ax.semilogy(ep, hg["train"], lw=1.8, color="#2c7fb8", label="训练")
ax.semilogy(ep, hg["val"], lw=1.8, color="#d62728", label="验证")
ax.set_xlabel("epoch"); ax.set_ylabel("MSE（对数轴）")
ax.set_title(f"GRU v2（{P['gru']:,} 参数）", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)
ratio = hg["val"][-1] / max(hg["train"][-1], 1e-9)
ax.text(0.5, 0.05, f"val/train = {ratio:.1f}x", transform=ax.transAxes,
        ha="center", fontsize=9, color="darkred", fontweight="bold")

ax = axes[1]
ep2 = np.arange(1, len(hr["train"]) + 1)
ax.plot(ep2, hr["train"], lw=1.8, color="#31a354", label="训练")
ax.plot(ep2, hr["val"], lw=1.8, color="#d62728", label="验证")
ax.set_xlabel("epoch"); ax.set_ylabel("总 loss")
ax.set_title(f"RSSM v2（{P['rssm']:,} 参数，kl_scale=0.3）",
             fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(ep2, hr["rec"], lw=1.6, color="#2c7fb8", label="重建")
ax.plot(ep2, hr["kl"], lw=1.6, color="#e6550d", label="KL（已×0.3）")
ax.plot(ep2, hr["multi"], lw=1.6, color="#31a354", label="多步预测")
ax.set_xlabel("epoch"); ax.set_ylabel("loss")
ax.set_title("RSSM 分项", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

fig.suptitle("改进版训练 v2 的 loss 曲线", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig34_v2_loss.png", bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════
# 图2：精度对比
# ═══════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.8))

COLORS = {"Persistence": "#999999", "ARX": "#e6550d",
          "GRU": "#2c7fb8", "RSSM": "#31a354"}
kt = np.arange(1, HOR + 1) * DT

ax = axes[0]
for nm in ["Persistence", "ARX", "GRU", "RSSM"]:
    c = R[nm]["curve"]
    ax.semilogy(kt, c, "o-", ms=2.5, lw=1.7, color=COLORS[nm], label=nm)
ax.set_xlabel("预测时域 [s]"); ax.set_ylabel("相对 RMSE（对数轴）")
ax.set_title(f"误差随时域（test，H={HOR} 步 = {HOR*DT:.1f} s）",
             fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

ax = axes[1]
names = ["Persistence", "ARX", "GRU", "RSSM"]
vals = [R[n]["test"] for n in names]
bars = ax.bar(range(4), vals, color=[COLORS[n] for n in names], alpha=0.85)
ax.set_xticks(range(4)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("相对 RMSE（对数轴）"); ax.set_yscale("log")
ax.set_title("测试集整体精度", fontweight="bold", fontsize=10.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}",
            ha="center", va="bottom", fontsize=9)

fig.suptitle("改进版 v2 精度对比", fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig35_v2_accuracy.png", bbox_inches="tight")
plt.close(fig)

# ═══════════════════════════════════════════════════════
print("=" * 82)
print("改进版 v2 结果")
print("=" * 82)
print(f"{'模型':<16}{'训练':>10}{'测试':>10}{'泛化比':>9}"
      f"{'k=1':>9}{'k=H':>9}{'增长':>8}")
print("-" * 74)
for nm in ["Persistence", "ARX", "GRU", "RSSM"]:
    r = R[nm]
    print(f"{nm:<16}{r['train']:>10.4f}{r['test']:>10.4f}"
          f"{r['test']/r['train']:>9.1f}x{r['k1']:>9.4f}{r['kH']:>9.4f}"
          f"{r['kH']/max(r['k1'],1e-12):>8.2f}x")

print()
print("=" * 82)
print("关键问题：RSSM 是否超过 Persistence？")
print("=" * 82)
p = R["Persistence"]["test"]
for nm in ["ARX", "GRU", "RSSM"]:
    v = R[nm]["test"]
    verdict = "✅ 更好" if v < p else "❌ 更差"
    print(f"  {nm:<14} {v:.4f} vs Persistence {p:.4f}  "
          f"({'好' if v<p else '差'} {max(v,p)/min(v,p):.1f}x)  {verdict}")

print()
print(f"  GRU vs RSSM: {R['GRU']['test']:.4f} vs {R['RSSM']['test']:.4f}")
if R["RSSM"]["test"] < R["GRU"]["test"]:
    print(f"    -> RSSM 更好 {R['GRU']['test']/R['RSSM']['test']:.2f}x")
else:
    print(f"    -> GRU 更好 {R['RSSM']['test']/R['GRU']['test']:.2f}x")

print()
print("=" * 82)
print("GRU 过拟合是否改善？")
print("=" * 82)
print(f"  val/train = {ratio:.1f}x   （v1: 166x）")
print(f"  {'✅ 显著改善' if ratio < 20 else '⚠ 仍过拟合'}")

print()
print(f"图:")
for f in ["fig34_v2_loss.png", "fig35_v2_accuracy.png"]:
    print(f"  {OUT/f}")
