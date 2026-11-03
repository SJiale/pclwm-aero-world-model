"""画 loss 曲线"""
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

C = Path(r"C:\pclwm\agtf30\curve")
OUT = Path(r"D:\涡轴WM\_backup\workspace_20261031_182906\figures")
h = json.loads((C / "curves.json").read_text())
g, r = h["gru"], h["rssm"]
ep = np.arange(1, len(g["train"]) + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))

# ① GRU
ax = axes[0, 0]
ax.semilogy(ep, g["train"], lw=1.8, color="#2c7fb8", label="训练 loss")
ax.semilogy(ep, g["val"], lw=1.8, color="#d62728", label="验证 loss")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE（对数轴）")
ax.set_title("GRU：训练 loss 降到 0.001，验证卡在 0.18", fontweight="bold",
             fontsize=10.5)
ax.legend(fontsize=8)
ax.annotate("严重过拟合", xy=(35, 0.18), xytext=(35, 0.02),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
            color="red", fontsize=9, fontweight="bold")

# ② RSSM 总 loss
ax = axes[0, 1]
ax.plot(ep, r["train"], lw=1.8, color="#31a354", label="训练 loss")
ax.plot(ep, r["val"], lw=1.8, color="#d62728", label="验证 loss")
ax.set_xlabel("epoch")
ax.set_ylabel("总 loss")
ax.set_title("RSSM：训练 loss 卡在 0.62（不下降）", fontweight="bold",
             fontsize=10.5)
ax.legend(fontsize=8)
ax.axhline(0.6, color="orange", ls="--", lw=1.2, alpha=0.8)
ax.text(5, 0.65, "训练停在 0.62 = KL 项主导", color="darkorange", fontsize=8.5)

# ③ RSSM 分项
ax = axes[1, 0]
ax.plot(ep, r["rec"], lw=1.6, color="#2c7fb8", label="重建 loss")
ax.plot(ep, r["kl"], lw=1.6, color="#e6550d", label="KL loss")
ax.plot(ep, r["multi"], lw=1.6, color="#31a354", label="多步预测 loss")
ax.set_xlabel("epoch")
ax.set_ylabel("loss")
ax.set_title("RSSM 分项（训练集）：重建和多步都降到 0.004，但 KL 卡在 0.61",
             fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ④ GRU 的 k=1 vs k=H
ax = axes[1, 1]
ax.plot(ep, g["val_k1"], lw=1.6, color="#9467bd", label="验证 k=1")
ax.plot(ep, g["val_kH"], lw=1.6, color="#ff7f0e", label="验证 k=16")
ax.plot(ep, r["val_rec"], lw=1.6, ls="--", color="#2c7fb8", label="RSSM 验证重建")
ax.plot(ep, r["val_multi"], lw=1.6, ls="--", color="#31a354",
        label="RSSM 验证多步")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE")
ax.set_title("验证集：GRU 首步误差反而比末步大（异常）", fontweight="bold",
             fontsize=10)
ax.legend(fontsize=7.5)

fig.suptitle("训练 loss 曲线诊断：GRU 严重过拟合，RSSM 被 KL 项锁死",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig33_loss_curves.png", bbox_inches="tight")
plt.close(fig)

print("=" * 80)
print("Loss 曲线诊断")
print("=" * 80)
print(f"\n[GRU]")
print(f"  训练 loss: {g['train'][0]:.5f} -> {g['train'][-1]:.5f}")
print(f"  验证 loss: {g['val'][0]:.5f} -> {g['val'][-1]:.5f}")
print(f"  过拟合比 (val/train): {g['val'][-1]/max(g['train'][-1],1e-9):.0f}x")
print(f"  >>> 训练 loss 比验证低 100 倍以上 = 严重过拟合")

print(f"\n[RSSM]")
print(f"  训练总 loss: {r['train'][0]:.4f} -> {r['train'][-1]:.4f}")
print(f"  验证总 loss: {r['val'][0]:.4f} -> {r['val'][-1]:.4f}")
print(f"  分项（末轮训练）: rec={r['rec'][-1]:.5f} kl={r['kl'][-1]:.5f} "
      f"multi={r['multi'][-1]:.5f}")
print(f"  KL 占总 loss: {r['kl'][-1]/r['train'][-1]*100:.1f}%")
print(f"  验证分项（末轮）: v_rec={r['val_rec'][-1]:.5f} "
      f"v_kl={r['val_kl'][-1]:.5f} v_multi={r['val_multi'][-1]:.5f}")
print(f"  >>> KL 占 99%，重建和多步已经很好，但 KL 无法下降")
print(f"      KL=0.61 且几乎不动 = 后验和先验差距大，latent 未被有效利用")

print(f"\n[关键问题]")
print(f"  GRU: 训练 0.0011 -> 验证 0.185，过拟合 160 倍")
print(f"  RSSM: 训练 0.619 卡死，KL 占 99%")
print(f"  验证 loss RSSM 0.997 vs GRU 0.185 -> RSSM 差 5.4 倍")

print(f"\n图: {OUT/'fig33_loss_curves.png'}")
