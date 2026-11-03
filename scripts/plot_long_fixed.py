"""长时评估图：修正时序后的结果"""
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
RES = PKG / "results"
OUT = PKG / "figures"
d = json.loads((RES / "long_eval_fixed.json").read_text(encoding="utf-8"))
agg = d["aggregate"]
pt = d["per_traj"]
dt = d["config"]["dt_s"]

COLOR = {"GRU": "#2c7fb8", "RSSM": "#31a354", "Persistence": "#999999"}

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

# ① 平均误差曲线
ax = axes[0]
for m in ["Persistence", "GRU", "RSSM"]:
    c = agg[m]["curve"]
    t = np.arange(1, len(c) + 1) * dt
    ax.semilogy(t, c, lw=2, color=COLOR[m], label=m)
ax.set_xlabel("预测时域 [s]")
ax.set_ylabel("相对 RMSE（对数轴）")
ax.set_title(f"长时自回归误差（{d['config']['time_s']:.0f} s，10 条 test）",
             fontweight="bold", fontsize=10.5)
ax.legend(fontsize=9)

# ② 逐轨迹散点
ax = axes[1]
rids = list(pt.keys())
x = np.arange(len(rids))
w = 0.27
for i, m in enumerate(["GRU", "RSSM", "Persistence"]):
    vals = [pt[r][m]["mean"] for r in rids]
    ax.bar(x + (i - 1) * w, vals, w, color=COLOR[m], alpha=0.85,
           label=m)
ax.set_xticks(x)
ax.set_xticklabels([r.split("_")[1] for r in rids], fontsize=7, rotation=30)
ax.set_ylabel("相对 RMSE")
ax.set_yscale("log")
ax.set_title("逐轨迹误差（按激励类型）", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

# ③ 首步 vs 末步
ax = axes[2]
for m in ["GRU", "RSSM"]:
    k1 = [pt[r][m]["k1"] for r in rids]
    kH = [pt[r][m]["kH"] for r in rids]
    ax.scatter(k1, kH, s=70, color=COLOR[m], alpha=0.8, label=m,
               edgecolors="k", linewidths=0.5)
lim = [1e-5, 1e0]
ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="y=x")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("首步误差 (k=1)")
ax.set_ylabel(f"末步误差 (k={d['config']['steps']})")
ax.set_title("首步 vs 末步：谁在长时劣化", fontweight="bold", fontsize=10.5)
ax.legend(fontsize=8)

fig.suptitle("长时递推评估（已修正动作时序 a[t-1]→o[t]）",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig36_long_eval_fixed.png", bbox_inches="tight")
plt.close(fig)

print("=" * 80)
print("修正时序后的结论")
print("=" * 80)
for m in ["GRU", "RSSM", "Persistence"]:
    a = agg[m]
    print(f"  {m:<14} k1={a['k1']:.4f}  末步={a['kH']:.4f}  "
          f"漂移={a['drift']:.1f}x  均值={a['mean']:.4f}")
print()
print("  >>> 首步：GRU 更准（%.4f vs %.4f，好 %.1f 倍）"
      % (agg["GRU"]["k1"], agg["RSSM"]["k1"],
         agg["RSSM"]["k1"] / agg["GRU"]["k1"]))
print("  >>> 末步：RSSM 更准（%.4f vs %.4f，好 %.1f 倍）"
      % (agg["RSSM"]["kH"], agg["GRU"]["kH"],
         agg["GRU"]["kH"] / agg["RSSM"]["kH"]))
print()
print("  这解释了为什么需要看长时：短时(3.36s)看 GRU 赢，")
print("  长时(40s)看 RSSM 赢。任务指标决定结论。")
print(f"\n图: {OUT/'fig36_long_eval_fixed.png'}")
