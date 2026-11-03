"""修复后的统一性能表 + 自由递推曲线

数据来源（全部实际运行结果）：
  - GRU:  gru_fair_w32.json   (WINDOW=32，公平重跑)
  - RSSM: rssm_mode_eval_fixed.json (显式 mode，全程 eval)
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(r"C:\pclwm\pclwm_experiments")
DIAG = OUT / "diagnosis"
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.linewidth": 0.9,
    "lines.linewidth": 1.7,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

gru = json.loads((DIAG / "gru_fair_w32.json").read_text(encoding="utf-8"))
rssm = json.loads((DIAG / "rssm_mode_eval_fixed.json").read_text(
    encoding="utf-8"))

# ══════════════════════════════════════════════════
# 1. 统一表
# ══════════════════════════════════════════════════
print("=" * 104)
print("修复后统一性能表（WINDOW=32，历史 3.36 s，HORIZON=32 步=3.36 s）")
print("=" * 104)

rows = []
# ARX 从 results_baselines.json
base = json.loads((OUT / "results_baselines.json").read_text(
    encoding="utf-8"))
if "ARX" in base:
    a = base["ARX"]
    rows.append(("ARX(na2,nb1)", a["params"],
                 a["test_id"]["overall_rel"], a["test_ood"]["overall_rel"],
                 a.get("long", {}), "—"))

for name in ["GRU96", "GRU128", "GRU192"]:
    if name not in gru:
        continue
    g = gru[name]
    rows.append((f"{name}", g["params"],
                 g["test_id"]["overall_rel"], g["test_ood"]["overall_rel"],
                 g.get("long", {}), "确定性"))

SH = rssm["short"]
LG = rssm["long"]
for mode in ["argmax", "prob", "gumbel"]:
    s_ti = SH[mode]["test_id"]
    s_od = SH[mode]["test_ood"]
    lng = {}
    for t in [10, 30, 60]:
        k = f"test_id_{t}s"
        if k in LG[mode]:
            lng[f"{t}s"] = {"final_rel": LG[mode][k]["final_mean"]}
    tag = "RSSM-d128"
    if mode == "gumbel":
        tag += " (gumbel)"
    elif mode == "prob":
        tag += " (prob)"
    else:
        tag += " (argmax)"
    rows.append((tag, 939909, s_ti["mean"], s_od["mean"], lng,
                 "随机潜状态" if mode != "argmax" else "确定性解码"))

print(f"\n{'模型':<24}{'参数':>10}{'test_id':>12}{'test_ood':>11}"
      f"{'长时10s':>10}{'长时30s':>10}{'长时60s':>10}")
print("-" * 92)
for name, npar, ti, od, lng, note in rows:
    def g(k):
        v = lng.get(k)
        if v is None:
            return "   -  "
        return f"{v['final_rel']:.4f}"
    print(f"{name:<24}{npar:>10,}{ti:>12.4f}{od:>11.4f}"
          f"{g('10s'):>10}{g('30s'):>10}{g('60s'):>10}")

# ══════════════════════════════════════════════════
# 2. 图：短期/长时对比 + argmax 崩溃
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# (a) test_id / test_ood 柱状
ax = axes[0]
names = [r[0] for r in rows]
ti = [r[2] for r in rows]
od = [r[3] for r in rows]
x = np.arange(len(names))
w = 0.38
b1 = ax.bar(x - w/2, ti, w, label="test_id", color="#1f77b4", alpha=0.85)
b2 = ax.bar(x + w/2, od, w, label="test_ood", color="#ff7f0e", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
ax.set_ylabel("相对 RMSE")
ax.set_title("(a) 短期 32 步预测误差", fontsize=11)
ax.legend(fontsize=8.5)
ax.set_yscale("log")
ax.axhline(1.0, color="r", ls="--", lw=0.8, alpha=0.6)
ax.text(len(names)-0.5, 1.05, "Persistence 水平", fontsize=7,
        color="r", ha="right")

# (b) 长时递推 final_rel
ax = axes[1]
gru_names = [n for n in ["GRU96", "GRU128", "GRU192"] if n in gru]
series = {
}
for n in gru_names:
    _l = gru[n]["long"]
    series[n] = [(_l.get(f"test_id_{t}s") if isinstance(_l.get(f"test_id_{t}s"), float)
                  else np.nan) for t in [10, 30, 60]]
for mode, lab, col in [("gumbel", "RSSM (gumbel)", "#2ca02c"),
                       ("prob", "RSSM (prob)", "#9467bd"),
                       ("argmax", "RSSM (argmax)", "#d62728")]:
    series[lab] = [LG[mode].get(f"test_id_{t}s", {}).get(
        "final_mean", np.nan) for t in [10, 30, 60]]
    if mode == "gumbel":
        series["RSSM (gumbel)_std"] = [LG[mode].get(
            f"test_id_{t}s", {}).get("final_std", 0) for t in [10, 30, 60]]

xs = [10, 30, 60]
colors = {"GRU96": "#999999", "GRU128": "#1f77b4", "GRU192": "#17becf",
          "RSSM (gumbel)": "#2ca02c", "RSSM (prob)": "#9467bd",
          "RSSM (argmax)": "#d62728"}
for k, v in series.items():
    if k.endswith("_std"):
        continue
    ax.plot(xs, v, "o-", label=k, color=colors.get(k, "#333"),
            ms=5)
if "RSSM (gumbel)_std" in series:
    g = series["RSSM (gumbel)"]
    s = series["RSSM (gumbel)_std"]
    ax.fill_between(xs, np.array(g)-np.array(s), np.array(g)+np.array(s),
                    color="#2ca02c", alpha=0.18)
ax.set_xlabel("自由递推时长 [s]")
ax.set_ylabel("final 相对误差")
ax.set_title("(b) 长时自由递推（test_id）", fontsize=11)
ax.legend(fontsize=8)
ax.set_yscale("log")
ax.set_xticks(xs)

# (c) 三种推理模式的 train/val/test 对比
ax = axes[2]
splits = ["train", "val", "test_id", "test_ood"]
modes = [("gumbel", "#2ca02c"), ("prob", "#9467bd"),
         ("argmax", "#d62728")]
for mode, col in modes:
    ys = [SH[mode][s]["mean"] for s in splits]
    es = [SH[mode][s]["std"] for s in splits]
    ax.errorbar(range(4), ys, yerr=es, marker="o", capsize=4,
                label=f"RSSM ({mode})", color=col)
ax.set_xticks(range(4))
ax.set_xticklabels(["train", "val", "test_id", "test_ood"], fontsize=9)
ax.set_ylabel("相对 RMSE")
ax.set_title("(c) 推理模式对同一检查点的影响", fontsize=11)
ax.legend(fontsize=8.5)
ax.set_yscale("log")
ax.axhline(0.08, color="gray", ls=":", lw=0.9, alpha=0.7)
ax.text(3, 0.085, "GRU-h128 水平", fontsize=7, color="gray", ha="right")

fig.suptitle("图 A  修复后基线对比（WINDOW=32，RSSM 显式推理模式，Gumbel 5 seeds）",
             fontsize=13, y=1.0)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(FIG / f"baseline_fixed_compare.{ext}")
plt.close(fig)
print(f"\n图 -> {FIG/'baseline_fixed_compare.png'}")

# ══════════════════════════════════════════════════
# 3. 自由递推曲线（gumbel）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))

for ax, split in zip(axes, ["test_id", "test_ood"]):
    for mode, col in [("gumbel", "#2ca02c"), ("prob", "#9467bd"),
                      ("argmax", "#d62728")]:
        for tgt in [30]:
            k = f"{split}_{tgt}s"
            if k not in LG[mode]:
                continue
            c = LG[mode][k]["curve"]
            if not c:
                continue
            t_axis = np.arange(len(c)) * 0.105
            ax.plot(t_axis, c, label=f"RSSM {mode} ({tgt}s rollout)",
                    color=col, lw=1.7)
    # GRU
    if "GRU128" in gru:
        ax.axhline(gru["GRU128"]["test_id"]["overall_rel"]
                   if split == "test_id"
                   else gru["GRU128"]["test_ood"]["overall_rel"],
                   color="#1f77b4", ls="--", lw=1.3,
                   label="GRU-h128 (32 步平均)")
    ax.set_xlabel("自由递推时间 [s]")
    ax.set_ylabel("相对 RMSE")
    ax.set_title(f"({'(a)' if split=='test_id' else '(b)'}) {split} "
                 f"自由递推误差增长", fontsize=11)
    ax.legend(fontsize=8)
    ax.set_yscale("log")

fig.suptitle("图 B  长时自由递推误差增长曲线（30 s 目标）",
             fontsize=13, y=1.0)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(FIG / f"rollout_curves.{ext}")
plt.close(fig)
print(f"图 -> {FIG/'rollout_curves.png'}")

# ══════════════════════════════════════════════════
# 4. 汇总 JSON
# ══════════════════════════════════════════════════
summary = {
    "config": {"window": 32, "horizon": 32, "dt": 0.105,
               "history_seconds": 3.36, "pred_seconds": 3.36},
    "table": [
        {"model": n, "params": int(p), "test_id": float(t),
         "test_ood": float(o),
         "long_10s": lng.get("10s", {}).get("final_rel"),
         "long_30s": lng.get("30s", {}).get("final_rel"),
         "long_60s": lng.get("60s", {}).get("final_rel"),
         "note": note}
        for n, p, t, o, lng, note in rows
    ],
}
(DIAG / "final_comparison.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n-> {DIAG/'final_comparison.json'}")

