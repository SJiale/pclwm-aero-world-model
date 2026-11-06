"""PSVA vs P_ctrl 配对差值图（替代原三条均值曲线图）

修正要点：
  1. 主图：逐时间步配对误差差值，y=0 参考线，正=PSVA改善
  2. Test-ID / Test-OOD 分开，并给逐轨迹
  3. ★ 明确标注：阴影是跨轨迹 std，不是跨训练种子 std
  4. 横轴统一"连续自主推演时间 (s)"
  5. 不扩大差异、不用误导坐标
  6. PNG + PDF
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = Path(r"C:\pclwm\pclwm_psva\results")
FIG = RES / "figures_v2"
FIG.mkdir(parents=True, exist_ok=True)

# ★ 中文字体：优先 SimSun（宋体），英文 Times New Roman
plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "SimHei",
                                    "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams.update({
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.3,
    "legend.frameon": True,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

P = json.loads((RES / "psva_results.json").read_text(encoding="utf-8"))
PCTRL = P["P_ctrl_seed0"]
PSVA = P["PSVA_seed0"]

DT = 0.105


def get_curves(cfg, split, tgt="60s"):
    """返回该配置在该 split 的逐轨迹误差曲线列表"""
    c = cfg["continuous"].get(f"{split}_{tgt}", {})
    if "curves" in c and c["curves"]:
        return [np.asarray(x) for x in c["curves"]]
    if "curve" in c and c["curve"]:
        return [np.asarray(c["curve"])]
    return []


def per_traj_curves(cfg, split, tgt="60s"):
    """逐轨迹曲线（若有 per_traj_curve 就分开；否则返回整体平均）"""
    return get_curves(cfg, split, tgt)


SPLITS = [
    ("test_id", "Test-ID", ["A11_msin_e3", "C6_mn_step"]),
    ("test_ood", "Test-OOD", ["A12_ramp_e3", "C7_threeway", "C8_multiphase"]),
]

print("=" * 88)
print("PSVA vs P_ctrl 配对差值绘图")
print("=" * 88)

# ══════════════════════════════════════════════════
# 数据准备：逐轨迹配对
# ══════════════════════════════════════════════════
# 注意：当前 results 中 curves 是把所有 (rep, traj) 展平的列表
# 需要按轨迹重新组织：每条轨迹有 3 次 rep
N_REP = 3

DATA = {}
for split, label, trajs in SPLITS:
    ctrl_all = get_curves(PCTRL, split)
    psva_all = get_curves(PSVA, split)
    n_traj = len(trajs)

    # 展平顺序：for r in range(N_REP): for rid in trajs
    # 所以索引 = r * n_traj + t_idx
    per_traj = {}
    for t_idx, rid in enumerate(trajs):
        idxs = [r * n_traj + t_idx for r in range(N_REP)]
        idxs = [i for i in idxs if i < len(ctrl_all) and i < len(psva_all)]
        if not idxs:
            continue
        L = min(min(len(ctrl_all[i]), len(psva_all[i])) for i in idxs)
        C = np.array([ctrl_all[i][:L] for i in idxs])   # [n_rep, L]
        Pv = np.array([psva_all[i][:L] for i in idxs])
        per_traj[rid] = {"ctrl": C, "psva": Pv}
    DATA[split] = {"label": label, "trajs": trajs, "per_traj": per_traj}
    print(f"  {split}: {len(per_traj)} 条轨迹匹配")


# ============================================================
# 图 1（主图）：逐时间步配对差值曲线
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

for ax, (split, label, trajs) in zip(axes, SPLITS):
    pt = DATA[split]["per_traj"]
    if not pt:
        ax.text(0.5, 0.5, "no data", ha="center", transform=ax.transAxes)
        continue

    L = min(min(len(v["ctrl"][0]), len(v["psva"][0])) for v in pt.values())
    t = np.arange(L) * DT

    # 每条轨迹的差值（PSVA - P_ctrl，正=改善）
    diffs = []
    for rid in trajs:
        if rid not in pt:
            continue
        d = pt[rid]["ctrl"][:, :L].mean(0) - pt[rid]["psva"][:, :L].mean(0)
        diffs.append(d)
    D = np.array(diffs)

    # 平滑（窗口 41 步 ≈ 4.3s）
    w = 41
    k = np.ones(w) / w
    for i, rid in enumerate([r for r in trajs if r in pt]):
        d_s = np.convolve(D[i], k, mode="same")
        # 边界修正（避免卷积边界效应）
        d_s[:w // 2] = D[i][:w // 2]
        d_s[-(w // 2):] = D[i][-(w // 2):]
        ax.plot(t, d_s, lw=1.2, alpha=0.85, label=rid)

    # 平均值
    mean_d = D.mean(0)
    mean_s = np.convolve(mean_d, k, mode="same")
    mean_s[:w // 2] = mean_d[:w // 2]
    mean_s[-(w // 2):] = mean_d[-(w // 2):]
    ax.plot(t, mean_s, "k-", lw=1.8, label="均值", zorder=5)

    # y=0 参考线
    ax.axhline(0, color="0.35", lw=1.0, ls="-", zorder=1)

    ax.set_xlabel("连续自主推演时间 (s)")
    ax.set_ylabel("误差差值  $e_{P\\_ctrl} - e_{PSVA}$")
    ax.set_title(f"({chr(97 + SPLITS.index((split, label, trajs)))}) {label}"
                 f"  ({len(pt)} 条轨迹)", fontsize=10)
    ax.set_xlim(0, t[-1])
    ax.legend(fontsize=7.5, loc="best", ncol=1)

    # 标注区域含义
    ax.text(0.02, 0.96, "↑ PSVA 改善", transform=ax.transAxes,
            fontsize=7.5, va="top", color="darkgreen")
    ax.text(0.02, 0.04, "↓ PSVA 退化", transform=ax.transAxes,
            fontsize=7.5, va="bottom", color="darkred")

fig.suptitle("图 1  PSVA 相对 P_ctrl 的逐时间步误差差值（正=PSVA 改善；"
             "曲线已做 4.3 s 滑动平均）", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig1_paired_diff_60s.{e}")
plt.close(fig)
print("\n  图1: fig1_paired_diff_60s  （主图）")

# ============================================================
# 图 2：逐轨迹相对改善（柱状图）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2),
                         gridspec_kw={"width_ratios": [2, 3]})

for ax, (split, label, trajs) in zip(axes, SPLITS):
    pt = DATA[split]["per_traj"]
    names, vals = [], []
    for rid in trajs:
        if rid not in pt:
            continue
        c_mean = pt[rid]["ctrl"].mean()
        p_mean = pt[rid]["psva"].mean()
        rel = (c_mean - p_mean) / c_mean * 100
        names.append(rid)
        vals.append(rel)

    if not names:
        continue

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in vals]
    bars = ax.barh(names, vals, color=colors, alpha=0.8, height=0.55)
    ax.axvline(0, color="0.3", lw=1.0)
    ax.set_xlabel("相对改善 (%)   [正=PSVA 更好]")
    ax.set_title(f"{label}", fontsize=10)
    ax.invert_yaxis()

    for b, v in zip(bars, vals):
        x = v + (0.15 if v > 0 else -0.15)
        ha = "left" if v > 0 else "right"
        ax.text(x, b.get_y() + b.get_height() / 2, f"{v:+.2f}%",
                va="center", ha=ha, fontsize=8)

    # 自适应坐标范围（不夸大）
    lim = max(abs(min(vals)), abs(max(vals))) * 1.45
    ax.set_xlim(-lim, lim)

fig.suptitle("图 2  逐轨迹 60 s 平均误差的相对改善（PSVA vs P_ctrl）",
             fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig2_per_traj_relative.{e}")
plt.close(fig)
print("  图2: fig2_per_traj_relative")

# ============================================================
# 图 3：分时段对照（0-10/10-30/30-60 s）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

SEGS = [(0, 10), (10, 30), (30, 60)]
SEG_LBL = ["0–10 s", "10–30 s", "30–60 s"]

for ax, (split, label, trajs) in zip(axes, SPLITS):
    pt = DATA[split]["per_traj"]
    x = np.arange(len(SEGS))
    w = 0.26
    n = len([r for r in trajs if r in pt])

    for i, rid in enumerate([r for r in trajs if r in pt]):
        vals = []
        for lo, hi in SEGS:
            i0 = int(round(lo / DT))
            i1 = None if hi >= 60 else int(round(hi / DT))
            c = pt[rid]["ctrl"][:, i0:i1].mean()
            p = pt[rid]["psva"][:, i0:i1].mean()
            vals.append((c - p) / c * 100)
        ax.bar(x + (i - (n - 1) / 2) * w, vals, w, label=rid, alpha=0.85)

    ax.axhline(0, color="0.3", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(SEG_LBL)
    ax.set_ylabel("相对改善 (%)")
    ax.set_xlabel("推演时段")
    ax.set_title(label, fontsize=10)
    ax.legend(fontsize=7.5)

fig.suptitle("图 3  分时段相对改善（PSVA vs P_ctrl）", fontsize=11)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig3_segment_relative.{e}")
plt.close(fig)
print("  图3: fig3_segment_relative")

# ============================================================
# 图 4：分离度（保留，但标注清楚）
# ============================================================
e3ref = json.loads((RES / "e3_reference_split.json").read_text(encoding="utf-8"))
fig, ax = plt.subplots(figsize=(6.4, 4.2))

names = ["E3", "P_ctrl", "PSVA"]
vals = [
    e3ref["Separation"]["sep_total"],
    PCTRL["probe"]["Separation"]["sep_total"],
    PSVA["probe"]["Separation"]["sep_total"],
]
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
bars = ax.bar(names, vals, color=cols, alpha=0.85, width=0.55)
ax.axhline(0, color="0.3", lw=0.9)
ax.axhline(0.10, color="#d62728", ls="--", lw=1.4,
           label="设计门槛 $\\mathrm{Sep}>0.10$")
ax.set_ylabel("$\\mathrm{Sep}_{\\mathrm{total}}$")
ax.set_title("图 4  快慢状态分工程度（h 前/后 64 维解码 R² 差值）",
             fontsize=10)
ax.legend(fontsize=8)

for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2,
            v + (0.003 if v >= 0 else -0.008),
            f"{v:+.4f}", ha="center", fontsize=8.5)

ax.set_ylim(-0.025, 0.12)
ax.text(0.98, 0.95, "门槛为本研究预设，非文献标准",
        transform=ax.transAxes, fontsize=6.5, ha="right", va="top",
        color="0.4", style="italic")

fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"fig4_separation.{e}")
plt.close(fig)
print("  图4: fig4_separation")

# ============================================================
# 数值摘要（文本，供报告引用）
# ============================================================
print("\n" + "=" * 88)
print("数值摘要（真实数值，未做任何视觉放大）")
print("=" * 88)

print(f"\n  {'split':<10}{'轨迹':<16}{'P_ctrl均值':>12}{'PSVA均值':>11}"
      f"{'差值':>11}{'相对':>9}{'方向':>6}")
print("  " + "-" * 76)
for split, label, trajs in SPLITS:
    pt = DATA[split]["per_traj"]
    for rid in trajs:
        if rid not in pt:
            continue
        c = float(pt[rid]["ctrl"].mean())
        p = float(pt[rid]["psva"].mean())
        print(f"  {split:<10}{rid:<16}{c:>12.4f}{p:>11.4f}"
              f"{c-p:>+11.4f}{(c-p)/c*100:>+8.2f}%"
              f"{'改善' if p < c else '退化':>6}")

# 汇总（跨轨迹）
print(f"\n  跨轨迹汇总:")
for split, label, trajs in SPLITS:
    pt = DATA[split]["per_traj"]
    cs = np.concatenate([pt[r]["ctrl"].ravel() for r in pt])
    ps = np.concatenate([pt[r]["psva"].ravel() for r in pt])
    n_better = sum(1 for r in pt if pt[r]["psva"].mean() < pt[r]["ctrl"].mean())
    print(f"    {label:<10} P_ctrl={cs.mean():.4f}  PSVA={ps.mean():.4f}  "
          f"相对={(cs.mean()-ps.mean())/cs.mean()*100:+.2f}%  "
          f"改善轨迹={n_better}/{len(pt)}")

print(f"\n  -> {FIG}")

