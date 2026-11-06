"""PSVA 阶段 1 曲线图：长期预测误差曲线"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(r"C:\pclwm\pclwm_psva\results")
FIG = RES / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "font.serif": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False, "figure.facecolor": "white",
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
    "axes.linewidth": 0.9, "lines.linewidth": 1.6,
    "savefig.dpi": 300, "savefig.bbox": "tight"})

P = json.loads((RES / "psva_results.json").read_text(encoding="utf-8"))
M = json.loads(Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
               .read_text(encoding="utf-8"))

E3 = M["seed0"]["cfgs"]["E3_both"]
PCTRL = P["P_ctrl_seed0"]
PSVA = P["PSVA_seed0"]

CFG = [
    ("E3 (PC-RSSM)", E3, "#1f77b4"),
    ("P_ctrl (不切分)", PCTRL, "#ff7f0e"),
    ("PSVA (64/64 切分)", PSVA, "#2ca02c"),
]

# ══════════════════════════════════════════════════
# 图 1：60s 连续推演误差曲线（均值 ± 标准差阴影）
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, split, title in zip(
        axes, ["test_id", "test_ood"],
        ["(a) Test-ID（2 条轨迹）", "(b) Test-OOD（3 条轨迹）"]):
    for name, cfg, col in CFG:
        cur = cfg["continuous"].get(f"{split}_60s", {})
        # 兼容两种格式：curves（多条）或 curve（单条均值）
        if "curves" in cur and cur["curves"]:
            curves = cur["curves"]
        elif "curve" in cur and cur["curve"]:
            curves = [cur["curve"]]
        else:
            continue
        # 对齐长度
        L = min(len(c) for c in curves)
        C = np.array([c[:L] for c in curves])
        t = np.arange(L) * 0.105
        mean = C.mean(0)
        std = C.std(0)
        # 平滑（100 点窗口）避免锯齿
        w = min(50, L // 10) or 1
        k = np.ones(w) / w
        m_s = np.convolve(mean, k, mode="valid")
        s_s = np.convolve(std, k, mode="valid")
        t_s = t[:len(m_s)]
        ax.plot(t_s, m_s, color=col, label=f"{name}  mean={cur.get('mean_rel', 0):.4f}")
        ax.fill_between(t_s, m_s - s_s, m_s + s_s, color=col, alpha=0.18)
    ax.set_xlabel("故障后时间 [s]")
    ax.set_ylabel("相对误差 (逐通道 RMSE / mean sd)")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(0, 60)

fig.suptitle("图 1  60s 连续自主推演误差曲线（mean ± std 跨轨迹）", fontsize=13)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"psva_curves_60s.{e}")
plt.close(fig)
print("  图1: psva_curves_60s")

# ══════════════════════════════════════════════════
# 图 2：多时间尺度对比
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, split, title in zip(
        axes, ["test_id", "test_ood"],
        ["(a) Test-ID", "(b) Test-OOD"]):
    x = np.arange(3)
    tgts = ["10s", "30s", "60s"]
    w = 0.26
    for i, (name, cfg, col) in enumerate(CFG):
        vals = [cfg["continuous"].get(f"{split}_{t}", {}).get("mean_rel", np.nan)
                for t in tgts]
        ax.bar(x + (i - 1) * w, vals, w, label=name, color=col, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["10 s", "30 s", "60 s"])
    ax.set_ylabel("相对误差")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8)

fig.suptitle("图 2  多时间尺度连续推演误差", fontsize=13)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"psva_multiscale.{e}")
plt.close(fig)
print("  图2: psva_multiscale")

# ══════════════════════════════════════════════════
# 图 3：Separation 对比
# ══════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))

names = ["E3", "P_ctrl", "PSVA"]
seps = [
    json.loads((RES / "e3_reference_split.json").read_text(encoding="utf-8"))
    ["Separation"]["sep_total"],
    PCTRL["probe"]["Separation"]["sep_total"],
    PSVA["probe"]["Separation"]["sep_total"],
]
cols = ["#1f77b4", "#ff7f0e", "#2ca02c"]
bars = ax.bar(names, seps, color=cols, alpha=0.85)
ax.axhline(0.10, color="red", ls="--", lw=1.5, label="成功门槛 0.10")
ax.axhline(0, color="gray", ls="-", lw=0.8)
ax.set_ylabel("Sep_total")
ax.set_title("图 3  潜状态快慢分工程度", fontsize=12)
ax.legend(fontsize=9)
for b, v in zip(bars, seps):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:+.4f}",
            ha="center", fontsize=9)
ax.set_ylim(-0.02, 0.12)

fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"psva_separation.{e}")
plt.close(fig)
print("  图3: psva_separation")

# ══════════════════════════════════════════════════
# 图 4：训练曲线
# ══════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, key, title in zip(
        axes, ["val", "l_fast", "l_dyn"],
        ["(a) 验证损失", "(b) L_fast（快变量）", "(c) L_dyn（动态）"]):
    for name, cfg, col in [("P_ctrl", PCTRL, "#ff7f0e"),
                            ("PSVA", PSVA, "#2ca02c")]:
        h = cfg["hist"].get(key, [])
        if h:
            ax.plot(h, color=col, label=name)
    ax.set_xlabel("Epoch")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    if key != "val":
        ax.set_yscale("log")

fig.suptitle("图 4  P_ctrl vs PSVA 训练曲线", fontsize=13)
fig.tight_layout()
for e in ["png", "pdf"]:
    fig.savefig(FIG / f"psva_train_curves.{e}")
plt.close(fig)
print("  图4: psva_train_curves")

print(f"\n  -> {FIG}")

