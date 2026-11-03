"""端到端预测演示图"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130,
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})

PKG = Path(r"C:\pclwm\handoff\PC-LWM_handoff_20261102")
d = np.load(PKG / "results" / "prediction_demo.npz", allow_pickle=True)
OUT = PKG / "figures"

tgt = d["target"]          # (384, 38)
pg = d["pred_gru"]
pr = d["pred_rssm"]
act = d["action"]
cols = [str(c) for c in d["obs_cols"]]
dt = 0.105
t = np.arange(len(tgt)) * dt

R2K = lambda x: x * 5 / 9
PSI2KPA = lambda x: x * 6.89476


def get(arr, name):
    return arr[:, cols.index(name)]


fig, axes = plt.subplots(4, 2, figsize=(15, 12))

# ① 燃油（动作，输入）
ax = axes[0, 0]
ax.plot(t, act, lw=1.6, color="#e6550d")
ax.set_ylabel("燃油 Wf [lb/s]")
ax.set_title("① 输入：未来燃油指令（唯一给出的未来信息）",
             fontweight="bold", fontsize=10)
ax2 = axes[0, 0].twinx()
ax2.set_ylabel("", color="w")

# ② 压气机出口温度
ax = axes[0, 1]
ax.plot(t, R2K(get(tgt, "S36_Tt")), "k-", lw=2.2, label="真值", alpha=0.85)
ax.plot(t, R2K(get(pg, "S36_Tt")), "--", lw=1.8, color="#2c7fb8",
        label="GRU-h128 预测")
ax.plot(t, R2K(get(pr, "S36_Tt")), ":", lw=1.8, color="#31a354",
        label="RSSM 预测")
ax.set_ylabel("S36 总温 [K]")
ax.set_title("② 压气机出口温度", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ③ 涡轮前温度
ax = axes[1, 0]
ax.plot(t, R2K(get(tgt, "S4_Tt")), "k-", lw=2.2, label="真值", alpha=0.85)
ax.plot(t, R2K(get(pg, "S4_Tt")), "--", lw=1.8, color="#2c7fb8",
        label="GRU-h128")
ax.plot(t, R2K(get(pr, "S4_Tt")), ":", lw=1.8, color="#31a354", label="RSSM")
ax.set_ylabel("S4 总温 [K]")
ax.set_title("③ 涡轮前温度 T4", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ④ 高压压气机出口压力
ax = axes[1, 1]
ax.plot(t, PSI2KPA(get(tgt, "S36_Pt")), "k-", lw=2.2, label="真值", alpha=0.85)
ax.plot(t, PSI2KPA(get(pg, "S36_Pt")), "--", lw=1.8, color="#2c7fb8",
        label="GRU-h128")
ax.plot(t, PSI2KPA(get(pr, "S36_Pt")), ":", lw=1.8, color="#31a354",
        label="RSSM")
ax.set_ylabel("S36 总压 [kPa]")
ax.set_title("④ 压气机出口压力", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ⑤ 逐通道相对误差
ax = axes[2, 0]
KEEP = [c for c in cols if c not in
        ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]]
sd = tgt.std(0) + 1e-9
eg = [np.sqrt(((get(pg, c) - get(tgt, c)) ** 2).mean()) / sd[cols.index(c)]
      for c in KEEP]
er = [np.sqrt(((get(pr, c) - get(tgt, c)) ** 2).mean()) / sd[cols.index(c)]
      for c in KEEP]
x = np.arange(len(KEEP))
w = 0.38
ax.bar(x - w / 2, eg, w, color="#2c7fb8", alpha=0.85, label="GRU-h128")
ax.bar(x + w / 2, er, w, color="#31a354", alpha=0.85, label="RSSM")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in KEEP], fontsize=6, rotation=0)
ax.set_ylabel("相对 RMSE")
ax.set_title("⑤ 逐通道误差", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ⑥ 误差随预测时域增长
ax = axes[2, 1]
for name, arr, c in [("GRU-h128", pg, "#2c7fb8"), ("RSSM", pr, "#31a354")]:
    curve = []
    for k in range(len(tgt)):
        e = (arr[k, [cols.index(x) for x in KEEP]]
             - tgt[k, [cols.index(x) for x in KEEP]]) / sd[[cols.index(x) for x in KEEP]]
        curve.append(np.sqrt((e ** 2).mean()))
    ax.plot(t, curve, lw=1.8, color=c, label=name)
ax.set_xlabel("预测时间 [s]")
ax.set_ylabel("相对 RMSE")
ax.set_title("⑥ 误差随时域增长", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ⑦ 温升对比（关键性能指标）
ax = axes[3, 0]
dT4_true = R2K(get(tgt, "S4_Tt")) - R2K(get(tgt, "S36_Tt"))
dT4_g = R2K(get(pg, "S4_Tt")) - R2K(get(pg, "S36_Tt"))
dT4_r = R2K(get(pr, "S4_Tt")) - R2K(get(pr, "S36_Tt"))
ax.plot(t, dT4_true, "k-", lw=2.2, label="真值", alpha=0.85)
ax.plot(t, dT4_g, "--", lw=1.8, color="#2c7fb8", label="GRU-h128")
ax.plot(t, dT4_r, ":", lw=1.8, color="#31a354", label="RSSM")
ax.set_xlabel("预测时间 [s]")
ax.set_ylabel("ΔT = T4 - T36 [K]")
ax.set_title("⑦ 燃烧温升（物理量，模型能追踪）", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

# ⑧ 总压比
ax = axes[3, 1]
opr_t = get(tgt, "S36_Pt") / get(tgt, "S2_Pt")
opr_g = get(pg, "S36_Pt") / get(pg, "S2_Pt")
opr_r = get(pr, "S36_Pt") / get(pr, "S2_Pt")
ax.plot(t, opr_t, "k-", lw=2.2, label="真值", alpha=0.85)
ax.plot(t, opr_g, "--", lw=1.8, color="#2c7fb8", label="GRU-h128")
ax.plot(t, opr_r, ":", lw=1.8, color="#31a354", label="RSSM")
ax.set_xlabel("预测时间 [s]")
ax.set_ylabel("总压比 OPR")
ax.set_title("⑧ 总压比（从 9.6 到 40）", fontweight="bold", fontsize=10)
ax.legend(fontsize=8)

for a in axes[:2].ravel():
    a.set_xlabel("预测时间 [s]")

fig.suptitle("端到端预测演示：给定 6.7 s 历史 + 未来燃油指令，自主推算 40 s\n"
             "（不使用任何未来传感器读数）",
             fontsize=13.5, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "fig40_prediction_demo.png", bbox_inches="tight")
plt.close(fig)
print(f"图: {OUT/'fig40_prediction_demo.png'}")
