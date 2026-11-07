"""因果诊断：读 CSV 分析（MPC 是否预测到超调 + 为什么仍选该动作）"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

CD = Path(r"C:\pclwm\pclwm_stage3a\causal")
DT = 0.105

S = pd.read_csv(CD / "steps.csv")
P = pd.read_csv(CD / "pred_n2_matrix.csv", header=None).to_numpy()
PS = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()
IT = pd.read_csv(CD / "cem_iters.csv")

cfg = {}
for ln in (CD / "cfg.txt").read_text().split("\n"):
    if "=" in ln:
        k, v = ln.split("=")
        cfg[k] = float(v)

N2_REF = cfg["n2_ref"]
N2_0 = cfg["n2_0"]
H = int(cfg["H"])
n_short = int(cfg["n_short"])
w_short = cfg["w_short"]

# 权重
wv = np.ones(H)
wv[:n_short] = w_short
wv = wv / wv.sum()

print("=" * 98)
print("因果诊断：MPC 是否预测到超调，为什么仍选该动作")
print("=" * 98)
print(f"  步数 N={len(S)}, H={H}, n_short={n_short}, w_short={w_short}")
print(f"  N2_ref={N2_REF:.2f}, N2_0={N2_0:.2f}")
print(f"  前 {n_short} 步权重 wv[0]={wv[0]:.5f}, 后步 wv[-1]={wv[-1]:.5f}, "
      f"比值={wv[0]/wv[-1]:.1f}x")

# ══════════════════════════════════════════════════
# [A] 是否预测到超调
# ══════════════════════════════════════════════════
print("\n" + "=" * 98)
print("[A] MPC 是否预测到 N2 超过 5099？")
print("=" * 98)

ov = S["pred_overshoot"].to_numpy()
pos = np.where(ov > 0)[0]

print(f"\n  预测到超调的步数: {len(pos)} / {len(S)}")
if len(pos):
    i0 = pos[0]
    print(f"  首次预测到超调: k={int(S['k'].iloc[i0])}, t={S['t'].iloc[i0]:.2f}s")
    print(f"    当时 N2={S['n2_before'].iloc[i0]:.1f}, "
          f"err={S['err_before'].iloc[i0]:+.1f}")
    print(f"    选 PLA={S['pla_apply'].iloc[i0]:.2f}")
    print(f"    预测峰值={S['pred_peak'].iloc[i0]:.1f}, "
          f"超调={ov[i0]:+.1f}")

# 预测到超调后仍加大 PLA 的步
print(f"\n  ── 已预测到超调、仍加大 PLA 的步 ──")
print(f"  {'k':>3}{'t':>7}{'PLA':>8}{'ΔPLA':>8}{'预测超调':>10}"
      f"{'预测峰值':>10}{'真实N2':>10}")
print("  " + "-" * 60)
n_agg = 0
for i in range(1, len(S)):
    if ov[i] > 0 and S["pla_apply"].iloc[i] > S["pla_apply"].iloc[i-1] + 1e-9:
        d = S["pla_apply"].iloc[i] - S["pla_apply"].iloc[i-1]
        print(f"  {int(S['k'].iloc[i]):>3}{S['t'].iloc[i]:>7.2f}"
              f"{S['pla_apply'].iloc[i]:>8.2f}{d:>+8.2f}"
              f"{ov[i]:>+10.1f}{S['pred_peak'].iloc[i]:>10.1f}"
              f"{S['n2_before'].iloc[i]:>10.1f}")
        n_agg += 1
print(f"\n  -> 共 {n_agg} 步")

# ══════════════════════════════════════════════════
# [D] 未来 PLA 序列形状
# ══════════════════════════════════════════════════
print("\n" + "=" * 98)
print("[D] MPC 预测的未来 PLA 序列：是否很快下降？")
print("=" * 98)

print(f"\n  {'k':>3}{'t':>7}{'执行PLA':>9}{'PLA[1]':>8}{'PLA[8]':>8}"
      f"{'PLA[16]':>8}{'PLA[24]':>8}{'PLA[32]':>8}{'峰位':>6}{'变化':>9}")
print("  " + "-" * 76)
for i in range(len(S)):
    k = int(S["k"].iloc[i])
    ps = PS[i]
    im = int(np.argmax(ps)) + 1
    print(f"  {k:>3}{S['t'].iloc[i]:>7.2f}{S['pla_apply'].iloc[i]:>9.2f}"
          f"{ps[0]:>8.2f}{ps[7]:>8.2f}{ps[15]:>8.2f}{ps[23]:>8.2f}"
          f"{ps[31]:>8.2f}{im:>6}{ps[31]-ps[0]:>+9.2f}")

# 统计
dec = np.sum(PS[:, -1] < PS[:, 0] - 0.5)
print(f"\n  PLA[32] 比 PLA[1] 低超过 0.5° 的步数: {dec} / {len(S)}")
print(f"  -> {'✅ MPC 计划后续大幅收油门' if dec > len(S)*0.5 else '❌ MPC 未计划明显回收'}")

# ══════════════════════════════════════════════════
# [B] 代价分解：为什么仍选该动作
# ══════════════════════════════════════════════════
print("\n" + "=" * 98)
print("[B] 代价分解：前段跟踪 vs 后段超调")
print("=" * 98)

print(f"\n  {'k':>3}{'t':>7}{'前8步MSE':>12}{'后24步MSE':>12}"
      f"{'加权j_track':>13}{'超调项':>11}{'超调占比':>10}")
print("  " + "-" * 70)
KEY = [10, 14, 15, 16, 17, 20, 25]
for i in range(len(S)):
    k = int(S["k"].iloc[i])
    if k not in KEY:
        continue
    e = P[i] - N2_REF
    mse_s = float(np.mean(e[:n_short]**2))
    mse_l = float(np.mean(e[n_short:]**2))
    jt = float(np.sum(wv * e**2))
    ovp = np.maximum(e, 0)
    ov_c = float(np.sum(wv * ovp**2))
    pct = ov_c / jt * 100 if jt > 0 else 0
    print(f"  {k:>3}{S['t'].iloc[i]:>7.2f}{mse_s:>12.1f}{mse_l:>12.1f}"
          f"{jt:>13.1f}{ov_c:>11.1f}{pct:>9.1f}%")

# ══════════════════════════════════════════════════
# [C] 预测准确性：MPC 预测峰值 vs 实际后续真实值
# ══════════════════════════════════════════════════
print("\n" + "=" * 98)
print("[C] 预测准确性：MPC 预测峰值 vs 后续真实到达的峰值")
print("=" * 98)

# 实际 N2 轨迹
n2_real = S["n2_before"].to_numpy()
t_real = S["t"].to_numpy()
# 加最后一步的 after
n2_real_full = np.append(n2_real, S["n2_after"].iloc[-1])
t_real_full = np.append(t_real, t_real[-1] + DT)

print(f"\n  {'k':>3}{'t':>7}{'预测峰值':>10}{'预测峰时':>10}"
      f"{'实际未来峰值':>13}{'误差':>10}")
print("  " + "-" * 56)
for i in range(0, len(S), 2):
    k = int(S["k"].iloc[i])
    t_now = S["t"].iloc[i]
    # 实际未来 H 步内的峰值
    future_mask = (t_real_full > t_now) & (t_real_full <= t_now + H*DT + 1e-9)
    if future_mask.sum() > 0:
        actual_peak = float(np.max(n2_real_full[future_mask]))
    else:
        actual_peak = np.nan
    pred_peak = S["pred_peak"].iloc[i]
    err = pred_peak - actual_peak if not np.isnan(actual_peak) else np.nan
    print(f"  {k:>3}{t_now:>7.2f}{pred_peak:>10.1f}"
          f"{S['pred_peak_t'].iloc[i]:>10.2f}{actual_peak:>13.1f}"
          f"{err:>+10.1f}" if not np.isnan(err) else
          f"  {k:>3}{t_now:>7.2f}{pred_peak:>10.1f}"
          f"{S['pred_peak_t'].iloc[i]:>10.2f}{'n/a':>13}{'n/a':>10}")

# ══════════════════════════════════════════════════
# CEM 迭代收敛
# ══════════════════════════════════════════════════
print("\n" + "=" * 98)
print("[附加] CEM 迭代收敛情况（k=17 附近）")
print("=" * 98)

for k in [14, 16, 17, 20]:
    sub = IT[IT["k"] == k]
    if len(sub) == 0:
        continue
    print(f"\n  ── k={k} ──")
    print(f"    {'it':>3}{'J_best':>12}{'j_track':>12}{'j_du':>10}"
          f"{'PLA[1]':>9}{'预测峰值':>10}")
    for _, r in sub.iterrows():
        print(f"    {int(r['it']):>3}{r['J_best']:>12.1f}{r['j_track']:>12.1f}"
              f"{r['j_du']:>10.2f}{r['pla1']:>9.2f}{r['pred_peak']:>10.1f}")

# 保存
res = {
    "A_predicted_overshoot": int(len(pos)),
    "A_first_k": int(S["k"].iloc[pos[0]]) if len(pos) else None,
    "A_first_t": float(S["t"].iloc[pos[0]]) if len(pos) else None,
    "A_n_agg_after_pred": int(n_agg),
    "D_pla_decrease_steps": int(dec),
    "D_n_steps": int(len(S)),
    "cfg": cfg,
}
(CD / "causal_summary.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {CD/'causal_summary.json'}")
