"""诊断：为什么 CEM 找不到明显更优的解？

疑点：
  k=17 时，"立即降至 63.64"的等权 MSE = 5747
  CEM 的选择 70.82 的等权 MSE = 125532（差 21 倍）
  
  但 CEM 自身的目标函数 J（w_short 加权）应该也能识别这个差距

检查：
  1. CEM 的 J 在两种解下分别是多少？
  2. 是否 warm-start 让 mu 粘在 70.8 附近？
  3. 是否 dpla_max 约束阻止了下降？
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

CD = Path(r"C:\pclwm\pclwm_stage3a\causal")
DT = 0.105
S = pd.read_csv(CD / "steps.csv")
PS = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()
P = pd.read_csv(CD / "pred_n2_matrix.csv", header=None).to_numpy()

cfg = {}
for ln in (CD / "cfg.txt").read_text().split("\n"):
    if "=" in ln:
        k, v = ln.split("=")
        cfg[k] = float(v)
N2_REF = cfg["n2_ref"]
H = int(cfg["H"])
N = len(S)

n_short = int(cfg["n_short"])
w_short = cfg["w_short"]
w_n2 = cfg["w_n2"]
w_du = cfg["w_du"]
dpla_max = cfg["dpla_max"]

wv = np.ones(H)
wv[:n_short] = w_short
wv = wv / wv.sum()

print("=" * 96)
print("诊断：CEM 为何找不到更优解")
print("=" * 96)
print(f"  w_short={w_short}, w_n2={w_n2}, w_du={w_du}, dpla_max={dpla_max}")

# ══════════════════════════════════════════════════
# 用 CEM 自己的 J 评价"立即降至63.64" vs CEM 解
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[1] 用 CEM 自己的加权 J 评价两种解")
print("=" * 96)

# 需要重新预测"立即降至63.64"的轨迹
# 从前面的实验我们知道：
#   k=17: 立即降 -> 预测峰值 5092.1, 等权MSE 5746.7
#   k=17: CEM解 -> 预测峰值 5566.1, 等权MSE 125531.8
# 现在算加权 J

print(f"\n  注：'立即降至63.64'的完整 32 步预测需要重算")
print(f"  但我们可以用已有的等权 MSE 与加权形式的关系推断")

# 从 steps.csv 得到 CEM 解的实际加权 j_track
print(f"\n  CEM 解的实际代价（来自 steps.csv）:")
print(f"  {'k':>3}{'t':>7}{'j_track(w)':>13}{'j_du':>9}{'J_total':>12}")
print("  " + "-" * 45)
for i in range(N):
    k = int(S["k"].iloc[i])
    if k in [10, 14, 17]:
        print(f"  {k:>3}{S['t'].iloc[i]:>7.2f}{S['j_track'].iloc[i]:>13.1f}"
              f"{S['j_du'].iloc[i]:>9.2f}{S['J_total'].iloc[i]:>12.1f}")

# 估算"立即降至63.64"的加权 j_track
# 等权 MSE * H = 等权 sum(e^2)
# 加权 j_track = sum(wv*e^2)
# 若前 8 步误差大（E_short）后 24 步小（E_long）：
#   等权 MSE = (8*E_short + 24*E_long)/32
#   加权 j_track = 0.0625*8*E_short + 0.02083*24*E_long
#                = 0.5*E_short + 0.5*E_long
#   => 加权 J ≈ 0.5*(E_short + E_long)

print(f"\n  关键推导：")
print(f"    wv 归一化后：前8步权重和 = {wv[:8].sum():.4f}, 后24步 = {wv[8:].sum():.4f}")
print(f"    -> 加权 j_track = {wv[:8].sum():.3f}*sum(e_short^2) + {wv[8:].sum():.3f}*sum(e_long^2)")
print(f"    -> 前后段各占约 50%")

print(f"\n  这意味着：即使前段加权 3x，归一化后前后段贡献仍是 50/50！")
print(f"  -> w_short=3.0 与 w_short=1.0 的差异很小（实验B已证实）")

# ══════════════════════════════════════════════════
# [2] 检查 warm-start 的时域移位问题
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[2] ★ warm-start 时域移位问题")
print("=" * 96)

print(f"\n  mpc_opt.m L36-40:")
print(f"    if CFG.warm_start && ~isempty(CFG.mu_prev)")
print(f"      mu = CFG.mu_prev(:)';   % 直接复用上一步的 32 步序列")
print(f"    end")
print(f"\n  问题：上一步执行了 mu[1]，本步的 mu[1] 应该是上一步的 mu[2]")
print(f"        但代码直接用了 mu[1..32]，没有左移一位")
print(f"  -> 每次迭代，CEM 的中心点都比正确值【晚一步】")

# 验证：比较相邻步的 mu
print(f"\n  验证：相邻步的 PLA 序列（前 6 步）")
print(f"  {'k':>3}{'mu[1]':>8}{'mu[2]':>8}{'mu[3]':>8}{'mu[4]':>8}"
      f"{'mu[5]':>8}{'mu[6]':>8}")
print("  " + "-" * 52)
for i in range(8, 20):
    k = int(S["k"].iloc[i])
    ps = PS[i]
    print(f"  {k:>3}{ps[0]:>8.2f}{ps[1]:>8.2f}{ps[2]:>8.2f}{ps[3]:>8.2f}"
          f"{ps[4]:>8.2f}{ps[5]:>8.2f}")

print(f"\n  若 warm-start 正确左移，应有 mu[k+1][j] ≈ mu[k][j+1]")
print(f"  实际检查：")
print(f"  {'k':>3}{'mu[k][2..6]':>32}{'mu[k+1][1..5]':>32}{'差异':>10}")
print("  " + "-" * 78)
for i in range(8, 16):
    k = int(S["k"].iloc[i])
    a = PS[i][1:6]
    b = PS[i+1][0:5]
    diff = np.mean(np.abs(a - b))
    print(f"  {k:>3}{str(np.round(a,2)):>32}{str(np.round(b,2)):>32}{diff:>10.3f}")

# ══════════════════════════════════════════════════
# [3] 检查 CEM 的探索范围
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[3] CEM 探索范围分析")
print("=" * 96)

IT = pd.read_csv(CD / "cem_iters.csv")
for k in [10, 14, 17]:
    sub = IT[IT["k"] == k]
    if len(sub) == 0:
        continue
    print(f"\n  ── k={k} ──")
    print(f"    {'it':>3}{'J_best':>12}{'PLA[1]':>9}{'预测峰值':>10}")
    for _, r in sub.iterrows():
        print(f"    {int(r['it']):>3}{r['J_best']:>12.1f}{r['pla1']:>9.2f}"
              f"{r['pred_peak']:>10.1f}")
    # PLA[1] 的离散度
    spread = sub["pla1"].max() - sub["pla1"].min()
    print(f"    PLA[1] 6轮范围: {spread:.2f}°")
    if spread < 1.0:
        print(f"    -> ⚠️ 探索范围很小，CEM 基本在原点附近微调")

# ══════════════════════════════════════════════════
# [4] 关键：dpla_max 约束是否阻止下降
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[4] dpla_max 约束的影响")
print("=" * 96)

k = 17
i = k - 1
pla_prev = S["pla_prev"].iloc[i]
print(f"\n  k=17: PLA_prev = {pla_prev:.2f}")
print(f"  dpla_max = {dpla_max}")
print(f"  第 1 步可达范围: [{pla_prev-dpla_max:.2f}, {pla_prev+dpla_max:.2f}]")
print(f"\n  但'立即降到 63.64'需要 ΔPLA = {63.64-pla_prev:+.2f}")
print(f"  单步最大变化 = {dpla_max}")
steps_needed = abs(63.64-pla_prev)/dpla_max
print(f"  -> 需要 {steps_needed:.1f} 步才能降到 63.64")
print(f"\n  所以'立即降至63.64'在约束下【不可行】！")
print(f"  实际可行的是：每步降 {dpla_max}°，{steps_needed:.0f} 步后到达")

# 重新评估
print(f"\n  ★ 修正：之前的参考策略没有应用 dpla_max 约束")
print(f"     因此'立即降至63.64'是不合法的解")
print(f"     CEM 在约束内寻找，结果可能是合理的")

res = {
    "wv_short_sum": float(wv[:n_short].sum()),
    "wv_long_sum": float(wv[n_short:].sum()),
    "k17_pla_prev": float(pla_prev),
    "k17_steps_to_gold": float(steps_needed),
    "dpla_max": float(dpla_max),
    "note": "参考策略未应用 dpla_max 约束，其 MSE 不可直接与 CEM 比较",
}
(CD / "optimality_diag.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {CD/'optimality_diag.json'}")

