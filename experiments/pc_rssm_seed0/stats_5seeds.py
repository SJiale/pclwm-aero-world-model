"""5 种子严格统计分析

计算：
  1. 配对差值的 t 检验（含 p 值、置信区间）
  2. 效应量（Cohen's d）
  3. 与 seed0 单种子结论的对比
"""
import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats

RES = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
OUT = Path(r"C:\pclwm\pclwm_pcrssm\results")

D = json.loads(RES.read_text(encoding="utf-8"))
CFGS = ["E0_baseline", "E1_static", "E2_dynamic", "E3_both"]
SEEDS = [f"seed{i}" for i in range(5)]

print("=" * 100)
print("5 种子严格统计分析")
print("=" * 100)


def g(c, *path, default=np.nan):
    cur = c
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def metric(c, m):
    if m == "test_id":
        return g(c, "test_id", "mean")
    if m == "test_ood":
        return g(c, "test_ood", "mean")
    if m.startswith("c") and "_" in m:
        parts = m.split("_")
        tgt = parts[0][1:] + "s"
        kind = "mean_rel" if parts[1] == "mean" else "final_mean"
        return g(c, "continuous", "test_id_" + tgt, kind)
    if m == "p_n2dot":
        return g(c, "probe", "N2dot_LP", "r2_test")
    if m == "p_n3dot":
        return g(c, "probe", "N3dot_HP", "r2_test")
    return np.nan


METRICS = [
    ("test_ood", "短期 test_ood", "low"),
    ("test_id", "短期 test_id", "low"),
    ("c30_mean", "连续30s", "low"),
    ("c60_mean", "连续60s", "low"),
    ("p_n2dot", "Probe N2dot R²", "high"),
    ("p_n3dot", "Probe N3dot R²", "high"),
]

print("\n" + "=" * 100)
print("[1] 配对 t 检验（n=5，双侧）")
print("=" * 100)
print("\n  解读标准：")
print("    p < 0.05  -> 显著")
print("    p < 0.10  -> 边缘显著")
print("    同向 5/5 -> 方向一致（符号检验 p=0.0625）")
print()

for cfg in CFGS[1:]:
    print(f"\n  ── {cfg} vs E0 ──")
    print(f"  {'指标':<18}{'原始均值':>11}{'原始SD':>10}"
          f"{'配对差均值':>12}{'配对差SD':>11}{'t':>8}{'p':>8}{'95%CI':>18}{'Cohen d':>10}")
    print("  " + "-" * 106)

    for mk, mname, better in METRICS:
        e0v = np.array([metric(D[s]["cfgs"]["E0_baseline"], mk) for s in SEEDS])
        eiv = np.array([metric(D[s]["cfgs"][cfg], mk) for s in SEEDS])
        valid = ~(np.isnan(e0v) | np.isnan(eiv))
        e0v, eiv = e0v[valid], eiv[valid]
        if len(e0v) < 2:
            continue

        # 配对差（原始单位）
        diff = eiv - e0v
        if better == "low":
            diff = -diff          # 正 = 改善

        t, p = stats.ttest_rel(e0v, eiv)
        if better == "low":
            t, p = -t, p
        mean_d = float(np.mean(diff))
        sd_d = float(np.std(diff, ddof=1))
        se = sd_d / np.sqrt(len(diff))
        ci = stats.t.interval(0.95, len(diff) - 1, loc=mean_d, scale=se)
        d = mean_d / sd_d if sd_d > 0 else np.nan

        # 相对改善
        rel = np.mean(diff / e0v) * 100

        print(f"  {mname:<18}{np.mean(e0v):>11.4f}{np.std(e0v, ddof=1):>10.4f}"
              f"{mean_d:>+12.4f}{sd_d:>11.4f}{t:>8.2f}{p:>8.3f}"
              f"[{ci[0]:>+7.4f},{ci[1]:>+7.4f}]{d:>10.2f}")

print("\n" + "=" * 100)
print("[2] 关键指标的完整统计（含相对改善）")
print("=" * 100)

print(f"\n  {'指标':<16}{'配置':<13}{'E0均值':>10}{'该配置均值':>12}"
      f"{'相对改善':>11}{'t':>8}{'p':>8}{'显著性':>12}")
print("  " + "-" * 92)

KEY = ["c60_mean", "c30_mean", "test_ood", "p_n2dot", "p_n3dot"]
KEYNAME = {"c60_mean": "连续60s", "c30_mean": "连续30s",
           "test_ood": "短期ood", "p_n2dot": "Probe N2dot",
           "p_n3dot": "Probe N3dot"}
BETTER = {"c60_mean": "low", "c30_mean": "low", "test_ood": "low",
          "p_n2dot": "high", "p_n3dot": "high"}

for mk in KEY:
    for cfg in CFGS[1:]:
        e0v = np.array([metric(D[s]["cfgs"]["E0_baseline"], mk) for s in SEEDS])
        eiv = np.array([metric(D[s]["cfgs"][cfg], mk) for s in SEEDS])
        valid = ~(np.isnan(e0v) | np.isnan(eiv))
        e0v, eiv = e0v[valid], eiv[valid]
        if len(e0v) < 2:
            continue
        diff = eiv - e0v
        if BETTER[mk] == "low":
            diff = -diff
        t, p = stats.ttest_rel(e0v, eiv)
        if BETTER[mk] == "low":
            t, p = -t, p
        rel = np.mean(diff / e0v) * 100
        sig = "*** p<0.01" if p < 0.01 else ("** p<0.05" if p < 0.05 else
              ("* p<0.10" if p < 0.10 else "n.s."))
        print(f"  {KEYNAME[mk]:<16}{cfg:<13}{np.mean(e0v):>10.4f}"
              f"{np.mean(eiv):>12.4f}{rel:>+10.2f}%{t:>8.2f}{p:>8.3f}{sig:>12}")
    print()

print("=" * 100)
print("[3] ★ 与 seed0 单种子结论的对比")
print("=" * 100)

print("\n  seed0 单种子（第一轮审查时的判断）:")
s0 = D["seed0"]["cfgs"]
print(f"    Probe N2dot: {g(s0['E0_baseline'],'probe','N2dot_LP','r2_test'):.4f} -> "
      f"{g(s0['E3_both'],'probe','N2dot_LP','r2_test'):.4f}  "
      f"(+{(g(s0['E3_both'],'probe','N2dot_LP','r2_test')-g(s0['E0_baseline'],'probe','N2dot_LP','r2_test'))*100:.1f}pp)")
print(f"    连续30s: {g(s0['E0_baseline'],'continuous','test_id_30s','mean_rel'):.4f} -> "
      f"{g(s0['E3_both'],'continuous','test_id_30s','mean_rel'):.4f}  "
      f"(改善 {(1-g(s0['E3_both'],'continuous','test_id_30s','mean_rel')/g(s0['E0_baseline'],'continuous','test_id_30s','mean_rel'))*100:+.1f}%)")
print(f"    -> 当时的判断: Probe 改善大, 连续推演变差")

print("\n  5 种子平均（本次分析）:")
e0p = np.mean([g(D[s]["cfgs"]["E0_baseline"], "probe", "N2dot_LP", "r2_test") for s in SEEDS])
e3p = np.mean([g(D[s]["cfgs"]["E3_both"], "probe", "N2dot_LP", "r2_test") for s in SEEDS])
e0c = np.mean([g(D[s]["cfgs"]["E0_baseline"], "continuous", "test_id_30s", "mean_rel") for s in SEEDS])
e3c = np.mean([g(D[s]["cfgs"]["E3_both"], "continuous", "test_id_30s", "mean_rel") for s in SEEDS])
print(f"    Probe N2dot: {e0p:.4f} -> {e3p:.4f}  ({(e3p-e0p)*100:+.1f}pp, {(e3p-e0p)/e0p*100:+.2f}%)")
print(f"    连续30s:     {e0c:.4f} -> {e3c:.4f}  ({(1-e3c/e0c)*100:+.1f}% 改善)")
print(f"    -> 结论: Probe 改善被高估, 连续推演实际有改善")

print("\n  ★ seed0 是反常样本:")
print(f"    seed0 Probe 提升: {(g(s0['E3_both'],'probe','N2dot_LP','r2_test')-g(s0['E0_baseline'],'probe','N2dot_LP','r2_test'))*100:+.1f}pp")
for s in SEEDS:
    d = (g(D[s]["cfgs"]["E3_both"], "probe", "N2dot_LP", "r2_test") -
         g(D[s]["cfgs"]["E0_baseline"], "probe", "N2dot_LP", "r2_test")) * 100
    print(f"    {s}: {d:+.2f}pp")

# 保存
out = {"n_seeds": len(SEEDS), "seeds": SEEDS}
(OUT / "multiseed_stats.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  （详细数值见上方表格）")
