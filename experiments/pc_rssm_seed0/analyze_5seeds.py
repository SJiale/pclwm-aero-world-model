"""PC-RSSM 5 种子完整配对分析

回答核心问题：
  物理锚定改善的是 (a) 潜空间可解码性、(b) 长期动力学预测能力、还是 (c) 两者兼有？

严格配对：
  同一 seed 内 E0/E1/E2/E3 共享主干初始权重
  配对差异 = E_i - E0
"""
import json
import sys
from pathlib import Path
import numpy as np

RES = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
OUT = Path(r"C:\pclwm\pclwm_pcrssm\results")

D = json.loads(RES.read_text(encoding="utf-8"))
CFGS = ["E0_baseline", "E1_static", "E2_dynamic", "E3_both"]
SEEDS = [f"seed{i}" for i in range(5)]

print("=" * 104)
print("PC-RSSM 5 种子完整配对分析")
print("=" * 104)

missing = [s for s in SEEDS if s not in D]
if missing:
    print(f"  ⚠ 缺失种子: {missing}")
    SEEDS = [s for s in SEEDS if s in D]
print(f"  完整种子: {SEEDS} ({len(SEEDS)}/5)")


def g(c, *path, default=np.nan):
    cur = c
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def metric(c, m):
    """提取指标"""
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
    if m == "rmse_n2dot":
        return g(c, "probe", "N2dot_LP", "rmse_phys")
    if m == "rmse_n3dot":
        return g(c, "probe", "N3dot_HP", "rmse_phys")
    return np.nan


# ══════════════════════════════════════════════════
# 表 1：全部种子 × 全部配置
# ══════════════════════════════════════════════════
print("\n" + "=" * 104)
print("[表1] 全部结果（5 种子 × 4 配置）")
print("=" * 104)

print(f"\n  {'seed':<8}{'配置':<13}{'test_id':>9}{'test_ood':>9}"
      f"{'30s':>9}{'60s':>9}{'N2dot R²':>10}{'N3dot R²':>10}{'best_val':>10}")
print("  " + "-" * 96)
for s in SEEDS:
    for cfg in CFGS:
        c = D[s]["cfgs"][cfg]
        print(f"  {s:<8}{cfg:<13}"
              f"{metric(c, 'test_id'):>9.4f}"
              f"{metric(c, 'test_ood'):>9.4f}"
              f"{metric(c, 'c30_mean'):>9.4f}"
              f"{metric(c, 'c60_mean'):>9.4f}"
              f"{metric(c, 'p_n2dot'):>10.4f}"
              f"{metric(c, 'p_n3dot'):>10.4f}"
              f"{c['best_val']:>10.5f}")
    print("  " + "-" * 96)

# ══════════════════════════════════════════════════
# 表 2：各配置的均值 ± 标准差
# ══════════════════════════════════════════════════
print("\n" + "=" * 104)
print("[表2] 各配置 5 种子均值 ± 标准差")
print("=" * 104)

METRICS = [
    ("test_id", "短期 test_id", "low"),
    ("test_ood", "短期 test_ood", "low"),
    ("c10_mean", "连续10s(mean)", "low"),
    ("c30_mean", "连续30s(mean)", "low"),
    ("c60_mean", "连续60s(mean)", "low"),
    ("p_n2dot", "Probe N2dot R²", "high"),
    ("p_n3dot", "Probe N3dot R²", "high"),
    ("rmse_n2dot", "Probe N2dot RMSE", "low"),
    ("rmse_n3dot", "Probe N3dot RMSE", "low"),
]

print(f"\n  {'指标':<20}" + "".join(f"{c.replace('_baseline','').replace('_static','').replace('_dynamic','').replace('_both',''):>18}"
                                  for c in CFGS))
print("  " + "-" * (20 + 18 * len(CFGS)))

for mk, mname, better in METRICS:
    row = f"  {mname:<20}"
    for cfg in CFGS:
        vals = [metric(D[s]["cfgs"][cfg], mk) for s in SEEDS]
        vals = [v for v in vals if not np.isnan(v)]
        if vals:
            row += f"{np.mean(vals):>10.4f}±{np.std(vals):<7.4f}"
        else:
            row += f"{'n/a':>18}"
    print(row)

# ══════════════════════════════════════════════════
# 表 3：配对改善（关键）
# ══════════════════════════════════════════════════
print("\n" + "=" * 104)
print("[表3] ★ 配对改善（相对同 seed 的 E0，正=改善）")
print("=" * 104)

for cfg in CFGS[1:]:
    print(f"\n  ── {cfg} vs E0_baseline ──")
    print(f"  {'指标':<20}" + "".join(f"{s:>9}" for s in SEEDS)
          + f"{'均值':>10}{'标准差':>9}{'同向数':>8}{'t值':>8}")
    print("  " + "-" * (20 + 9 * len(SEEDS) + 35))

    for mk, mname, better in METRICS:
        rels = []
        for s in SEEDS:
            e0 = metric(D[s]["cfgs"]["E0_baseline"], mk)
            ei = metric(D[s]["cfgs"][cfg], mk)
            if np.isnan(e0) or np.isnan(ei) or e0 == 0:
                rels.append(np.nan)
                continue
            r = (ei - e0) / e0 * 100
            if better == "low":
                r = -r
            rels.append(r)
        rels = np.array(rels)
        valid = rels[~np.isnan(rels)]
        if len(valid) == 0:
            print(f"  {mname:<20}{'n/a':>9}")
            continue
        mean_r = float(np.mean(valid))
        std_r = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0
        same = int(np.sum(np.sign(valid) == np.sign(mean_r)))
        # 配对 t 检验
        if len(valid) > 1 and std_r > 0:
            t = mean_r / (std_r / np.sqrt(len(valid)))
        else:
            t = np.nan
        cells = "".join(f"{v:>8.2f}%" if not np.isnan(v) else f"{'n/a':>9}"
                        for v in rels)
        print(f"  {mname:<20}{cells}{mean_r:>9.2f}%{std_r:>8.2f}%"
              f"{same:>5}/{len(valid)}{t:>8.2f}")

# ══════════════════════════════════════════════════
# 表 4：核心问题回答
# ══════════════════════════════════════════════════
print("\n" + "=" * 104)
print("[表4] ★★ 核心问题：物理锚定改善了什�么？")
print("=" * 104)


def paired_summary(cfg, mk, better):
    rels = []
    for s in SEEDS:
        e0 = metric(D[s]["cfgs"]["E0_baseline"], mk)
        ei = metric(D[s]["cfgs"][cfg], mk)
        if np.isnan(e0) or np.isnan(ei) or e0 == 0:
            continue
        r = (ei - e0) / e0 * 100
        if better == "low":
            r = -r
        rels.append(r)
    if not rels:
        return np.nan, np.nan, 0, 0
    rels = np.array(rels)
    m = float(np.mean(rels))
    sd = float(np.std(rels, ddof=1)) if len(rels) > 1 else 0.0
    same = int(np.sum(np.sign(rels) == np.sign(m)))
    return m, sd, same, len(rels)


print("\n  ── (a) 潜空间可解码性 ──")
for cfg in CFGS[1:]:
    m, sd, same, n = paired_summary(cfg, "p_n2dot", "high")
    m3, sd3, same3, n3 = paired_summary(cfg, "p_n3dot", "high")
    print(f"    {cfg:<13} N2dot R² {m:+6.2f}% (同向 {same}/{n})   "
          f"N3dot R² {m3:+6.2f}% (同向 {same3}/{n3})")

print("\n  ── (b) 长期动力学预测 ──")
for cfg in CFGS[1:]:
    m30, sd30, s30, n30 = paired_summary(cfg, "c30_mean", "low")
    m60, sd60, s60, n60 = paired_summary(cfg, "c60_mean", "low")
    print(f"    {cfg:<13} 30s {m30:+6.2f}% (同向 {s30}/{n30})   "
          f"60s {m60:+6.2f}% (同向 {s60}/{n60})")

print("\n  ── (c) 短期预测 ──")
for cfg in CFGS[1:]:
    m, sd, same, n = paired_summary(cfg, "test_id", "low")
    mo, sdo, sameo, no = paired_summary(cfg, "test_ood", "low")
    print(f"    {cfg:<13} test_id {m:+6.2f}% (同向 {same}/{n})   "
          f"ood {mo:+6.2f}% (同向 {sameo}/{no})")

# ══════════════════════════════════════════════════
# 表 5：E3 逐轨迹
# ══════════════════════════════════════════════════
print("\n" + "=" * 104)
print("[表5] E3 vs E0 逐轨迹（test_id）")
print("=" * 104)

print(f"\n  {'seed':<8}" + "".join(f"{rid:>20}" for rid in
                                  ["A11_msin_e3", "C6_mn_step"]))
print("  " + "-" * 48)
for s in SEEDS:
    row = f"  {s:<8}"
    for rid in ["A11_msin_e3", "C6_mn_step"]:
        e0 = g(D[s]["cfgs"]["E0_baseline"], "test_id", "per_traj", rid, "mean")
        e3 = g(D[s]["cfgs"]["E3_both"], "test_id", "per_traj", rid, "mean")
        if not np.isnan(e0) and not np.isnan(e3):
            row += f"{e0:.4f}->{e3:.4f}({(e0-e3)/e0*100:+5.1f}%)"
        else:
            row += f"{'n/a':>20}"
    print(row)

print(f"\n  解读：若 A11 变差而 C6 变好（或反之），说明 E3 的改善不稳定。")

# 保存
summary = {}
for cfg in CFGS:
    cfg_s = {}
    for mk, _, _ in METRICS:
        vals = [metric(D[s]["cfgs"][cfg], mk) for s in SEEDS]
        vals = [v for v in vals if not np.isnan(v)]
        cfg_s[mk] = {"mean": float(np.mean(vals)),
                     "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                     "n": len(vals)}
    summary[cfg] = cfg_s

(OUT / "multiseed_summary.json").write_text(
    json.dumps({"seeds": SEEDS, "summary": summary}, ensure_ascii=False,
               indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'multiseed_summary.json'}")
