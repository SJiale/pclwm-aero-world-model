"""多种子配对分析 v2（按实际 JSON 结构）

JSON 结构：
  seed{N} -> cfgs -> {E0_baseline,...} -> {
    best_val, best_epoch, train_time_s, params,
    test_id: {mean, std, per_channel, per_traj, n_rep},
    test_ood: {...},
    continuous: {test_id_10s: {final_mean, final_std, mean_rel, mean_rel_std, n, curve}, ...},
    probe: {N2dot_LP: {r2_test, r2_val, rmse_phys, lam}, N3dot_HP: {...}},
    kl: {...}, hist: {...}
  }
"""
import json
import sys
from pathlib import Path
import numpy as np

RES = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
OUT = Path(r"C:\pclwm\pclwm_pcrssm\results")

D = json.loads(RES.read_text(encoding="utf-8"))
CFGS = ["E0_baseline", "E1_static", "E2_dynamic", "E3_both"]

print("=" * 100)
print("PC-RSSM 多种子配对分析")
print("=" * 100)
print(f"  已加载: {list(D.keys())}")

complete = [k for k, v in D.items()
            if all(c in v.get("cfgs", {}) for c in CFGS)]
print(f"  完整 seed: {complete}")

if not complete:
    print("  无完整 seed")
    sys.exit(0)


def g(c, *path, default=np.nan):
    """安全取值"""
    cur = c
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# ══════════════════════════════════════════════════
# 表 1：统一对比
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表1] 已完成配置统一对比")
print("=" * 100)

for sk in complete:
    print(f"\n  ── {sk} ──")
    hdr = (f"  {'配置':<13}{'best_val':>10}{'@ep':>4}{'耗时':>7}"
           f"{'test_id':>9}{'test_ood':>9}"
           f"{'30s(mean)':>11}{'60s(mean)':>11}"
           f"{'N2dot R²':>10}{'N3dot R²':>10}")
    print(hdr)
    print("  " + "-" * 96)
    for cfg in CFGS:
        c = D[sk]["cfgs"][cfg]
        ti = g(c, "test_id", "mean")
        od = g(c, "test_ood", "mean")
        m30 = g(c, "continuous", "test_id_30s", "mean_rel")
        m60 = g(c, "continuous", "test_id_60s", "mean_rel")
        p2 = g(c, "probe", "N2dot_LP", "r2_test")
        p3 = g(c, "probe", "N3dot_HP", "r2_test")
        print(f"  {cfg:<13}{c['best_val']:>10.5f}{c['best_epoch']:>4}"
              f"{c['train_time_s']/60:>6.0f}m"
              f"{ti:>9.4f}{od:>9.4f}{m30:>11.4f}{m60:>11.4f}"
              f"{p2:>10.4f}{p3:>10.4f}")

# ══════════════════════════════════════════════════
# 表 2：配对分析
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表2] 配对分析（相对同 seed 的 E0，正=改善）")
print("=" * 100)

METRICS = [
    ("test_id", "短期 test_id", "low"),
    ("test_ood", "短期 test_ood", "low"),
    ("c10_mean", "连续10s(mean)", "low"),
    ("c30_mean", "连续30s(mean)", "low"),
    ("c60_mean", "连续60s(mean)", "low"),
    ("c10_final", "连续10s(final)", "low"),
    ("c30_final", "连续30s(final)", "low"),
    ("c60_final", "连续60s(final)", "low"),
    ("p_n2dot", "Probe N2dot R²", "high"),
    ("p_n3dot", "Probe N3dot R²", "high"),
]


def metric(c, m):
    if m == "test_id":
        return g(c, "test_id", "mean")
    if m == "test_ood":
        return g(c, "test_ood", "mean")
    if m.startswith("c") and "_" in m:
        parts = m.split("_")
        tgt = parts[0][1:] + "s"      # c10 -> 10s
        kind = "mean_rel" if parts[1] == "mean" else "final_mean"
        return g(c, "continuous", "test_id_" + tgt, kind)
    if m == "p_n2dot":
        return g(c, "probe", "N2dot_LP", "r2_test")
    if m == "p_n3dot":
        return g(c, "probe", "N3dot_HP", "r2_test")
    return np.nan


for cfg in CFGS[1:]:
    print(f"\n  ── {cfg} vs E0_baseline ──")
    print(f"  {'指标':<18}" + "".join(f"{s:>12}" for s in complete)
          + f"{'平均':>10}{'一致':>8}")
    print("  " + "-" * (18 + 12 * len(complete) + 20))

    for mk, mname, better in METRICS:
        row = []
        for sk in complete:
            e0 = metric(D[sk]["cfgs"]["E0_baseline"], mk)
            ei = metric(D[sk]["cfgs"][cfg], mk)
            if np.isnan(e0) or np.isnan(ei) or e0 == 0:
                row.append(np.nan)
                continue
            rel = (ei - e0) / e0 * 100
            if better == "low":
                rel = -rel
            row.append(rel)
        row = np.array(row)
        valid = row[~np.isnan(row)]
        avg = np.mean(valid) if len(valid) else np.nan
        same = int(np.sum(np.sign(valid) == np.sign(avg))) if len(valid) else 0
        cells = "".join(f"{v:>11.2f}%" if not np.isnan(v) else f"{'n/a':>12}"
                        for v in row)
        print(f"  {mname:<18}{cells}{avg:>9.2f}%{same:>5}/{len(valid)}")

# ══════════════════════════════════════════════════
# 表 3：seed0 E3 详细分析
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表3] seed0：E3 为何短期只改善 1.3%")
print("=" * 100)

for sk in complete[:1]:
    e0 = D[sk]["cfgs"]["E0_baseline"]
    e3 = D[sk]["cfgs"]["E3_both"]

    print(f"\n  ── 短期 ──")
    for k, nm in [("test_id", "test_id"), ("test_ood", "test_ood")]:
        v0 = g(e0, k, "mean")
        v3 = g(e3, k, "mean")
        print(f"    {nm:<10} E0={v0:.4f}  E3={v3:.4f}  "
              f"改善={(v0-v3)/v0*100:+.2f}%")

    print(f"\n  ── 逐轨迹（test_id）──")
    for rid in e0["test_id"]["per_traj"]:
        v0 = g(e0, "test_id", "per_traj", rid, "mean")
        v3 = g(e3, "test_id", "per_traj", rid, "mean")
        print(f"    {rid:<16} E0={v0:.4f}  E3={v3:.4f}  "
              f"改善={(v0-v3)/v0*100:+.2f}%")

    print(f"\n  ── 连续推演 ──")
    print(f"    {'目标':<8}{'E0 mean':>10}{'E3 mean':>10}{'改善':>9}"
          f"{'E0 final':>11}{'E3 final':>11}{'改善':>9}")
    print("    " + "-" * 68)
    for tgt in ["10s", "30s", "60s"]:
        m0 = g(e0, "continuous", f"test_id_{tgt}", "mean_rel")
        m3 = g(e3, "continuous", f"test_id_{tgt}", "mean_rel")
        f0 = g(e0, "continuous", f"test_id_{tgt}", "final_mean")
        f3 = g(e3, "continuous", f"test_id_{tgt}", "final_mean")
        print(f"    {tgt:<8}{m0:>10.4f}{m3:>10.4f}{(m0-m3)/m0*100:>8.2f}%"
              f"{f0:>11.4f}{f3:>11.4f}{(f0-f3)/f0*100:>8.2f}%")

    print(f"\n  ── 物理 Probe ──")
    for pn in ["N2dot_LP", "N3dot_HP"]:
        r0 = g(e0, "probe", pn, "r2_test")
        r3 = g(e3, "probe", pn, "r2_test")
        e0_ = g(e0, "probe", pn, "rmse_phys")
        e3_ = g(e3, "probe", pn, "rmse_phys")
        print(f"    {pn:<10} E0: R²={r0:.4f} RMSE={e0_:.1f}  "
              f"E3: R²={r3:.4f} RMSE={e3_:.1f}  "
              f"ΔR²={r3-r0:+.4f}")

    print(f"\n  ── KL ──")
    for kn in ["kl_raw", "kl_clip", "kl_frac"]:
        v0 = g(e0, "kl", kn)
        v3 = g(e3, "kl", kn)
        print(f"    {kn:<10} E0={v0:.6f}  E3={v3:.6f}")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
summary = {"complete_seeds": complete,
           "n_complete": len(complete),
           "n_total_expected": 5}
(OUT / "interim_pairing.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'interim_pairing.json'}")
