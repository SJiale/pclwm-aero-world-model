"""PSVA vs P_ctrl 阶段 1 分析

回答：
  1. 60s 连续预测差异
  2. Test-OOD 差异
  3. Separation 差异
  4. 逐轨迹差异
  5. ★ 关键：区分"潜状态分工更明显"与"长期预测更准确"
"""
import json
import sys
from pathlib import Path
import numpy as np

RES = Path(r"C:\pclwm\pclwm_psva\results\psva_results.json")
OUT = Path(r"C:\pclwm\pclwm_psva\results")

D = json.loads(RES.read_text(encoding="utf-8"))
print("=" * 100)
print("PSVA vs P_ctrl 阶段 1 对比分析")
print("=" * 100)
print(f"  已有配置: {list(D.keys())}")


def g(c, *path, default=np.nan):
    cur = c
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


# 找 P_ctrl 和 PSVA
_ctrl = [k for k in D if k.startswith("P_ctrl")]
_psva = [k for k in D if k.startswith("PSVA")]
if not _ctrl or not _psva:
    print(f"  ⚠ 缺少配置: ctrl={_ctrl}, psva={_psva}")
    sys.exit(0)

C = D[_ctrl[0]]
P = D[_psva[0]]

print(f"\n  P_ctrl: {_ctrl[0]}  参数={C['params']:,}")
print(f"  PSVA  : {_psva[0]}  参数={P['params']:,}")

# ══════════════════════════════════════════════════
# 表 1：核心指标
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表1] ★ 核心指标对比")
print("=" * 100)

METRICS = [
    ("test_id", "短期 Test-ID", "low"),
    ("test_ood", "★ 短期 Test-OOD", "low"),
    ("c10", "连续 10s", "low"),
    ("c30", "连续 30s", "low"),
    ("c60", "★ 连续 60s", "low"),
]


def getv(c, m):
    if m in ("test_id", "test_ood"):
        return g(c, m, "mean")
    if m.startswith("c") and m[1:].isdigit():
        return g(c, "continuous", f"test_id_{m[1:]}s", "mean_rel")
    return np.nan


print(f"\n  {'指标':<20}{'P_ctrl':>12}{'PSVA':>12}{'差异':>12}{'相对':>10}{'胜者':>8}")
print("  " + "-" * 76)

for mk, mname, better in METRICS:
    vc = getv(C, mk)
    vp = getv(P, mk)
    if np.isnan(vc) or np.isnan(vp):
        continue
    diff = vp - vc
    rel = diff / vc * 100
    if better == "low":
        win = "PSVA" if vp < vc else "P_ctrl"
    else:
        win = "PSVA" if vp > vc else "P_ctrl"
    print(f"  {mname:<20}{vc:>12.4f}{vp:>12.4f}{diff:>+12.4f}"
          f"{rel:>+9.2f}%{win:>8}")

# ══════════════════════════════════════════════════
# 表 2：Separation（潜状态分工）
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表2] ★ Separation（潜状态分工程度）")
print("=" * 100)

print(f"\n  {'指标':<24}{'P_ctrl':>12}{'PSVA':>12}{'差异':>12}")
print("  " + "-" * 60)
for key, name in [("sep_fast", "Sep_fast（快变量分化）"),
                  ("sep_slow", "Sep_slow（慢变量分化）"),
                  ("sep_total", "★ Sep_total")]:
    vc = g(C, "probe", "Separation", key)
    vp = g(P, "probe", "Separation", key)
    print(f"  {name:<24}{vc:>+12.4f}{vp:>+12.4f}{vp-vc:>+12.4f}")

print(f"\n  参照基准:")
print(f"    E3 任意 64/64 切分: Sep_total = +0.0049（无分工）")
print(f"    成功门槛:           Sep_total > +0.10")

# 交叉 Probe 明细
print(f"\n  交叉 Probe 明细:")
print(f"  {'Probe':<14}{'P_ctrl h_f':>12}{'P_ctrl h_s':>12}"
      f"{'PSVA h_f':>12}{'PSVA h_s':>12}")
print("  " + "-" * 62)
for k in ["Ndot_N2", "Ndot_N3", "N_N2", "N_N3"]:
    row = f"  {k:<14}"
    for cfg in [C, P]:
        row += f"{g(cfg, 'probe', k, 'h_f'):>12.4f}"
        row += f"{g(cfg, 'probe', k, 'h_s'):>12.4f}"
    print(row)

# ══════════════════════════════════════════════════
# 表 3：逐轨迹
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表3] 逐轨迹差异")
print("=" * 100)

for split in ["test_id", "test_ood"]:
    print(f"\n  ── {split} ──")
    trajs = list(g(C, split, "per_traj", default={}).keys())
    if not trajs:
        continue
    print(f"  {'轨迹':<18}{'P_ctrl':>10}{'PSVA':>10}{'差异':>10}{'相对':>10}{'胜者':>8}")
    print("  " + "-" * 66)
    for rid in trajs:
        vc = g(C, split, "per_traj", rid, "mean")
        vp = g(P, split, "per_traj", rid, "mean")
        if np.isnan(vc) or np.isnan(vp):
            continue
        rel = (vp - vc) / vc * 100
        win = "PSVA" if vp < vc else "P_ctrl"
        print(f"  {rid:<18}{vc:>10.4f}{vp:>10.4f}{vp-vc:>+10.4f}"
              f"{rel:>+9.2f}%{win:>8}")

# ══════════════════════════════════════════════════
# 表 4：★ 关键判断
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表4] ★★ 关键判断：分工 vs 预测")
print("=" * 100)

sep_c = g(C, "probe", "Separation", "sep_total")
sep_p = g(P, "probe", "Separation", "sep_total")
c60_c = getv(C, "c60")
c60_p = getv(P, "c60")
ood_c = getv(C, "test_ood")
ood_p = getv(P, "test_ood")

print(f"""
  两个维度必须分开看：

  (1) 潜状态分工（Separation）
      P_ctrl = {sep_c:+.4f}
      PSVA   = {sep_p:+.4f}
      变化   = {sep_p-sep_c:+.4f}
      -> {'PSVA 分工更明显' if sep_p > sep_c else 'PSVA 未改善分工'}

  (2) 长期动力学预测（连续 60s）
      P_ctrl = {c60_c:.4f}
      PSVA   = {c60_p:.4f}
      变化   = {(c60_p-c60_c)/c60_c*100:+.2f}%
      -> {'PSVA 更准确' if c60_p < c60_c else 'PSVA 未改善预测'}

  (3) 跨工况泛化（Test-OOD）
      P_ctrl = {ood_c:.4f}
      PSVA   = {ood_p:.4f}
      变化   = {(ood_p-ood_c)/ood_c*100:+.2f}%
""")

# 四种组合
sep_up = sep_p > sep_c + 0.02
pred_up = c60_p < c60_c * 0.98

print("  ── 四种可能组合 ──")
if sep_up and pred_up:
    print("  ✅ 分工↑ + 预测↑  -> 两者兼有，切分有实质价值")
    verdict = "both"
elif sep_up and not pred_up:
    print("  ⚠️ 分工↑ + 预测无改善 -> **分工更明显 ≠ 预测更准确**")
    print("      这正是需要警惕的情况：Separation 提升不能作为有效性的证据")
    verdict = "sep_only"
elif not sep_up and pred_up:
    print("  ⚠️ 分工无改善 + 预测↑ -> 收益可能来自其他因素（如监督头容量）")
    verdict = "pred_only"
else:
    print("  ❌ 分工无改善 + 预测无改善 -> 切分无效")
    verdict = "neither"

print(f"\n  本次判定: {verdict}")

# ══════════════════════════════════════════════════
# 表 5：训练稳定性
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表5] 训练稳定性")
print("=" * 100)

print(f"\n  {'指标':<20}{'P_ctrl':>14}{'PSVA':>14}")
print("  " + "-" * 50)
for key, name in [("best_val", "best_val"),
                  ("best_epoch", "best_epoch"),
                  ("train_time_s", "训练时间(s)"),
                  ("params", "参数量")]:
    vc = C.get(key)
    vp = P.get(key)
    fmt = ",.0f" if key == "params" else ".5f" if key == "best_val" else ".0f"
    print(f"  {name:<20}{vc:>14{fmt}}{vp:>14{fmt}}")

# 损失曲线终点
print(f"\n  损失曲线（末 epoch）:")
for key, name in [("l_fast", "L_fast"), ("l_slow", "L_slow"),
                  ("l_dyn", "L_dyn")]:
    hc = g(C, "hist", key, default=[])
    hp = g(P, "hist", key, default=[])
    vc = hc[-1] if hc else np.nan
    vp = hp[-1] if hp else np.nan
    print(f"  {name:<20}{vc:>14.6f}{vp:>14.6f}")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
summary = {
    "verdict": verdict,
    "separation": {"P_ctrl": float(sep_c), "PSVA": float(sep_p),
                   "delta": float(sep_p - sep_c)},
    "c60": {"P_ctrl": float(c60_c), "PSVA": float(c60_p),
            "rel_change": float((c60_p - c60_c) / c60_c * 100)},
    "test_ood": {"P_ctrl": float(ood_c), "PSVA": float(ood_p),
                 "rel_change": float((ood_p - ood_c) / ood_c * 100)},
}
(OUT / "psva_stage1_analysis.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'psva_stage1_analysis.json'}")
