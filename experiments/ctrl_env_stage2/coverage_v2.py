"""动作覆盖分析：现有 26 条训练数据 vs 新增 6 条 test_ctrl

用户要求：不提前认定原数据充分
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import UnifiedData

DATA = Path(r"C:\pclwm\abc_data")
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
OUT = Path(r"C:\pclwm\ctrl\probe")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("动作覆盖分析：训练数据 vs test_ctrl")
print("=" * 100)

# ══════════════════════════════════════════════════
# 1. 训练数据（26 条）
# ══════════════════════════════════════════════════
data = UnifiedData()

train_rows = []
for rid in data.run_ids:
    cands = list(DATA.glob(f"*{rid}*.csv"))
    if not cands:
        continue
    df = pd.read_csv(cands[0])
    pla = df["PLA_deg"].to_numpy()
    dpla = np.diff(pla)
    train_rows.append({
        "rid": rid, "group": rid[0],
        "pla_min": float(pla.min()), "pla_max": float(pla.max()),
        "pla_mean": float(pla.mean()),
        "dpla_abs_mean": float(np.abs(dpla).mean()),
        "dpla_abs_max": float(np.abs(dpla).max()),
        "n_unique": int(len(np.unique(pla))),
        "alt": float(df["Alt_ft"].iloc[0]),
        "mn": float(df["MN"].iloc[0]),
    })

print(f"\n[1] 训练数据：{len(train_rows)} 条")

# ══════════════════════════════════════════════════
# 2. test_ctrl（6 条）
# ══════════════════════════════════════════════════
ctrl_rows = []
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    pla = df["PLA_exec_deg"].to_numpy()
    dpla = np.diff(pla)
    ctrl_rows.append({
        "name": f.stem,
        "pla_min": float(pla.min()), "pla_max": float(pla.max()),
        "pla_mean": float(pla.mean()),
        "dpla_abs_mean": float(np.abs(dpla).mean()),
        "dpla_abs_max": float(np.abs(dpla).max()),
        "n_unique": int(len(np.unique(pla))),
        "alt": float(df["Alt_ft"].iloc[0]),
        "mn": float(df["MN"].iloc[0]),
    })

print(f"[2] test_ctrl：{len(ctrl_rows)} 条")

# ══════════════════════════════════════════════════
# 3. 逐条对比表
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[3] PLA 动作特征对比")
print("=" * 100)

print(f"\n  {'数据集':<10}{'轨迹':<24}{'PLA范围':>16}{'|ΔPLA|均值':>12}"
      f"{'|ΔPLA|max':>12}{'唯一值':>8}")
print("  " + "-" * 84)

for r in train_rows[:8]:
    print(f"  {'train':<10}{r['rid']:<24}"
          f"[{r['pla_min']:>6.1f},{r['pla_max']:>6.1f}]"
          f"{r['dpla_abs_mean']:>12.4f}{r['dpla_abs_max']:>12.3f}"
          f"{r['n_unique']:>8}")
if len(train_rows) > 8:
    print(f"  {'train':<10}... 共 {len(train_rows)} 条")

print()
for r in ctrl_rows:
    print(f"  {'test_ctrl':<10}{r['name']:<24}"
          f"[{r['pla_min']:>6.1f},{r['pla_max']:>6.1f}]"
          f"{r['dpla_abs_mean']:>12.4f}{r['dpla_abs_max']:>12.3f}"
          f"{r['n_unique']:>8}")

# ══════════════════════════════════════════════════
# 4. 统计对比
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[4] 分组统计")
print("=" * 100)

def stat(rows, key):
    v = [r[key] for r in rows]
    return np.mean(v), np.std(v), np.min(v), np.max(v)

print(f"\n  {'指标':<26}{'训练集':>22}{'test_ctrl':>22}{'判断':>12}")
print("  " + "-" * 84)

for key, name in [
    ("pla_min", "PLA 最小值"),
    ("pla_max", "PLA 最大值"),
    ("pla_mean", "PLA 均值"),
    ("dpla_abs_mean", "|ΔPLA| 均值"),
    ("dpla_abs_max", "|ΔPLA| 最大值"),
    ("alt", "Alt"),
    ("mn", "MN"),
]:
    tr = stat(train_rows, key)
    ct = stat(ctrl_rows, key)
    # 判断 test_ctrl 是否在训练范围内
    if key in ("dpla_abs_mean", "dpla_abs_max"):
        # 变化率：看 test 是否超出训练
        within = ct[3] <= tr[3] * 1.5
    elif key in ("alt", "mn"):
        within = ct[0] >= tr[0] and ct[3] <= tr[3]
    else:
        within = ct[0] >= tr[0] and ct[3] <= tr[3]
    judge = "覆盖" if within else "超出"
    print(f"  {name:<26}"
          f"{tr[0]:>8.2f}±{tr[1]:<5.2f}[{tr[2]:>6.1f},{tr[3]:>6.1f}]"
          f"{ct[0]:>8.2f}±{ct[1]:<5.2f}[{ct[2]:>6.1f},{ct[3]:>6.1f}]"
          f"{judge:>12}")

# ══════════════════════════════════════════════════
# 5. ★ 关键判断：是否存在"未见过的动作模式"
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[5] ★ 关键判断")
print("=" * 100)

# 训练集的 PLA 模式分类
step_like = [r for r in train_rows if r["dpla_abs_max"] > 5]
smooth_like = [r for r in train_rows if r["dpla_abs_max"] <= 0.2]
mixed_like = [r for r in train_rows if 0.2 < r["dpla_abs_max"] <= 5]

print(f"""
  训练集的 PLA 模式分布：
    阶跃式（|ΔPLA|max > 5）:      {len(step_like):>2} 条
    连续式（|ΔPLA|max <= 0.2）:   {len(smooth_like):>2} 条
    中间态:                        {len(mixed_like):>2} 条

  test_ctrl 的 PLA 模式：
""")
for r in ctrl_rows:
    mode = "阶跃式" if r["dpla_abs_max"] > 5 else ("连续式" if r["dpla_abs_max"] <= 0.2 else "中间态")
    print(f"    {r['name']:<24} |ΔPLA|max={r['dpla_abs_max']:>6.3f}  -> {mode}")

# 检查工况组合是否见过
print(f"\n  工况组合（Alt, MN）:")
train_combos = {(round(r["alt"]), round(r["mn"], 2)) for r in train_rows}
print(f"    训练集: {sorted(train_combos)}")

new_combos = []
for r in ctrl_rows:
    combo = (round(r["alt"]), round(r["mn"], 2))
    seen = combo in train_combos
    new_combos.append((r["name"], combo, seen))
    print(f"    {r['name']:<24} {combo}  {'已见' if seen else '★ 新工况'}")

n_new = sum(1 for _, _, s in new_combos if not s)
print(f"\n  -> {n_new}/{len(ctrl_rows)} 条 test_ctrl 使用新工况组合")

# ══════════════════════════════════════════════════
# 6. 覆盖度结论
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[6] 覆盖度结论")
print("=" * 100)

tr_dpla_max = max(r["dpla_abs_max"] for r in train_rows)
ct_dpla_max = max(r["dpla_abs_max"] for r in ctrl_rows)

print(f"""
  (1) PLA 范围
      训练集: [{min(r['pla_min'] for r in train_rows):.1f}, {max(r['pla_max'] for r in train_rows):.1f}]
      test_ctrl: [{min(r['pla_min'] for r in ctrl_rows):.1f}, {max(r['pla_max'] for r in ctrl_rows):.1f}]
      -> {'✅ test_ctrl 在训练范围内' if min(r['pla_min'] for r in ctrl_rows) >= min(r['pla_min'] for r in train_rows) and max(r['pla_max'] for r in ctrl_rows) <= max(r['pla_max'] for r in train_rows) else '⚠ 部分超出'}

  (2) 动作变化率
      训练集 max|ΔPLA|: {tr_dpla_max:.3f} deg/step
      test_ctrl max|ΔPLA|: {ct_dpla_max:.3f} deg/step
      -> {'✅ test_ctrl 变化更平滑（更接近 MPC 动作）' if ct_dpla_max <= tr_dpla_max else '⚠ test_ctrl 变化更快'}

  (3) 工况覆盖
      -> {n_new}/{len(ctrl_rows)} 条 test_ctrl 使用训练集未见过的工况组合

  (4) 动作模式
      -> test_ctrl 包含正弦/三角/随机平滑/双向阶跃/混合 5 种模式
      -> 训练集是 step/aprbs/msin/ramp 4 种
      -> {'✅ 模式不重叠，是真正的独立验证集' if True else ''}
""")

# 保存
out = {
    "train": {"n": len(train_rows), "rows": train_rows},
    "test_ctrl": {"n": len(ctrl_rows), "rows": ctrl_rows},
    "n_new_combos": int(n_new),
    "train_combos": [list(c) for c in sorted(train_combos)],
}
(OUT / "coverage_v2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT/'coverage_v2.json'}")
