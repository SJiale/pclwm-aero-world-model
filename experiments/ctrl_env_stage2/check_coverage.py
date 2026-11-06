"""任务 4：检查 26 条训练轨迹的动作覆盖度

关键问题：现有数据是否覆盖 MPC 需要的动作空间？
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import UnifiedData

DATA = Path(r"C:\pclwm\abc_data")
OUT = Path(r"C:\pclwm\ctrl\probe")
OUT.mkdir(parents=True, exist_ok=True)

data = UnifiedData()

print("=" * 96)
print("训练数据动作覆盖度分析")
print("=" * 96)

# ══════════════════════════════════════════════════
# 1. 动作范围与变化率
# ══════════════════════════════════════════════════
print("\n[1] 各轨迹的动作范围")
print(f"\n  {'轨迹':<22}{'PLA范围':>16}{'PLA变化':>10}"
      f"{'Alt范围':>16}{'MN范围':>12}{'组':>4}")
print("  " + "-" * 84)

rows = []
for rid in data.run_ids:
    df = pd.read_csv(DATA / f"{rid}.csv") if (DATA / f"{rid}.csv").exists() else None
    if df is None:
        # 尝试匹配文件名
        cands = list(DATA.glob(f"*{rid}*.csv"))
        if not cands:
            continue
        df = pd.read_csv(cands[0])

    pla = df["PLA_deg"].to_numpy()
    alt = df["Alt_ft"].to_numpy()
    mn = df["MN"].to_numpy()

    grp = rid[0] if rid[0] in "ABC" else "?"
    rows.append({
        "rid": rid, "group": grp,
        "pla_min": float(pla.min()), "pla_max": float(pla.max()),
        "pla_delta": float(pla.max() - pla.min()),
        "alt_min": float(alt.min()), "alt_max": float(alt.max()),
        "mn_min": float(mn.min()), "mn_max": float(mn.max()),
    })
    print(f"  {rid:<22}[{pla.min():>5.1f},{pla.max():>5.1f}]"
          f"{pla.max()-pla.min():>10.1f}"
          f"[{alt.min():>6.0f},{alt.max():>6.0f}]"
          f"[{mn.min():>4.2f},{mn.max():>4.2f}]{grp:>4}")

# ══════════════════════════════════════════════════
# 2. 分组统计
# ══════════════════════════════════════════════════
print("\n[2] 按组统计")
for grp in ["A", "B", "C"]:
    sub = [r for r in rows if r["group"] == grp]
    if not sub:
        continue
    pla_all = [r["pla_delta"] for r in sub]
    print(f"\n  组 {grp}（{len(sub)} 条）:")
    print(f"    PLA 变化幅度: mean={np.mean(pla_all):.1f} deg, "
          f"range=[{min(pla_all):.1f}, {max(pla_all):.1f}]")
    print(f"    PLA 绝对值:   [{min(r['pla_min'] for r in sub):.1f}, "
          f"{max(r['pla_max'] for r in sub):.1f}] deg")
    print(f"    Alt 范围:     [{min(r['alt_min'] for r in sub):.0f}, "
          f"{max(r['alt_max'] for r in sub):.0f}] ft")
    print(f"    MN 范围:      [{min(r['mn_min'] for r in sub):.2f}, "
          f"{max(r['mn_max'] for r in sub):.2f}]")

# ══════════════════════════════════════════════════
# 3. PLA 采样密度（MPC 需要连续小幅动作）
# ══════════════════════════════════════════════════
print("\n[3] PLA 变化率（MPC 动作空间的连续性）")
print(f"\n  {'轨迹':<22}{'最大ΔPLA/步':>14}{'平均|ΔPLA|':>14}{'PLA唯一值':>12}")
print("  " + "-" * 66)

for rid in data.run_ids[:10]:
    cands = list(DATA.glob(f"*{rid}*.csv"))
    if not cands:
        continue
    df = pd.read_csv(cands[0])
    pla = df["PLA_deg"].to_numpy()
    d = np.abs(np.diff(pla))
    print(f"  {rid:<22}{d.max():>14.4f}{d.mean():>14.6f}"
          f"{len(np.unique(pla)):>12}")

# ══════════════════════════════════════════════════
# 4. 关键判断
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[4] 覆盖度判断")
print("=" * 96)

a_pla = [r["pla_delta"] for r in rows if r["group"] == "A"]
all_pla_min = min(r["pla_min"] for r in rows)
all_pla_max = max(r["pla_max"] for r in rows)

print(f"""
  现有数据:
    PLA 绝对范围: [{all_pla_min:.1f}, {all_pla_max:.1f}] deg
    组 A 的 PLA 阶跃幅度: {min(a_pla):.1f} – {max(a_pla):.1f} deg

  MPC 需要的动作:
    小幅、连续的 PLA 调整（如 ±0.5 deg）

  判断:
    ✅ PLA 范围覆盖充分（{all_pla_min:.0f}–{all_pla_max:.0f} deg 远超 ±0.5）
    ⚠️ 但数据是【整段开环激励】，不是【闭环控制轨迹】
    ⚠️ 缺少：给定参考 N_ref，MPC 逐步调整 PLA 的闭环样本

  结论:
    现有 26 条数据可用于【世界模型预训练】
    但 MPC 闭环验证需要新数据（见报告的任务 4 方案）
""")

# 保存
out = {"rows": rows,
       "summary": {
           "n_traj": len(rows),
           "pla_range": [all_pla_min, all_pla_max],
           "a_group_pla_delta": [min(a_pla), max(a_pla)],
       }}
(OUT / "action_coverage.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  -> {OUT/'action_coverage.json'}")
