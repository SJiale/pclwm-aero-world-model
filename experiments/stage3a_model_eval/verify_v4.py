"""P0-1：核实 V4 的 PLA 阶跃时刻 vs rollout 窗口"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask"]

from evaluate import UnifiedData, WINDOW, HORIZON, DT

data = UnifiedData()
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7

print("=" * 95)
print("P0-1：核实 V4 阶跃时刻 vs rollout 窗口")
print("=" * 95)

print(f"\n  WINDOW={WINDOW} ({WINDOW*DT:.2f}s)  HORIZON={HORIZON}")
print(f"  rollout 起点 = WINDOW 步 = {WINDOW*DT:.2f} s")
print(f"  10s rollout 终点 = {WINDOW*DT + 10:.2f} s")

for f in sorted(CTRL.glob("V4*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    pla = d["PLA_cmd_deg"].to_numpy()
    t = d["time_s"].to_numpy()
    n2 = d["N2int"].to_numpy()

    print(f"\n  ── {f.stem} ──")
    print(f"    总时长: {t[-1]:.2f} s, {len(t)} 点 (0.105s)")

    # 找阶跃
    dp = np.diff(pla)
    jumps = np.where(np.abs(dp) > 0.5)[0]
    print(f"    PLA 阶跃（|Δ|>0.5°）共 {len(jumps)} 处:")
    for j in jumps:
        print(f"      t={t[j+1]:>6.2f} s:  {pla[j]:.2f} -> {pla[j+1]:.2f}  "
              f"(Δ={dp[j]:+.2f})")

    # 关键：我的 10s rollout 覆盖范围
    t_start = WINDOW * DT
    t_end = t_start + 10
    print(f"\n    我的 10s rollout 覆盖: [{t_start:.2f}, {t_end:.2f}] s")
    covered = [t[j+1] for j in jumps if t_start <= t[j+1] <= t_end]
    missed = [t[j+1] for j in jumps if not (t_start <= t[j+1] <= t_end)]
    print(f"      覆盖到的阶跃: {covered if covered else '【无】'}")
    print(f"      未覆盖的阶跃: {missed if missed else '无'}")

    # N2 在该窗口内的变化
    i0 = int(round(t_start / DT))
    i1 = int(round(t_end / DT))
    seg = n2[i0:min(i1, len(n2))]
    print(f"\n    N2 在 rollout 窗口内: [{seg.min():.1f}, {seg.max():.1f}] "
          f"变化 {seg.max()-seg.min():.1f} rpm")
    print(f"    N2 全程: [{n2.min():.1f}, {n2.max():.1f}] 变化 {n2.max()-n2.min():.1f} rpm")

print("\n" + "=" * 95)
print("结论")
print("=" * 95)
print("""
  如果 V4 的阶跃在 t=15s 之后，而 rollout 只覆盖 [3.36, 13.36]s，
  则图 1 中 V4 的"剧烈波动"与阶跃无关，我之前的归因是错的。

  需要做事件对齐的 rollout（围绕每个阶跃时刻），才能评价阶跃响应。
""")
