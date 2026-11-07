"""任务5：B2 数据 PLA-N2 时序分析

从 B2 第一段有效数据分析：
  - PLA 峰值 vs N2 峰值的时间关系
  - N2 何时越过目标
  - PLA 何时停止上升
"""
import re
import json
import numpy as np
from pathlib import Path

IIID = Path(r"C:\pclwm\pclwm_stage3a\iiid")
OUT = Path(r"C:\pclwm\pclwm_stage3a\replay")
OUT.mkdir(parents=True, exist_ok=True)

txt = (IIID / "b2.txt").read_text(encoding="utf-8")
rows = []
for ln in txt.split("\n"):
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+"
                 r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        rows.append([float(x) for x in m.groups()])

L = np.array(rows)
# 只取第一段（k<=60，参考正确的 +750 段）
L1 = L[L[:, 0] <= 60]

k = L1[:, 0]
t = L1[:, 1]
n2 = L1[:, 2]
ref = L1[:, 3]
err = L1[:, 4]
pla = L1[:, 5]
wf = L1[:, 6]
T45 = L1[:, 7]

N2_TARGET = 5099.29
N2_0 = 4349.29

print("=" * 90)
print("任务 5：B2 第一段 PLA-N2 时序分析")
print("=" * 90)
print(f"  步数: {len(k)} (k=1..{int(k[-1])})")
print(f"  时长: 0 - {t[-1]:.2f} s")

# ══ 关键事件时刻 ══
i_pla_max = np.argmax(pla)
i_n2_max = np.argmax(n2)
i_cross = np.where(n2 >= N2_TARGET)[0]

print(f"\n[关键事件]")
print(f"  PLA 峰值:        {pla[i_pla_max]:.2f}° @ k={int(k[i_pla_max])}, "
      f"t={t[i_pla_max]:.2f}s")
print(f"  N2 峰值:         {n2[i_n2_max]:.1f} rpm @ k={int(k[i_n2_max])}, "
      f"t={t[i_n2_max]:.2f}s")
if len(i_cross):
    print(f"  N2 首次越过目标: k={int(k[i_cross[0]])}, t={t[i_cross[0]]:.2f}s "
          f"(N2={n2[i_cross[0]]:.1f})")
print(f"  PLA 峰值到 N2 峰值的时间差: {t[i_n2_max]-t[i_pla_max]:+.2f} s")

# ══ 分段统计 ══
print(f"\n[分段]")
print(f"  {'区间':<18}{'k范围':<12}{'N2范围':<22}{'PLA范围':<20}")
print("  " + "-" * 74)
segs = [
    ("0-1 s", 0, 1.0), ("1-2 s", 1.0, 2.0), ("2-3 s", 2.0, 3.0),
    ("3-4 s", 3.0, 4.0), ("4-5 s", 4.0, 5.0), ("5-6.3 s", 5.0, 6.31),
]
for nm, t0, t1 in segs:
    m = (t >= t0) & (t < t1)
    if m.sum() == 0:
        continue
    print(f"  {nm:<18}{int(k[m][0])}-{int(k[m][-1]):<8}"
          f"[{n2[m].min():>7.1f},{n2[m].max():>7.1f}]  "
          f"[{pla[m].min():>5.2f},{pla[m].max():>5.2f}]")

# ══ 关键：PLA 何时停止上升 ══
dpla = np.diff(pla)
i_pla_stop = None
for i in range(len(dpla) - 1, 0, -1):
    if abs(dpla[i]) > 0.05:
        i_pla_stop = i
        break

print(f"\n[PLA 动作特性]")
print(f"  PLA 总变化: {pla[0]:.2f} → {pla[-1]:.2f} (Δ={pla[-1]-pla[0]:+.2f})")
print(f"  PLA 峰值: {pla.max():.2f} (超过目标工作点)"
      if pla.max() > 63.64 else f"  PLA 峰值: {pla.max():.2f}")
print(f"  PLA 振荡次数（方向反转）: {int(np.sum(np.diff(np.sign(dpla[dpla!=0]))!=0))}")

# ══ 用可行性映射估算"PLA 峰值对应多少稳态 N2" ══
FEAS_PLA = np.array([55, 58, 60, 62, 65, 70, 75])
FEAS_N2 = np.array([4349.29, 4621.21, 4795.20, 4967.66,
                    5208.97, 5575.92, 5927.54])

print(f"\n[★ 关键：PLA 峰值对应的稳态 N2]")
print(f"  PLA 峰值 = {pla.max():.2f}°")
n2_at_pla_max = np.interp(pla.max(), FEAS_PLA, FEAS_N2)
print(f"  -> 该 PLA 对应稳态 N2 ≈ {n2_at_pla_max:.1f} rpm")
print(f"  -> 目标 N2 = {N2_TARGET:.1f} rpm")
print(f"  -> 超出目标 {n2_at_pla_max - N2_TARGET:+.1f} rpm")

# 正确的 PLA（对应 5099）
pla_correct = np.interp(N2_TARGET, FEAS_N2, FEAS_PLA)
print(f"\n[正确的 PLA]")
print(f"  N2={N2_TARGET:.1f} 对应 PLA = {pla_correct:.2f}°")
print(f"  MPC 实际用了 {pla.max():.2f}°，超调 {(pla.max()-pla_correct)/pla_correct*100:+.1f}%")

# ══ 逐时间步详表（关键区间）══
print(f"\n[关键区间详表]")
print(f"  {'k':>4}{'t':>7}{'N2':>10}{'ref':>9}{'err':>9}{'PLA':>8}"
      f"{'dPLA':>8}{'Wf':>9}")
print("  " + "-" * 66)
d_pla_full = np.diff(pla, prepend=pla[0])
for i in range(0, len(k), 3):
    print(f"  {int(k[i]):>4}{t[i]:>7.2f}{n2[i]:>10.1f}{ref[i]:>9.1f}"
          f"{err[i]:>+9.1f}{pla[i]:>8.2f}{d_pla_full[i]:>+8.2f}{wf[i]:>9.4f}")

# ══ 保存 ══
res = {
    "n_steps": int(len(k)),
    "pla_peak": float(pla.max()),
    "pla_peak_t": float(t[i_pla_max]),
    "pla_peak_k": int(k[i_pla_max]),
    "n2_peak": float(n2.max()),
    "n2_peak_t": float(t[i_n2_max]),
    "n2_peak_k": int(k[i_n2_max]),
    "n2_cross_target_t": float(t[i_cross[0]]) if len(i_cross) else None,
    "n2_cross_target_k": int(k[i_cross[0]]) if len(i_cross) else None,
    "delay_pla_to_n2_peak_s": float(t[i_n2_max] - t[i_pla_max]),
    "pla_correct": float(pla_correct),
    "n2_at_pla_peak_steady": float(n2_at_pla_max),
    "overshoot_from_pla": float(n2_at_pla_max - N2_TARGET),
    "data": {
        "k": k.tolist(), "t": t.tolist(), "n2": n2.tolist(),
        "ref": ref.tolist(), "err": err.tolist(), "pla": pla.tolist(),
        "wf": wf.tolist(), "T45": T45.tolist(),
    },
}
(OUT / "b2_timing.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'b2_timing.json'}")
