"""参数量匹配策略（第二轮实验配置）

问题：DT-RSSM 双 GRU cell 导致参数量天然更高
  - 基线 RSSM-d128 : 939,909
  - DT-RSSM (96+96) : 1,484,773  (1.58x)

两种匹配方式：
  A. 缩小 DT  -> (48+48) = 1,069,045 (1.14x)  仍不完美，且削弱 DT 表达力
  B. 加宽基线 -> RSSM-d? 使 ≈ 1,484,773       ✅ 更干净：同容量比结构

选 B 为主，A 为补充。

第二轮实验矩阵：
  RSSM-d128     939,909   基线（已有）
  RSSM-wide     ~1.48M    容量对照  ← 新增，回答"提升是否来自容量"
  DT-K1  (96+96) 1,484,773  K=1（已有/进行中）
  DT-K4  (96+96) 1,484,773  K=4
  DT-K8  (96+96) 1,484,773  K=8
  DT-K4-matched (48+48) 1,069,045  低容量对照
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_dt")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))
from rssm_impl import RSSM

C_O, C_A = 5, 3
TARGET = 1_484_773   # DT(96+96) 的参数量

print("=" * 80)
print("参数量匹配：加宽基线 RSSM")
print("=" * 80)
print(f"  目标参数量 = {TARGET:,}（DT-RSSM 96+96）")
print()

print(f"  {'deter':>8}{'hidden':>8}{'参数量':>14}{'vs 目标':>10}")
print("  " + "-" * 42)

best = None
for deter in range(128, 320, 8):
    for hidden in [128, 160, 192, 224, 256]:
        m = RSSM(C_O, C_A, deter=deter, stoch=32, classes=32,
                 hidden=hidden, free_bits=1.0)
        n = sum(p.numel() for p in m.parameters())
        if best is None or abs(n - TARGET) < abs(best[2] - TARGET):
            best = (deter, hidden, n)

# 打印附近几个
for deter in range(144, 224, 8):
    m = RSSM(C_O, C_A, deter=deter, stoch=32, classes=32,
             hidden=192, free_bits=1.0)
    n = sum(p.numel() for p in m.parameters())
    print(f"  {deter:>8}{192:>8}{n:>14,}{n/TARGET:>10.3f}")

print()
print(f"  ✅ 最佳匹配: deter={best[0]}, hidden={best[1]}, "
      f"参数={best[2]:,}")
print(f"     相对目标 {best[2]/TARGET:.3f}x, "
      f"绝对差 {abs(best[2]-TARGET)/TARGET*100:.1f}%")

# 验证
m = RSSM(C_O, C_A, deter=best[0], stoch=32, classes=32,
         hidden=best[1], free_bits=1.0)
n_actual = sum(p.numel() for p in m.parameters())
print(f"     复核: {n_actual:,}")

print()
print("=" * 80)
print("第二轮实验配置确定")
print("=" * 80)
print(f"""
  RSSM-wide：deter={best[0]}, hidden={best[1]}, 参数={n_actual:,}
     作用：容量对照。若 DT 与 RSSM-wide 性能相近，
           说明 DT 的收益来自容量而非时间尺度结构。
""")
