import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from psva_data import PSVAData
from psva import PSVA_WorldModel

pd_ = PSVAData()
tr = pd_.windows("train", stride=4)
batch = tr[:64]
rb = {k: torch.from_numpy(np.stack([w[k] for w in batch])).float()
      for k in ["obs", "act", "fut", "act_fut", "y_fast", "y_slow"]}
rd = {k: torch.from_numpy(np.stack([w["y_delta"][k] for w in batch])).float()
      for k in [1, 4, 8, 16]}
C_O = 5
T = 32
m_mask = torch.ones(64, T, C_O)

print("=" * 80)
print("l_dyn 梯度量级诊断")
print("=" * 80)

allv = np.concatenate([rd[k].numpy().ravel() for k in rd])
print()
print("真实 ΔN（归一化）: std=%.4f  range=[%.2f, %.2f]"
      % (allv.std(), allv.min(), allv.max()))
print()
print("  注意：ΔN 已用慢变量 sd 归一化（N2int sd=783.8 rpm）")
print("        单个窗口的 ΔN 通常 < 2 rpm，归一化后 < 0.003")
print("        但跨窗口拼接后 std=%.3f，因为有的大幅机动窗口 ΔN 大" % allv.std())

print()
print("=" * 80)
print("λ 扫描")
print("=" * 80)

# 用零初始化（正式训练的方式）
mm = PSVA_WorldModel(C_O, 3, d_fast=64, d_slow=64,
                     lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
mm.train()
o = mm(rb["obs"], m_mask, rb["act"], rb["act_fut"], rb["fut"],
       obs_true=rb["obs"], y_fast=rb["y_fast"], y_slow=rb["y_slow"],
       y_delta=rd, mode="gumbel")
print()
print("零初始化下（正式训练初期）:")
print("  l_fast=%.6f  l_slow=%.6f  l_dyn=%.6e"
      % (float(o["l_fast"]), float(o["l_slow"]), float(o["l_dyn"])))

# 各分量梯度
comps = [("rec", o["rec"], 1.0), ("multi", o["multi"], 1.0),
         ("l_fast", o["l_fast"], 0.01), ("l_slow", o["l_slow"], 0.01),
         ("l_dyn", o["l_dyn"], 0.003)]
gv = {}
for cn, term, w in comps:
    mm.zero_grad(set_to_none=True)
    if float(term.detach()) == 0.0:
        continue
    (w * term).backward(retain_graph=True)
    g = mm.rssm.cell.weight_hh.grad
    gv[cn] = float(g.norm()) if g is not None else 0.0

base = gv.get("multi", 1e-9)
print()
print("  梯度范数（相对 multi）:")
for k, v in gv.items():
    print("    %-10s %.4e  %8.3f x" % (k, v, v / base))

print()
print("  λ 扫描（目标 l_dyn ≈ multi 的 10-30%）:")
print("  %8s %12s %12s" % ("λ_dyn", "占比", "判断"))
print("  " + "-" * 36)
gd = gv.get("l_dyn", 0.0) / 0.003      # 归一化到 λ=1
for lam in [0.0001, 0.0003, 0.001, 0.003, 0.01]:
    r = lam * gd / base
    if 0.08 <= r <= 0.30:
        v = "✅ 合适"
    elif r < 0.08:
        v = "过弱"
    else:
        v = "过强"
    print("  %8.4f %11.1f%% %12s" % (lam, r * 100, v))

print()
print("  静态损失:")
gs = gv.get("l_fast", 0.0) / 0.01
gl = gv.get("l_slow", 0.0) / 0.01
print("    λ_fast 扫描（目标 5-15%）:")
for lam in [0.001, 0.003, 0.01, 0.03]:
    r = lam * gs / base
    v = "✅" if 0.03 <= r <= 0.20 else ("小" if r < 0.03 else "大")
    print("      λ=%.4f -> %.1f%%  %s" % (lam, r * 100, v))
print("    λ_slow 扫描（目标 5-15%）:")
for lam in [0.001, 0.003, 0.01, 0.03]:
    r = lam * gl / base
    v = "✅" if 0.03 <= r <= 0.20 else ("小" if r < 0.03 else "大")
    print("      λ=%.4f -> %.1f%%  %s" % (lam, r * 100, v))
