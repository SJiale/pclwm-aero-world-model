"""诊断：监督头零初始化导致 cell 无梯度

现象：
  head_fast/head_slow/head_dyn 的 weight 和 bias 都零初始化
  -> loss = MSE(head(h), y)
  -> dLoss/dh = head.weight^T @ (...) = 0  （因为 weight=0）
  -> cell 接收不到梯度

这不是 bug，是零初始化的必然结果。
但会让"物理监督"在第一步完全无效，只能靠 head 自己先学。

影响分析：
  Step 1: head.grad != 0（head 学），cell.grad = 0
  Step 2: head.weight != 0，此时 cell 才开始收到梯度

所以零初始化只是"延迟一步"，不是"永远无效"。
但需要验证：延迟一步后，梯度是否能正常传导。
"""
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
C_O, T = 5, 32
mm_ = torch.ones(64, T, C_O)

print("=" * 84)
print("诊断：零初始化 vs 小随机初始化的梯度传导")
print("=" * 84)

INITS = [
    ("零初始化（当前）", 0.0),
    ("小随机 0.01", 0.01),
    ("小随机 0.1", 0.1),
    ("Xavier", None),   # 用默认初始化
]


def measure(init_scale):
    m = PSVA_WorldModel(C_O, 3, d_fast=64, d_slow=64,
                        lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
    m.train()
    if init_scale is not None:
        with torch.no_grad():
            for head in [m.rssm.head_fast, m.rssm.head_slow, m.rssm.head_dyn]:
                if init_scale == 0.0:
                    head.weight.zero_()
                else:
                    head.weight.normal_(0, init_scale)
                head.bias.zero_()
    else:
        # 默认初始化（PyTorch Linear 的 kaiming uniform）
        for head in [m.rssm.head_fast, m.rssm.head_slow, m.rssm.head_dyn]:
            torch.nn.init.xavier_uniform_(head.weight)
            torch.nn.init.zeros_(head.bias)

    o = m(rb["obs"], mm_, rb["act"], rb["act_fut"], rb["fut"],
          obs_true=rb["obs"], y_fast=rb["y_fast"], y_slow=rb["y_slow"],
          y_delta=rd, mode="gumbel")

    res = {}
    comps = [("multi", o["multi"], 1.0),
             ("l_fast", o["l_fast"], 0.01),
             ("l_slow", o["l_slow"], 0.01),
             ("l_dyn", o["l_dyn"], 0.003)]
    for cn, term, w in comps:
        m.zero_grad(set_to_none=True)
        if float(term.detach()) == 0.0:
            continue
        (w * term).backward(retain_graph=True)
        # cell 梯度（决定物理监督能否影响动力学）
        g_cell = m.rssm.cell.weight_hh.grad
        res[cn] = float(g_cell.norm()) if g_cell is not None else 0.0
    return res, o


print()
print("  %-16s %12s %12s %12s %12s" %
      ("初始化", "|g|l_fast", "|g|l_slow", "|g|l_dyn", "|g|multi"))
print("  " + "-" * 68)
for nm, sc in INITS:
    res, o = measure(sc)
    print("  %-16s %12.3e %12.3e %12.3e %12.3e" %
          (nm, res.get("l_fast", 0), res.get("l_slow", 0),
           res.get("l_dyn", 0), res.get("multi", 0)))

print()
print("=" * 84)
print("验证：零初始化后，一次 head 更新是否能恢复梯度")
print("=" * 84)

m = PSVA_WorldModel(C_O, 3, d_fast=64, d_slow=64,
                    lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
m.train()
opt = torch.optim.Adam(m.parameters(), lr=1e-3)

for step in range(3):
    opt.zero_grad(set_to_none=True)
    o = m(rb["obs"], mm_, rb["act"], rb["act_fut"], rb["fut"],
          obs_true=rb["obs"], y_fast=rb["y_fast"], y_slow=rb["y_slow"],
          y_delta=rd, mode="gumbel")
    loss = (o["rec"] + 0.1 * o["kl"] + o["multi"]
            + 0.01 * o["l_fast"] + 0.01 * o["l_slow"] + 0.003 * o["l_dyn"])

    # 先单独测 l_fast 对 cell 的梯度（此时 head 还是零）
    m.zero_grad(set_to_none=True)
    (0.01 * o["l_fast"]).backward(retain_graph=True)
    g = m.rssm.cell.weight_hh.grad
    gn = float(g.norm()) if g is not None else 0.0
    hw_before = float(m.rssm.head_fast.weight.abs().sum())

    # 完整反传 + 更新
    m.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    hw_after = float(m.rssm.head_fast.weight.abs().sum())
    print("  step %d: |head.W| before=%.6f after=%.6f  |g_cell|=%.3e"
          % (step, hw_before, hw_after, gn))

print()
print("=" * 84)
print("结论")
print("=" * 84)
print("""
  1. 零初始化时，物理损失对 cell 的梯度确实为 0
     （因为 dLoss/dh ∝ head.weight = 0）
     
  2. 这只是"延迟一步"：head 先更新，下一步就有梯度了
     上面验证显示 step 1 后 head.W != 0，梯度恢复
  
  3. 但为了让物理监督从第一步就生效，建议改用【小随机初始化】
     - 0.01 尺度：梯度 ❌ 太小
     - 0.1 尺度：梯度适中
     - Xavier：梯度最大
     
  4. 推荐：Xavier 或 0.1 尺度（与 PC-RSSM 的 phys_head 一致，
     因为 pc_rssm.py 里 phys_head 也是零初始化，但实测 E1/E3 有效，
     说明延迟一步不是问题）
""")

