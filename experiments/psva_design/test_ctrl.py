"""P_ctrl 对照测试：不切分 h 但相同监督"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from evaluate import WINDOW, HORIZON
from psva import PSVA_WorldModel

DEV = "cpu"
C_O, C_A = 5, 3
B, T, H = 4, WINDOW, HORIZON
torch.manual_seed(0)

print("=" * 84)
print("P_ctrl 对照：不切分 h + 相同监督")
print("=" * 84)

obs = torch.randn(B, T, C_O)
mask = torch.ones(B, T, C_O)
act = torch.randn(B, T, C_A)
af = torch.randn(B, H, C_A)
fut = torch.randn(B, H, C_O)
y_f = torch.randn(B, T, 2)
y_s = torch.randn(B, T, 2)
y_d = {k: torch.randn(B, 2) * 0.1 for k in [1, 4, 8, 16]}

n_pass = n_fail = 0
def check(nm, cond, d=""):
    global n_pass, n_fail
    if cond: n_pass += 1; print(f"  OK {nm}" + (f"  {d}" if d else ""))
    else: n_fail += 1; print(f"  FAIL {nm}  {d}")

# 切分版
m_split = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64, unsplit=False,
                          lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
# 不切分版
m_ctrl = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64, unsplit=True,
                         lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)

print()
print("[1] 结构")
check("split 模式 d_fast=64", m_split.rssm.d_fast == 64)
check("ctrl 模式 unsplit=True", m_ctrl.rssm.unsplit is True)
check("split head_fast 输入 64", m_split.rssm.head_fast.in_features == 64)
check("ctrl  head_fast 输入 128", m_ctrl.rssm.head_fast.in_features == 128)
check("ctrl  head_slow 输入 128", m_ctrl.rssm.head_slow.in_features == 128)
check("ctrl  head_dyn  输入 128", m_ctrl.rssm.head_dyn.in_features == 128)

h = torch.randn(2, 128)
hf_s, hs_s = m_split.rssm.split_h(h)
hf_c, hs_c = m_ctrl.rssm.split_h(h)
check("split: h_f 是前 64 维", torch.allclose(hf_s, h[:, :64]))
check("split: h_f != h_s", not torch.allclose(hf_s, hs_s))
check("ctrl: h_f == h == h_s", torch.allclose(hf_c, h) and torch.allclose(hs_c, h))

print()
print("[2] 前向与损失")
for nm, m in [("split", m_split), ("ctrl", m_ctrl)]:
    m.train()
    with torch.no_grad():
        for hd in [m.rssm.head_fast, m.rssm.head_slow, m.rssm.head_dyn]:
            hd.weight.normal_(0, 0.1); hd.bias.zero_()
    o = m(obs, mask, act, af, fut, obs_true=obs, y_fast=y_f,
          y_slow=y_s, y_delta=y_d, mode="gumbel")
    check(f"{nm}: loss 有限", torch.isfinite(o["loss"]),
          f"loss={float(o['loss']):.5f}")
    check(f"{nm}: l_fast>0", float(o["l_fast"]) > 0,
          f"{float(o['l_fast']):.5f}")
    check(f"{nm}: l_slow>0", float(o["l_slow"]) > 0,
          f"{float(o['l_slow']):.6f}")
    check(f"{nm}: l_dyn>0", float(o["l_dyn"]) > 0,
          f"{float(o['l_dyn']):.6e}")

print()
print("[3] 梯度")
for nm, m in [("split", m_split), ("ctrl", m_ctrl)]:
    m.zero_grad(set_to_none=True)
    o = m(obs, mask, act, af, fut, obs_true=obs, y_fast=y_f, y_slow=y_s,
          y_delta=y_d, mode="gumbel")
    o["loss"].backward()
    ng = sum(1 for p in m.parameters() if p.grad is not None)
    na = sum(1 for _ in m.parameters())
    check(f"{nm}: 全部参数有梯度", ng == na, f"{ng}/{na}")
    bad = any(p.grad is not None and not torch.isfinite(p.grad).all()
              for p in m.parameters())
    check(f"{nm}: 无 NaN", not bad)

print()
print("[4] 参数量")
n_s = sum(p.numel() for p in m_split.parameters())
n_c = sum(p.numel() for p in m_ctrl.parameters())
print(f"    split: {n_s:,}")
print(f"    ctrl : {n_c:,}")
print(f"    差   : {n_c - n_s:+,}  ({n_c/n_s - 1:+.2%})")
check("ctrl 参数更多（头更大）", n_c > n_s,
      f"+{(n_c-n_s)} ({(n_c/n_s-1)*100:.2f}%)")

print()
print("=" * 84)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 84)
sys.exit(0 if n_fail == 0 else 1)