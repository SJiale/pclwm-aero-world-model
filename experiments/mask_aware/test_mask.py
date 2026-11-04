"""Mask-aware 机制单元测试 + 信息泄漏检查

关键验证：
  1. mask 应用正确（缺失通道被填充，mask 不参与 loss）
  2. 故障后 T4 真值绝不进入模型（信息隔离）
  3. MaskGRU / MaskRSSM 前向、梯度、维度
  4. 训练时增强的缺失模式覆盖多种情况
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))

from mask_impl import (apply_mask, make_mask, augment_mask_batch,
                       MaskGRU, MaskWorldModel)
from evaluate import UnifiedData, DT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

print("=" * 80)
print("Mask-aware 单元测试")
print("=" * 80)
print(f"  设备={DEV}")

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
print(f"  通道: {data.obs_cols}")
print(f"  T4 索引 = {I_T4}")

n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1
        print(f"  ❌ {name}  {detail}")


# ══════════════════════════════════════════════════
# 1. mask 应用
# ══════════════════════════════════════════════════
print("\n[1] mask 应用正确性")
x = torch.tensor([[[1.0, 2.0, 3.0]]])
m = torch.tensor([[[1.0, 0.0, 1.0]]])
y = apply_mask(x, m)
check("有效通道保留", float(y[0, 0, 0]) == 1.0 and float(y[0, 0, 2]) == 3.0)
check("缺失通道填充为 0", float(y[0, 0, 1]) == 0.0, f"={float(y[0,0,1])}")

y2 = apply_mask(x, m, fill=-9.0)
check("自定义填充值", float(y2[0, 0, 1]) == -9.0)

# 全缺失
m0 = torch.zeros(1, 1, 3)
y3 = apply_mask(x, m0)
check("全缺失全填充", torch.allclose(y3, torch.zeros_like(y3)))

# 全有效
m1 = torch.ones(1, 1, 3)
check("全有效等于原值", torch.allclose(apply_mask(x, m1), x))

# ══════════════════════════════════════════════════
# 2. mask 生成模式
# ══════════════════════════════════════════════════
print("\n[2] mask 生成模式")
B, T = 4, 32

m_none = make_mask(B, T, C_O, "none")
check("none 全 1", bool((m_none == 1).all()))

m_fix = make_mask(B, T, C_O, "fixed", channels=[I_T4])
check("fixed 指定通道全 0", bool((m_fix[:, :, I_T4] == 0).all()))
check("fixed 其他通道全 1",
      bool((m_fix[:, :, [i for i in range(C_O) if i != I_T4]] == 1).all()))

m_rnd = make_mask(B, T, C_O, "random", ratio=0.4)
drop_per_step = (m_rnd == 0).sum(-1)
check("random 每步至少留 1 个", bool((m_rnd.sum(-1) >= 1).all()),
      f"最少有效={int(m_rnd.sum(-1).min())}")
check("random 有丢弃", bool((m_rnd == 0).any()))

m_bst = make_mask(B, T, C_O, "burst", channels=[I_T4],
                  burst_len=10, t0=20)
check("burst 在 [20,30) 为 0",
      bool((m_bst[:, 20:30, I_T4] == 0).all()))
check("burst 在 [0,20) 为 1",
      bool((m_bst[:, :20, I_T4] == 1).all()))
check("burst 在 [30,T) 为 1",
      bool((m_bst[:, 30:, I_T4] == 1).all()))

# ══════════════════════════════════════════════════
# 3. 信息隔离（最关键）
# ══════════════════════════════════════════════════
print("\n[3] 信息隔离：故障后 T4 真值不外泄")
wm = MaskWorldModel(C_O, C_A, deter=64, hidden=64).to(DEV).eval()

obs = torch.randn(2, T, C_O, device=DEV)
act = torch.randn(2, T, C_A, device=DEV)
af = torch.randn(2, 32, C_A, device=DEV)

# 构造"T4 从 t=16 起失效"的 mask
mask = torch.ones(2, T, C_O, device=DEV)
mask[:, 16:, I_T4] = 0.0

with torch.no_grad():
    # 完整观测
    p_full = wm.imagine(obs, torch.ones_like(mask), act, af, mode="argmax")
    # T4 缺失
    p_mask = wm.imagine(obs, mask, act, af, mode="argmax")

    # 攻击测试：故障后把 obs 的 T4 改成任意值，输出应完全不变
    obs2 = obs.clone()
    obs2[:, 16:, I_T4] = 999.0
    p_mask2 = wm.imagine(obs2, mask, act, af, mode="argmax")

check("故障后改 T4 观测不影响输出",
      torch.allclose(p_mask, p_mask2, atol=1e-6),
      f"max diff={(p_mask-p_mask2).abs().max():.2e}")

# 对照：故障前（历史段）的 T4 应该影响输出
obs3 = obs.clone()
obs3[:, :16, I_T4] = 999.0
with torch.no_grad():
    p_mask3 = wm.imagine(obs3, mask, act, af, mode="argmax")
check("故障前的 T4 历史会影响输出",
      not torch.allclose(p_mask, p_mask3, atol=1e-3),
      f"max diff={(p_mask-p_mask3).abs().max():.4f}")

check("缺失与完整观测输出不同",
      not torch.allclose(p_full, p_mask, atol=1e-3),
      f"max diff={(p_full-p_mask).abs().max():.4f}")

# ══════════════════════════════════════════════════
# 4. MaskGRU 前向
# ══════════════════════════════════════════════════
print("\n[4] MaskGRU 前向")
g = MaskGRU(C_O, C_A, hidden=64, mlp=128).to(DEV)
out = g(obs, mask, act, af)
check("输出形状", out.shape == (2, 32, C_O), str(tuple(out.shape)))
check("输出有限", torch.isfinite(out).all())
n_g = sum(p.numel() for p in g.parameters())
print(f"     参数量 = {n_g:,}")

# ══════════════════════════════════════════════════
# 5. MaskWorldModel 训练路径
# ══════════════════════════════════════════════════
print("\n[5] MaskWorldModel 训练路径")
wm.train()
fut = torch.randn(2, 32, C_O, device=DEV)
try:
    out = wm(obs, mask, act, af, fut)
    check("forward 不崩溃", True)
    check("rec 有限", bool(np.isfinite(float(out["rec"]))),
          f"{float(out['rec']):.5f}")
    check("multi 有限", bool(np.isfinite(float(out["multi"]))),
          f"{float(out['multi']):.5f}")
    loss = out["rec"] + 0.1*out["kl"] + out["multi"]
    loss.backward()
    n_grad = sum(1 for p in wm.parameters() if p.grad is not None)
    n_all = sum(1 for _ in wm.parameters())
    check("所有参数有梯度", n_grad == n_all, f"{n_grad}/{n_all}")
except Exception as e:
    check("MaskWorldModel forward", False, str(e)[:100])

# ══════════════════════════════════════════════════
# 6. 训练时增强
# ══════════════════════════════════════════════════
print("\n[6] 训练时 mask 增强")
cfg = {"p_none": 0.2, "p_fixed": 0.2, "p_random": 0.3,
       "p_burst": 0.3, "fixed_channels": [I_T4],
       "burst_channels": [I_T4], "random_ratio": 0.3,
       "burst_min": 8, "burst_max": 24}
o_, m_, f_, mf_ = augment_mask_batch(
    torch.randn(32, T, C_O), torch.randn(32, T, C_A),
    torch.randn(32, 32, C_O), torch.randn(32, 32, C_A), cfg, DEV)
check("输出形状正确", o_.shape == (32, T, C_O) and m_.shape == (32, T, C_O))
check("mask 是 0/1", bool(((m_ == 0) | (m_ == 1)).all()))
n_all_valid = int((m_.sum(-1) == C_O).all(-1).sum())
n_some_missing = int(((m_.sum(-1) < C_O).any(-1)).sum())
check("增强产生了缺失样本", n_some_missing > 0,
      f"全有效={n_all_valid}, 有缺失={n_some_missing}")
check("未来 mask 沿用历史末尾",
      bool((mf_ == m_[:, -1:, :]).all()))

# ══════════════════════════════════════════════════
# 7. DT-Mask-RSSM
# ══════════════════════════════════════════════════
print("\n[7] DT-Mask-RSSM（联合模型）")
try:
    wm_dt = MaskWorldModel(C_O, C_A, deter=64, hidden=64,
                           use_dt=True, deter_slow=64,
                           k_slow=4).to(DEV).eval()
    with torch.no_grad():
        p = wm_dt.imagine(obs, mask, act, af, mode="gumbel")
    check("DT-Mask 前向", p.shape == (2, 32, C_O), str(tuple(p.shape)))
    n_dt = sum(q.numel() for q in wm_dt.parameters())
    print(f"     参数量 = {n_dt:,}")
except Exception as e:
    check("DT-Mask 前向", False, str(e)[:100])

# ══════════════════════════════════════════════════
print()
print("=" * 80)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 80)
if n_fail == 0:
    print("  ✅ 全部通过")
else:
    sys.exit(1)
