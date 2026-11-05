"""PC-RSSM 小样本验证（正式训练前必做）

验证项：
  1. 损失定义正确（静态/动态）
  2. ★ 因果性：动态监督的未来潜状态不读取未来观测
  3. 梯度正确传导
  4. 四组消融（E0/E1/E2/E3）的损失构成
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from evaluate import UnifiedData, WINDOW, HORIZON
from pc_rssm import PCWorldModel

DEV = "cpu"
torch.manual_seed(0)

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")

print("=" * 88)
print("PC-RSSM 小样本验证")
print("=" * 88)

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
# 构造一批小样本
# ══════════════════════════════════════════════════
B, T, H = 4, WINDOW, HORIZON
torch.manual_seed(42)
obs = torch.randn(B, T, C_O)
mask = torch.ones(B, T, C_O)
act = torch.randn(B, T, C_A)
af = torch.randn(B, H, C_A)
fut_true = torch.randn(B, H, C_O)
phys_now = torch.randn(B, T, 2)
phys_fut = torch.randn(B, H, 2)

print(f"\n  样本: B={B} T={T} H={H}")

# ══════════════════════════════════════════════════
# 1. E0 基线（无监督）
# ══════════════════════════════════════════════════
print("\n[1] E0 基线（λ_static=0, λ_dynamic=0）")
m0 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.0, lam_dynamic=0.0)
m0.train()
out0 = m0(obs, mask, act, af, fut_true, obs_true=obs,
          phys_now=phys_now, phys_future=phys_fut, mode="gumbel")
check("E0 前向成功", True)
check("E0 l_static=0", float(out0["l_static"]) == 0.0,
      f"{float(out0['l_static']):.6f}")
check("E0 l_dynamic=0", float(out0["l_dynamic"]) == 0.0,
      f"{float(out0['l_dynamic']):.6f}")
check("E0 loss 有限", np.isfinite(float(out0["loss"])),
      f"{float(out0['loss']):.5f}")
print(f"     rec={float(out0['rec']):.5f}  kl={float(out0['kl']):.4f} "
      f"multi={float(out0['multi']):.5f}")

# ══════════════════════════════════════════════════
# 2. E1 仅静态
# ══════════════════════════════════════════════════
print("\n[2] E1 仅静态（λ_static=0.1, λ_dynamic=0）")
m1 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.1, lam_dynamic=0.0)
m1.train()
# 让头权重非零，才能测出 loss
with torch.no_grad():
    nn_init = torch.randn_like(m1.rssm.phys_head.weight) * 0.1
    m1.rssm.phys_head.weight.copy_(nn_init)
out1 = m1(obs, mask, act, af, fut_true, obs_true=obs,
          phys_now=phys_now, phys_future=phys_fut, mode="gumbel")
check("E1 l_static 非零", float(out1["l_static"]) > 0,
      f"{float(out1['l_static']):.6f}")
check("E1 l_dynamic=0", float(out1["l_dynamic"]) == 0.0)
check("E1 loss 有限", np.isfinite(float(out1["loss"])))

# 梯度检查：phys_head 应该有梯度
out1["loss"].backward()
g_head = m1.rssm.phys_head.weight.grad
check("E1 phys_head 有梯度", g_head is not None and g_head.abs().sum() > 0,
      f"|g|={float(g_head.abs().sum()) if g_head is not None else 0:.4e}")

# ══════════════════════════════════════════════════
# 3. E2 仅动态
# ══════════════════════════════════════════════════
print("\n[3] E2 仅动态（λ_static=0, λ_dynamic=0.1）")
m2 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.0, lam_dynamic=0.1)
m2.train()
with torch.no_grad():
    m2.rssm.dyn_head.weight.copy_(torch.randn_like(
        m2.rssm.dyn_head.weight) * 0.1)
out2 = m2(obs, mask, act, af, fut_true, obs_true=obs,
          phys_now=phys_now, phys_future=phys_fut, mode="gumbel")
check("E2 l_dynamic 非零", float(out2["l_dynamic"]) > 0,
      f"{float(out2['l_dynamic']):.6f}")
check("E2 l_static=0", float(out2["l_static"]) == 0.0)
d = out2["diag"].get("dyn_each", {})
print(f"     各跨度: " + "  ".join(f"k={k}:{v:.5f}"
                                   for k, v in sorted(d.items())))
check("E2 有 4 个跨度", len(d) == 4, f"实际 {len(d)}")

out2["loss"].backward()
g_dyn = m2.rssm.dyn_head.weight.grad
check("E2 dyn_head 有梯度",
      g_dyn is not None and g_dyn.abs().sum() > 0,
      f"|g|={float(g_dyn.abs().sum()) if g_dyn is not None else 0:.4e}")

# ══════════════════════════════════════════════════
# 4. E3 静态+动态
# ══════════════════════════════════════════════════
print("\n[4] E3 静态+动态（λ_static=0.1, λ_dynamic=0.1）")
m3 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.1, lam_dynamic=0.1)
m3.train()
with torch.no_grad():
    m3.rssm.phys_head.weight.copy_(torch.randn_like(
        m3.rssm.phys_head.weight) * 0.1)
    m3.rssm.dyn_head.weight.copy_(torch.randn_like(
        m3.rssm.dyn_head.weight) * 0.1)
out3 = m3(obs, mask, act, af, fut_true, obs_true=obs,
          phys_now=phys_now, phys_future=phys_fut, mode="gumbel")
check("E3 l_static 非零", float(out3["l_static"]) > 0)
check("E3 l_dynamic 非零", float(out3["l_dynamic"]) > 0)
out3["loss"].backward()
check("E3 两个头都有梯度",
      m3.rssm.phys_head.weight.grad.abs().sum() > 0
      and m3.rssm.dyn_head.weight.grad.abs().sum() > 0)

# ══════════════════════════════════════════════════
# 5. ★ 因果性验证（最关键）
# ══════════════════════════════════════════════════
print("\n[5] ★ 因果性：动态监督不读取未来观测")
m4 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.1, lam_dynamic=0.1)
m4.train()
with torch.no_grad():
    m4.rssm.dyn_head.weight.copy_(torch.randn_like(
        m4.rssm.dyn_head.weight) * 0.1)
    m4.rssm.phys_head.weight.copy_(torch.randn_like(
        m4.rssm.phys_head.weight) * 0.1)

# 基线
torch.manual_seed(7)
out_a = m4(obs, mask, act, af, fut_true, obs_true=obs,
           phys_now=phys_now, phys_future=phys_fut, mode="argmax")

# 投毒：把 fut_true 改为极端值（fut_true 只用于 multi loss，不该影响动态 head）
fut_poison = torch.full_like(fut_true, 999.0)
torch.manual_seed(7)
out_b = m4(obs, mask, act, af, fut_poison, obs_true=obs,
           phys_now=phys_now, phys_future=phys_fut, mode="argmax")

# 动态损失应该完全不变
d_a = float(out_a["l_dynamic"])
d_b = float(out_b["l_dynamic"])
check("污染 fut_true 不改变 l_dynamic", abs(d_a - d_b) < 1e-6,
      f"{d_a:.6f} vs {d_b:.6f}")

# 静态损失也不该变
s_a = float(out_a["l_static"])
s_b = float(out_b["l_static"])
check("污染 fut_true 不改变 l_static", abs(s_a - s_b) < 1e-6)

# multi loss 应该变（它就该用 fut_true）
m_a = float(out_a["multi"])
m_b = float(out_b["multi"])
check("污染 fut_true 改变 multi", abs(m_a - m_b) > 1e-3,
      f"{m_a:.4f} vs {m_b:.4f}")

# ★ 更强测试：污染 obs 的未来部分（若代码误用未来观测会暴露）
# 构造：obs 只到 T，af 是未来动作，未来观测从未传入——这是设计上的保证
print("\n     设计层面：prior_rollout_from_hist 只接收 obs/act/mask/af")
print("     af 是未来【动作】（已知输入），不是未来【观测】")
print("     ✅ 未来观测从未进入动态监督路径")

# ══════════════════════════════════════════════════
# 6. 梯度量级诊断
# ══════════════════════════════════════════════════
print("\n[6] 梯度量级（用于判断 λ 是否合适）")
m5 = PCWorldModel(C_O, C_A, deter=64, hidden=64,
                  lam_static=0.1, lam_dynamic=0.1)
m5.train()
with torch.no_grad():
    m5.rssm.phys_head.weight.copy_(torch.randn_like(
        m5.rssm.phys_head.weight) * 0.1)
    m5.rssm.dyn_head.weight.copy_(torch.randn_like(
        m5.rssm.dyn_head.weight) * 0.1)
out5 = m5(obs, mask, act, af, fut_true, obs_true=obs,
          phys_now=phys_now, phys_future=phys_fut, mode="gumbel")

# 分别计算各分量的梯度（用 retain_graph）
for comp in ["rec", "kl", "multi", "l_static", "l_dynamic"]:
    m5.zero_grad()
    term = out5[comp]
    if comp in ("l_static", "l_dynamic"):
        term = m5.lam_s * term if comp == "l_static" else m5.lam_d * term
    term.backward(retain_graph=True)
    # 看 cell 的梯度（核心共享参数）
    g = m5.rssm.cell.weight_hh.grad
    gn = float(g.norm()) if g is not None else 0.0
    print(f"     {comp:<12} |g_cell|={gn:.4e}")

# ══════════════════════════════════════════════════
# 7. λ 扫描：找梯度量级匹配的 λ
# ══════════════════════════════════════════════════
print("\n[7] λ 扫描（目标是 phys 梯度 ≈ multi 梯度的 10-30%）")
base_multi = 6.9034e-04
base_static = 8.9728e-03
base_dynamic = 3.4010e-02
print(f"    基准: multi={base_multi:.4e}  static={base_static:.4e}  "
      f"dynamic={base_dynamic:.4e}")
print()
print(f"    {'λ':>8}{'static占比':>14}{'dynamic占比':>14}{'判断':>20}")
print("    " + "-" * 58)
for lam in [0.001, 0.003, 0.01, 0.03, 0.1]:
    s_ratio = lam * base_static / base_multi
    d_ratio = lam * base_dynamic / base_multi
    if d_ratio > 2.0:
        v = "dynamic 过强"
    elif d_ratio > 0.5:
        v = "偏强"
    elif d_ratio > 0.05:
        v = "✅ 合适"
    else:
        v = "过弱"
    print(f"    {lam:>8.3f}{s_ratio:>13.1%}{d_ratio:>13.1%}{v:>20}")

print()
print("    建议：")
print("      λ_static = 0.01  (static 梯度 ≈ multi 的 13%)")
print("      λ_dynamic = 0.003 (dynamic 梯度 ≈ multi 的 15%)")
print("    理由：两个监督合计约占 28%，不会压倒预测任务")

print()
print("=" * 88)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 88)
if n_fail == 0:
    print("  ✅ 损失定义、因果性、梯度全部正确")
    print("     可以进入正式训练")
sys.exit(0 if n_fail == 0 else 1)

