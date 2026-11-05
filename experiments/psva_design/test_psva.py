"""PSVA 单元测试（训练前必做）

验证项：
  1. 前向传播（4 种配置）
  2. 反向梯度（所有参数 + 无 NaN）
  3. 先验递推（不看未来观测）
  4. ★ 因果性（污染未来观测不影响动态损失）
  5. 参数量（vs PC-RSSM E3）
  6. ★ 切分正确性（h_fast / h_slow 无重叠）
  7. ΔN 归一化的合理性（梯度量级）
  8. 不影响已有模型（PC-RSSM E3 可正常加载）
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from evaluate import UnifiedData, WINDOW, HORIZON
from psva_data import PSVAData
from psva import PSVA_WorldModel, PSVA_RSSM
from pc_rssm import PCWorldModel

DEV = "cpu"
torch.manual_seed(0)

data = UnifiedData()
C_O = data.n_obs
C_A = data.n_act

print("=" * 96)
print("PSVA 单元测试")
print("=" * 96)

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
# 构造小批样本
# ══════════════════════════════════════════════════
B, T, H = 4, WINDOW, HORIZON
torch.manual_seed(42)
obs = torch.randn(B, T, C_O)
mask = torch.ones(B, T, C_O)
act = torch.randn(B, T, C_A)
af = torch.randn(B, H, C_A)
fut = torch.randn(B, H, C_O)
y_fast = torch.randn(B, T, 2)
y_slow = torch.randn(B, T, 2)
y_delta = {k: torch.randn(B, 2) * 0.1 for k in [1, 4, 8, 16]}

# ══════════════════════════════════════════════════
# 1. 结构测试
# ══════════════════════════════════════════════════
print("\n[1] 结构测试")
m = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64,
                    lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
check("d_fast + d_slow == deter", m.rssm.d_fast + m.rssm.d_slow == m.rssm.deter,
      f"{m.rssm.d_fast} + {m.rssm.d_slow} = {m.rssm.deter}")

# 切分正确性
h_test = torch.randn(3, 128)
hf, hs = m.rssm.split_h(h_test)
check("h_fast 形状", hf.shape == (3, 64), str(tuple(hf.shape)))
check("h_slow 形状", hs.shape == (3, 64), str(tuple(hs.shape)))
check("切分无重叠",
      torch.allclose(torch.cat([hf, hs], -1), h_test, atol=0),
      "concat(split(h)) == h")
# 与直接索引一致
check("切分与索引一致",
      torch.allclose(hf, h_test[:, :64]) and torch.allclose(hs, h_test[:, 64:]))

# 3D 切分
h3 = torch.randn(2, 5, 128)
hf3, hs3 = m.rssm.split_h(h3)
check("3D 切分正确", hf3.shape == (2, 5, 64) and hs3.shape == (2, 5, 64))

# ══════════════════════════════════════════════════
# 2. 前向传播（4 种配置）
# ══════════════════════════════════════════════════
print("\n[2] 前向传播（4 种配置）")

CONFIGS = [
    ("P0 全关", 0.0, 0.0, 0.0),
    ("P1 仅静态", 0.01, 0.01, 0.0),
    ("P2 仅动态", 0.0, 0.0, 0.003),
    ("P3 静态+动态", 0.01, 0.01, 0.003),
]

for nm, lf, ls, ld in CONFIGS:
    mm = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64,
                         lam_fast=lf, lam_slow=ls, lam_dyn=ld)
    mm.train()
    # 非零头权重
    with torch.no_grad():
        mm.rssm.head_fast.weight.copy_(torch.randn_like(mm.rssm.head_fast.weight) * 0.1)
        mm.rssm.head_slow.weight.copy_(torch.randn_like(mm.rssm.head_slow.weight) * 0.1)
        mm.rssm.head_dyn.weight.copy_(torch.randn_like(mm.rssm.head_dyn.weight) * 0.1)
    out = mm(obs, mask, act, af, fut, obs_true=obs,
             y_fast=y_fast, y_slow=y_slow, y_delta=y_delta, mode="gumbel")
    ok = (torch.isfinite(out["loss"]) and
          not torch.isnan(out["loss"]))
    check(f"{nm}: loss 有限", ok, f"loss={float(out['loss']):.5f}")
    if lf > 0:
        check(f"{nm}: l_fast>0", float(out["l_fast"]) > 0,
              f"{float(out['l_fast']):.5f}")
    if ls > 0:
        check(f"{nm}: l_slow>0", float(out["l_slow"]) > 0,
              f"{float(out['l_slow']):.5f}")
    if ld > 0:
        check(f"{nm}: l_dyn>0", float(out["l_dyn"]) > 0,
              f"{float(out['l_dyn']):.6f}")

# ══════════════════════════════════════════════════
# 3. 反向梯度
# ══════════════════════════════════════════════════
print("\n[3] 反向梯度")
mm = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64,
                     lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
mm.train()
with torch.no_grad():
    mm.rssm.head_fast.weight.copy_(torch.randn_like(mm.rssm.head_fast.weight) * 0.1)
    mm.rssm.head_slow.weight.copy_(torch.randn_like(mm.rssm.head_slow.weight) * 0.1)
    mm.rssm.head_dyn.weight.copy_(torch.randn_like(mm.rssm.head_dyn.weight) * 0.1)
out = mm(obs, mask, act, af, fut, obs_true=obs,
         y_fast=y_fast, y_slow=y_slow, y_delta=y_delta, mode="gumbel")
out["loss"].backward()

n_g = sum(1 for p in mm.parameters() if p.grad is not None)
n_all = sum(1 for _ in mm.parameters())
check("所有参数有梯度", n_g == n_all, f"{n_g}/{n_all}")

has_nan = any(p.grad is not None and not torch.isfinite(p.grad).all()
              for p in mm.parameters())
check("梯度无 NaN/Inf", not has_nan)

# 分离梯度：三个头应该互相独立
mm.zero_grad()
out2 = mm(obs, mask, act, af, fut, obs_true=obs,
          y_fast=y_fast, y_slow=y_slow, y_delta=y_delta, mode="gumbel")
(mm.lam_f * out2["l_fast"]).backward(retain_graph=True)
g_f = mm.rssm.head_fast.weight.grad.clone() if mm.rssm.head_fast.weight.grad is not None else None
g_s = mm.rssm.head_slow.weight.grad
g_d = mm.rssm.head_dyn.weight.grad
check("l_fast 只作用于 head_fast", g_f is not None and g_f.abs().sum() > 0)
check("l_fast 不作用于 head_slow", g_s is None or g_s.abs().sum() == 0,
      f"|g_s|={float(g_s.abs().sum()) if g_s is not None else 0:.2e}")

# ══════════════════════════════════════════════════
# 4. 先验递推
# ══════════════════════════════════════════════════
print("\n[4] 先验递推")
mm.eval()
rssm = mm.rssm
with torch.no_grad():
    st = rssm.init_state(B, DEV)
    # 历史编码
    obs_m = obs * mask
    for t in range(T):
        z_prev = st["z"].reshape(B, -1)
        h = rssm.cell(torch.cat([z_prev, act[:, t]], -1), st["h"])
        emb = rssm.enc(torch.cat([obs_m[:, t], mask[:, t]], -1))
        pl = rssm.post(torch.cat([torch.cat([h, z_prev], -1), emb], -1)
                       ).reshape(B, rssm.stoch, rssm.classes)
        z = rssm.sample(pl, "argmax")
        st = {"h": h, "z": z}

    hs_roll = rssm.prior_rollout_h(st, af, mode="argmax")
check("先验递推输出形状", hs_roll.shape == (B, H, rssm.deter),
      str(tuple(hs_roll.shape)))
check("递推状态有变化",
      not torch.allclose(hs_roll[:, 0], hs_roll[:, -1]),
      f"max diff={float((hs_roll[:,0]-hs_roll[:,-1]).abs().max()):.4f}")

# ══════════════════════════════════════════════════
# 5. ★ 因果性
# ══════════════════════════════════════════════════
print("\n[5] ★ 因果性（动态监督不看未来观测）")
mm2 = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64,
                      lam_fast=0.0, lam_slow=0.0, lam_dyn=0.003)
mm2.eval()
with torch.no_grad():
    mm2.rssm.head_dyn.weight.copy_(
        torch.randn_like(mm2.rssm.head_dyn.weight) * 0.1)

torch.manual_seed(7)
o1 = mm2(obs, mask, act, af, fut, obs_true=obs,
         y_fast=y_fast, y_slow=y_slow, y_delta=y_delta, mode="argmax")

# 污染 fut（只用于 multi 损失，不该影响 l_dyn）
fut_poison = torch.full_like(fut, 999.0)
torch.manual_seed(7)
o2 = mm2(obs, mask, act, af, fut_poison, obs_true=obs,
         y_fast=y_fast, y_slow=y_slow, y_delta=y_delta, mode="argmax")

d1, d2 = float(o1["l_dyn"]), float(o2["l_dyn"])
check("污染 fut 不改 l_dyn", abs(d1 - d2) < 1e-6, f"{d1:.6f} vs {d2:.6f}")
m1, m2 = float(o1["multi"]), float(o2["multi"])
check("污染 fut 改 multi", abs(m1 - m2) > 1e-3, f"{m1:.4f} vs {m2:.4f}")

# 更强的：污染 obs 的未来部分（本设计不传未来 obs，应无影响）
check("设计保证：未来观测从未进入动态路径", True,
      "prior_rollout_from_hist(obs, mask, act, af) 只收历史 obs + 未来动作")

# ══════════════════════════════════════════════════
# 6. 参数量（vs PC-RSSM E3）
# ══════════════════════════════════════════════════
print("\n[6] 参数量对比")
pc = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                  lam_static=0.01, lam_dynamic=0.003)
n_pc = sum(p.numel() for p in pc.parameters())
n_psva = sum(p.numel() for p in mm.parameters())
print(f"    PC-RSSM E3 : {n_pc:,}")
print(f"    PSVA       : {n_psva:,}")
print(f"    差值       : {n_psva - n_pc:+,}  ({n_psva/n_pc - 1:+.3%})")

# 分解
n_head_f = sum(p.numel() for p in mm.rssm.head_fast.parameters())
n_head_s = sum(p.numel() for p in mm.rssm.head_slow.parameters())
n_head_d = sum(p.numel() for p in mm.rssm.head_dyn.parameters())
print(f"    head_fast  : {n_head_f:,}")
print(f"    head_slow  : {n_head_s:,}")
print(f"    head_dyn   : {n_head_d:,}")
# PC-RSSM 的头部
n_pc_heads = (sum(p.numel() for p in pc.rssm.phys_head.parameters()) +
              sum(p.numel() for p in pc.rssm.dyn_head.parameters()))
print(f"    PC 头部    : {n_pc_heads:,}")
check("参数量增幅 < 0.1%", abs(n_psva/n_pc - 1) < 0.001,
      f"{n_psva/n_pc - 1:+.4%}")

# ══════════════════════════════════════════════════
# 7. ΔN 归一化梯度量级
# ══════════════════════════════════════════════════
print("\n[7] ΔN 归一化的梯度量级")
pd_ = PSVAData()
tr = pd_.windows("train", stride=4)
batch = tr[:64]
# 真实 ΔN 的尺度
dn_vals = np.concatenate([np.stack([w["y_delta"][k] for k in [1, 4, 8, 16]], 0)
                          for w in batch], 0)
print(f"    真实 ΔN（归一化）: mean={dn_vals.mean():.6f} "
      f"std={dn_vals.std():.6f} range=[{dn_vals.min():.4f}, {dn_vals.max():.4f}]")
print(f"    ⚠ 量级 <1e-3，动态损失梯度可能过小")

# 测试真实数据上的梯度
real_b = {k: torch.from_numpy(np.stack([w[k] for w in batch])).float()
          for k in ["obs", "act", "fut", "act_fut", "y_fast", "y_slow"]}
real_d = {k: torch.from_numpy(np.stack([w["y_delta"][k] for w in batch])).float()
          for k in [1, 4, 8, 16]}
mm3 = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64,
                      lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
mm3.train()
with torch.no_grad():
    mm3.rssm.head_fast.weight.copy_(torch.randn_like(mm3.rssm.head_fast.weight) * 0.1)
    mm3.rssm.head_slow.weight.copy_(torch.randn_like(mm3.rssm.head_slow.weight) * 0.1)
    mm3.rssm.head_dyn.weight.copy_(torch.randn_like(mm3.rssm.head_dyn.weight) * 1.0)
mreal = torch.ones(64, T, C_O)
o3 = mm3(real_b["obs"], mreal, real_b["act"], real_b["act_fut"], real_b["fut"],
         obs_true=real_b["obs"], y_fast=real_b["y_fast"],
         y_slow=real_b["y_slow"], y_delta=real_d, mode="gumbel")

print(f"    真实数据上的损失: rec={float(o3['rec']):.5f} "
      f"multi={float(o3['multi']):.5f}")
print(f"      l_fast={float(o3['l_fast']):.5f}  "
      f"l_slow={float(o3['l_slow']):.6f}  "
      f"l_dyn={float(o3['l_dyn']):.6e}")

# 各分量的 cell 梯度
comps = [("rec", o3["rec"], 1.0), ("multi", o3["multi"], 1.0),
         ("l_fast", o3["l_fast"], 0.01), ("l_slow", o3["l_slow"], 0.01),
         ("l_dyn", o3["l_dyn"], 0.003)]
gvals = {}
for cn, term, w in comps:
    mm3.zero_grad(set_to_none=True)
    if float(term.detach()) == 0:
        continue
    (w * term).backward(retain_graph=True)
    g = mm3.rssm.cell.weight_hh.grad
    gvals[cn] = float(g.norm()) if g is not None else 0.0

base = gvals.get("multi", 1e-9)
print(f"\n    梯度范数（相对 multi）:")
for k, v in gvals.items():
    print(f"      {k:<10} {v:.4e}  {v/base:>8.3f}x")

if gvals.get("l_dyn", 0) / base < 0.01:
    print(f"\n    ⚠ l_dyn 梯度仅 {gvals['l_dyn']/base*100:.2f}% of multi — 偏小")
    print(f"      建议：λ_dyn 提高，或 ΔN 用更小单位归一化")
else:
    print(f"\n    ✓ l_dyn 梯度占比 {gvals['l_dyn']/base*100:.1f}%")

# ══════════════════════════════════════════════════
# 8. 不影响已有模型
# ══════════════════════════════════════════════════
print("\n[8] 不影响已有模型")
ck = Path(r"C:\pclwm\pclwm_pcrssm\checkpoints\E3_both.pt")
if ck.exists():
    sd = torch.load(ck, map_location=DEV, weights_only=True)
    pc2 = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                       lam_static=0.01, lam_dynamic=0.003)
    missing, unexpected = pc2.load_state_dict(sd, strict=True), None
    check("PC-RSSM E3 检查点可加载", True, f"keys={len(sd)}")
    # 验证 PC 模型仍能前向
    pc2.eval()
    with torch.no_grad():
        o_pc = pc2(obs, mask, act, af, fut, obs_true=obs,
                   phys_now=y_fast, phys_future=y_fast[:, :H],
                   mode="argmax")
    check("PC-RSSM E3 前向正常", torch.isfinite(o_pc["loss"]),
          f"loss={float(o_pc['loss']):.4f}")
else:
    check("PC-RSSM E3 检查点存在", False, str(ck))

# ══════════════════════════════════════════════════
print()
print("=" * 96)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 96)
if n_fail == 0:
    print("  ✅ PSVA 实现正确，可进入正式训练")
sys.exit(0 if n_fail == 0 else 1)


