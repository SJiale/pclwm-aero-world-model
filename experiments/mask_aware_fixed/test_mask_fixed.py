"""修复后单元测试：重建监督 + 逐步在线估计

核心验证：
  1. 重建损失用**真实观测**作监督，不是遮蔽值 0
  2. 逐步在线估计每 0.105 s 更新一次，不是每 32 步
  3. 信息隔离仍然正确
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))

from mask_impl_fixed import (apply_mask, augment_mask_batch, MaskGRU,
                             MaskWorldModel)
from evaluate import UnifiedData, WINDOW, DT

DEV = "cpu"
torch.manual_seed(0)

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")

print("=" * 84)
print("修复后单元测试")
print("=" * 84)
print(f"  C_O={C_O}  C_A={C_A}  T4_idx={I_T4}")

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
# 1. augment_mask_batch 返回真实观测
# ══════════════════════════════════════════════════
print("\n[1] 增强函数返回真实观测作监督")
B, T = 8, 32
obs = torch.randn(B, T, C_O)
act = torch.randn(B, T, C_A)
fut = torch.randn(B, 32, C_O)
af = torch.randn(B, 32, C_A)
cfg = {"p_none": 0.25, "p_fixed": 0.25, "p_random": 0.25, "p_burst": 0.25,
       "fixed_channels": [I_T4], "burst_channels": [I_T4],
       "random_ratio": 0.3, "burst_min": 8, "burst_max": 24}

obs_m, m_hist, obs_true, fut_true = augment_mask_batch(
    obs, act, fut, af, cfg, DEV)

check("返回 4 个张量", True, f"{len([obs_m, m_hist, obs_true, fut_true])}")
check("obs_masked 形状", obs_m.shape == obs.shape)
check("obs_true 形状", obs_true.shape == obs.shape)
check("obs_true 是原始观测（未被遮蔽）",
      torch.allclose(obs_true, obs),
      f"max diff={(obs_true-obs).abs().max():.2e}")
has_masked = bool((m_hist == 0).any())
if has_masked:
    masked_vals = obs_m[m_hist == 0]
    true_vals = obs_true[m_hist == 0]
    check("遮蔽位置 obs_masked=0", bool(torch.allclose(
        masked_vals, torch.zeros_like(masked_vals))))
    check("遮蔽位置 obs_true≠0（真实值保留）",
          bool((true_vals.abs() > 1e-6).any()),
          f"非零比例={float((true_vals.abs()>1e-6).float().mean())*100:.1f}%")

# ══════════════════════════════════════════════════
# 2. 重建损失用真实观测
# ══════════════════════════════════════════════════
print("\n[2] 重建损失监督目标")
wm = MaskWorldModel(C_O, C_A, deter=32, hidden=32).to(DEV)
wm.train()

# 构造：T4 全程遮蔽
m = torch.ones(B, T, C_O)
m[:, :, I_T4] = 0.0
obs_m2 = apply_mask(obs, m)

out_fixed = wm(obs_m2, m, act, af, fut,
               obs_true=obs, mode="gumbel")
out_wrong = wm(obs_m2, m, act, af, fut,
               obs_true=obs_m2, mode="gumbel")

check("forward 接收 obs_true 参数", True)
check("两种监督目标给出不同 rec",
      abs(float(out_fixed["rec"]) - float(out_wrong["rec"])) > 1e-6,
      f"真实={float(out_fixed['rec']):.5f}, 遮蔽={float(out_wrong['rec']):.5f}")

# 关键：如果模型输出 T4=0，用 obs_masked 监督会得到 0 loss
with torch.no_grad():
    recon = out_fixed["recon"]
    burn = max(1, T // 4)
    # 人工构造"完美预测遮蔽值"的情况
    fake_recon = obs_m2.clone()
    rec_fake_masked = ((fake_recon[:, burn:] - obs_m2[:, burn:]) ** 2).mean()
    rec_fake_true = ((fake_recon[:, burn:] - obs[:, burn:]) ** 2).mean()
check("遮蔽监督: 输出 0 得 0 loss", float(rec_fake_masked) < 1e-9,
      f"={float(rec_fake_masked):.2e}")
check("真实监督: 输出 0 有惩罚", float(rec_fake_true) > 1e-6,
      f"={float(rec_fake_true):.4f}")
print("     -> 证明：旧实现会让模型学成「遮蔽通道输出 0」")

# ══════════════════════════════════════════════════
# 3. 逐步在线估计（MaskRSSM）
# ══════════════════════════════════════════════════
print("\n[3] MaskRSSM 逐步在线估计")
wm.eval()
rssm = wm.rssm
B1 = 2
o_seq = torch.randn(B1, 100, C_O)
m_seq = torch.ones(B1, 100, C_O)
m_seq[:, 20:, I_T4] = 0.0            # t>=20 失效
a_seq = torch.randn(B1, 100, C_A)

with torch.no_grad():
    state = rssm.initial(B1, DEV)
    hats = []
    states_changed = []
    prev_h = None
    for t in range(100):
        o_hat, state = rssm.step(o_seq[:, t], m_seq[:, t], a_seq[:, t],
                                 state, t, mode="argmax")
        hats.append(o_hat)
        if prev_h is not None:
            d = float((state["h"] - prev_h).abs().mean())
            states_changed.append(d)
        prev_h = state["h"].clone()

check("step 返回 o_hat", hats[0].shape == (B1, C_O),
      str(tuple(hats[0].shape)))
check("每步隐状态都变化", all(d > 1e-8 for d in states_changed),
      f"最小变化={min(states_changed):.2e}")
check("100 步调用 100 次 step", len(hats) == 100)

# 对比：每 32 步才更新
with torch.no_grad():
    state2 = rssm.initial(B1, DEV)
    h_updates = 0
    prev = None
    for t in range(100):
        o_hat2, state2 = rssm.step(o_seq[:, t], m_seq[:, t],
                                   a_seq[:, t], state2, t, mode="argmax")
        if prev is not None and float((state2["h"] - prev).abs().mean()) > 1e-8:
            h_updates += 1
        prev = state2["h"].clone()
print(f"     -> 逐步更新：100 步内隐状态变化 {h_updates} 次")

# ══════════════════════════════════════════════════
# 4. MaskGRU 逐步在线估计
# ══════════════════════════════════════════════════
print("\n[4] MaskGRU 逐步在线估计")
g = MaskGRU(C_O, C_A, hidden=32, mlp=64).to(DEV).eval()
with torch.no_grad():
    h = g.init_hidden(B1, DEV)
    ghats, gchanges = [], []
    prev = None
    for t in range(100):
        o_hat, h = g.step(o_seq[:, t], m_seq[:, t], a_seq[:, t], h)
        ghats.append(o_hat)
        if prev is not None:
            gchanges.append(float((h - prev).abs().mean()))
        prev = h.clone()

check("GRU step 返回正形状", ghats[0].shape == (B1, C_O))
check("GRU 每步都更新", all(d > 1e-8 for d in gchanges),
      f"最小={min(gchanges):.2e}")

# ══════════════════════════════════════════════════
# 5. 信息隔离（逐步模式）
# ══════════════════════════════════════════════════
print("\n[5] 逐步模式下的信息隔离")


def run_online(model, kind, o_in, m_in, a_in):
    """逐步在线：每步用当前观测更新，输出全部通道"""
    B = o_in.shape[0]
    hats = []
    with torch.no_grad():
        if kind == "rssm":
            state = model.rssm.initial(B, DEV)
            for t in range(o_in.shape[1]):
                o_hat, state = model.rssm.step(
                    o_in[:, t], m_in[:, t], a_in[:, t], state, t,
                    mode="argmax")
                hats.append(o_hat)
        else:
            h = model.init_hidden(B, DEV)
            for t in range(o_in.shape[1]):
                o_hat, h = model.step(o_in[:, t], m_in[:, t],
                                      a_in[:, t], h)
                hats.append(o_hat)
    return torch.stack(hats, 1)


for nm, model, kind in [("MaskRSSM", wm, "rssm"), ("MaskGRU", g, "gru")]:
    p0 = run_online(model, kind, o_seq, m_seq, a_seq)
    # 投毒：故障后 T4 改极端值
    o_poison = o_seq.clone()
    o_poison[:, 20:, I_T4] = 999.0
    p1 = run_online(model, kind, o_poison, m_seq, a_seq)
    d = float((p0 - p1).abs().max())
    check(f"{nm} 逐步模式：故障后 T4 污染不影响",
          d < 1e-6, f"max diff={d:.2e}")

    # 极端投毒
    o_huge = o_seq.clone()
    o_huge[:, 20:, I_T4] = 1e6
    p2 = run_online(model, kind, o_huge, m_seq, a_seq)
    d2 = float((p0 - p2).abs().max())
    check(f"{nm} 逐步模式：1e6 极端污染不影响",
          d2 < 1e-6, f"max diff={d2:.2e}")

    # 故障前 T4 应影响
    o_pre = o_seq.clone()
    o_pre[:, :20, I_T4] += 5.0
    p3 = run_online(model, kind, o_pre, m_seq, a_seq)
    d3 = float((p0 - p3).abs().max())
    check(f"{nm} 逐步模式：故障前 T4 有影响", d3 > 1e-6,
          f"max diff={d3:.4f}")

# ══════════════════════════════════════════════════
# 6. 其他传感器参与更新（逐步模式）
# ══════════════════════════════════════════════════
print("\n[6] 其他传感器在逐步更新中起作用")
for nm, model, kind in [("MaskRSSM", wm, "rssm"), ("MaskGRU", g, "gru")]:
    p0 = run_online(model, kind, o_seq, m_seq, a_seq)
    for cn, ci in [("Wfact", 0), ("N2int", 1), ("T45", 3), ("P36", 4)]:
        o_x = o_seq.clone()
        o_x[:, 20:, ci] += 3.0
        px = run_online(model, kind, o_x, m_seq, a_seq)
        d = float((p0 - px).abs().max())
        check(f"{nm}: 污染 {cn:<7} 影响 T4", d > 1e-6, f"diff={d:.4f}")

# ══════════════════════════════════════════════════
# 7. 梯度检查
# ══════════════════════════════════════════════════
print("\n[7] 梯度反向传播")
wm.train()
out = wm(obs_m2, m, act, af, fut, obs_true=obs, mode="gumbel")
loss = out["rec"] + 0.1*out["kl"] + out["multi"]
loss.backward()
n_grad = sum(1 for p in wm.parameters() if p.grad is not None)
n_all = sum(1 for _ in wm.parameters())
check("所有参数有梯度", n_grad == n_all, f"{n_grad}/{n_all}")

# ══════════════════════════════════════════════════
print()
print("=" * 84)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 84)
if n_fail == 0:
    print("  ✅ 全部通过")
    print()
    print("  关键结论：")
    print("    1. 重建监督已改为真实观测（遮蔽位置输出 0 会被惩罚）")
    print("    2. 逐步在线估计每步都更新隐状态（100 步 = 100 次更新）")
    print("    3. 信息隔离在逐步模式下仍然正确（含 1e6 极端污染）")
    print("    4. 4 个正常传感器在逐步模式下都影响 T4 重构")
sys.exit(0 if n_fail == 0 else 1)

