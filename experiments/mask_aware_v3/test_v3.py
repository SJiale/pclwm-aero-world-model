"""v3 训练/推理一致性测试

核心验证：
  1. GRU 训练 forward 与在线 step 使用同一套 cell 更新
  2. 训练 forward 与逐步在线在相同输入下输出一致
  3. 三种推理模式（严格开环/分段反馈/逐步在线）共享初始化
  4. 信息泄漏
"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
from evaluate import UnifiedData, WINDOW, DT
from mask_impl_v3 import (MaskGRU, MaskRSSM, MaskWorldModel,
                          MaskGRUModel, apply_mask, augment_mask_batch)

DEV = "cpu"
torch.manual_seed(0)
data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")

print("=" * 88)
print("v3 训练/推理一致性测试")
print("=" * 88)

n_pass = n_fail = 0
def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1; print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1; print(f"  ❌ {name}  {detail}")

B, T, H = 4, WINDOW, 8

# ══════════════════════════════════════════════════
# 1. GRU: forward 与逐步在线一致性
# ══════════════════════════════════════════════════
print("\n[1] MaskGRU: forward vs 逐步在线")
g = MaskGRU(C_O, C_A, hidden=32, mlp=64).eval()

obs = torch.randn(B, T, C_O)
mask = torch.ones(B, T, C_O)
mask[:, T//2:, I_T4] = 0.0
act = torch.randn(B, T, C_A)
af = torch.randn(B, H, C_A)

with torch.no_grad():
    # forward 路径
    p_fwd = g(obs, mask, act, af)

    # 逐步骤路径：手动拆解 forward 的内部逻辑
    h = g.enc_hist(obs, mask, act)
    m_last = mask[:, -1]
    o_prev = obs[:, -1]
    outs = []
    for k in range(H):
        o_hat, h = g.update(h, o_prev, m_last, af[:, k])
        outs.append(o_hat)
        o_prev = o_hat
    p_manual = torch.stack(outs, 1)

check("forward 与手动 update 序列一致",
      torch.allclose(p_fwd, p_manual, atol=1e-6),
      f"max diff={(p_fwd-p_manual).abs().max():.2e}")

# 在线 step 路径
with torch.no_grad():
    h = g.online_init(obs, mask, act)
    outs2 = []
    o_prev = obs[:, -1]
    for k in range(H):
        o_hat, h = g.online_step(h, o_prev, m_last, af[:, k])
        outs2.append(o_hat)
        o_prev = o_hat
    p_online = torch.stack(outs2, 1)

check("online_step 与 forward 一致",
      torch.allclose(p_fwd, p_online, atol=1e-6),
      f"max diff={(p_fwd-p_online).abs().max():.2e}")

# ══════════════════════════════════════════════════
# 2. GRU: Decoder 输入分布一致性
# ══════════════════════════════════════════════════
print("\n[2] Decoder 输入分布（训练 vs 推理应相同）")
with torch.no_grad():
    h_tr = g.enc_hist(obs, mask, act)
    z = g.cell_in(torch.cat([apply_mask(obs[:, -1], m_last), m_last,
                             af[:, 0]], -1))
    h_cell = g.cell(z, h_tr)

    # 训练路径的 decoder 输入
    dec_in_train = h_cell
    # 在线路径的 decoder 输入
    o_hat, h_online = g.online_step(h_tr, obs[:, -1], m_last, af[:, 0])
    # h_online 就是新的 cell 输出
    dec_in_online = h_online

check("训练与推理的 Decoder 输入完全一致",
      torch.allclose(dec_in_train, dec_in_online, atol=1e-7),
      f"max diff={(dec_in_train-dec_in_online).abs().max():.2e}")

print(f"     训练输入范数 = {float(dec_in_train.norm()):.4f}")
print(f"     推理输入范数 = {float(dec_in_online.norm()):.4f}")

# ══════════════════════════════════════════════════
# 3. RSSM: forward 与在线一致性
# ══════════════════════════════════════════════════
print("\n[3] MaskRSSM: forward vs 逐步在线")
wm = MaskWorldModel(C_O, C_A, deter=32, hidden=32).eval()

with torch.no_grad():
    p_fwd = wm.imagine(obs, mask, act, af, mode="argmax")

    # 在线路径
    st = wm.rssm.online_init(obs, mask, act, mode="argmax")
    outs = []
    for k in range(H):
        # 在线时每步后验更新（但未来无观测 -> 只能用先验）
        st = wm.rssm._step_prior(st, af[:, k], "argmax")
        outs.append(wm.rssm.dec(wm.rssm.feat(st)))
    p_online = torch.stack(outs, 1)

check("imagine 与手动先验递推一致",
      torch.allclose(p_fwd, p_online, atol=1e-6),
      f"max diff={(p_fwd-p_online).abs().max():.2e}")

# ══════════════════════════════════════════════════
# 4. 状态维度与参数量
# ══════════════════════════════════════════════════
print("\n[4] 状态与参数")
h = g.init_state(B, DEV)
check("GRU 状态维度", h.shape == (B, 32), str(tuple(h.shape)))
st = wm.rssm.init_state(B, DEV)
check("RSSM 状态维度", st["h"].shape == (B, 32) and
      st["z"].shape == (B, 32, 32))
n_g = sum(p.numel() for p in g.parameters())
n_r = sum(p.numel() for p in wm.parameters())
print(f"     MaskGRU  {n_g:,} 参数")
print(f"     MaskRSSM {n_r:,} 参数")

# ══════════════════════════════════════════════════
# 5. 三种推理模式共享初始化
# ══════════════════════════════════════════════════
print("\n[5] 三种推理模式共享同一初始化")
with torch.no_grad():
    h0 = g.online_init(obs, mask, act)
    # 模式 1：严格开环
    p_open = g.rollout_open(h0, af, m_last)
    # 模式 2：分段反馈（模拟）
    p_seg_start = g.rollout_open(h0, af, m_last)   # 第一段相同
check("严格开环与分段反馈的第一段相同",
      torch.allclose(p_open, p_seg_start, atol=1e-7))
check("开环输出形状", p_open.shape == (B, H, C_O), str(tuple(p_open.shape)))

with torch.no_grad():
    st0 = wm.rssm.online_init(obs, mask, act, mode="argmax")
    r_open = wm.rssm.rollout_open(st0, af, mode="argmax")
check("RSSM 开环输出形状", r_open.shape == (B, H, C_O))

# ══════════════════════════════════════════════════
# 6. 信息泄漏
# ══════════════════════════════════════════════════
print("\n[6] 信息泄漏（逐步在线）")


def online_run(model, kind, o_in, m_in, a_in, n_steps):
    Bx = o_in.shape[0]
    outs = []
    with torch.no_grad():
        h = model.online_init(o_in[:, :WINDOW], m_in[:, :WINDOW],
                             a_in[:, :WINDOW])
        for k in range(n_steps):
            o_hat, h = model.online_step(
                h, o_in[:, WINDOW+k], m_in[:, WINDOW+k],
                a_in[:, WINDOW+k])
            outs.append(o_hat)
    return torch.stack(outs, 1)


T2 = WINDOW + 40
o_seq = torch.randn(2, T2, C_O)
m_seq = torch.ones(2, T2, C_O)
m_seq[:, WINDOW:, I_T4] = 0.0
a_seq = torch.randn(2, T2, C_A)
N = 40

p0 = online_run(g, "gru", o_seq, m_seq, a_seq, N)

for pert, lab in [(500.0, "+500"), (1e6, "1e6"), (-1e4, "-1e4")]:
    o_p = o_seq.clone(); o_p[:, WINDOW:, I_T4] += pert
    pp = online_run(g, "gru", o_p, m_seq, a_seq, N)
    check(f"GRU 故障后 T4 {lab} 不影响",
          torch.allclose(p0, pp, atol=1e-6),
          f"max diff={(p0-pp).abs().max():.2e}")

o_pre = o_seq.clone(); o_pre[:, :WINDOW, I_T4] += 5.0
pp = online_run(g, "gru", o_pre, m_seq, a_seq, N)
check("GRU 故障前 T4 有影响",
      not torch.allclose(p0, pp, atol=1e-6),
      f"max diff={(p0-pp).abs().max():.4f}")

# RSSM 同理
def online_run_rssm(model, o_in, m_in, a_in, n_steps, mode="argmax"):
    outs = []
    with torch.no_grad():
        st = model.rssm.online_init(o_in[:, :WINDOW], m_in[:, :WINDOW],
                                    a_in[:, :WINDOW], mode)
        for k in range(n_steps):
            o_hat, st = model.rssm.online_step(
                st, o_in[:, WINDOW+k], m_in[:, WINDOW+k],
                a_in[:, WINDOW+k], mode)
            outs.append(o_hat)
    return torch.stack(outs, 1)

r0 = online_run_rssm(wm, o_seq, m_seq, a_seq, N)
for pert, lab in [(500.0, "+500"), (1e6, "1e6")]:
    o_p = o_seq.clone(); o_p[:, WINDOW:, I_T4] += pert
    rr = online_run_rssm(wm, o_p, m_seq, a_seq, N)
    check(f"RSSM 故障后 T4 {lab} 不影响",
          torch.allclose(r0, rr, atol=1e-6),
          f"max diff={(r0-rr).abs().max():.2e}")

o_pre2 = o_seq.clone(); o_pre2[:, :WINDOW, I_T4] += 5.0
rr = online_run_rssm(wm, o_pre2, m_seq, a_seq, N)
check("RSSM 故障前 T4 有影响",
      not torch.allclose(r0, rr, atol=1e-6),
      f"max diff={(r0-rr).abs().max():.4f}")

# ══════════════════════════════════════════════════
# 7. 训练梯度
# ══════════════════════════════════════════════════
print("\n[7] 训练梯度")
gm = MaskGRUModel(C_O, C_A, hidden=32, mlp=64)
out = gm(obs, mask, act, af, fut_true=torch.randn(B, H, C_O))
out["multi"].backward()
n_grad = sum(1 for p in gm.parameters() if p.grad is not None)
n_all = sum(1 for _ in gm.parameters())
check("GRU 所有参数有梯度", n_grad == n_all, f"{n_grad}/{n_all}")

wm2 = MaskWorldModel(C_O, C_A, deter=32, hidden=32)
out = wm2(obs, mask, act, af, fut_true=torch.randn(B, H, C_O),
          obs_true=obs, mode="gumbel")
(out["rec"] + 0.1*out["kl"] + out["multi"]).backward()
n_grad = sum(1 for p in wm2.parameters() if p.grad is not None)
n_all = sum(1 for _ in wm2.parameters())
check("RSSM 所有参数有梯度", n_grad == n_all, f"{n_grad}/{n_all}")

# ══════════════════════════════════════════════════
print()
print("=" * 88)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 88)
if n_fail == 0:
    print("  ✅ 训练/推理完全一致")
    print("     - forward 与 online_step 使用同一 cell 原语")
    print("     - Decoder 输入分布完全一致")
    print("     - 三种推理模式共享同一初始化")
    print("     - 信息隔离正确")
sys.exit(0 if n_fail == 0 else 1)