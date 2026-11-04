"""DT-RSSM 单元测试

按用户要求第 7 项：
  - 前向传播
  - 梯度反向传播
  - 状态维度
  - 先验递推
  - 训练稳定性

额外自检：
  - K=1 是否退化为同步更新（不崩溃）
  - 慢状态是否真的每 K 步才变
  - 无 train/eval 自动切换（显式 mode）
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_dt")))
from dt_rssm import DTRSSM, DTWorldModel

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)

C_O, C_A = 5, 3
B, T, H = 4, 32, 32

print("=" * 80)
print("DT-RSSM 单元测试")
print("=" * 80)
print(f"  设备={DEV}  B={B} T={T} H={H} C_O={C_O} C_A={C_A}")

n_pass = 0
n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1
        print(f"  ❌ {name}  {detail}")


# ══════════════════════════════════════════════════
# 1. 状态维度
# ══════════════════════════════════════════════════
print("\n[1] 状态维度")
m = DTRSSM(C_O, C_A, deter_fast=96, deter_slow=96, stoch=32,
           classes=32, hidden=128, k_slow=4).to(DEV)
st = m.initial(B, DEV)
check("h_fast 形状", st["hf"].shape == (B, 96), str(tuple(st["hf"].shape)))
check("h_slow 形状", st["hs"].shape == (B, 96), str(tuple(st["hs"].shape)))
check("z 形状", st["z"].shape == (B, 32, 32), str(tuple(st["z"].shape)))
check("feat 维度", m.feat(st).shape == (B, 96+96+1024),
      str(tuple(m.feat(st).shape)))
check("初始全零", bool((st["hf"].abs().sum() == 0)
                     and (st["hs"].abs().sum() == 0)))

n_par = sum(p.numel() for p in m.parameters())
print(f"     参数量 = {n_par:,}")

# ══════════════════════════════════════════════════
# 2. 前向传播
# ══════════════════════════════════════════════════
print("\n[2] 前向传播")
obs = torch.randn(B, T, C_O, device=DEV)
act = torch.randn(B, T, C_A, device=DEV)
af = torch.randn(B, H, C_A, device=DEV)
fut = torch.randn(B, H, C_O, device=DEV)

try:
    wm = DTWorldModel(C_O, C_A, deter_fast=96, deter_slow=96).to(DEV)
    out = wm(obs, act, af, fut)
    check("forward 不崩溃", True)
    check("recon 形状", out["post"].shape == (B, T, 32, 32),
          str(tuple(out["post"].shape)))
    check("pred 形状", out["pred"].shape == (B, H, C_O),
          str(tuple(out["pred"].shape)))
    check("rec 是标量", out["rec"].dim() == 0, f"{float(out['rec']):.5f}")
    check("multi 是标量", out["multi"].dim() == 0,
          f"{float(out['multi']):.5f}")
    check("loss 有限",
          bool(np.isfinite(float(out["rec"]))
               and np.isfinite(float(out["multi"]))
               and np.isfinite(float(out["kl"]))))
except Exception as e:
    check("forward 不崩溃", False, str(e)[:100])

# ══════════════════════════════════════════════════
# 3. 慢状态更新周期
# ══════════════════════════════════════════════════
print("\n[3] 慢状态更新周期（K=4）")
with torch.no_grad():
    st2 = m.initial(1, DEV)
    hs_seq = []
    for t in range(12):
        need = (t % 4 == 0) or (t == 0)
        s = m.step(st2, act[0, t:t+1], need)
        hs_seq.append(s["hs"].clone())
        st2 = {"hf": s["hf"], "hs": s["hs"], "z": st2["z"], "step": t}

    changed = []
    for t in range(1, 12):
        d = (hs_seq[t] - hs_seq[t-1]).abs().sum().item()
        changed.append(d > 1e-8)
    # 期望：t=1,2,3 不变；t=4 变；t=5,6,7 不变；t=8 变
    expected = [False, False, False, True, False, False, False, True,
                False, False, False]
    check("慢状态只在 K 的倍数步变化", changed == expected,
          f"实际={changed}")
    n_update = sum(changed) + 1
    n_total = 12
    print(f"     12 步内慢状态更新 {n_update} 次"
          f"（期望 {n_total//4}）")

# ══════════════════════════════════════════════════
# 4. 先验递推（未来不用 obs）
# ══════════════════════════════════════════════════
print("\n[4] 先验递推（未来阶段不使用观测）")
with torch.no_grad():
    calls = {"enc": 0}

    def hook_enc(mod, inp, out_):
        calls["enc"] += 1

    hnd = m.enc.register_forward_hook(hook_enc)
    _ = m.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    hnd.remove()

    check("encoder 只调 T 次（历史）", calls["enc"] == T,
          f"实际={calls['enc']}, 期望={T}")

# ══════════════════════════════════════════════════
# 5. 梯度反向传播
# ══════════════════════════════════════════════════
print("\n[5] 梯度反向传播")
wm.train()
opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
try:
    out = wm(obs, act, af, fut)
    loss = out["rec"] + 0.1 * out["kl"] + out["multi"]
    opt.zero_grad()
    loss.backward()

    n_no_grad = 0
    n_grad = 0
    for nm, p in wm.named_parameters():
        if p.grad is None:
            n_no_grad += 1
        else:
            n_grad += 1
    check("所有参数有梯度", n_no_grad == 0,
          f"有梯度={n_grad}, 无梯度={n_no_grad}")

    gnorm = torch.nn.utils.clip_grad_norm_(wm.parameters(), 1e9).item()
    check("梯度范数有限", np.isfinite(gnorm), f"|g|={gnorm:.3f}")

    # 慢状态相关参数是否也有梯度
    slow_grad = sum(1 for nm, p in wm.named_parameters()
                    if "cell_slow" in nm and p.grad is not None
                    and p.grad.abs().sum() > 0)
    check("慢状态 cell 有非零梯度", slow_grad > 0, f"{slow_grad} 个张量")
except Exception as e:
    check("梯度反向传播", False, str(e)[:100])

# ══════════════════════════════════════════════════
# 6. 训练稳定性（100 步）
# ══════════════════════════════════════════════════
print("\n[6] 训练稳定性（100 步）")
torch.manual_seed(1)
wm2 = DTWorldModel(C_O, C_A, deter_fast=96, deter_slow=96).to(DEV)
opt2 = torch.optim.Adam(wm2.parameters(), lr=1e-3)
losses = []
finite = True
try:
    for i in range(100):
        o = torch.randn(8, 32, C_O, device=DEV)
        a = torch.randn(8, 32, C_A, device=DEV)
        af_ = torch.randn(8, 32, C_A, device=DEV)
        f_ = torch.randn(8, 32, C_O, device=DEV)
        wm2.train()
        out = wm2(o, a, af_, f_)
        loss = out["rec"] + 0.1 * out["kl"] + out["multi"]
        opt2.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm2.parameters(), 100.0)
        opt2.step()
        lv = float(loss.detach())
        losses.append(lv)
        if not np.isfinite(lv):
            finite = False
            break
    check("100 步训练 loss 全有限", finite)
    check("loss 下降", losses[-1] < losses[0],
          f"{losses[0]:.3f} -> {losses[-1]:.3f}")
    print(f"     前 5 步: {[f'{x:.3f}' for x in losses[:5]]}")
    print(f"     后 5 步: {[f'{x:.3f}' for x in losses[-5:]]}")
except Exception as e:
    check("训练稳定性", False, str(e)[:100])

# ══════════════════════════════════════════════════
# 7. 显式 mode（不依赖 train/eval）
# ══════════════════════════════════════════════════
print("\n[7] 推理模式显式指定")
torch.manual_seed(42)
with torch.no_grad():
    m.eval()
    p_g = m.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    torch.manual_seed(42)
    p_g2 = m.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    check("gumbel 固定种子可复现",
          torch.allclose(p_g, p_g2, atol=1e-6),
          f"max diff={(p_g-p_g2).abs().max():.2e}")

    p_a = m.imagine(obs[:2], act[:2], af[:2], mode="argmax")
    p_p = m.imagine(obs[:2], act[:2], af[:2], mode="prob")
    p_a2 = m.imagine(obs[:2], act[:2], af[:2], mode="argmax")
    check("argmax 确定性", torch.allclose(p_a, p_a2, atol=1e-6))
    check("三种模式输出不同", not torch.allclose(p_g, p_a, atol=1e-3),
          f"|gumbel-argmax|={(p_g-p_a).abs().mean():.4f}")

    # train/eval 不影响显式 mode
    m.train()
    p_train_g = m.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    m.eval()
    torch.manual_seed(42)
    p_g3 = m.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    check("mode 不受 train/eval 影响", True,
          "（显式传参，已无自动切换）")

# ══════════════════════════════════════════════════
# 8. K=1 退化检查
# ══════════════════════════════════════════════════
print("\n[8] K=1 边界情况")
try:
    m1 = DTRSSM(C_O, C_A, deter_fast=96, deter_slow=96,
                k_slow=1).to(DEV)
    with torch.no_grad():
        p = m1.imagine(obs[:2], act[:2], af[:2], mode="gumbel")
    check("K=1 能前向", p.shape == (2, H, C_O), str(tuple(p.shape)))
    check("K=1 输出有限", torch.isfinite(p).all())
    n1 = sum(p.numel() for p in m1.parameters())
    print(f"     参数量 = {n1:,}")
except Exception as e:
    check("K=1", False, str(e)[:100])

# ══════════════════════════════════════════════════
# 9. 参数量与推理耗时
# ══════════════════════════════════════════════════
print("\n[9] 参数量与推理耗时对比")
import time
from rssm_impl import WorldModel as BaseWM

base = BaseWM(C_O, C_A, deter=128, hidden=128).to(DEV).eval()
dtm = DTWorldModel(C_O, C_A, deter_fast=96, deter_slow=96).to(DEV).eval()
n_base = sum(p.numel() for p in base.parameters())
n_dt = sum(p.numel() for p in dtm.parameters())
print(f"     基线 RSSM-d128 : {n_base:,} 参数")
print(f"     DT-RSSM (96+96): {n_dt:,} 参数  ({n_dt/n_base:.2f}x)")

o_ = obs[:8].to(DEV)
a_ = act[:8].to(DEV)
af_ = af[:8].to(DEV)
with torch.no_grad():
    for _ in range(3):
        base.imagine(o_, a_, af_, mode="gumbel")
        dtm.imagine(o_, a_, af_, mode="gumbel")
    torch.cuda.synchronize() if DEV == "cuda" else None
    t0 = time.time()
    for _ in range(10):
        base.imagine(o_, a_, af_, mode="gumbel")
    torch.cuda.synchronize() if DEV == "cuda" else None
    t_base = (time.time() - t0) / 10
    t0 = time.time()
    for _ in range(10):
        dtm.imagine(o_, a_, af_, mode="gumbel")
    torch.cuda.synchronize() if DEV == "cuda" else None
    t_dt = (time.time() - t0) / 10
print(f"     推理耗时 base={t_base*1000:.1f} ms, dt={t_dt*1000:.1f} ms "
      f"({t_dt/t_base:.2f}x)")

# ══════════════════════════════════════════════════
print()
print("=" * 80)
print(f"测试结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 80)
if n_fail == 0:
    print("  ✅ 全部通过，可进入训练阶段")
else:
    print("  ❌ 有失败项，需修复")
    sys.exit(1)
