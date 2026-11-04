"""诊断：为何逐步在线估计反而更差

观察（Mask-GRU 修复版）：
  永久失效  逐步 37.31 vs 分段 29.15  (-28%)
  缺失 5s   逐步 30.73 vs 分段 19.88  (-55%)
  缺失 10s  逐步 29.54 vs 分段 18.45  (-60%)
  缺失 30s  逐步 35.86 vs 分段 24.92  (-44%)
  test_ood 更差（-144% ~ -168%）

假设：
  H1. 隐状态漂移：逐步更新时 T4 缺失状态与训练分布不匹配
  H2. 误差累积：每步的预测误差反馈进隐状态（但逐步模式不回填预测）
  H3. 训练/推理不一致：训练时 T4 缺失是随机短片段，推理时是长期
  H4. 单条轨迹初始化问题：逐步模式从 t=0 开始，分段模式从 i_fail-W 开始
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))

from evaluate import UnifiedData, WINDOW, DT
from mask_impl_fixed import MaskGRU, MaskWorldModel, apply_mask

DEV = "cpu"
data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
SD_T4 = float(data.obs_sd[I_T4])
MU_T4 = float(data.obs_mu[I_T4])
MODE = "gumbel"

print("=" * 88)
print("诊断：逐步在线为何更差")
print("=" * 88)

# 加载修复版 MaskGRU
ck = Path(r"C:\pclwm\pclwm_mask\checkpoints_fixed\MaskGRU.pt")
if not ck.exists():
    print(f"  检查点不存在: {ck}")
    sys.exit(1)
g = MaskGRU(C_O, C_A, hidden=128, mlp=256).to(DEV)
g.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
g.eval()
print(f"  已加载 {ck.name}")


def online_stepwise(model, kind, rid, t_fail, dur=None, freeze_mode="zero"):
    """复现逐步在线，可控制缺失 T4 的填充方式"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    errs = []
    torch.manual_seed(42)
    with torch.no_grad():
        h = model.init_hidden(1, DEV)
        # 故障前：全传感器
        for t in range(i_fail):
            ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
            at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
            mt = torch.ones(1, C_O, device=DEV)
            _, h = model.step(ot[:, 0], mt, at, h)

        # 故障后：T4 mask=0
        last_t4 = o[i_fail-1, I_T4]
        for t in range(i_fail, i_end):
            ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
            # freeze_mode 决定 ot 里 T4 的实际值（都会被 mask 掉，但测敏感度）
            if freeze_mode == "freeze":
                ot[:, 0, I_T4] = torch.tensor(float(last_t4))
            at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
            mt = torch.ones(1, C_O, device=DEV)
            mt[:, I_T4] = 0.0
            o_hat, h = model.step(ot[:, 0], mt, at, h)
            e = float(o_hat[0, I_T4]) - float(o[t, I_T4])
            errs.append(e * SD_T4)
    return np.array(errs)


def online_segmented(model, kind, rid, t_fail, dur=None, horizon=32):
    """分段方案（每 32 步重编码历史）"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    cur_obs = o[i_fail-WINDOW:i_fail].copy()
    cur_mask = np.ones((WINDOW, C_O), np.float32)
    errs, t_idx = [], i_fail
    torch.manual_seed(42)
    with torch.no_grad():
        while t_idx < i_end:
            H = min(horizon, i_end - t_idx)
            if H <= 0: break
            obs_t = torch.from_numpy(cur_obs[None]).float().to(DEV)
            mask_t = torch.from_numpy(cur_mask[None]).float().to(DEV)
            act_t = torch.from_numpy(a[t_idx-WINDOW:t_idx][None]).float().to(DEV)
            af_t = torch.from_numpy(a[t_idx-1:t_idx-1+H][None]).float().to(DEV)
            p = model(obs_t, mask_t, act_t, af_t)[0].numpy()
            gt = o[t_idx:t_idx+H]
            errs.extend(((p[:, I_T4]-gt[:, I_T4])*SD_T4).tolist())
            nxt = gt.copy(); nxt[:, I_T4] = p[:, I_T4]
            cur_obs = np.concatenate([cur_obs, nxt], 0)[-WINDOW:]
            nm = np.ones((H, C_O), np.float32); nm[:, I_T4] = 0.0
            cur_mask = np.concatenate([cur_mask, nm], 0)[-WINDOW:]
            t_idx += H
    return np.array(errs)


# ══════════════════════════════════════════════════
# 1. 复现两种方案，逐轨迹对比
# ══════════════════════════════════════════════════
print("\n[1] 逐轨迹对比（永久失效 @20s）")
print(f"  {'轨迹':<22}{'逐步':>10}{'分段':>10}{'差异':>10}")
print("  " + "-" * 54)
for split in ["test_id", "test_ood"]:
    for rid in data.splits[split]:
        e_s = online_stepwise(g, "gru", rid, 20.0)
        e_g = online_segmented(g, "gru", rid, 20.0)
        if e_s is None or e_g is None:
            continue
        r_s = float(np.sqrt((e_s**2).mean()))
        r_g = float(np.sqrt((e_g**2).mean()))
        print(f"  {rid:<22}{r_s:>10.2f}{r_g:>10.2f}"
              f"{(r_s-r_g)/max(r_g,1e-9)*100:>9.0f}%")

# ══════════════════════════════════════════════════
# 2. 误差随时间演变
# ══════════════════════════════════════════════════
print("\n[2] 误差随故障后时间的演变（test_id 首条）")
rid = data.splits["test_id"][0]
e_s = online_stepwise(g, "gru", rid, 20.0)
e_g = online_segmented(g, "gru", rid, 20.0)
print(f"  {'区间':<12}{'逐步RMSE':>12}{'分段RMSE':>12}")
print("  " + "-" * 38)
for lo, hi in [(0,5),(5,15),(15,30),(30,60),(60,200)]:
    i0 = int(lo/DT); i1 = min(len(e_s), int(hi/DT))
    if i1 <= i0: continue
    a_ = float(np.sqrt((e_s[i0:i1]**2).mean()))
    i0g = int(lo/DT); i1g = min(len(e_g), int(hi/DT))
    b_ = float(np.sqrt((e_g[i0g:i1g]**2).mean())) if i1g > i0g else 0
    print(f"  {f'{lo}-{hi}s':<12}{a_:>12.2f}{b_:>12.2f}")

# ══════════════════════════════════════════════════
# 3. 隐状态漂移检测
# ══════════════════════════════════════════════════
print("\n[3] 隐状态漂移：逐步更新时隐状态范数演变")
torch.manual_seed(42)
o = data.norm_obs(rid).astype(np.float32)
a = data.norm_act(rid).astype(np.float32)
i_fail = int(round(20.0/DT))
with torch.no_grad():
    h = g.init_hidden(1, DEV)
    norms_pre, norms_post = [], []
    for t in range(i_fail):
        ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
        at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
        mt = torch.ones(1, C_O, device=DEV)
        _, h = g.step(ot[:, 0], mt, at, h)
        norms_pre.append(float(h.norm()))
    for t in range(i_fail, min(len(o), i_fail+300)):
        ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
        at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
        mt = torch.ones(1, C_O, device=DEV); mt[:, I_T4] = 0.0
        _, h = g.step(ot[:, 0], mt, at, h)
        norms_post.append(float(h.norm()))

print(f"  故障前 |h| 均值 = {np.mean(norms_pre):.4f} (std={np.std(norms_pre):.4f})")
print(f"  故障后 |h| 均值 = {np.mean(norms_post):.4f} (std={np.std(norms_post):.4f})")
print(f"  变化 {(np.mean(norms_post)-np.mean(norms_pre))/max(np.mean(norms_pre),1e-9)*100:+.1f}%")

# ══════════════════════════════════════════════════
# 4. 填充值敏感性
# ══════════════════════════════════════════════════
print("\n[4] 缺失 T4 的填充值敏感性（理论上应无影响）")
e_zero = online_stepwise(g, "gru", rid, 20.0, freeze_mode="zero")
e_free = online_stepwise(g, "gru", rid, 20.0, freeze_mode="freeze")
d = float(np.abs(e_zero - e_free).max())
print(f"  zero 填充 vs freeze 填充: max diff = {d:.2e}")
print(f"  -> {'✅ 无影响（mask 正确封闭）' if d < 1e-6 else '❌ 有影响（mask 失效）'}")

# ══════════════════════════════════════════════════
# 5. 关键对比：分段 vs 逐步的信息量
# ══════════════════════════════════════════════════
print("\n[5] 核心分析：为什么分段更好？")
print("""
  分段方案 (每 32 步)：
    - 每次用 32 步真实历史（含故障前的 T4）重编码
    - 未来 32 步纯外推
    - 感受野：32 步真实 T4 + 32 步外推

  逐步方案 (每步)：
    - 从 t=0 开始维护一个连续隐状态
    - 故障后 T4 一直是 mask=0
    - 隐状态中的 T4 信息随时间衰减（GRU 的遗忘）

  可能原因：
    1. 逐步方案中 T4 信息单调衰减，无法"刷新"
    2. 分段方案每 32 步重新注入真实 T4 历史（虽然是故障前的）
    3. 训练时 T4 缺失片段最长 32 步，推理时永久缺失 -> 分布外
""")

# ══════════════════════════════════════════════════
# 6. 混合方案：定期刷新 + 逐步更新
# ══════════════════════════════════════════════════
print("[6] 测试混合方案：每 N 步用真实历史刷新隐状态")
print(f"  {'刷新周期':<12}{'RMSE':>10}")
print("  " + "-" * 24)

def online_hybrid(model, rid, t_fail, refresh_every):
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    if i_fail < WINDOW or i_fail >= n - 5: return None
    errs = []
    torch.manual_seed(42)
    with torch.no_grad():
        # 初始：用故障前 WINDOW 步编码
        h = model.init_hidden(1, DEV)
        for t in range(i_fail - WINDOW, i_fail):
            ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
            at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1,-1)
            mt = torch.ones(1, C_O, device=DEV)
            _, h = model.step(ot[:, 0], mt, at, h)
        # 故障后：逐步 + 定期刷新（刷新用最近 WINDOW 步的预测填充历史）
        hist_obs = o[i_fail-WINDOW:i_fail].copy()
        hist_mask = np.ones((WINDOW, C_O), np.float32)
        for t in range(i_fail, n):
            ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
            at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1,-1)
            mt = torch.ones(1, C_O, device=DEV); mt[:, I_T4] = 0.0
            o_hat, h_new = model.step(ot[:, 0], mt, at, h)
            e = float(o_hat[0, I_T4]) - float(o[t, I_T4])
            errs.append(e * SD_T4)
            # 更新历史缓冲
            nxt = o[t:t+1].copy(); nxt[0, I_T4] = float(o_hat[0, I_T4])
            hist_obs = np.concatenate([hist_obs, nxt], 0)[-WINDOW:]
            nm = np.ones((1, C_O), np.float32); nm[0, I_T4] = 0.0
            hist_mask = np.concatenate([hist_mask, nm], 0)[-WINDOW:]
            # 定期刷新：用当前历史缓冲重新编码
            if refresh_every and (t - i_fail + 1) % refresh_every == 0:
                h = model.init_hidden(1, DEV)
                for k in range(WINDOW):
                    ok = torch.from_numpy(hist_obs[k:k+1][None]).float().to(DEV)
                    ak = torch.from_numpy(a[t-WINDOW+1+k:t-WINDOW+2+k][None]).float().to(DEV).reshape(1,-1)
                    mk = torch.from_numpy(hist_mask[k:k+1][None]).float().to(DEV)
                    _, h = model.step(ok[:, 0], mk[:, 0], ak, h)
            else:
                h = h_new
    return np.array(errs)

rids = data.splits["test_id"] + data.splits["test_ood"]
for re_ in [0, 64, 32, 16, 8]:
    vals = []
    for rid in rids:
        e = online_hybrid(g, rid, 20.0, re_)
        if e is not None and len(e) > 0:
            vals.append(float(np.sqrt((e**2).mean())))
    if vals:
        label = "纯逐步" if re_ == 0 else f"每 {re_} 步"
        print(f"  {label:<12}{np.mean(vals):>10.2f}")

print("\n" + "=" * 88)
print("诊断结论")
print("=" * 88)
print("""
  见上方各节输出。关键判断：
    1. 若逐轨迹差异一致 -> 系统性现象，非个别轨迹
    2. 若误差随时间单调增长 -> 隐状态漂移
    3. 若 |h| 显著变化 -> 分布外导致状态偏移
    4. 若混合方案（定期刷新）接近分段 -> 说明需要"信息刷新"
""")

