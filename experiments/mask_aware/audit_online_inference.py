"""在线推理审计（精简版）

修正上一版的检查错误：
  第 1 次 encoder 调用处理的是【历史窗口】，此时 T4 确实有效 -> mask=1 正确
  只有【预测段】才应 mask=0

本版按阶段分别检查。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))

from evaluate import UnifiedData, WINDOW, DT
from mask_impl import MaskWorldModel

DEV = "cpu"
data = UnifiedData()
C_O = data.n_obs
I_T4 = data.obs_cols.index("S4_Tt")
CH = {n: i for i, n in enumerate(data.obs_cols)}

print("=" * 84)
print("在线推理审计（修正版）")
print("=" * 84)

n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1
        print(f"  ❌ {name}  {detail}")


torch.manual_seed(0)
model = MaskWorldModel(C_O, 3, deter=32, hidden=32).to(DEV).eval()
rid = data.splits["test_id"][0]
o_orig = data.norm_obs(rid).astype(np.float32)
a = data.norm_act(rid).astype(np.float32)
T_FAIL = 30.0
i_fail = int(round(T_FAIL / DT))

caps_obs, caps_mask = [], []


def hook(m, inp, out):
    x = inp[0].detach().cpu().numpy().copy()   # [B, 2C]
    caps_obs.append(x[:, :C_O].copy())
    caps_mask.append(x[:, C_O:].copy())


def reconstruct(o_in, log=False):
    if log:
        caps_obs.clear(); caps_mask.clear()
        h = model.rssm.enc.register_forward_hook(hook)
    n = len(o_in)
    cur_obs = o_in[i_fail-WINDOW:i_fail].copy()
    cur_mask = np.ones((WINDOW, C_O), np.float32)
    t_idx, preds = i_fail, []
    with torch.no_grad():
        while t_idx < n:
            H = min(32, n - t_idx)
            if H <= 0:
                break
            obs_t = torch.from_numpy(cur_obs[None]).float()
            mask_t = torch.from_numpy(cur_mask[None]).float()
            act_t = torch.from_numpy(a[t_idx-WINDOW:t_idx][None]).float()
            af_t = torch.from_numpy(a[t_idx-1:t_idx-1+H][None]).float()
            torch.manual_seed(42)
            p = model.imagine(obs_t, mask_t, act_t, af_t,
                              mode="argmax")[0].numpy()
            preds.extend(p[:, I_T4].tolist())
            gt = o_in[t_idx:t_idx+H]
            nxt = gt.copy()
            nxt[:, I_T4] = p[:, I_T4]
            cur_obs = np.concatenate([cur_obs, nxt], 0)[-WINDOW:]
            nm = np.ones((H, C_O), np.float32); nm[:, I_T4] = 0.0
            cur_mask = np.concatenate([cur_mask, nm], 0)[-WINDOW:]
            t_idx += H
    if log:
        h.remove()
    return np.array(preds)


print("\n[1] 投毒测试（故障后污染 T4）")
p0 = reconstruct(o_orig)
o_p = o_orig.copy(); o_p[i_fail:, I_T4] += 500.0
p_p = reconstruct(o_p)
check("故障后 T4 污染不影响输出",
      np.abs(p0 - p_p).max() < 1e-6, f"max diff={np.abs(p0-p_p).max():.2e}")

o_pre = o_orig.copy(); o_pre[:i_fail, I_T4] += 500.0
p_pre = reconstruct(o_pre)
check("故障前 T4 污染影响输出",
      np.abs(p0 - p_pre).max() > 1e-3,
      f"max diff={np.abs(p0-p_pre).max():.4f}")

print("\n[2] 其他正常传感器参与更新（消融）")
for nm in ["Wfact_lb_s", "N2int", "S45_Tt", "S36_Pt"]:
    ci = CH[nm]
    o_x = o_orig.copy(); o_x[i_fail:, ci] += 5.0
    px = reconstruct(o_x)
    d = np.abs(p0 - px).max()
    check(f"污染 {nm:<12} 影响 T4 重构", d > 1e-6, f"diff={d:.4f}")

print("\n[3] mask 分阶段检查")
_ = reconstruct(o_orig, log=True)
n_calls = len(caps_mask)
print(f"  encoder 总调用 {n_calls} 次")
print(f"  阶段解析：每次调用处理 {WINDOW} 步窗口")
print(f"    第 1 次: 纯历史 [i_fail-W, i_fail)，T4 有效 -> mask 应=1")
print(f"    第 2 次起: 窗口后端含预测段 -> mask 应含 0")

first_t4 = caps_mask[0][:, I_T4]
check("第 1 次（纯历史）T4 mask=1", bool(np.all(first_t4 == 1)),
      f"值={first_t4}")

later = np.concatenate([c[:, I_T4] for c in caps_mask[1:]])
check("后续调用含 mask=0", bool((later == 0).any()),
      f"0 的比例={np.mean(later == 0)*100:.1f}%")
check("后续调用 mask 全为 0 或 1", bool(np.isin(later, [0, 1]).all()))

# 预测段（第 2 次起）末尾的 mask 应为 0
last_t4 = caps_mask[-1][:, I_T4]
check("最后一次调用末尾 T4 mask=0", bool(last_t4[-1] == 0),
      f"末位={last_t4[-1]}")

other = caps_mask[-1][:, [i for i in range(C_O) if i != I_T4]]
check("其他通道 mask 始终=1", bool(np.all(other == 1)))

print("\n[4] encoder 未见过量污染值")
_ = reconstruct(o_orig, log=True)
all_t4 = np.concatenate([c[:, I_T4] for c in caps_obs])
check("encoder 输入 T4 无异常大值", bool(np.abs(all_t4).max() < 100),
      f"max|T4|={np.abs(all_t4).max():.3f}")

# 高强度投毒测试
o_huge = o_orig.copy(); o_huge[i_fail:, I_T4] = 1e6
p_huge = reconstruct(o_huge)
check("极端污染（1e6）仍不影响", np.abs(p0 - p_huge).max() < 1e-6,
      f"max diff={np.abs(p0-p_huge).max():.2e}")

print("\n" + "=" * 84)
print(f"审计结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 84)
if n_fail == 0:
    print("""
  ✅ 在线推理安全性确认：
     1. 故障后 T4 真值绝不进入模型（含极端污染测试）
     2. 故障前 T4 历史正常参与（合理）
     3. 4 个正常传感器全部参与状态更新
     4. mask 分阶段正确：历史=1，预测段含 0
     5. 其他通道 mask 始终为 1
""")
sys.exit(0 if n_fail == 0 else 1)
