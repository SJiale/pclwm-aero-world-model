"""诊断 RSSM 的异常表现

现象：
  窗口级（test_id）：1-step=0.713, 32-step=0.712  <- 1-step 就错得离谱
  连续推演：0.447                              <- 反而最好

矛盾：如果 1 步预测就 0.71，连续推演不应优于 GRU(0.057)

可能原因：
  1. RSSM.imagine 的输入协议与我理解的不同（可能它内部先 observe 再 prior）
  2. RSSM 的 mode 参数（gumbel/post/prior）默认值不对
  3. 窗口级评价时我传的 act 顺序错了
"""
import sys
import numpy as np
import torch

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, DT
from rssm_impl import RSSM
from pc_rssm import PCWorldModel

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
obs_sd = np.asarray(data.obs_sd).ravel()
REF = float(ev.ref_std)
DEV = "cuda"

print("=" * 100)
print("诊断 RSSM")
print("=" * 100)

sd = torch.load(r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt",
                map_location=DEV, weights_only=True)
if "state_dict" in sd:
    sd = sd["state_dict"]

rssm = RSSM(C_O, C_A).to(DEV)
miss = rssm.load_state_dict(sd, strict=False)
print(f"  load_state_dict: missing={len(miss.missing_keys)} unexpected={len(miss.unexpected_keys)}")
if miss.missing_keys:
    print(f"    missing: {miss.missing_keys[:8]}")
if miss.unexpected_keys:
    print(f"    unexpected: {miss.unexpected_keys[:8]}")
rssm.eval()

# 看 imagine 源码
import inspect
print("\n=== RSSM.imagine 源码 ===")
src = inspect.getsource(RSSM.imagine)
for i, line in enumerate(src.split("\n"), 1):
    print(f"  {i:>3}: {line}")

print("\n=== RSSM.initial / observe 签名 ===")
for nm in ["initial", "observe", "prior", "feat"]:
    if hasattr(rssm, nm):
        try:
            print(f"  {nm}: {inspect.signature(getattr(rssm, nm))}")
        except Exception:
            pass

# ══════════════════════════════════════════════════
# 用真实窗口测试
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("真实窗口测试")
print("=" * 100)

ws = data.windows("test_id", stride=8)[:4]
obs = torch.from_numpy(np.stack([w["obs"] for w in ws])).float().to(DEV)
act = torch.from_numpy(np.stack([w["act"] for w in ws])).float().to(DEV)
af = torch.from_numpy(np.stack([w["act_fut"] for w in ws])).float().to(DEV)
fut = torch.from_numpy(np.stack([w["fut"] for w in ws])).float().to(DEV)

print(f"  obs={tuple(obs.shape)} act={tuple(act.shape)} af={tuple(af.shape)} fut={tuple(fut.shape)}")


def rel(p, t):
    d = (p - t) * torch.from_numpy(obs_sd).float().to(p.device)
    return float(((d ** 2).mean(-1).sqrt().mean()) / REF)


for mode in [None, "gumbel", "post", "prior", "argmax"]:
    try:
        torch.manual_seed(777)
        with torch.no_grad():
            p = rssm.imagine(obs, act, af, mode=mode)
        print(f"  mode={str(mode):<8} 输出={tuple(p.shape)}  rel={rel(p, fut):.5f}  "
              f"有限={bool(torch.isfinite(p).all())}")
    except Exception as e:
        print(f"  mode={str(mode):<8} ❌ {type(e).__name__}: {str(e)[:60]}")

# 检查预测值本身
print("\n  预测 vs 真值（第一个窗口，前 3 步）:")
torch.manual_seed(777)
with torch.no_grad():
    p = rssm.imagine(obs, act, af, mode="gumbel")
print(f"    预测[0,:3] = {p[0,:3].cpu().numpy()}")
print(f"    真值[0,:3] = {fut[0,:3].cpu().numpy()}")
print(f"    历史末[0]  = {obs[0,-1].cpu().numpy()}")

# 关键：预测是否接近"历史末值的复制"？
last = obs[:, -1:].expand(-1, p.shape[1], -1)
print(f"\n  如果预测=历史末值复制，rel = {rel(last, fut):.5f}")
mean_pred = p.mean(0, keepdim=True).expand_as(p)
print(f"  预测的时间均值 rel = {rel(mean_pred, fut):.5f}")
