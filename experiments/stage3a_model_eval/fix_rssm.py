"""P0-1：修复 RSSM 检查点加载

从检查点反推超参数：
  post.0.weight (128, 256) -> deter + embed_dim = 256
  prior.0.weight (128, 128) -> deter = 128
  cell.weight_ih (384, 1027) -> 3*deter = 384 (deter=128), 1027 = stoch*classes + c_act
  dec.0.weight (128, 1152) -> feat_dim = 1152
"""
import sys
import torch
import inspect

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from rssm_impl import RSSM

print("=" * 100)
print("P0-1：反推 RSSM 超参数")
print("=" * 100)

CKPT = r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt"
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
if "state_dict" in sd:
    sd = sd["state_dict"]

# 去前缀
sd = {k[len("rssm."):] if k.startswith("rssm.") else k: v for k, v in sd.items()}

print("\n  去前缀后的完整张量列表:")
for k, v in sd.items():
    print(f"    {k:<26} {tuple(v.shape)}")

# ══════════════════════════════════════════════════
# 反推
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("反推超参数")
print("=" * 100)

C_O, C_A = 5, 3

deter = sd["cell.weight_hh"].shape[1]
print(f"  deter = {deter}   (cell.weight_hh 第二维)")

n_gate_rows = sd["cell.weight_ih"].shape[0]
print(f"  cell 输出维 = {n_gate_rows}  = 3*deter = {3*deter}  "
      f"{'✅' if n_gate_rows == 3*deter else '❌'}")

cell_in = sd["cell.weight_ih"].shape[1]
print(f"  cell 输入维 = {cell_in}")
print(f"    其中 c_act = {C_A}")
print(f"    -> stoch*classes = {cell_in - C_A}")

# prior 输出 = stoch*classes
prior_out = sd["prior.2.weight"].shape[0]
print(f"  prior 输出 = {prior_out}  (stoch*classes)")

# 推断 stoch / classes
# 常见：(8,32)=256, (16,32)=512, (32,32)=1024, (4,32)=128
for stoch, classes in [(8, 32), (16, 32), (32, 32), (4, 32), (8, 16), (4, 16)]:
    if stoch * classes == prior_out:
        print(f"  ✅ 推断: stoch={stoch}, classes={classes}")

prior_in = sd["prior.0.weight"].shape[1]
print(f"  prior 输入 = {prior_in}  = deter = {deter}  "
      f"{'✅' if prior_in == deter else '❌'}")

# embed 维度
post_in = sd["post.0.weight"].shape[1]
print(f"  post 输入 = {post_in} = deter + embed = {deter} + {post_in - deter}")
embed_dim = post_in - deter

enc_out = sd["enc.2.weight"].shape[0]
print(f"  enc 输出 = {enc_out} (embed_dim)  {'✅' if enc_out == embed_dim else '❌'}")

# dec 输入
dec_in = sd["dec.0.weight"].shape[0]
feat_dim = sd["dec.0.weight"].shape[1]
print(f"  dec 输入 = {feat_dim} (feat_dim)")
print(f"    feat = h({deter}) + z({prior_out}) = {deter + prior_out}  "
      f"{'✅' if feat_dim == deter + prior_out else '❌'}")

# ══════════════════════════════════════════════════
# 用反推的超参数构造模型
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("用反推超参数加载")
print("=" * 100)

print("\n  RSSM.__init__ 签名:")
print("   ", inspect.signature(RSSM.__init__))

# 尝试构造
attempts = [
    dict(deter=deter, stoch=8, classes=32),
    dict(deter=deter, stoch=16, classes=32),
    dict(deter=deter, stoch=32, classes=32),
    dict(deter=deter, stoch=4, classes=32),
]

for kw in attempts:
    try:
        m = RSSM(C_O, C_A, **kw)
        r = m.load_state_dict(sd, strict=True)
        n = sum(p.numel() for p in m.parameters())
        print(f"  ✅ {kw}  strict=True 成功!  参数={n:,}")
        break
    except Exception as e:
        msg = str(e).split("\n")[0][:80]
        print(f"  ❌ {kw}: {msg}")
