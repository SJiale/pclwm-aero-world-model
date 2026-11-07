"""P0-1b：精确找出 RSSM 加载失败的具体张量"""
import sys
import torch
import inspect

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from rssm_impl import RSSM

CKPT = r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt"
sd = torch.load(CKPT, map_location="cpu", weights_only=True)
if "state_dict" in sd:
    sd = sd["state_dict"]
sd = {k[len("rssm."):] if k.startswith("rssm.") else k: v for k, v in sd.items()}

print("=" * 100)
print("精确诊断 RSSM 加载失败")
print("=" * 100)

m = RSSM(5, 3, deter=128, stoch=32, classes=32)
mod = m.state_dict()

print(f"\n  检查点张量数: {len(sd)}")
print(f"  模型张量数:   {len(mod)}")

print(f"\n  {'key':<26}{'检查点':>18}{'模型':>18}{'状态':>8}")
print("  " + "-" * 72)

all_keys = sorted(set(sd.keys()) | set(mod.keys()))
mismatch = []
for k in all_keys:
    ck = tuple(sd[k].shape) if k in sd else None
    md = tuple(mod[k].shape) if k in mod else None
    if ck == md:
        st = "✅"
    elif ck is None:
        st = "缺检查点"
        mismatch.append(k)
    elif md is None:
        st = "缺模型"
        mismatch.append(k)
    else:
        st = "形状不符"
        mismatch.append(k)
    if st != "✅" or k in ["enc.0.weight", "post.0.weight", "cell.weight_ih",
                            "prior.0.weight", "dec.0.weight"]:
        print(f"  {k:<26}{str(ck):>18}{str(md):>18}{st:>8}")

print(f"\n  不匹配的 key ({len(mismatch)}): {mismatch}")

# 若只有 hidden 相关不匹配，检查 hidden
if mismatch:
    print("\n  === 模型结构 ===")
    for name, mod_ in m.named_modules():
        if name and "." not in name:
            print(f"    {name}: {mod_}")

# 检查模型里 hidden 用在哪
print("\n  === 模型源码的 hidden 用法 ===")
src = inspect.getsource(RSSM.__init__)
for line in src.split("\n"):
    if "hidden" in line:
        print(f"    {line.strip()}")
