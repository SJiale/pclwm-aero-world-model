"""P0-1c：确认 RSSM 完全匹配"""
import sys
import torch

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from rssm_impl import RSSM

sd = torch.load(r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt",
                map_location="cpu", weights_only=True)
if "state_dict" in sd:
    sd = sd["state_dict"]
sd = {k[len("rssm."):] if k.startswith("rssm.") else k: v for k, v in sd.items()}

print("=" * 100)
print("确认 RSSM 超参数: deter=128, stoch=32, classes=32, hidden=128")
print("=" * 100)

m = RSSM(5, 3, deter=128, stoch=32, classes=32, hidden=128)
r = m.load_state_dict(sd, strict=True)
n = sum(p.numel() for p in m.parameters())

print(f"\n  ✅ strict=True 加载成功")
print(f"  参数量 = {n:,}")
print(f"  missing = {len(r.missing_keys)}")
print(f"  unexpected = {len(r.unexpected_keys)}")

# 验证权重确实不同（不是随机初始化）
import torch.nn as nn
m2 = RSSM(5, 3, deter=128, stoch=32, classes=32, hidden=128)
k = "enc.0.weight"
diff = (m.state_dict()[k] - m2.state_dict()[k]).abs().max()
print(f"\n  权重差异检查 ({k}): max|diff| = {diff:.6f}")
print(f"  -> {'✅ 权重已正确加载（非随机）' if diff > 1e-6 else '❌ 仍是随机'}")

# 保存真实超参数
import json
from pathlib import Path
hp = {
    "c_obs": 5, "c_act": 3,
    "deter": 128, "stoch": 32, "classes": 32, "hidden": 128,
    "n_params": n,
    "prefix_stripped": "rssm.",
    "load_strict": True,
}
out = Path(r"C:\pclwm\pclwm_stage3a\results")
out.mkdir(parents=True, exist_ok=True)
(out / "rssm_hyperparams.json").write_text(
    json.dumps(hp, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {out/'rssm_hyperparams.json'}")
