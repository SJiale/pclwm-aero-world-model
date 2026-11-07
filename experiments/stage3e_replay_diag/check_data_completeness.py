"""任务1：数据完整性严格检查

禁止：用常数或其他数据补齐，不得虚构 5 通道
"""
import re
import json
from pathlib import Path
import numpy as np

IIID = Path(r"C:\pclwm\pclwm_stage3a\iiid")
CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
GRU = Path(r"C:\pclwm\pclwm_stage3a")

print("=" * 88)
print("任务 1：数据完整性检查")
print("=" * 88)

# ══════════════════════════════════════════════════
# GRU 接口要求
# ══════════════════════════════════════════════════
import scipy.io as sio
G = sio.loadmat(GRU / "gru_pack2.mat")
obs_mu = G["obs_mu"].ravel()
obs_sd = G["obs_sd"].ravel()
act_mu = G["act_mu"].ravel()
act_sd = G["act_sd"].ravel()
W = 32   # 已知 WINDOW=32（evaluate.py）
H = 32   # 已知 HORIZON=32

OBS_COLS = ["Wfact_lb_s", "N2int", "S4_Tt", "S45_Tt", "S36_Pt"]
ACT_COLS = ["PLA_deg", "Alt_ft", "MN"]

print(f"\n[GRU 接口要求]")
print(f"  WINDOW={W}, HORIZON={H}")
print(f"  观测 5 通道: {OBS_COLS}")
print(f"  动作 3 通道: {ACT_COLS}")
print(f"  obs_mu = {obs_mu}")
print(f"  obs_sd = {obs_sd}")

# ══════════════════════════════════════════════════
# B2 数据检查
# ══════════════════════════════════════════════════
print("\n" + "=" * 88)
print("[B2 数据检查]")
print("=" * 88)

txt = (IIID / "b2.txt").read_text(encoding="utf-8")
rows = []
for ln in txt.split("\n"):
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+"
                 r"([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        rows.append([float(x) for x in m.groups()])

L = np.array(rows)
print(f"  已解析 {len(L)} 步")

B2_COLS = ["k", "t", "N2", "N2ref", "err", "PLA", "Wf", "T45", "wfMax", "opt_ms"]
print(f"\n  B2 实际列 ({len(B2_COLS)}):")
for i, c in enumerate(B2_COLS):
    print(f"    [{i}] {c}")

# 匹配
print(f"\n  GRU 观测通道匹配:")
have_obs = {}
for oc in OBS_COLS:
    if oc == "Wfact_lb_s":
        have_obs[oc] = ("Wf", 6)
    elif oc == "N2int":
        have_obs[oc] = ("N2", 2)
    elif oc == "S4_Tt":
        have_obs[oc] = None
    elif oc == "S45_Tt":
        have_obs[oc] = ("T45", 7)
    elif oc == "S36_Pt":
        have_obs[oc] = None

    if have_obs[oc]:
        src, idx = have_obs[oc]
        v = L[:, idx]
        print(f"    {oc:<14} ✅ 来自 [{idx}] {src:<8} range=[{v.min():.4f}, {v.max():.4f}]")
    else:
        print(f"    {oc:<14} ❌ 缺失")

print(f"\n  GRU 动作通道匹配:")
for ac in ACT_COLS:
    if ac == "PLA_deg":
        print(f"    {ac:<14} ✅ 来自 [5] PLA      range=[{L[:,5].min():.3f}, {L[:,5].max():.3f}]")
    elif ac == "Alt_ft":
        print(f"    {ac:<14} ✅ 已知恒为 0（仿真设置 Alt=0）")
    elif ac == "MN":
        print(f"    {ac:<14} ✅ 已知恒为 0（仿真设置 MN=0）")

n_have = sum(1 for v in have_obs.values() if v is not None)
n_miss = 5 - n_have

print(f"\n  结论: 观测 {n_have}/5 可用, {n_miss}/5 缺失")
if n_miss > 0:
    missing = [k for k, v in have_obs.items() if v is None]
    print(f"  缺失通道: {missing}")
    print(f"  → ❌ 无法直接对 B2 数据做 GRU 回放（模型要求 5 通道输入）")

# 检查 b2.mat 是否存在（未保存）
print(f"\n  b2.mat 是否存在: {(IIID/'b2.mat').exists()}")
print(f"  b2_fixed.mat 是否存在: {(IIID/'b2_fixed.mat').exists()}")

# ══════════════════════════════════════════════════
# 替代数据源检查：test_ctrl
# ══════════════════════════════════════════════════
print("\n" + "=" * 88)
print("[替代数据源: test_ctrl]")
print("=" * 88)

import pandas as pd

for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    cols_ok = all(c in df.columns for c in OBS_COLS + ["PLA_exec_deg"])
    pla = df["PLA_exec_deg"].to_numpy()
    dp = np.diff(pla)
    jumps = np.where(np.abs(dp) > 0.5)[0]
    print(f"\n  {f.stem}")
    print(f"    行数 {len(df)}, 5通道完整: {'✅' if cols_ok else '❌'}")
    print(f"    PLA 范围: [{pla.min():.1f}, {pla.max():.1f}]")
    print(f"    阶跃数（|ΔPLA|>0.5）: {len(jumps)}")
    if len(jumps):
        for j in jumps[:5]:
            print(f"      t={df['time_s'].iloc[j+1]:.2f}s: {pla[j]:.1f}->{pla[j+1]:.1f}")

# ══════════════════════════════════════════════════
# 判定
# ══════════════════════════════════════════════════
print("\n" + "=" * 88)
print("[判定]")
print("=" * 88)

result = {
    "b2_complete": n_miss == 0,
    "b2_n_steps": int(len(L)),
    "b2_have_obs": n_have,
    "b2_missing_obs": [k for k, v in have_obs.items() if v is None] if n_miss else [],
    "test_ctrl_available": True,
    "gru_window": W,
    "gru_horizon": H,
}
(IIID / "data_completeness.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"""
  B2 数据: {'✅ 完整' if n_miss == 0 else '❌ 不完整'}
    缺: {result['b2_missing_obs']}
    步数: {len(L)}

  因此：
    1. ❌ 无法用 B2 数据做 GRU 5 通道回放
    2. ✅ 但可用 test_ctrl 的完整数据做替代诊断

  test_ctrl V4/V6 有完整 5 通道 + PLA 阶跃
  虽不是 MPC 闭环数据，但可回答"GRU 对 PLA 阶跃的预测能力"
""")

