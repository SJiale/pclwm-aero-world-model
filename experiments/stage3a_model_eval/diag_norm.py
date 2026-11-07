"""P0-2 关键核查：动作归一化是否一致？

这是最可能的高误差根源：
  如果 test_ctrl 的动作归一化与训练数据不同，
  模型收到的动作就是错的 -> 预测必然崩坏
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm"]

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, DT

data = UnifiedData()
ev = UnifiedEval(data)

print("=" * 100)
print("P0-2 核查：动作归一化")
print("=" * 100)

# ══════════════════════════════════════════════════
# 1. UnifiedData 的实际动作归一化
# ══════════════════════════════════════════════════
print("\n[1] UnifiedData 的窗口动作值（训练数据）")
ws = data.windows("test_id", stride=8)
w = ws[0]
print(f"  act[0] = {w['act'][0]}")
print(f"  act[-1] = {w['act'][-1]}")
print(f"  act_fut[0] = {w['act_fut'][0]}")

# 反推归一化规律
print("\n  原始 CSV 中的对应值:")
DATA = Path(r"C:\pclwm\abc_data")
rid = w.get("rid", "A11_msin_e3")
print(f"    rid = {rid}")
cands = list(DATA.glob(f"*{rid}*.csv"))
if cands:
    df = pd.read_csv(cands[0])
    t0 = w.get("t0", 0)
    print(f"    t0 = {t0}")
    print(f"    PLA[{t0}:{t0+3}] = {df['PLA_deg'].iloc[t0:t0+3].to_numpy()}")
    print(f"    Alt[{t0}:{t0+3}] = {df['Alt_ft'].iloc[t0:t0+3].to_numpy()}")
    print(f"    MN[{t0}:{t0+3}]  = {df['MN'].iloc[t0:t0+3].to_numpy()}")

    # 验证假设
    pla = df["PLA_deg"].iloc[t0]
    alt = df["Alt_ft"].iloc[t0]
    mn = df["MN"].iloc[t0]
    a = w["act"][0]
    print(f"\n    验证归一化公式:")
    print(f"      PLA={pla:.3f} -> act[0]={a[0]:.6f}")
    print(f"        PLA/100       = {pla/100:.6f}  {'✅' if abs(pla/100-a[0])<1e-4 else '❌'}")
    print(f"        (PLA-55)/45   = {(pla-55)/45:.6f}")
    print(f"      Alt={alt:.1f} -> act[1]={a[1]:.6f}")
    print(f"        Alt/25000     = {alt/25000:.6f}  {'✅' if abs(alt/25000-a[1])<1e-4 else '❌'}")
    print(f"      MN={mn:.3f} -> act[2]={a[2]:.6f}")
    print(f"        MN 原值       = {mn:.6f}  {'✅' if abs(mn-a[2])<1e-4 else '❌'}")

# ══════════════════════════════════════════════════
# 2. test_ctrl 的动作范围（归一化后）
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[2] test_ctrl 归一化后的动作范围")
print("=" * 100)

CTRL = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7

print(f"\n  {'轨迹':<22}{'PLA原':>16}{'PLA归一':>18}{'Alt归一':>10}{'MN归一':>10}")
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    pla = d["PLA_cmd_deg"].to_numpy()
    alt = d["Alt_ft"].to_numpy()
    mn = d["MN"].to_numpy()
    pn = pla / 100.0
    an = alt / 25000.0
    print(f"  {f.stem:<22}[{pla.min():>5.1f},{pla.max():>5.1f}]"
          f"[{pn.min():>7.4f},{pn.max():>7.4f}]"
          f"{an[0]:>10.4f}{mn[0]:>10.4f}")

# 训练数据动作范围对比
print(f"\n  训练数据动作范围（归一化后）:")
all_act = np.concatenate([w["act"] for w in data.windows("train", stride=8)])
print(f"    PLA归一: [{all_act[:,0].min():.4f}, {all_act[:,0].max():.4f}]")
print(f"    Alt归一: [{all_act[:,1].min():.4f}, {all_act[:,1].max():.4f}]")
print(f"    MN归一:  [{all_act[:,2].min():.4f}, {all_act[:,2].max():.4f}]")

# ══════════════════════════════════════════════════
# 3. ★ 关键：test_ctrl 起始状态是否在训练分布内
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[3] ★ 观测归一化：test_ctrl vs 训练数据")
print("=" * 100)

obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()

print(f"\n  {'轨迹':<22}" + "".join(f"{c[:10]:>12}" for c in data.obs_cols))
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    d = df.iloc[::STEP].reset_index(drop=True)
    raw = d[["Wfact_lb_s", "N2int", "S4_Tt", "S45_Tt", "S36_Pt"]].to_numpy()
    # 起始 32 步的归一化后均值
    o32 = raw[:32]
    z = (o32.mean(0) - obs_mu) / obs_sd
    print(f"  {f.stem:<22}" + "".join(f"{v:>12.3f}" for v in z))

# 训练数据
print(f"\n  训练数据起始窗口（归一化后均值绝对值）:")
ws_tr = data.windows("train", stride=64)
ztr = np.array([w["obs"].mean(0) for w in ws_tr])
print(f"    {'':<22}" + "".join(f"{v:>12.3f}" for v in np.abs(ztr).mean(0)))
print(f"    最大值: {'':<15}" + "".join(f"{v:>12.3f}" for v in np.abs(ztr).max(0)))

# ══════════════════════════════════════════════════
# 4. test_ctrl 是否从冷启动开始（未稳定）
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[4] test_ctrl 是否从稳态开始？")
print("=" * 100)

print(f"\n  {'轨迹':<22}{'N2[0]':>10}{'N2[last]':>10}{'N2变化':>12}{'前10步N2':>14}")
for f in sorted(CTRL.glob("V*.csv")):
    df = pd.read_csv(f)
    n2 = df["N2int"].to_numpy()
    d = n2[::STEP]
    print(f"  {f.stem:<22}{n2[0]:>10.1f}{n2[-1]:>10.1f}"
          f"{n2[-1]-n2[0]:>12.1f}"
          f"[{d[:10].min():>6.1f},{d[:10].max():>6.1f}]")

# 训练数据的对应值
print(f"\n  训练数据（前 10 步 N2 波动）:")
for rid in list(data.splits["train"])[:3]:
    c = list(DATA.glob(f"*{rid}*.csv"))
    if c:
        df = pd.read_csv(c[0])
        n2 = df["N2int"].to_numpy()[::STEP]
        print(f"    {rid:<20} N2[0]={n2[0]:>8.1f}  前10步波动="
              f"[{n2[:10].min():.1f},{n2[:10].max():.1f}]")
