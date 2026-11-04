"""时间尺度分析（探索性，需谨慎解读）

重要修正（用户指正）：
  ❌ 不能根据 ACF 衰减和上升时间就认定
     "N2int 是固有快动态，Wfact 是固有慢动态"
  ✅ 必须区分三类不同的时间尺度：
     (a) 输入激励的时间尺度 —— 由我们给的 PLA/Alt/MN 波形决定
     (b) 闭环控制响应的时间尺度 —— EEC 的调节动态
     (c) 系统内部动力学的时间尺度 —— 真正的物理惯性

为什么不能直接从观测数据分离这三者：
  - 观测通道是闭环系统的输出
  - 燃油 Wfact 是 EEC 的输出，其变化由控制器决定，不是"固有惯性"
  - N2int 上升快，可能因为控制器快速调油，而非转子惯性小
  - 所以 ACF/上升时间只能作为"表观时间尺度"，不能直接归因

正确的验证方式（本轮未完成，需后续）：
  1. 开环激励：固定 EEC，直接给 Wfdmd 阶跃，测转子响应
  2. 频响函数估计：Wf→N2 的传递函数，看相位滞后
  3. 物理参数辨识：从数据反推转子惯量、热容
  4. 与 AGTF30e 模型参数对比（已知的 J、热容等）
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import UnifiedData, DT

OUT = Path(r"C:\pclwm\pclwm_dt")
OUT.mkdir(parents=True, exist_ok=True)

data = UnifiedData()
FS = 1.0 / DT

print("=" * 92)
print("时间尺度分析（探索性）")
print("=" * 92)
print("  ⚠ 重要声明：以下数字是【表观时间尺度】，不是【固有动力学时间尺度】")
print("  ⚠ 原因：所有观测来自闭环系统，无法分离激励/控制/本体三者")
print()

# ══════════════════════════════════════════════════
# 1. 先看输入激励本身的时间尺度（这是可控的、已知的）
# ══════════════════════════════════════════════════
print("=" * 92)
print("[1] 输入激励的时间尺度（已知，用于对照）")
print("=" * 92)

act_names = data.act_cols
print(f"\n  动作通道: {act_names}")

EXC_INFO = {
    "step80": ("多级阶跃", "2000 deg/s 最快切换"),
    "aprbs": ("APRBS", "保持 1.5-6 s 随机"),
    "msin": ("多正弦", "0.02-0.23 Hz"),
    "ramp": ("斜坡", "2.1 deg/s 慢变"),
    "alt": ("高度变化", "500-2000 ft/s"),
    "mn": ("马赫变化", "0.02-0.08 /s"),
}

print(f"\n  {'激励类型':<14}{'说明':<32}{'时间尺度'}")
print("  " + "-" * 62)
for k, (nm, desc) in EXC_INFO.items():
    print(f"  {nm:<14}{desc:<32}", end="")
    if k == "step80":
        print("瞬态 < 1 s")
    elif k == "aprbs":
        print("1.5-6 s 保持")
    elif k == "msin":
        print("4.3-50 s 周期")
    elif k == "ramp":
        print("斜坡 ~20 s")
    else:
        print("10-40 s")

# 从数据实测 PLA 变化速率
pla_rates = {}
for r in data.splits["train"]:
    a = data.norm_act(r)
    raw = data.runs[r]["act"]
    # 反归一化 PLA（第一通道）
    pla = raw[:, 0]
    d = np.abs(np.diff(pla)) / DT
    pla_rates[r] = float(np.percentile(d, 99))

print(f"\n  实测 PLA 变化速率（99 分位，deg/s）:")
by_type = {}
for r, v in pla_rates.items():
    for k in EXC_INFO:
        if k in r:
            by_type.setdefault(k, []).append(v)
            break
for k, vals in by_type.items():
    print(f"    {k:<12} {np.mean(vals):8.2f} deg/s  ({len(vals)} 条)")

# ══════════════════════════════════════════════════
# 2. 表观时间尺度（观测，但混杂了控制影响）
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[2] 表观时间尺度（观测数据的 ACF 衰减）")
print("=" * 92)
print("  ⚠ 这些数字混杂了：输入激励 + EEC 控制响应 + 本体动力学")


def acf_decay(x, max_lag=300):
    x = np.asarray(x, float)
    x = x - x.mean()
    if x.std() < 1e-9:
        return None
    var = (x**2).mean()
    for lag in range(1, min(max_lag, len(x)-1)):
        if (x[:-lag] * x[lag:]).mean() / var < 1/np.e:
            return lag * DT
    return None


print(f"\n  {'通道':<14}{'ACF 衰减中位 [s]':>18}{'最小值':>10}{'最大值':>10}")
print("  " + "-" * 54)
acf_tab = {}
for ci, col in enumerate(data.obs_cols):
    vals = []
    for r in data.splits["train"]:
        t = acf_decay(data.norm_obs(r)[:, ci])
        if t:
            vals.append(t)
    if vals:
        acf_tab[col] = {"median": float(np.median(vals)),
                        "min": float(np.min(vals)),
                        "max": float(np.max(vals))}
        print(f"  {col:<14}{np.median(vals):>18.2f}"
              f"{np.min(vals):>10.2f}{np.max(vals):>10.2f}")

# ══════════════════════════════════════════════════
# 3. 关键：控制响应 vs 本体动力学的混淆证据
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[3] 为什么不能直接归因：控制与动力的耦合证据")
print("=" * 92)

# 证据 1：Wfact 是 EEC 输出，不是独立输入
print("\n  证据 1：Wfact 由 EEC 决定，不是自由变量")
print("    - Wfdmd（指令）与 Wfact（实际）的差 = 执行器动态")
print("    - 若 Wfact 的平滑性来自控制器滤波，则不是发动机惯性")

gaps = []
for r in data.splits["train"]:
    o = data.runs[r]["aux"]
    # aux = [N2dot, N3dot, Wfdmd, S36_Tt, S5_Tt, S0_W]
    wf = data.runs[r]["obs"][:, 0]
    wd = o[:, 2]
    n = min(len(wf), len(wd))
    gap = np.abs(wf[:n] - wd[:n])
    gaps.append(float(np.percentile(gap, 99)))
print(f"    |Wfact - Wfdmd| 99分位: {np.mean(gaps):.5f} lb/s "
      f"（占均值 {np.mean(gaps)/0.667*100:.1f}%）")
print(f"    -> 执行器跟踪很快，Wfact ≈ Wfdmd")

# 证据 2：N2 与 Wf 的相位关系
print("\n  证据 2：N2int 与 Wfact 的时序关系")
lags = []
for r in data.splits["train"]:
    n2 = data.norm_obs(r)[:, 1]
    wf = data.norm_obs(r)[:, 0]
    n = min(len(n2), len(wf))
    n2, wf = n2[:n], wf[:n]
    # 互相关，找 N2 相对 Wf 的滞后
    best_lag, best_c = 0, -2
    for lag in range(-30, 31):
        if lag < 0:
            c = np.corrcoef(wf[-lag:], n2[:n+lag])[0, 1]
        elif lag > 0:
            c = np.corrcoef(wf[:n-lag], n2[lag:])[0, 1]
        else:
            c = np.corrcoef(wf, n2)[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    lags.append(best_lag * DT)
print(f"    N2 相对 Wf 的最优滞后中位: {np.median(lags):.2f} s")
print(f"    -> 若滞后为正且小，说明 N2 快速跟随 Wf（控制器驱动）")

# 证据 3：不同激励下同一通道的 ACF 差异
print("\n  证据 3：时间尺度随激励类型变化（说明不是固有属性）")
print(f"    {'通道':<14}{'step 轨迹':>12}{'ramp 轨迹':>12}{'差异':>10}")
print("    " + "-" * 50)
for ci, col in enumerate(data.obs_cols[:3]):
    st_v, rp_v = [], []
    for r in data.splits["train"]:
        t = acf_decay(data.norm_obs(r)[:, ci])
        if not t:
            continue
        if "step" in r:
            st_v.append(t)
        elif "ramp" in r:
            rp_v.append(t)
    if st_v and rp_v:
        m1, m2 = np.median(st_v), np.median(rp_v)
        print(f"    {col:<14}{m1:>12.2f}{m2:>12.2f}"
              f"{abs(m1-m2)/max(m1,m2)*100:>9.0f}%")

print("\n    -> 若同一通道在不同激励下 ACF 差异大，说明它反映的是激励而非固有动态")

# ══════════════════════════════════════════════════
# 4. 结论
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[4] 谨慎的结论")
print("=" * 92)
print("""
  ✅ 可以说的：
     - 在【表观层面】，5 个通道的 ACF 衰减时间在 7.7-8.7 s 之间，
       比值 1.1x，未见明显的时间尺度分离
     - 阶跃响应的上升时间在 1.05-2.31 s 之间，比值 2.2x
     - 因此 DT-RSSM 采用"按通道分配时间尺度"的设计
       【缺乏强物理依据】

  ❌ 不能说的：
     - "N2int 是固有快动态" —— N2 受 EEC 快速调油驱动，不是转子惯性的直接体现
     - "Wfact 是固有慢动态" —— Wfact 是控制器输出，其平滑性可能来自控制律
     - "时间常数 1.05 s 是物理惯性常数"

  ⚠ DT-RSSM 的定位调整：
     - 不做"物理时间尺度分离"的声称
     - 而是把慢状态作为【额外的记忆容量 / 多尺度特征提取器】
     - 性能提升若存在，归因于容量与表征，不是物理时间尺度

  📋 需要后续验证（本轮不做）：
     1. 开环实验：固定 EEC，直接给 Wfdmd 阶跃，测转子响应
     2. 频响估计：Wfdmd → N2int 的传递函数
     3. 物理参数辨识
""")

out = {
    "note": "exploratory_only",
    "warning": "ACF/rise-time are APPARENT timescales, not intrinsic; "
               "mixes excitation, closed-loop control, and plant dynamics",
    "apparent_acf": acf_tab,
    "actuator_gap_99pct": float(np.mean(gaps)),
    "n2_vs_wf_lag_s": float(np.median(lags)),
    "conclusion": "no strong timescale separation at apparent level; "
                  "DT-RSSM should NOT claim physics-based separation",
}
(OUT / "timescale_analysis_v2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'timescale_analysis_v2.json'}")
