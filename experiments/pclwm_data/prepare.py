"""阶段 1：统一数据接口 + 修正数据划分

修正的问题：
  - 之前划分只有 25 条，遗漏 C8
  - 旧划分在 dataset2（30 条旧数据）上做的，与新 26 条不匹配
  - 需要按完 整轨迹划分，防止时间窗口泄漏

输出：
  - C:\\pclwm\\pclwm_data\\prepared.npz  统一张量
  - C:\\pclwm\\pclwm_data\\splits.json   4 类划分
  - C:\\pclwm\\pclwm_data\\norm.npz      归一化参数（持久化）
  - C:\\pclwm\\pclwm_data\\SPEC.md       接口说明
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ABC = Path(r"C:\pclwm\abc_data")
OUT = Path(r"C:\pclwm\pclwm_data")
OUT.mkdir(parents=True, exist_ok=True)

DT_RAW = 0.015

# ══════════════════════════════════════════════════════
# 1. 确定训练用信号
# ══════════════════════════════════════════════════════
# 外部输入（已知、可控）
ACT_COLS = ["PLA_deg", "Alt_ft", "MN"]

# 预测目标（关键动态量）
OBS_COLS = [
    "Wfact_lb_s",   # 实际燃油
    "N2int",        # 高压轴转速
    "S4_Tt",        # 涡轮前温度 T4
    "S45_Tt",       # 高压涡轮出口 T45
    "S36_Pt",       # 压气机出口总压 P36
]

# 辅助（可用于物理约束，暂不作为主要目标）
AUX_COLS = ["N2dot", "N3dot", "Wfdmd_lb_s", "S36_Tt", "S5_Tt", "S0_W"]

# ══════════════════════════════════════════════════════
# 2. 采样率选择（依据 fig04 频谱结果）
# ══════════════════════════════════════════════════════
# 频谱分析结论：系统带宽 << 4.76 Hz（0.105s 的奈奎斯特）
# 保留 0.105 s（step=7）作为训练步长
STEP = 7
DT = DT_RAW * STEP     # 0.105 s
FS = 1.0 / DT          # 9.52 Hz

print("=" * 78)
print("阶段 1：数据接口与划分")
print("=" * 78)
print(f"  原始采样: {DT_RAW} s")
print(f"  训练步长: {DT:.3f} s (step={STEP})")
print(f"  采样频率: {FS:.2f} Hz, 奈奎斯特 {FS/2:.2f} Hz")

# ══════════════════════════════════════════════════════
# 3. 加载 26 条轨迹
# ══════════════════════════════════════════════════════
FILES = sorted(ABC.glob("*.csv"))
print(f"\n  找到 {len(FILES)} 条轨迹")

runs = {}
meta = {}
for p in FILES:
    num = int(p.name[:2])
    tag = p.stem[3:]
    df = pd.read_csv(p)
    if df.shape[1] < 10:
        print(f"  SKIP {tag}: 只有 {df.shape[1]} 列")
        continue

    # 降采样
    d = df.iloc[::STEP].copy()
    n = len(d)
    if n < 100:
        print(f"  SKIP {tag}: 降采样后仅 {n} 点")
        continue

    obs = d[OBS_COLS].to_numpy(np.float32)
    act = d[ACT_COLS].to_numpy(np.float32)
    aux = d[AUX_COLS].to_numpy(np.float32)
    t = d["time_s"].to_numpy(np.float64)

    # 有限性检查
    if not np.isfinite(obs).all() or not np.isfinite(act).all():
        print(f"  SKIP {tag}: 含非有限值")
        continue

    runs[tag] = {"obs": obs, "act": act, "aux": aux, "time": t,
                 "num": num}
    meta[tag] = {
        "num": num, "group": tag[0], "n_points": n,
        "obs_range": {c: [float(obs[:, i].min()), float(obs[:, i].max())]
                      for i, c in enumerate(OBS_COLS)},
        "act_range": {c: [float(act[:, i].min()), float(act[:, i].max())]
                      for i, c in enumerate(ACT_COLS)},
    }

print(f"  有效轨迹: {len(runs)} 条")
print(f"  每条点数: {set(len(v['time']) for v in runs.values())}")

# ══════════════════════════════════════════════════════
# 4. 数据划分（按完整轨迹）
# ══════════════════════════════════════════════════════
# 设计原则：
#   - 按组分层，保证每组都有 train/val/test
#   - test_ood 选复合/未见组合
SPLITS = {
    # A 组 12 条：train 9 / val 2 / test_id 1
    "train": [
        "A1_step_e1", "A2_aprbs_e1", "A3_msin_e1", "A4_ramp_e1",
        "A5_step_e2", "A6_aprbs_e2", "A7_msin_e2", "A8_ramp_e2",
    ],
    "val": ["A9_step_e3", "A10_aprbs_e3"],
    "test_id": ["A11_msin_e3"],
    "test_ood": ["A12_ramp_e3"],          # 未见：e3 环境的 ramp

    # B 组 6 条：train 3 / val 1 / test_id 1 / test_ood 1
    # （train 部分追加到上面）
}
# 重新整理（按 17/4/2/3 分配）
SPLITS = {
    "train": [
        # A 组 8 条
        "A1_step_e1", "A2_aprbs_e1", "A3_msin_e1", "A4_ramp_e1",
        "A5_step_e2", "A6_aprbs_e2", "A7_msin_e2", "A8_ramp_e2",
        # B 组 5 条
        "B1_alt_slow", "B2_alt_fast", "B3_mn_slow", "B4_mn_fast",
        "B5_flight_climb",
        # C 组 4 条
        "C1_seq_pla_first", "C2_seq_env_first", "C3_opposite",
        "C4_alternate",
    ],
    "val": [
        "A9_step_e3", "A10_aprbs_e3",       # A 组未见的 e3 环境
        "B6_flight_desc",                    # B 组下降飞行
        "C5_partial_overlap",                # C 组部分重叠
    ],
    "test_id": [
        "A11_msin_e3",                       # A 组 e3 多正弦
        "C6_mn_step",                        # C 组 MN 阶跃
    ],
    "test_ood": [
        "A12_ramp_e3",                       # A 组 e3 斜坡（组合未见）
        "C7_threeway",                       # C 组三者异步（最复杂）
        "C8_multiphase",                     # C 组 4 相位（完全未见）
    ],
}

# 校验
all_splits = [t for v in SPLITS.values() for t in v]
assert len(all_splits) == len(set(all_splits)), "划分有重复"
assert set(all_splits) == set(runs.keys()), \
    f"划分不完整: 缺 {set(runs.keys())-set(all_splits)}, " \
    f"多 {set(all_splits)-set(runs.keys())}"

print(f"\n  划分（共 {len(all_splits)} 条）:")
for k, v in SPLITS.items():
    print(f"    {k:<10} {len(v):>2} 条: {v}")

# ══════════════════════════════════════════════════════
# 5. 归一化（只用 train 拟合）
# ══════════════════════════════════════════════════════
tr_obs = np.concatenate([runs[t]["obs"] for t in SPLITS["train"]], 0)
tr_act = np.concatenate([runs[t]["act"] for t in SPLITS["train"]], 0)

obs_mu = tr_obs.mean(0).astype(np.float64)
obs_sd = tr_obs.std(0).astype(np.float64) + 1e-8
act_mu = tr_act.mean(0).astype(np.float64)
act_sd = tr_act.std(0).astype(np.float64) + 1e-8

print(f"\n  归一化参数（train 拟合）:")
print(f"    {'通道':<14}{'mean':>12}{'std':>12}")
for i, c in enumerate(OBS_COLS):
    print(f"    {c:<14}{obs_mu[i]:>12.3f}{obs_sd[i]:>12.3f}")

# ══════════════════════════════════════════════════════
# 6. 保存
# ══════════════════════════════════════════════════════
np.savez_compressed(
    OUT / "prepared.npz",
    **{f"{k}_obs": v["obs"] for k, v in runs.items()},
    **{f"{k}_act": v["act"] for k, v in runs.items()},
    **{f"{k}_aux": v["aux"] for k, v in runs.items()},
    **{f"{k}_time": v["time"] for k, v in runs.items()},
    obs_cols=np.array(OBS_COLS),
    act_cols=np.array(ACT_COLS),
    aux_cols=np.array(AUX_COLS),
    run_ids=np.array(list(runs.keys())),
)

np.savez_compressed(OUT / "norm.npz",
                    obs_mu=obs_mu, obs_sd=obs_sd,
                    act_mu=act_mu, act_sd=act_sd,
                    obs_cols=np.array(OBS_COLS),
                    act_cols=np.array(ACT_COLS))

(OUT / "splits.json").write_text(
    json.dumps({"splits": SPLITS, "step": STEP, "dt": DT,
                "fs": FS, "obs_cols": OBS_COLS, "act_cols": ACT_COLS,
                "aux_cols": AUX_COLS},
               ensure_ascii=False, indent=1), encoding="utf-8")

(OUT / "meta.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")

print(f"\n  保存:")
for f in ["prepared.npz", "norm.npz", "splits.json", "meta.json"]:
    sz = (OUT / f).stat().st_size / 1024
    print(f"    {f:<18} {sz:>8.1f} KB")
