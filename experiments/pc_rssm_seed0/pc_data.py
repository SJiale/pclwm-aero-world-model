"""PC-RSSM 数据加载：把物理标签（N2dot/N3dot）接入现有窗口

关键约束：
  1. 物理标签只在【训练损失】中使用
  2. 推理/评价阶段完全不读物理标签
  3. 归一化参数只用 train 拟合
  4. 时间对齐必须与 prepared.npz 完全一致

轴对应（已核实）：
  N2dot → 低压轴（LP: 风扇+LPC+LPT），rpm/s
  N3dot → 高压轴（HP: HPC+HPT），rpm/s
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import UnifiedData, WINDOW, HORIZON

PHYS_DIR = Path(r"C:\pclwm\phys_labels")
STEP = 7          # 降采样步长（与 prepared.npz 一致）
DT = 0.105

# 物理目标列（labels.csv 中的列名）
PHYS_COLS = ["loop_N2dot", "loop_N3dot"]
PHYS_NAMES = ["N2dot_LP", "N3dot_HP"]   # 修正后的物理命名


def load_phys(run_ids):
    """加载物理标签并按 step=7 降采样

    返回 dict: rid -> np.array [n, 2]
    """
    # 从 phys_labels 目录匹配轨迹名
    files = {f.stem.replace(".labels", ""): f
             for f in PHYS_DIR.glob("*.labels.csv")}
    # 文件名格式: 01_A1_step_e1.labels.csv -> tag = A1_step_e1
    tag2file = {}
    for stem, f in files.items():
        # stem 形如 "01_A1_step_e1"
        parts = stem.split("_", 1)
        if len(parts) == 2 and parts[0].isdigit():
            tag2file[parts[1]] = f
        else:
            tag2file[stem] = f

    out = {}
    missing = []
    for rid in run_ids:
        if rid not in tag2file:
            missing.append(rid)
            continue
        df = pd.read_csv(tag2file[rid])
        arr = df[PHYS_COLS].to_numpy(np.float64)[::STEP]
        out[rid] = arr.astype(np.float32)
    if missing:
        print(f"  ⚠ 缺失物理标签: {missing}")
    return out


def fit_phys_norm(phys, train_ids):
    """归一化参数只用 train 拟合"""
    arrs = [phys[r] for r in train_ids if r in phys]
    if not arrs:
        raise RuntimeError("无 train 物理标签")
    A = np.concatenate(arrs, 0)
    mu = A.mean(0).astype(np.float64)
    sd = A.std(0).astype(np.float64) + 1e-8
    return mu, sd


class PCData:
    """带物理标签的窗口构造"""

    def __init__(self, data=None):
        self.d = data if data is not None else UnifiedData()
        self.phys_raw = load_phys(self.d.run_ids)
        self.phys_mu, self.phys_sd = fit_phys_norm(
            self.phys_raw, self.d.splits["train"])

        # 打印归一化参数
        print("  物理标签归一化参数（train 拟合）:")
        for i, nm in enumerate(PHYS_NAMES):
            print(f"    {nm:<10} mu={self.phys_mu[i]:>10.3f}  "
                  f"sd={self.phys_sd[i]:>10.3f}")

    def norm_phys(self, rid):
        return (self.phys_raw[rid] - self.phys_mu) / self.phys_sd

    def windows(self, split, stride=4, horizon=HORIZON):
        """构造带物理监督的窗口

        每个窗口含：
          obs      [T, C_O]  历史观测（归一化）
          act      [T, C_A]
          fut      [H, C_O]  未来观测（归一化，作预测目标）
          act_fut  [H, C_A]
          phys_now [T, 2]    历史段物理量（归一化，静态监督用）
          phys_fut [H, 2]    未来段物理量（归一化，动态监督目标）
          rid, t0
        """
        ids = self.d.splits[split] if isinstance(split, str) else split
        W = []
        for r in ids:
            if r not in self.phys_raw:
                continue
            o = self.d.norm_obs(r).astype(np.float32)
            a = self.d.norm_act(r).astype(np.float32)
            p = self.norm_phys(r).astype(np.float32)
            t = self.d.runs[r]["time"]
            n = len(o)
            n_p = len(p)
            # 长度对齐检查
            if abs(n - n_p) > 2:
                print(f"  ⚠ {r} 长度不对齐: obs={n} phys={n_p}")
            n_use = min(n, n_p)
            for st in range(1, n_use - WINDOW - horizon, stride):
                W.append({
                    "obs": o[st:st+WINDOW],
                    "act": a[st-1:st-1+WINDOW],
                    "fut": o[st+WINDOW:st+WINDOW+horizon],
                    "act_fut": a[st-1+WINDOW:st-1+WINDOW+horizon],
                    "phys_now": p[st:st+WINDOW],
                    "phys_fut": p[st+WINDOW:st+WINDOW+horizon],
                    "rid": r,
                    "t0": float(t[st+WINDOW]),
                })
        return W


# 物理目标的正规化（供监督头使用）
PHYS_KEYS = ["phys_now", "phys_fut"]


if __name__ == "__main__":
    print("=" * 84)
    print("PC-RSSM 数据加载自检")
    print("=" * 84)

    pd_ = PCData()
    d = pd_.d
    print(f"\n  轨迹数: {len(d.run_ids)}")
    print(f"  有物理标签: {len(pd_.phys_raw)}")

    # 长度对齐检查
    print("\n  长度对齐检查:")
    bad = 0
    for rid in d.run_ids:
        if rid not in pd_.phys_raw:
            continue
        n_o = len(d.runs[rid]["obs"])   # 已是降采样后长度
        n_p = len(pd_.phys_raw[rid])
        if abs(n_o - n_p) > 2:
            print(f"    ⚠ {rid}: obs={n_o} phys={n_p}")
            bad += 1
    print(f"    不对齐: {bad} / {len(pd_.phys_raw)}")

    # 构造窗口
    tr = pd_.windows("train", stride=4)
    va = pd_.windows("val", stride=8)
    te = pd_.windows("test_id", stride=8)
    ood = pd_.windows("test_ood", stride=8)
    print(f"\n  窗口数: train={len(tr)} val={len(va)} "
          f"te={len(te)} ood={len(ood)}")

    # 检查一个窗口
    w = tr[0]
    print(f"\n  窗口0: {w['rid']} t0={w['t0']:.3f}")
    for k in ["obs", "act", "fut", "act_fut", "phys_now", "phys_fut"]:
        print(f"    {k:<10} shape={w[k].shape}")

    # ★ 时间对齐验证
    print("\n  ★ 时间对齐验证")
    rid = w["rid"]
    o = d.norm_obs(rid)
    p = pd_.norm_phys(rid)
    pos = int(round(w["t0"] / DT))
    c1 = np.allclose(w["obs"], o[pos-WINDOW:pos], atol=1e-5)
    c2 = np.allclose(w["phys_now"], p[pos-WINDOW:pos], atol=1e-5)
    c3 = np.allclose(w["fut"], o[pos:pos+HORIZON], atol=1e-5)
    c4 = np.allclose(w["phys_fut"], p[pos:pos+HORIZON], atol=1e-5)
    print(f"    obs      = o[pos-W:pos]      {c1}")
    print(f"    phys_now = p[pos-W:pos]      {c2}")
    print(f"    fut      = o[pos:pos+H]      {c3}")
    print(f"    phys_fut = p[pos:pos+H]      {c4}")

    # 物理量统计
    P = np.concatenate([w["phys_now"] for w in tr], 0)
    print(f"\n  物理目标统计（归一化后，train）：")
    for i, nm in enumerate(PHYS_NAMES):
        print(f"    {nm:<10} mean={P[:,i].mean():>8.4f} sd={P[:,i].std():>8.4f} "
              f"range=[{P[:,i].min():.2f}, {P[:,i].max():.2f}]")

    # 物理量自相关（检查是否真的动态）
    print(f"\n  物理量自相关（lag=1，train 首条）:")
    for rid in d.splits["train"][:1]:
        pp = pd_.norm_phys(rid)
        for i, nm in enumerate(PHYS_NAMES):
            ac = np.corrcoef(pp[:-1, i], pp[1:, i])[0, 1]
            print(f"    {nm:<10} lag1 ACF={ac:>8.4f}")

    ok = c1 and c2 and c3 and c4 and bad == 0
    print(f"\n  自检: {'✅ 通过' if ok else '❌ 失败'}")

