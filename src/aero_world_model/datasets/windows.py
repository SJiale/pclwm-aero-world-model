"""
数据加载器：把多工况轨迹切成 (历史 obs, 动作, 未来 obs) 窗口

═══════════════════════════════════════════════════════════════════════════
关键设计决策（依据本课题实测）
═══════════════════════════════════════════════════════════════════════════

【决策 1】window 长度 = 512 步（10.24 s @ dt=0.02）
    依据：[实测] 最长时间常数 τ_max = 5.51 s（Np/Load）
    3·τ_max = 16.5 s = 825 步
    512 步 = 10.24 s ≈ 1.86·τ_max，覆盖约 84% 的阶跃响应；
    1024 步 = 20.5 s ≈ 3.7·τ_max，完整覆盖。
    标准 DreamerV3 的 batch_length=64（1.28 s）只覆盖 0.23·τ_max，**远远不够**。

【决策 2】按 run 划分，窗口绝不跨 run
    禁止跨轨迹拼接：不同 run 的边界处物理状态不连续。

【决策 3】每条轨迹只用激励段（t >= 200 s）
    理由：[实测] 0-123 s 是所有轨迹共有的起动段（内容相同）。
    保留 t>=200 s 可确保：稳态已建立 + 不重复训练相同内容。
    但保留 200-350 s 的稳态段，让模型学到"稳态应保持不变"。

【决策 4】归一化：只在 train 上拟合，保存 scaler 参数
    用 (x - mean) / std，按通道独立。
    不用 min-max（对异常值敏感，而 T45 有已知的起动峰值）。

【决策 5】动作与观测的时间对齐
    严格按 a_t -> o_{t+1}：
        obs[t]   = [Ng, Np, TIT, T45, Load] 在时刻 t
        act[t]   = fuel 在 [t, t+1) 生效
        obs[t+1] = 下一时刻状态
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["WindowDataset", "Scaler", "load_runs", "make_datasets"]


# ══════════════════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════════════════

OBS_COLS = ["Ng_rpm", "Np_rpm", "TIT_K", "T45_K", "T5_K", "Load_W"]
# ⚠ 动作必须是【指令】，不是系统的实际执行值（自审修正）
#   [实测] Fuel_kg_s 是燃调系统输出，含起动逻辑+速率限制，
#   在 t>=200s 时仍滞后于指令 0.03-0.07 kg/s（13/18 条轨迹验证）。
#   正确的动作是 FuelCmd_kg_s（From Workspace 输入）。
ACT_COLS = ["FuelCmd_kg_s"]
# 实际燃油作为【观测】的一部分（系统状态）
EXTRA_OBS_COLS = ["Fuel_kg_s"]


@dataclass
class DataConfig:
    window: int = 512            # 历史窗口步数（决策 1）
    horizon: int = 16            # 多步预测步数
    stride: int = 64             # 窗口滑动步长（避免过度重叠）
    dt: float = 0.1              # 目标采样间隔（合并数据为 0.1 s）
    keep_after_s: float = 200.0  # 只保留该时刻之后（决策 3）
    obs_cols: list[str] = None
    act_cols: list[str] = None
    include_actual_fuel: bool = True   # 是否把实际燃油也作为观测

    def __post_init__(self):
        if self.obs_cols is None:
            cols = list(OBS_COLS)
            if self.include_actual_fuel:
                cols = cols + list(EXTRA_OBS_COLS)
            self.obs_cols = cols
        if self.act_cols is None:
            self.act_cols = list(ACT_COLS)


# ══════════════════════════════════════════════════════════════════════════
# 归一化
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class Scaler:
    """按通道的 z-score 归一化。只在 train 上拟合。"""
    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @classmethod
    def fit(cls, obs: np.ndarray, act: np.ndarray) -> "Scaler":
        """obs: [N,C_o], act: [N,C_a]"""
        o_std = obs.std(0)
        a_std = act.std(0)
        o_std[o_std < 1e-8] = 1.0
        a_std[a_std < 1e-8] = 1.0
        return cls(obs.mean(0), o_std, act.mean(0), a_std)

    def transform_obs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.obs_mean) / self.obs_std

    def transform_act(self, x: np.ndarray) -> np.ndarray:
        return (x - self.act_mean) / self.act_std

    def inverse_obs(self, x: np.ndarray) -> np.ndarray:
        return x * self.obs_std + self.obs_mean

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "act_mean": self.act_mean.tolist(),
            "act_std": self.act_std.tolist(),
        }, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(np.array(d["obs_mean"]), np.array(d["obs_std"]),
                   np.array(d["act_mean"]), np.array(d["act_std"]))


# ══════════════════════════════════════════════════════════════════════════
# 数据读取
# ══════════════════════════════════════════════════════════════════════════

def load_runs(
    raw_dirs: list[Path],
    cfg: DataConfig,
) -> dict[str, dict[str, np.ndarray]]:
    """从多个 raw 目录读取轨迹，下采样到统一 dt。

    Returns:
        {run_id: {"obs": [T,C_o], "act": [T,C_a], "time": [T]}}
    """
    out: dict[str, dict[str, np.ndarray]] = {}
    for raw in raw_dirs:
        if not Path(raw).exists():
            continue
        for d in sorted(Path(raw).iterdir()):
            if not d.is_dir():
                continue
            rid = d.name
            try:
                # 读所有需要的列
                cols = {}
                t_ref = None
                for name in cfg.obs_cols + cfg.act_cols:
                    f = d / f"{name}.csv"
                    if not f.exists():
                        raise FileNotFoundError(f)
                    v = pd.read_csv(f)
                    tcols = v.iloc[:, 0].to_numpy()
                    if t_ref is None:
                        t_ref = tcols
                    if len(tcols) != len(t_ref):
                        raise ValueError(f"{name} 长度不一致")
                    cols[name] = v.iloc[:, 1].to_numpy()

                t = t_ref
                # 下采样：按 dt 网格取最近点
                src_dt = float(np.median(np.diff(t)))
                step = int(round(cfg.dt / src_dt))
                if step < 1:
                    step = 1
                idx = np.arange(0, len(t), step)
                t_ds = t[idx]

                # 只保留 keep_after_s 之后
                keep = t_ds >= cfg.keep_after_s - 1e-9
                idx = idx[keep]
                t_ds = t_ds[keep]
                if len(idx) < cfg.window + cfg.horizon + 2:
                    print(f"  [SKIP] {rid}: 采后点数 {len(idx)} 不足")
                    continue

                obs = np.stack([cols[n][idx] for n in cfg.obs_cols], axis=1)
                act = np.stack([cols[n][idx] for n in cfg.act_cols], axis=1)

                out[rid] = {"obs": obs.astype(np.float32),
                            "act": act.astype(np.float32),
                            "time": t_ds.astype(np.float64)}
            except Exception as e:
                print(f"  [FAIL] {rid}: {e}")
    return out


# ══════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════

class WindowDataset(Dataset):
    """滑动窗口数据集。

    每个样本：
        obs   [T, C_o]   历史观测（已归一化）
        act   [T, C_a]   动作（已归一化）。act[t] 作用于 t -> t+1
        fut   [H, C_o]   未来 H 步的真实观测（已归一化），用于多步损失

    T = cfg.window, H = cfg.horizon
    """

    def __init__(self, runs: dict[str, dict[str, np.ndarray]],
                 scaler: Scaler, cfg: DataConfig):
        self.cfg = cfg
        self.samples: list[tuple[str, int]] = []
        self.data: dict[str, dict[str, np.ndarray]] = {}

        need = cfg.window + cfg.horizon
        for rid, d in runs.items():
            obs = scaler.transform_obs(d["obs"]).astype(np.float32)
            act = scaler.transform_act(d["act"]).astype(np.float32)
            self.data[rid] = {"obs": obs, "act": act}
            n = len(obs)
            if n < need + 1:
                continue
            # 起点范围
            last = n - need - 1
            # P0-4 修复：动作前移一拍，需要 s-1 >= 0，故起点从 1 开始
            for start in range(1, last + 1, cfg.stride):
                self.samples.append((rid, start))
        if not self.samples:
            raise RuntimeError(
                f"没有有效窗口。检查 window={cfg.window} 是否超过轨迹长度"
                f"（轨迹约 {(len(next(iter(self.data.values()))['obs'])
                                if self.data else 0)} 点）")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        rid, s = self.samples[i]
        d = self.data[rid]
        T, H = self.cfg.window, self.cfg.horizon

        # ══════════════════════════════════════════════════════════════════
        # P0-4 修复（审查报告：动作与观测时序错开一拍，已独立复核成立）
        # ══════════════════════════════════════════════════════════════════
        # 合同：u_t 作用于 x_t → x_{t+1}，即  x_{t+1} = f(x_t, u_t)
        #
        # 旧实现（错）：
        #   obs     = x[s   : s+T]        即 x_s … x_{s+T-1}
        #   act     = u[s   : s+T]        即 u_s … u_{s+T-1}
        #   fut     = x[s+T : s+T+H]      即 x_{s+T} …
        #   act_fut = u[s+T : s+T+H]      即 u_{s+T} …
        # 问题：act[k] 与 obs[k] 同索引，但按合同 x_{s+k} 应由 u_{s+k-1} 产生，
        #       act_fut[0] 应为 u_{s+T-1}（产生 x_{s+T} 的动作），却取了 u_{s+T}。
        #
        # 新实现（对齐合同）：动作整体前移一拍
        #   act[k]     = u[s+k-1]  产生观测 x[s+k]
        #   act_fut[k] = u[s+T-1+k] 产生目标 x[s+T+k]
        #
        # 为保证 s-1 >= 0，WindowDataset 的样本起点从 1 开始（见 __init__）。
        # ══════════════════════════════════════════════════════════════════
        s_act = s - 1

        obs = d["obs"][s: s + T]
        act = d["act"][s_act: s_act + T]          # u_{s-1} … u_{s+T-2}
        fut = d["obs"][s + T: s + T + H]          # x_{s+T} …
        act_fut = d["act"][s_act + T: s_act + T + H]   # u_{s+T-1} …

        return {
            "obs": torch.from_numpy(obs),
            "act": torch.from_numpy(act),
            "fut": torch.from_numpy(fut),
            "act_fut": torch.from_numpy(act_fut),
        }


# ══════════════════════════════════════════════════════════════════════════
# 工厂
# ══════════════════════════════════════════════════════════════════════════

def make_datasets(
    raw_dirs: list[Path],
    splits: dict[str, list[str]],
    cfg: DataConfig,
) -> tuple[dict[str, WindowDataset], Scaler, dict]:
    """构建 train/val/test 数据集。

    Args:
        raw_dirs: 原始数据目录列表
        splits:   {split_name: [run_id, ...]}
        cfg:      数据配置

    Returns:
        (datasets, scaler, stats)
    """
    runs = load_runs(raw_dirs, cfg)
    print(f"共读到 {len(runs)} 条轨迹")

    # 按 split 分组
    grouped: dict[str, dict] = {}
    for name, ids in splits.items():
        sub = {k: v for k, v in runs.items() if k in ids}
        if sub:
            grouped[name] = sub
    print("分组：" + ", ".join(f"{k}={len(v)}" for k, v in grouped.items()))

    # scaler 只在 train 上拟合
    tr = grouped.get("train", {})
    if not tr:
        raise RuntimeError("train 为空")
    all_obs = np.concatenate([v["obs"] for v in tr.values()], 0)
    all_act = np.concatenate([v["act"] for v in tr.values()], 0)
    scaler = Scaler.fit(all_obs, all_act)

    datasets = {k: WindowDataset(v, scaler, cfg) for k, v in grouped.items()}

    stats = {
        "n_runs": {k: len(v) for k, v in grouped.items()},
        "n_windows": {k: len(d) for k, d in datasets.items()},
        "n_points": {k: sum(len(v["obs"]) for v in g.values())
                     for k, g in grouped.items()},
        "window": cfg.window,
        "horizon": cfg.horizon,
        "dt": cfg.dt,
        "obs_cols": cfg.obs_cols,
        "act_cols": cfg.act_cols,
        "scaler": {
            "obs_mean": scaler.obs_mean.tolist(),
            "obs_std": scaler.obs_std.tolist(),
        },
    }
    return datasets, scaler, stats
