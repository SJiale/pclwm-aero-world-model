"""
W1.1 统一重量化模块（Unified Resampling）

问题背景（[实测] _probe/out_sig_enum.txt）：
    T700 模型导出的 82 个信号混杂四种采样栅格：
        0.002 s (350001 点) / 0.02 s (35001 点) / 0.1 s (7001 点) / 单点常数
    若按行号直接拼表，最大索引错位可达 50 倍，模型会学到虚假因果，
    表现为训练 loss 低但开环推演完全失效。

本模块的唯一职责：
    把任意采样率的信号安全地对齐到统一训练栅格（默认 0.02 s），
    并显式记录每个通道的「有效带宽」与「因果性代价」，不掩盖信息损失。

三条硬性规则：
    1. 连续量（转速/温度/功率/流量）：先抗混叠，再重采样。禁止直接抽点。
    2. 离散量（阶段/许可标志）：零阶保持（ZOH）。严禁插值。
       插值会造出「阶段 4.5」这类不存在的状态。
    3. 上采样不产生新信息；下采样不得把未来信息泄漏到过去。

----------------------------------------------------------------------
关于抗混叠滤波器的因果性（本模块最关键的设计决策）
----------------------------------------------------------------------
初版曾用零相位 filtfilt（前向+反向），理由是「避免群延迟破坏 a_t → o_{t+1} 对齐」。
自审时实测发现这是错的：对 t=5.00 s 处的阶跃，

    输出首次非零出现在 t=4.76 s（提前 0.24 s = 12 个目标采样），
    且阶跃前存在负向预振（-0.043 @ 4.98 s）。

即零相位滤波把未来样本抹进了过去。这在离线数据构建中是**信息泄漏**：
当模型从 o_t 预测 o_{t+1} 时，o_t 已含 (t, t+0.24s) 的观测，
会造成「训练指标好看但开环推演失效」——正是数据协议 §4 警告的首要失败情形。

修正方案：改用**因果滑动平均**（trailing moving average）做抗混叠。
    - 严格因果：y[n] 只依赖 x[0..n]，acausal_extent = 0
    - 无群延迟补偿导致的再泄漏
    - 引入已知的固定滞后 lag = (ma_len-1)/2 * dt_src（秒级可算，写入报告）
    - 长度 2q（q=下采样比）使首个频谱零点落在目标 Nyquist 上

因果滤波必然带滞后。为保证**跨通道不产生错位**，align_signals() 默认做
「统一滞后对齐」：所有通道额外右移，使总滞后相同（= 最大通道滞后）。
代价是整条轨迹有统一延迟 Δ，这在时间序列数据集中无害（只是把时间轴整体平移），
而跨通道错位才是有害的。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SignalSpec",
    "ResampleReport",
    "build_target_grid",
    "resample_continuous",
    "resample_discrete",
    "align_signals",
    "ReportWriter",
]


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------

@dataclass
class SignalSpec:
    """单个信号的重量化规格。

    Attributes:
        name:        信号名（与导出变量名一致）
        role:        action / obs / exog / mask
        resample:    'continuous' 或 'discrete'
        unit:        物理单位（仅记录）
        source_dt_s: 源采样间隔（秒），由实测确定
    """
    name: str
    role: str
    resample: str
    unit: str
    source_dt_s: float | None = None

    def __post_init__(self) -> None:
        if self.resample not in ("continuous", "discrete"):
            raise ValueError(
                f"{self.name}: resample 必须是 'continuous' 或 'discrete'，"
                f"实际为 {self.resample!r}"
            )
        if self.role not in ("action", "obs", "exog", "mask"):
            raise ValueError(f"{self.name}: 未知 role {self.role!r}")


@dataclass
class ResampleReport:
    """单个通道的重量化记录。必须随数据一起归档，不得丢弃。"""
    name: str
    mode: str                       # 'decimate' | 'identity' | 'upsample'
    source_dt_s: float
    target_dt_s: float
    n_source: int
    n_target: int
    # 有效带宽：由源采样率决定，上采样不会提升
    effective_bandwidth_hz: float
    # 抗混叠配置
    anti_alias_ma_len: int | None = None
    # 因果性代价（核心审计字段）
    causal: bool = True
    lag_s: float = 0.0              # 该通道的固有滞后（秒，正=滞后）
    acausal_extent_s: float = 0.0   # 该通道「用到了多少未来数据」（秒）；0=严格因果
    # 净时间偏移：lag_s - acausal_extent_s/2。正=滞后，负=超前。
    # 这是跨通道对齐的依据，而非仅 lag_s（否则上采样通道的超前会被漏掉）。
    net_offset_s: float = 0.0
    # 对齐后统一偏移（align_signals 填充）
    total_lag_s: float | None = None
    n_edge_padded: int = 0
    # 检查结果
    n_nan_source: int = 0
    n_nan_output: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "source_dt_s": self.source_dt_s,
            "target_dt_s": self.target_dt_s,
            "n_source": self.n_source,
            "n_target": self.n_target,
            "effective_bandwidth_hz": round(self.effective_bandwidth_hz, 6),
            "anti_alias_ma_len": self.anti_alias_ma_len,
            "causal": self.causal,
            "lag_s": round(self.lag_s, 9),
            "acausal_extent_s": round(self.acausal_extent_s, 9),
            "net_offset_s": round(self.net_offset_s, 9),
            "total_lag_s": (None if self.total_lag_s is None
                            else round(self.total_lag_s, 9)),
            "n_edge_padded": self.n_edge_padded,
            "n_nan_source": self.n_nan_source,
            "n_nan_output": self.n_nan_output,
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# 目标栅格
# --------------------------------------------------------------------------

def build_target_grid(
    t_start: float,
    t_end: float,
    dt: float,
    *,
    atol: float = 1e-9,
) -> np.ndarray:
    """构造统一的训练时间栅格。

    用乘法构造而非累加，避免浮点漂移：
        t[k] = t_start + k*dt
    累加写法在 35000 步后会偏出 1e-8 量级，足以破坏后续的等距断言。

    Args:
        t_start: 起点（秒）
        t_end:   终点（秒），含端点
        dt:      栅格间隔（秒）
        atol:    端点容差

    Returns:
        等间隔时间向量，末点不超过 t_end + atol
    """
    if dt <= 0:
        raise ValueError(f"dt 必须为正，实际 {dt}")
    if t_end < t_start:
        raise ValueError(f"t_end({t_end}) 不得小于 t_start({t_start})")

    n = int(np.floor((t_end - t_start) / dt + atol)) + 1
    grid = t_start + np.arange(n, dtype=np.float64) * dt
    if grid[-1] > t_end + atol:
        grid = grid[:-1]
    return grid


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------

def _infer_uniform_dt(t: np.ndarray) -> float:
    """从时间向量推断采样间隔，并校验等距性。"""
    if t.size < 2:
        raise ValueError("时间向量少于 2 点，无法推断 dt")
    d = np.diff(t)
    dt = float(np.median(d))
    if not np.allclose(d, dt, rtol=1e-2, atol=1e-9):
        max_dev = float(np.max(np.abs(d - dt)))
        raise ValueError(
            f"时间向量非等距：中位 dt={dt:.9g}s，最大偏差 {max_dev:.3e}s。"
            f"请先对原始导出做严格等距检查。"
        )
    return dt


def _causal_moving_average(x: np.ndarray, ma_len: int) -> np.ndarray:
    """因果滑动平均：y[n] = mean(x[n-ma_len+1 : n+1])。

    严格因果，只看过去。为消除起始段的窗长不足（会削弱抗混叠），
    对左边界做**首值复制填充**，使全程窗长恒定。

    Args:
        x:      输入序列
        ma_len: 窗长（>=1）

    Returns:
        与 x 等长的因果均值序列
    """
    if ma_len <= 1:
        return x.copy()
    n = x.size
    pad = np.full(ma_len - 1, x[0], dtype=x.dtype)
    xp = np.concatenate([pad, x])
    c = np.concatenate([[0.0], np.cumsum(xp)])
    # y[i] = (c[i+ma_len] - c[i]) / ma_len, i = 0..n-1 （对应 xp 的滑动窗）
    out = (c[ma_len:ma_len + n] - c[:n]) / ma_len
    return out


# --------------------------------------------------------------------------
# 连续量：抗混叠 + 重采样
# --------------------------------------------------------------------------

def resample_continuous(
    t: np.ndarray,
    x: np.ndarray,
    t_target: np.ndarray,
    *,
    ma_len: int | None = None,
) -> tuple[np.ndarray, ResampleReport]:
    """连续量：抗混叠后重采样到目标栅格。

    三种情形：
        decimate (dt_src < dt_tgt) : 因果滑动平均抗混叠 + 直接取栅格点。严格因果。
        identity (dt_src == dt_tgt): 直接对齐。
        upsample (dt_src > dt_tgt) : 线性插值。**会用到下一个源样本**，
                                     故 acausal_extent_s = dt_src，如实记录。

    Args:
        t:        源时间（等距，秒）
        x:        源数值
        t_target: 目标时间栅格（秒）
        ma_len:   抗混叠窗长。None = 2*round(dt_tgt/dt_src)（首零点落在目标 Nyquist）

    Returns:
        (对齐后的值, 报告)

    Raises:
        ValueError: 长度不一致、时间轴含 NaN、非单调、非等距
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    x = np.asarray(x, dtype=np.float64).ravel()

    if t.size != x.size:
        raise ValueError(f"t 与 x 长度不一致：{t.size} vs {x.size}")
    if not np.all(np.isfinite(t)):
        raise ValueError("时间轴含 NaN/Inf")
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间轴非严格单调递增")

    src_dt = _infer_uniform_dt(t)
    src_fs = 1.0 / src_dt
    tgt_dt = float(np.median(np.diff(t_target)))

    n_nan_src = int(np.count_nonzero(~np.isfinite(x)))

    ratio = src_dt / tgt_dt
    if abs(ratio - 1.0) < 1e-6:
        mode = "identity"
    elif src_dt < tgt_dt:
        mode = "decimate"
    else:
        mode = "upsample"

    report = ResampleReport(
        name="",
        mode=mode,
        source_dt_s=src_dt,
        target_dt_s=tgt_dt,
        n_source=int(t.size),
        n_target=int(t_target.size),
        effective_bandwidth_hz=src_fs / 2.0,   # 由源决定；上采样不提升
        causal=True,
        lag_s=0.0,
        acausal_extent_s=0.0,
    )
    # 自审修正：初版漏了此字段赋值，导致 NaN 统计恒为 0（测试暴露）
    report.n_nan_source = n_nan_src

    # NaN 预处理
    x_work = x
    if n_nan_src > 0:
        ok = np.isfinite(x)
        if not np.any(ok):
            raise ValueError("该通道全为 NaN")
        x_work = np.interp(t, t[ok], x[ok])
        report.notes.append(
            f"源数据含 {n_nan_src} 个 NaN，已用线性插值填补；"
            f"下游必须复核这些点是否应参与训练"
        )

    # ---- 抗混叠（仅 decimate）----
    if mode == "decimate":
        q = int(round(tgt_dt / src_dt))
        if ma_len is None:
            # 长度 2q：归一化频率 1/(2q) 处为首零点，即 f = fs_src/(2q) = fs_tgt/2
            ma_len = 2 * q
        ma_len = max(1, int(ma_len))
        x_work = _causal_moving_average(x_work, ma_len)
        report.anti_alias_ma_len = ma_len
        report.causal = True
        report.lag_s = (ma_len - 1) / 2.0 * src_dt
        report.acausal_extent_s = 0.0
        report.net_offset_s = report.lag_s
        # 首零点位置与在目标 Nyquist 处的衰减
        null_hz = src_fs / ma_len
        report.notes.append(
            f"因果滑动平均 ma_len={ma_len}，首频谱零点 {null_hz:.2f} Hz"
            f"（目标 Nyquist {1/(2*tgt_dt):.2f} Hz），滞后 {report.lag_s*1000:.1f} ms；"
            f"严格因果（acausal_extent=0）"
        )
    elif mode == "upsample":
        # 线性插值会用到下一个源样本 —— 如实记录，不隐瞒
        report.causal = False
        report.lag_s = 0.0
        report.acausal_extent_s = src_dt
        # 净偏移为负=超前：两源样本之间的插值使变化提前半个源周期出现。
        # 自审修正：初版只用 lag_s 做跨通道对齐，漏掉了上采样通道的超前，
        # 实测造成 40 ms 错位（= 2 个目标采样）。必须用 net_offset_s 对齐。
        report.net_offset_s = -src_dt / 2.0
        tgt_nyq = 1.0 / (2.0 * tgt_dt)
        report.notes.append(
            f"上采样：线性插值，acausal_extent={src_dt*1000:.0f} ms（用到下一个源样本，"
            f"上采样无法避免），净超前 {src_dt/2*1000:.0f} ms；"
            f"有效带宽仍为 {src_fs/2:.2f} Hz，仅目标 Nyquist "
            f"{tgt_nyq:.2f} Hz 的 {100*src_fs/2/tgt_nyq:.0f}%，不得当作高带宽信号"
        )
    else:  # identity
        report.net_offset_s = 0.0
        report.notes.append("源栅格与目标一致，直接对齐")

    # ---- 取栅格点 ----
    if mode == "decimate" and abs(t_target[0] - t[0]) < 1e-9:
        # 栅格与源对齐：用索引映射，避免插值再引入误差
        idx = np.rint((t_target - t[0]) / src_dt).astype(np.int64)
        idx = np.clip(idx, 0, x_work.size - 1)
        out = x_work[idx]
    else:
        # identity / upsample / 起始不一致：线性插值
        # 注：decimate 走到这里说明栅格与源起点不对齐，需记录
        if mode == "decimate":
            report.notes.append("目标栅格起点与源不一致，改用线性插值取点")
        out = np.interp(t_target, t, x_work)

    n_before = int(np.count_nonzero(t_target < t[0] - 1e-9))
    n_after = int(np.count_nonzero(t_target > t[-1] + 1e-9))
    if n_before or n_after:
        report.notes.append(
            f"目标栅格有 {n_before} 点早于源起点、{n_after} 点晚于源终点，已用端点值延拓"
        )

    report.n_nan_output = int(np.count_nonzero(~np.isfinite(out)))
    return out, report


# --------------------------------------------------------------------------
# 离散量：零阶保持
# --------------------------------------------------------------------------

def resample_discrete(
    t: np.ndarray,
    x: np.ndarray,
    t_target: np.ndarray,
) -> tuple[np.ndarray, ResampleReport]:
    """离散量：零阶保持对齐。

    为什么严禁插值：
        Seq_Phase 这类阶段编号插值后会产出 4.5 这种不存在的状态，
        下游用它做分段或掩码会引入无法解释的行为。
        ZOH 的语义是「该时刻生效的最近一个已确定值」，与 Simulink 的
        zoh 采样器、与真实的采样保持电路都一致。

    ZOH 天然因果：只会回看过去最近的已确定值，acausal_extent = 0。

    Args:
        t:        源时间（升序）
        x:        源离散值（整数或类别）
        t_target: 目标栅格

    Returns:
        (对齐后的值, 报告)
    """
    t = np.asarray(t, dtype=np.float64).ravel()
    x = np.asarray(x).ravel()

    if t.size != x.size:
        raise ValueError(f"t 与 x 长度不一致：{t.size} vs {x.size}")
    if not np.all(np.diff(t) > 0):
        raise ValueError("时间轴非严格单调递增")

    src_dt = _infer_uniform_dt(t)
    tgt_dt = float(np.median(np.diff(t_target)))
    ratio = src_dt / tgt_dt
    mode = ("identity" if abs(ratio - 1.0) < 1e-6
            else "decimate" if src_dt < tgt_dt else "upsample")

    report = ResampleReport(
        name="",
        mode=mode,
        source_dt_s=src_dt,
        target_dt_s=tgt_dt,
        n_source=int(t.size),
        n_target=int(t_target.size),
        effective_bandwidth_hz=0.0,   # 离散量无带宽概念
        causal=True,
        lag_s=0.0,
        acausal_extent_s=0.0,
        notes=["离散量采用零阶保持（ZOH），未做任何插值，严格因果"],
    )

    idx = np.searchsorted(t, t_target, side="right") - 1
    idx = np.clip(idx, 0, t.size - 1)
    out = x[idx]

    # 校验：输出值域必须完全落在源值域内（防止插值引入新值）
    uniq_src = np.unique(x[np.isfinite(x)])
    uniq_out = np.unique(out)
    extra = np.setdiff1d(uniq_out, uniq_src)
    if extra.size > 0:
        raise AssertionError(
            f"ZOH 产生了源中不存在的值 {extra[:5].tolist()}，实现有误"
        )
    report.notes.append(
        f"输出 {uniq_out.size} 个唯一值，全部来自源值域（ZOH 校验通过）"
    )

    return out, report


# --------------------------------------------------------------------------
# 批量对齐（含跨通道统一滞后）
# --------------------------------------------------------------------------

def align_signals(
    raw: dict[str, tuple[np.ndarray, np.ndarray]],
    specs: list[SignalSpec],
    t_target: np.ndarray,
    *,
    ma_len: int | None = None,
    unify_lag: bool = True,
) -> tuple[dict[str, np.ndarray], list[ResampleReport]]:
    """把多通道原始信号批量对齐到统一栅格，并消除跨通道错位。

    跨通道统一滞后（unify_lag）：
        因果抗混叠必然引入滞后，且不同源采样率的滞后不同
        （如 0.002s 通道 ma_len=20 → 滞后 19 ms；0.1s 通道线性插值 → 滞后 0）。
        若不对齐，同一物理时刻在各通道上会相差 1–5 个目标采样。
        做法：所有通道额外右移 (max_lag - lag_c)，使总滞后统一为 max_lag。
        代价是整条轨迹统一延迟 max_lag，无害；跨通道错位才有害。

    Args:
        raw:      {信号名: (t, x)}
        specs:    每个信号的规格
        t_target: 目标时间栅格
        ma_len:   抗混叠窗长（None=自动）
        unify_lag: 是否做跨通道统一滞后

    Returns:
        (对齐结果 dict, 报告列表)

    Raises:
        KeyError: specs 里的信号在 raw 中缺失
    """
    out: dict[str, np.ndarray] = {}
    reports: list[ResampleReport] = []

    for spec in specs:
        if spec.name not in raw:
            raise KeyError(
                f"信号 {spec.name!r} 在 raw 数据中缺失。"
                f"可用：{sorted(raw.keys())}"
            )
        t, x = raw[spec.name]
        if spec.resample == "continuous":
            y, rep = resample_continuous(t, x, t_target, ma_len=ma_len)
        else:
            y, rep = resample_discrete(t, x, t_target)

        rep.name = spec.name
        if len(y) != len(t_target):
            raise AssertionError(
                f"{spec.name}: 对齐后长度 {len(y)} != 目标 {len(t_target)}"
            )
        out[spec.name] = y
        reports.append(rep)

    # ---- 跨通道统一偏移 ----
    # 自审修正：初版只按 lag_s 对齐，漏掉了上采样通道的「净超前」（-src_dt/2），
    # 实测造成 40 ms（= 2 个目标采样）错位。改为按 net_offset_s 对齐。
    if unify_lag and reports:
        tgt_dt = float(np.median(np.diff(t_target)))
        max_off = max(r.net_offset_s for r in reports)
        for r in reports:
            shift_s = max_off - r.net_offset_s
            k = int(round(shift_s / tgt_dt))
            if k > 0:
                arr = out[r.name]
                shifted = np.empty_like(arr)
                if k >= arr.size:
                    shifted[:] = arr[0]
                else:
                    shifted[:k] = arr[0]
                    shifted[k:] = arr[:-k]
                out[r.name] = shifted
                r.n_edge_padded = k
                r.notes.append(
                    f"为统一跨通道净偏移右移 {k} 个采样（{shift_s*1000:.1f} ms）"
                )
            r.total_lag_s = max_off

    return out, reports


# --------------------------------------------------------------------------
# 报告落盘
# --------------------------------------------------------------------------

class ReportWriter:
    """把重量化报告写成 JSON + 人可读 markdown，随数据一起归档。"""

    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(
        self,
        reports: list[ResampleReport],
        t_target: np.ndarray,
        *,
        stem: str = "resample_report",
        extra: dict[str, Any] | None = None,
    ) -> tuple[Path, Path]:
        tgt_dt = float(np.median(np.diff(t_target)))
        payload = {
            "target_grid": {
                "n_points": int(t_target.size),
                "t_start": float(t_target[0]),
                "t_end": float(t_target[-1]),
                "dt_s": tgt_dt,
            },
            "causality_audit": {
                "all_causal": all(r.causal for r in reports),
                "max_acausal_extent_s": max(
                    (r.acausal_extent_s for r in reports), default=0.0
                ),
                "uniform_total_offset_s": next(
                    (r.total_lag_s for r in reports if r.total_lag_s is not None),
                    None,
                ),
                "channels_using_future_data": [
                    r.name for r in reports if r.acausal_extent_s > 0
                ],
                "per_channel_net_offset_ms": {
                    r.name: round(r.net_offset_s * 1000, 3) for r in reports
                },
            },
            "channels": [r.to_dict() for r in reports],
        }
        if extra:
            payload["extra"] = extra

        jpath = self.out_dir / f"{stem}.json"
        jpath.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        audit = payload["causality_audit"]
        lines = [
            f"# 重量化报告 `{stem}`",
            "",
            f"- 目标栅格：{t_target.size} 点，"
            f"{t_target[0]:.3f}–{t_target[-1]:.3f} s，dt={tgt_dt:.4f} s",
            "",
            "## 因果性审计",
            "",
            f"- 全部通道严格因果：**{'是' if audit['all_causal'] else '否'}**",
            f"- 最大未来数据依赖：**{audit['max_acausal_extent_s']*1000:.1f} ms**",
            f"- 统一总偏移：**{(audit['uniform_total_offset_s'] or 0)*1000:.1f} ms**"
            f"（正值=整体滞后，无害；跨通道错位才有害）",
        ]
        if audit["channels_using_future_data"]:
            lines.append(
                f"- ⚠ 使用未来数据的通道："
                f"`{'`, `'.join(audit['channels_using_future_data'])}`"
                f"（上采样固有，见各通道备注）"
            )
        lines += [
            "",
            "### 各通道净时间偏移（对齐依据）",
            "",
            "| 通道 | 净偏移 |",
            "|---|---|",
        ]
        for nm, ms in audit["per_channel_net_offset_ms"].items():
            sign = "超前" if ms < 0 else "滞后"
            lines.append(f"| `{nm}` | {ms:+.1f} ms（{sign}） |")
        lines += [
            "",
            "## 通道明细",
            "",
            "| 通道 | 模式 | 源 dt | 有效带宽 | MA 窗长 | 固有滞后 | 未来依赖 | "
            "右移 | 源点数 | 备注 |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in reports:
            notes = "；".join(r.notes) if r.notes else "—"
            lines.append(
                f"| `{r.name}` | {r.mode} | {r.source_dt_s:.4f} s | "
                f"{r.effective_bandwidth_hz:.2f} Hz | "
                f"{r.anti_alias_ma_len if r.anti_alias_ma_len is not None else '—'} | "
                f"{r.lag_s*1000:.1f} ms | "
                f"{r.acausal_extent_s*1000:.0f} ms | "
                f"{r.n_edge_padded} | {r.n_source} | {notes} |"
            )
        mpath = self.out_dir / f"{stem}.md"
        mpath.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return jpath, mpath
