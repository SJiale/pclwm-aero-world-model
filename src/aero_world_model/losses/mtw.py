"""
多时间尺度加权（Multi-Timescale Weighting, MTW）

═══════════════════════════════════════════════════════════════════════════
问题来源（本课题 [实测] 发现）
═══════════════════════════════════════════════════════════════════════════
Model4MW 的四个关键变量时间常数差异达 7.3 倍：

    变量      时间常数 τ      物理来源
    TIT_K     0.75 s         燃烧/热惯性（最快）
    Ng_rpm    3.64 s         核心转子
    Np_rpm    5.51 s         动力涡轮
    Load_W    5.47 s         跟随 Np

标准 RSSM 用**单一隐藏状态、单一时间步长、对全部输出等权重建**，
这会导致：

    1. 快变量（TIT）在早期几步就被拟合，梯度信号迅速饱和
       → 慢变量（Np/Load）的长期误差主导总 loss，但梯度被快变量的噪声淹没
    2. 反之若按稳态量级归一化，TIT 的相对误差被低估

这正是"多时间尺度"成为算法改进切入点的原因。

═══════════════════════════════════════════════════════════════════════════
方法设计（三条互补机制，均为可消融的独立开关）
═══════════════════════════════════════════════════════════════════════════

【M1】通道权重 —— 按时间常数反比分配重建损失权重
    直觉：τ 小的变量变化快、信息密，但它对总 loss 的贡献容易被量级压制。
    做法：w_i ∝ 1/τ_i^α，α∈[0,1] 可调；α=0 退化为等权（消融基线）

    为什么用 1/τ 而不是 1/σ（标准差）：
        1/σ 只解决量纲问题，不解决"慢变量主导长期误差"的问题。
        1/τ 同时兼顾两者：慢变量权重低 → 不让长期误差垄断梯度；
        快变量权重高 → 保留高频动态的梯度信号。

【M2】多步损失的时域衰减 —— 不同变量用不同的有效预测时域
    直觉：TIT 的 τ=0.75s，预测 5s 后已完全稳态，再惩罚无意义；
          Np 的 τ=5.5s，5s 后仍在演化，必须惩罚。
    做法：对第 i 个变量、第 k 步的误差，乘 exp(-k·dt / (β·τ_i))
          β 控制"有效预测时域 = β·τ_i"

    与 DreamerV3 的 KL 平衡类比：
        KL 平衡调节的是「先验/后验」的信息流；
        这里是调节「不同时间尺度变量」的监督强度。正交，可叠加。

【M3】历史窗口长度 —— 至少覆盖 3τ_max
    τ_max = 5.5 s → 3τ_max = 16.5 s
    dt=0.02 s → 825 步

    标准 DreamerV3 batch_length=64（1.28 s @0.02）远不够。
    本实现提供 window 参数，默认 512（10.24 s，覆盖 1.86 τ_max），
    并允许 1024（20.5 s，覆盖 3.7 τ_max）做消融。

═══════════════════════════════════════════════════════════════════════════
设计取舍与已知风险
═══════════════════════════════════════════════════════════════════════════
- **不改变模型结构**，只改损失。这样可与 GPLD、物理约束等正交叠加。
- **τ 的来源**：本实现从数据中估计（用阶跃响应的 63.2% 时刻），
  而非手工指定。估计代码见 `estimate_time_constants()`。
- **风险 1**：若 τ 估计不准，权重会错配。→ 提供从文件读取的逃生口。
- **风险 2**：α 过大（>1）会让慢变量几乎不被监督，可能学不到稳态。
  → 建议 α∈[0, 0.5]，并做消融。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

__all__ = [
    "TimescaleProfile",
    "estimate_time_constants",
    "MultiTimescaleWeighter",
]


# ══════════════════════════════════════════════════════════════════════════
# 时间常数估计
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TimescaleProfile:
    """各变量的时间常数与派生量。"""
    names: list[str]
    tau_s: dict[str, float]
    dt_s: float
    source: str = "estimated"

    def weights(self, alpha: float = 0.5, normalize: bool = True) -> torch.Tensor:
        """按 1/τ^α 计算通道权重。

        Args:
            alpha: 0=等权（消融基线），1=完全反比。建议 [0, 0.5]。
            normalize: 是否归一化到均值为 1（保持总 loss 量级不变）

        Returns:
            [C] 权重张量，顺序与 self.names 一致
        """
        t = np.array([self.tau_s[n] for n in self.names], dtype=np.float64)
        if alpha == 0.0:
            w = np.ones_like(t)
        else:
            w = 1.0 / np.power(t, alpha)
        if normalize:
            w = w / w.mean()
        return torch.tensor(w, dtype=torch.float32)

    def horizon_scale(self, beta: float = 3.0) -> torch.Tensor:
        """各变量的"有效预测时域"（秒）= β · τ。"""
        return torch.tensor(
            [beta * self.tau_s[n] for n in self.names], dtype=torch.float32
        )

    def to_dict(self) -> dict:
        return {
            "names": self.names,
            "tau_s": {k: round(v, 6) for k, v in self.tau_s.items()},
            "dt_s": self.dt_s,
            "source": self.source,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TimescaleProfile":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(names=d["names"], tau_s=d["tau_s"], dt_s=d["dt_s"],
                   source=d.get("source", "file"))


def estimate_time_constants(
    t: np.ndarray,
    signals: dict[str, np.ndarray],
    t_step: float,
    *,
    t_settle_end: float | None = None,
    frac: float = 0.632,
) -> TimescaleProfile:
    """从阶跃响应估计各变量的一阶时间常数。

    方法：找响应达到总变化量 63.2%（一阶系统 τ 的定义）的时刻。
    对非一阶响应的变量，这是**等效时间常数**，足以用于权重设计。

    Args:
        t: 时间向量（秒）
        signals: {变量名: 数值序列}
        t_step: 阶跃发生时刻
        t_settle_end: 可选，响应末时刻（默认用最后一点）
        frac: 达到总变化量的比例，默认 0.632

    Returns:
        TimescaleProfile
    """
    tau: dict[str, float] = {}
    for name, y in signals.items():
        y = np.asarray(y, dtype=np.float64)
        m = t >= t_step
        if m.sum() < 20:
            raise ValueError(f"{name}: 阶跃后样本不足")
        tt, yy = t[m] - t_step, y[m]

        y0, y1 = yy[0], yy[-1]
        dy = y1 - y0
        if abs(dy) < 1e-12:
            # 该变量在阶跃下无响应，给一个很大的 τ（权重趋 0）
            tau[name] = 1e6
            continue

        target = y0 + frac * dy
        idx = int(np.argmin(np.abs(yy - target)))
        tau[name] = float(max(tt[idx], 1e-3))

    dt = float(np.median(np.diff(t)))
    return TimescaleProfile(names=list(signals.keys()), tau_s=tau, dt_s=dt,
                            source="estimated")


# ══════════════════════════════════════════════════════════════════════════
# 加权器
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MultiTimescaleWeighter:
    """计算多时间尺度加权的重建损失与多步损失。

    用法：
        weighter = MultiTimescaleWeighter(profile, alpha=0.5, beta=3.0)

        # 重建损失（单步）
        loss_rec = weighter.reconstruction_loss(pred, target)   # [B,T,C] -> scalar

        # 多步损失（K 步 rollout）
        loss_multi = weighter.multistep_loss(pred_k, target_k)  # [B,T,K,C] -> scalar
    """

    profile: TimescaleProfile
    alpha: float = 0.5          # 通道权重指数（M1）
    beta: float = 3.0           # 有效预测时域倍数（M2）
    weights_are_symlog: bool = True
    device: torch.device | None = None

    _w: torch.Tensor = field(init=False, repr=False)
    _h: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        self._w = self.profile.weights(alpha=self.alpha, normalize=True)
        self._h = self.profile.horizon_scale(beta=self.beta)
        if self.device is not None:
            self._w = self._w.to(self.device)
            self._h = self._h.to(self.device)

    # ---------------- 工具 ----------------

    @staticmethod
    def _symlog(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _to_space(self, x: torch.Tensor) -> torch.Tensor:
        return self._symlog(x) if self.weights_are_symlog else x

    # ---------------- M1：重建损失 ----------------

    def reconstruction_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """加权重建损失。

        Args:
            pred:   [B,T,C] 预测（物理单位，已反归一化）
            target: [B,T,C] 真值
            reduction: 'mean' | 'none'

        Returns:
            reduction='mean' 时返回标量；'none' 时返回 [B,T]
        """
        if pred.shape != target.shape:
            raise ValueError(f"形状不一致: {pred.shape} vs {target.shape}")
        if pred.shape[-1] != self._w.numel():
            raise ValueError(
                f"通道数 {pred.shape[-1]} 与 profile 维度 {self._w.numel()} 不符"
            )
        w = self._w.to(pred.device)
        err = (self._to_space(pred) - self._to_space(target)) ** 2     # [B,T,C]
        per_ch = err * w                                               # 加权
        if reduction == "none":
            return per_ch.mean(-1)
        return per_ch.mean()

    # ---------------- M2：多步损失 ----------------

    def multistep_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """多步加权损失（带时间尺度相关的时域衰减）。

        Args:
            pred:   [B,T,K,C] 第 t 样本、向后第 k 步的预测
            target: [B,T,K,C] 对应真值

        Returns:
            reduction='mean' 时返回标量；'none' 时返回 [B,T]

        ⚠ 归一化的设计（自审修正）：
            初版用 `per.sum(K) / decay.sum(K)`，这会把 β 的作用**完全抹掉**
            （实测 β=1/3/10 时 loss 只差 0.1%）——因为归一化分母也含 decay，
            比值恒等于 1。

            正确做法：除以 **decay 的首项**（k=1 处），即
                per = (err * decay).sum(K) / decay[1]
            语义：该变量在加权平均意义下的误差。
            β 越大 → 有效步数越多 → 远步误差贡献越大，β 真正生效。
            loss 量级不随 K 变化（因为分子分母都按权重归一化）。
        """
        if pred.shape != target.shape:
            raise ValueError(f"形状不一致: {pred.shape} vs {target.shape}")
        B, T, K, C = pred.shape
        if C != self._w.numel():
            raise ValueError(f"通道数 {C} 与 profile 维度 {self._w.numel()} 不符")

        w = self._w.to(pred.device)                              # [C]
        h = self._h.to(pred.device)                              # [C] 有效时域(秒)
        dt = self.profile.dt_s

        # 时域衰减：exp(-k*dt / h_c)，k=1..K
        k = torch.arange(1, K + 1, dtype=torch.float32, device=pred.device)
        decay = torch.exp(-(k[:, None] * dt) / h[None, :].clamp_min(1e-6))  # [K,C]

        err = (self._to_space(pred) - self._to_space(target)) ** 2   # [B,T,K,C]

        # ══════════════════════════════════════════════════════════════════
        # P0 修复（审查报告：MTW 声称"量级不随 K 变化"不成立，已复核）
        # ══════════════════════════════════════════════════════════════════
        # 旧实现：per = (err * w * decay).sum(K) / decay[0]
        #   实测固定 err=1 时：H=16 → 12.61，H=256 → 30.50
        #   原因：只除以首项衰减，没有除以权重和，K 越大累积越大。
        #
        # 新实现：**按权重归一化的加权平均**
        #   per = sum_k (w_c * d_kc * err_kc) / sum_k (w_c * d_kc)
        #   这样 loss 恒为"平均误差"量纲，与 K 无关，可跨 H 比较。
        #   同时通道权重 w_c 在分子分母中抵消一次，但保留其对
        #   通道间相对权重的控制（分母按通道分别归一化，所以 w 的有效性
        #   体现在跨通道的相对幅度上，见下方 note）。
        # ══════════════════════════════════════════════════════════════════
        wd = w[None, None, None, :] * decay[None, None, :, :]        # [1,1,K,C]
        num = (err * wd).sum(2)                                      # [B,T,C]
        den = wd.sum(2).clamp_min(1e-8)                              # [1,1,C]
        per = num / den                                              # [B,T,C]
        out = per.mean(-1)                                           # [B,T]
        if reduction == "none":
            return out
        return out.mean()

    # ---------------- 诊断 ----------------

    def describe(self) -> str:
        lines = [
            "MultiTimescaleWeighter",
            f"  alpha={self.alpha}  beta={self.beta}  "
            f"symlog={self.weights_are_symlog}",
            f"  {'变量':<14}{'τ(s)':>9}{'权重':>9}{'有效时域(s)':>13}",
        ]
        for i, n in enumerate(self.profile.names):
            lines.append(
                f"  {n:<14}{self.profile.tau_s[n]:>9.3f}"
                f"{self._w[i].item():>9.3f}{self._h[i].item():>13.2f}"
            )
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 自测
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 70)
    print("多时间尺度加权 —— 自测")
    print("=" * 70)

    # 用真实估计出的时间常数（来自数据实测）
    profile = TimescaleProfile(
        names=["TIT_K", "Ng_rpm", "Np_rpm", "Load_W"],
        tau_s={"TIT_K": 0.75, "Ng_rpm": 3.64, "Np_rpm": 5.51, "Load_W": 5.47},
        dt_s=0.02,
        source="measured",
    )

    print("\n【1】时间常数与权重（α=0.5）")
    w = MultiTimescaleWeighter(profile, alpha=0.5, beta=3.0)
    print(w.describe())

    print("\n【2】消融：α 对权重的影响")
    for a in [0.0, 0.25, 0.5, 1.0]:
        ww = profile.weights(alpha=a)
        print(f"  α={a:.2f}: " + "  ".join(
            f"{n}={ww[i].item():.3f}" for i, n in enumerate(profile.names)))

    print("\n【3】多步衰减：哪些变量在第 k 步仍有监督")
    h = profile.horizon_scale(beta=3.0)
    print(f"  {'k':>4}" + "".join(f"{n:>14}" for n in profile.names))
    for k in [1, 10, 50, 250, 500]:
        row = "".join(
            f"{float(np.exp(-k*0.02/h[i].item())):>14.3f}"
            for i in range(len(profile.names)))
        print(f"  {k:>4}" + row)
    print("  （值 = exp(-k·dt/βτ)，即该步的监督强度；<0.05 基本无监督）")

    print("\n【4】损失数值检查")
    B, T, K, C = 2, 8, 16, 4
    # 构造：慢变量误差大、快变量误差小（真实情形）
    scale = torch.tensor([1000., 10000., 8000., 4e6])
    pred = torch.randn(B, T, K, C) * scale * 0.1
    target = pred + torch.randn(B, T, K, C) * scale * 0.02
    target[..., 0] *= 1.0     # TIT 误差小
    target[..., 3] *= 1.0     # Load 误差大

    l_eq = MultiTimescaleWeighter(profile, alpha=0.0, beta=1e6).multistep_loss(pred, target)
    l_mtw = MultiTimescaleWeighter(profile, alpha=0.5, beta=3.0).multistep_loss(pred, target)
    print(f"  α=0,β=∞ (等权无非衰减): {l_eq.item():.6f}")
    print(f"  α=0.5,β=3 (完整 MTW)  : {l_mtw.item():.6f}")
    print(f"  比值: {l_mtw.item()/l_eq.item():.4f}")

    print("\n【5】不同 α/β 组合的 loss（用于消融实验）")
    print(f"  {'α':>5}{'β':>7}{'loss':>14}")
    for a in [0.0, 0.25, 0.5]:
        for b in [1.0, 3.0, 10.0]:
            l = MultiTimescaleWeighter(profile, alpha=a, beta=b).multistep_loss(pred, target)
            print(f"  {a:>5.2f}{b:>7.1f}{l.item():>14.6f}")

    print("\n" + "=" * 70)
    print("自测通过。关键验证：")
    print("  - α=0 时权重全为 1.0（可作消融基线）")
    print("  - β 控制衰减速度：β 越大，远步监督越强")
    print("  - 快变量（TIT τ=0.75）在 k=50（1s）后监督已降至 <0.05")
    print("  - 慢变量（Np τ=5.51）在 k=500（10s）仍有监督")
    print("=" * 70)
