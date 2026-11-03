"""
Two-Hot 输出编码（PyTorch 实现）

⚠ 用途说明（重要，防止误用）：
    官方 DreamerV3 中，two-hot **仅用于 reward head 与 value head**，
    **观测重建（decoder）用的是 symlog_mse**。
    依据：`vendor/dreamerv3/dreamerv3/rssm.py:299-300`

        o1, o2 = 'categorical', ('symlog_mse' if self.symlog else 'mse')
        outputs = {k: o1 if v.discrete else o2 for k, v in spaces.items()}

    以及 `dreamerv3/configs.yaml:98,101`：
        rewhead: {..., output: symexp_twohot, bins: 255}
        value:   {..., output: symexp_twohot, bins: 255}

    本课题第一阶段**不做强化学习、不训 Actor-Critic**（方案 §6 红线），
    因而**没有 reward/value 目标**，主训练路径**不使用本模块**。

    保留它的原因：第二阶段若引入控制/规划（如评估候选动作代价、TD-MPC 式规划），
    可能需要 value head，届时可直接复用。

对齐来源：官方 `embodied/jax/outs.py:273-330` 与 `embodied/jax/heads.py:132-144`
          （danijar/dreamerv3, MIT）。

核心设计（严格照搬官方）：
    1. 分箱在 symlog 空间等距，经 symexp 后变为**非均匀**的物理量网格：
       小值区分辨率高、大值区分辨率低，匹配物理量"小值敏感"的需求。
    2. 目标值由**相邻两个箱**的线性插值表示（two-hot 名字来源），
       权重按距离反比分配。
    3. pred() 用**对称求和**，保证对称分箱 + 均匀概率时输出恰好为 0。

⚠ 使用前提（自测发现）：
    分箱覆盖 symexp(20)≈±4.85e8，即使远端箱概率 1e-6 也贡献 ~485 的误差。
    因此**输入必须归一化到 O(1) 量级**，否则 mean() 不收敛。
    这条对 symlog_mse 同样成立（Ng 的 symlog ≈ 10.7，与燃油的 0.11 量级差异大）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "make_bins",
    "symlog",
    "symexp",
    "TwoHotDist",
]


# --------------------------------------------------------------------------
# symlog / symexp
# --------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    """对称对数压缩：sign(x) * log(1 + |x|)。

    与官方 `nets.symlog` 完全一致（官方 outs.py 中用 sg 截断梯度，见 TwoHotDist.loss）。
    """
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


def symexp(x: torch.Tensor) -> torch.Tensor:
    """symlog 的逆：sign(x) * (exp(|x|) - 1)。"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


# --------------------------------------------------------------------------
# 分箱
# --------------------------------------------------------------------------

def make_bins(n_bins: int, *, vmin: float = -20.0, vmax: float = 0.0,
              device=None, dtype=torch.float32) -> torch.Tensor:
    """构造 two-hot 的对称分箱。

    对齐官方 heads.py:136-143：
        half = linspace(vmin, vmax, n_half)   # symlog 空间等距
        half = symexp(half)                   # 转回物理量（非均匀）
        bins = concat([half, -half[::-1] 或 -half[:-1][::-1]])

    ⚠ 关于偶数分箱（自审发现，与官方有细微差异）：
        官方对偶数 n 用 `bins = concat([half, -half[::-1]])`，
        而 half 的末端恰好是 symexp(0) = 0，导致 **0 处出现重复箱**：
            half        = [-1, ..., -0.05, 0]
            -half[::-1] = [0, 0.05, ..., 1]
            concat      = [..., 0, 0, ...]   ← 重复
        searchsorted 在重复值上会让 below/above 落在同一箱附近，
        two-hot 退化为 one-hot。官方只用 n=255（奇数）故不受影响。
        本实现对偶数 n 改用 `endpoint=False` 生成无 0 的 half，消除重复。

        若要与官方逐位一致，请只用奇数分箱（官方默认 255）。

    Args:
        n_bins: 分箱总数。官方默认 255（奇数）。本课题建议同为奇数。
        vmin, vmax: symlog 空间的范围。官方用 (-20, 0)。
            ⚠ 注意：symexp(-20) ≈ -4.85e8，即分箱覆盖 ±4.85e8 的物理范围。
            但 symlog 空间等距意味着**相对分辨率**在物理空间约 15%
            （127 步覆盖 [-20,0]，步长 0.157 in log space）。
            **因此输入必须归一化到 O(1) 量级**，否则大数值（如 Ng=4.5e4）
            的相对精度会很粗。这是使用 two-hot 的前提条件。

    Returns:
        形状 [n_bins] 的升序分箱（无重复）
    """
    if n_bins < 2:
        raise ValueError(f"n_bins 至少为 2，实际 {n_bins}")

    if n_bins % 2 == 1:
        # 奇数：中间恰好一个 0
        n_half = (n_bins - 1) // 2 + 1
        half = torch.linspace(vmin, vmax, n_half, device=device, dtype=dtype)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)], dim=0)
    else:
        # 偶数：endpoint=False 避免 0 重复（见上方说明）
        n_half = n_bins // 2
        half = torch.linspace(vmin, vmax, n_half + 1, device=device, dtype=dtype)[:-1]
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)], dim=0)

    # 校验：严格升序（无重复）
    d = bins[1:] - bins[:-1]
    if not torch.all(d > 0):
        bad = int(torch.argmin(d).item())
        raise AssertionError(
            f"分箱非严格升序：位置 {bad} 处 diff={d[bad].item():.3e}，"
            f"相邻值 {bins[bad].item():.6g} -> {bins[bad+1].item():.6g}"
        )
    # 校验：关于 0 对称
    if not torch.allclose(bins, -bins.flip(0), atol=1e-5):
        raise AssertionError("分箱不关于 0 对称")
    return bins


# --------------------------------------------------------------------------
# TwoHot 分布
# --------------------------------------------------------------------------

class TwoHotDist:
    """Two-hot 分类输出，接口模仿 PyTorch 的 `TorchDistribution`。

    与官方 `outs.TwoHot` 的对应关系：
        官方 pred()  -> 本类 mode() / mean()
        官方 loss()  -> 本类 negative_log_prob()

    使用方式：
        logits = head(feature)                 # [..., n_bins]
        dist = TwoHotDist(logits, bins)
        loss = dist.negative_log_prob(target)  # 标量场，逐元素
        y = dist.mean()                        # [..., 1] 物理量
    """

    def __init__(self, logits: torch.Tensor, bins: torch.Tensor):
        if logits.shape[-1] != bins.shape[0]:
            raise ValueError(
                f"logits 末维 {logits.shape[-1]} 与分箱数 {bins.shape[0]} 不符"
            )
        self.logits = logits
        self.probs = F.softmax(logits, dim=-1)
        # bins 与 logits 同 device
        self.bins = bins.to(dtype=logits.dtype, device=logits.device)
        self._n = logits.shape[-1]

    # ------------------------------------------------------------------
    # 预测
    # ------------------------------------------------------------------

    def mean(self) -> torch.Tensor:
        """期望值（物理量空间）。

        对齐官方 pred()：用**对称求和**避免浮点累积误差。
        官方注释原文：
            "The naive implementation results in a non-zero result even if the bins
             are symmetric and the probabilities uniform, because the sum operation
             goes left to right, accumulating numerical errors.
             Instead, we use a symmetric sum to ensure that the predicted rewards
             and values are actually zero at initialization."

        Returns:
            [..., 1]
        """
        n = self._n
        if n % 2 == 1:
            m = (n - 1) // 2
            p1, p2, p3 = self.probs[..., :m], self.probs[..., m:m + 1], self.probs[..., m + 1:]
            b1, b2, b3 = self.bins[..., :m], self.bins[..., m:m + 1], self.bins[..., m + 1:]
            wavg = (
                (p2 * b2).sum(-1)
                + ((p1 * b1).flip(-1) + (p3 * b3)).sum(-1)
            )
        else:
            p1, p2 = self.probs[..., :n // 2], self.probs[..., n // 2:]
            b1, b2 = self.bins[..., :n // 2], self.bins[..., n // 2:]
            wavg = ((p1 * b1).flip(-1) + (p2 * b2)).sum(-1)
        return wavg.unsqueeze(-1)

    def mode(self) -> torch.Tensor:
        """众数箱的值（不是 top-1 的箱值，而是期望值）。

        官方 TwoHot 没有单独的 mode()，decoder 评估时用 pred()。
        这里 `mode` 与 `mean` 等价，以匹配调用方对 `.mode()` 的预期。
        """
        return self.mean()

    # ------------------------------------------------------------------
    # 损失
    # ------------------------------------------------------------------

    def negative_log_prob(self, target: torch.Tensor) -> torch.Tensor:
        """two-hot 交叉熵。

        对齐官方 loss()：
            1. target 先做 symlog 压缩，且 **stop-gradient**（不通过目标反传）
            2. 找 target 落在哪两个相邻箱之间
            3. 权重按到两箱的距离**反比**分配
            4. 与 log_softmax 做交叉熵

        Args:
            target: 物理量空间的目标，形状 [..., 1] 或与 logits[...,0] 广播兼容

        Returns:
            逐元素负对数似然，形状 [...]
        """
        # target: [..., 1] -> [...]
        if target.shape[-1] == 1 and self.logits.dim() == target.dim():
            target = target.squeeze(-1)

        # stop-gradient：目标不参与反传（官方 sg(self.squash(target)))
        target = symlog(target.detach())

        n = self._n
        bins = self.bins

        # 官方逻辑：below = (#bins <= target) - 1；above = n - (#bins > target)
        # 展开 target 以广播
        tgt = target.unsqueeze(-1)                       # [..., 1]
        le = (bins <= tgt).to(torch.int64).sum(-1) - 1   # below
        gt = n - (bins > tgt).to(torch.int64).sum(-1)    # above
        below = le.clamp(0, n - 1)
        above = gt.clamp(0, n - 1)

        equal = (below == above)
        b_below = bins[below]
        b_above = bins[above]
        dist_below = torch.where(equal, torch.ones_like(target), (b_below - target).abs())
        dist_above = torch.where(equal, torch.ones_like(target), (b_above - target).abs())
        total = dist_below + dist_above
        # 防 0 除（equal 时 total=2，不会为 0；但 target 恰好落在箱上时 total=dist_above>0）
        w_below = dist_above / total
        w_above = dist_below / total

        # one-hot 合成 two-hot 目标
        onehot_below = F.one_hot(below, n).to(self.logits.dtype)
        onehot_above = F.one_hot(above, n).to(self.logits.dtype)
        twohot = (
            onehot_below * w_below.unsqueeze(-1)
            + onehot_above * w_above.unsqueeze(-1)
        )

        log_pred = F.log_softmax(self.logits, dim=-1)
        nll = -(twohot * log_pred).sum(-1)
        return nll

    # 便捷别名：调用方（decoder loss）习惯用 log_prob
    def log_prob(self, target: torch.Tensor) -> torch.Tensor:
        return -self.negative_log_prob(target)


# --------------------------------------------------------------------------
# 自测
# --------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=== 1. 分箱构造 ===")
    for n in (255, 254, 11, 10):
        b = make_bins(n)
        print(f"  n_bins={n:<4} 形状{tuple(b.shape)}  "
              f"min={b[0].item():+.4f}  max={b[-1].item():+.4f}  "
              f"中位={b[n//2].item():+.4f}")
        # 对称性检查
        sym = torch.allclose(b, -b.flip(0), atol=1e-6)
        print(f"           对称={sym}")

    print("\n=== 2. 初始化偏置检查（均匀概率 -> 预测应为 0）===")
    bins = make_bins(255)
    logits = torch.zeros(4, 3, 255)          # 均匀
    dist = TwoHotDist(logits, bins)
    pred = dist.mean()
    print(f"  预测值 max|.|={pred.abs().max().item():.3e}（应≈0）")

    print("\n=== 3. two-hot 目标构造（落在两箱之间）===")
    target = torch.tensor([[0.0], [1.0], [100.0], [-50.0]])
    logits = torch.zeros(4, 255)
    dist = TwoHotDist(logits, bins)
    nll = dist.negative_log_prob(target)
    print(f"  NLL={nll.detach().numpy().round(4)}")

    print("\n=== 4. 训练式拟合：让网络学会预测 3 个不同值 ===")
    torch.manual_seed(1)
    head = torch.nn.Linear(8, 255)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    targets_true = torch.tensor([10.0, 500.0, 20000.0])
    feat = torch.randn(3, 8)
    for it in range(400):
        logits = head(feat)
        d = TwoHotDist(logits, bins)
        loss = d.negative_log_prob(targets_true.unsqueeze(-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 100 == 0 or it == 399:
            with torch.no_grad():
                p = TwoHotDist(head(feat), bins).mean().squeeze(-1)
            print(f"  iter {it:<4} loss={loss.item():.4f} pred={p.numpy().round(2)}")

    print("\n=== 5. 大跨度精度检查（物理量关键需求）===")
    print(f"  真实值: {targets_true.numpy()}")
    with torch.no_grad():
        p = TwoHotDist(head(feat), bins).mean().squeeze(-1)
    rel = ((p - targets_true).abs() / targets_true.abs() * 100).numpy()
    print(f"  预测值: {p.numpy().round(2)}")
    print(f"  相对误差: {rel.round(3)} %")
    print(f"  （Ng 级 20000 与 燃油级 10 的相对误差应都在同一数量级，"
          f"证明 two-hot 能处理大跨度）")
