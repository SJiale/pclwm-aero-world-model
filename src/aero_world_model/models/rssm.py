"""
RSSM 核心（PC-LWM 版）

移植来源：`vendor/dreamerv3-torch/networks.py`（NM512/dreamerv3-torch, MIT）
参照核对：`vendor/dreamerv3/dreamerv3/rssm.py`（danijar/dreamerv3, MIT）
核对报告：`reports/G1.1_RSSM等价性核对_20261029.md`（修订版 R2）

本文件相对原实现的改动（M2/M3）：
    1. **移除 RL 相关**：不保留 ImagBehavior、reward/value head、
       continue predictor、policy。本课题只做观测重建 + 多步预测。
    2. **移除 CNN 路径**：观测是 ~10 维数值，不需要卷积。
       Encoder/Decoder 简化为纯 MLP。
    3. **移除 episode 机制**：离线数据按固定长度切窗，不涉及 is_first/is_terminal。
       保留 `reset` 接口以兼容 RSSM 的初值重置语义，但训练时全为 False。
    4. **暴露潜状态**：提供 `get_feat` / `get_latent` 供评估模块（LRC、探针）使用。
    5. **保留原实现的所有数值细节**：
       - `unimix_ratio` 在 OneHotDist 中实现
       - KL 用 `clip(min=free)` 而非 max（与原实现一致，见下方注释）
       - `dyn_scale=0.5` / `rep_scale=0.1` / `kl_free=1.0` 默认值
       - 离散采样用 straight-through

⚠ 与原实现的一处**已知差异**（不影响本课题，但要记录）：
    原实现 `kl_loss` 用 `torch.clip(rep_loss, min=free)`，
    注释说"original repo 用 maximum"。官方 JAX 版实际是：
        kl = jnp.maximum(kl, free)   # 对每个元素
    两者等价（都是逐元素下界裁剪），所以此处保留 PyTorch 写法。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as torchd

from .sampling import onehot_st

__all__ = [
    "RSSM",
    "MLPEncoder",
    "MLPDecoder",
    "RSSMConfig",
]


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

class RSSMConfig:
    """RSSM 超参。

    默认值取官方 small 配置（见 G1.1 核对报告 §1）：
        deter=512, hidden=512, stoch=32, classes=32
    官方 large 为 deter=8192, classes=64 —— 本课题不用（8GB 显存）。
    """

    def __init__(
        self,
        *,
        stoch: int = 32,
        classes: int = 32,
        deter: int = 512,
        hidden: int = 512,
        discrete: bool = True,
        act: str = "SiLU",
        norm: bool = True,
        unimix_ratio: float = 0.01,
        min_std: float = 0.1,
        # KL 平衡（官方默认）
        dyn_scale: float = 0.5,
        rep_scale: float = 0.1,
        kl_free: float = 1.0,
        # 本课题新增
        n_bins: int = 255,
    ):
        self.stoch = stoch
        self.classes = classes
        self.deter = deter
        self.hidden = hidden
        self.discrete = discrete
        self.act = act
        self.norm = norm
        self.unimix_ratio = unimix_ratio
        self.min_std = min_std
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.kl_free = kl_free
        self.n_bins = n_bins

    @property
    def feat_dim(self) -> int:
        """潜特征维度 = stoch*classes + deter（离散）或 stoch + deter（连续）。"""
        s = self.stoch * self.classes if self.discrete else self.stoch
        return s + self.deter


# --------------------------------------------------------------------------
# 工具层
# --------------------------------------------------------------------------

class _OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    """混合了 unimix 的 one-hot 分布。

    对齐原实现 `tools.OneHotDist`：
        - unimix：把概率向均匀分布混合，防止 logits 饱和
        - mode()：straight-through 梯度（前向 one-hot，反向走 logits）

    为什么需要 straight-through：
        离散采样不可微。前向用 argmax 得到 one-hot，
        反向把梯度直接接到 logits 上（`logits - logits.detach()` 实现）。
        这是 DreamerV3 能在离散潜变量上做端到端训练的关键。
    """

    def __init__(self, logits=None, probs=None, unimix_ratio: float = 0.0):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, dim=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()


class _SymlogDist:
    """symlog 空间的 MSE 分布（连续观测输出头）。

    对齐官方 `rssm.py:299` —— decoder 对连续观测用 `symlog_mse`，**不是 two-hot**。
    对齐原实现 `tools.SymlogDist`。
    """

    def __init__(self, mode: torch.Tensor, tol: float = 1e-8):
        self._mode = mode
        self._tol = tol

    def mode(self) -> torch.Tensor:
        return torch.sign(self._mode) * (torch.exp(torch.abs(self._mode)) - 1.0)

    # decoder 的可用输出
    def mean(self) -> torch.Tensor:
        return self.mode()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """负 MSE（symlog 空间）。返回的是 log_prob，越小越好时要取负。"""
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        value_s = torch.sign(value) * torch.log(torch.abs(value) + 1.0)
        distance = (self._mode - value_s) ** 2.0
        distance = torch.where(distance < self._tol, torch.zeros_like(distance), distance)
        # 对最后一维求和（多通道各自独立，此处按特征维聚合）
        return -distance.sum(list(range(len(distance.shape)))[2:])


def _act(name: str):
    return getattr(torch.nn, name)


# --------------------------------------------------------------------------
# RSSM
# --------------------------------------------------------------------------

class RSSM(nn.Module):
    """Recurrent State Space Model（确定性 h + 随机 z）。

    数据格式约定（与原实现一致）：
        输入 shape: (batch, time, ...)
        内部按 (time, batch, ...) 处理（static_scan 约定）
    """

    def __init__(self, action_dim: int, cfg: RSSMConfig):
        super().__init__()
        self.cfg = cfg
        self._stoch = cfg.stoch
        self._deter = cfg.deter
        self._hidden = cfg.hidden
        self._discrete = cfg.discrete
        self._unimix = cfg.unimix_ratio
        self._min_std = cfg.min_std
        self._action_dim = action_dim

        act_fn = _act(cfg.act)

        # 后验输入层需要 embed_dim，但 embed_dim 由 Encoder 决定，
        # 而 Encoder 的输出维度在构造时才知道 → 延迟到 build() 中构建。
        self._obs_out_layers: nn.Sequential | None = None
        self._built = False

        # 先验路径（img_step）
        self._img_in_layers = nn.Sequential(
            nn.Linear(self._stoch_dim() + action_dim, cfg.hidden),
            act_fn(),
            nn.LayerNorm(cfg.hidden) if cfg.norm else nn.Identity(),
        )
        self._cell = nn.GRUCell(cfg.hidden, cfg.deter)
        self._img_out_layers = nn.Sequential(
            nn.Linear(cfg.deter, cfg.hidden),
            act_fn(),
            nn.LayerNorm(cfg.hidden) if cfg.norm else nn.Identity(),
        )
        self._ims_stat_layer = nn.Linear(cfg.hidden, self._stoch_stat_dim())
        self._obs_stat_layer = nn.Linear(cfg.hidden, self._stoch_stat_dim())

    def _stoch_dim(self) -> int:
        return self._stoch * self.cfg.classes if self._discrete else self._stoch

    def _stoch_stat_dim(self) -> int:
        return self._stoch * self.cfg.classes if self._discrete else 2 * self._stoch

    def build(self, embed_dim: int):
        """按编码器输出维度构建后验输入层（延迟构建模式）。"""
        if self._built:
            return
        cfg = self.cfg
        act_fn = _act(cfg.act)
        self._obs_out_layers = nn.Sequential(
            nn.Linear(cfg.deter + embed_dim, cfg.hidden),
            act_fn(),
            nn.LayerNorm(cfg.hidden) if cfg.norm else nn.Identity(),
        )
        self._built = True

    # ---- 状态构造 ----

    def initial(self, batch_size: int, device=None) -> dict:
        """初始潜状态（零初始化）。

        对齐原实现 `initial()`。
        """
        dev = device or next(self.parameters()).device
        if self._discrete:
            stoch = torch.zeros(batch_size, self._stoch, self.cfg.classes, device=dev)
        else:
            stoch = torch.zeros(batch_size, self._stoch, device=dev)
        deter = torch.zeros(batch_size, self._deter, device=dev)
        return {"stoch": stoch, "deter": deter}

    def get_dist(self, state: dict):
        if self._discrete:
            logit = state["logit"]
            return torchd.independent.Independent(
                _OneHotDist(logit, unimix_ratio=self._unimix), 1
            )
        mean, std = state["mean"], state["std"]
        return torchd.Independent(torchd.Normal(mean, std), 1)

    def get_feat(self, state: dict) -> torch.Tensor:
        """潜特征 = concat(stoch, deter)。

        供 decoder、探针、LRC 使用。
        """
        stoch = state["stoch"]
        if self._discrete:
            shape = list(stoch.shape[:-2]) + [self._stoch * self.cfg.classes]
            stoch = stoch.reshape(shape)
        return torch.cat([stoch, state["deter"]], -1)

    def get_latent(self, state: dict) -> dict:
        """返回裸潜变量（供评估模块计算 LRC）。"""
        return {"stoch": state["stoch"], "deter": state["deter"]}

    # ---- 递推 ----

    def _suff_stats(self, layer: nn.Linear, x: torch.Tensor) -> dict:
        x = layer(x)
        if self._discrete:
            logit = x.reshape(*x.shape[:-1], self._stoch, self.cfg.classes)
            return {"logit": logit}
        x = x.reshape(*x.shape[:-1], 2, self._stoch)
        mean, std = torch.chunk(x, 2, -2)
        mean = mean.squeeze(-2)
        if self._min_std > 0:
            std = F.softplus(std.squeeze(-2)) + self._min_std
        else:
            std = torch.exp(std.squeeze(-2))
        return {"mean": mean, "std": std}

    def img_step(self, prev_state: dict, prev_action: torch.Tensor,
                 sample: bool = True) -> dict:
        """先验单步：只用动作推进，不看当前观测。"""
        prev_stoch = prev_state["stoch"]
        if self._discrete:
            prev_stoch = prev_stoch.reshape(
                *prev_stoch.shape[:-2], self._stoch * self.cfg.classes
            )
        x = torch.cat([prev_stoch, prev_action], -1)
        x = self._img_in_layers(x)
        deter = self._cell(x, prev_state["deter"])
        x = self._img_out_layers(deter)
        stats = self._suff_stats(self._ims_stat_layer, x)
        if self._discrete:
            # 快速路径：Gumbel-max 直接采样，绕过 torch.distributions（2.8x 加速）
            stoch = onehot_st(stats["logit"], sample=sample, unimix=self._unimix)
        else:
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
        return {"stoch": stoch, "deter": deter, **stats}

    def obs_step(self, prev_state: dict, prev_action: torch.Tensor,
                 embed: torch.Tensor, sample: bool = True) -> tuple[dict, dict]:
        """后验单步：先算先验，再用当前观测修正。"""
        assert self._built, "必须先调用 build(embed_dim)"
        prior = self.img_step(prev_state, prev_action, sample)
        x = torch.cat([prior["deter"], embed], -1)
        x = self._obs_out_layers(x)
        stats = self._suff_stats(self._obs_stat_layer, x)
        if self._discrete:
            stoch = onehot_st(stats["logit"], sample=sample, unimix=self._unimix)
        else:
            if sample:
                stoch = self.get_dist(stats).sample()
            else:
                stoch = self.get_dist(stats).mode()
        post = {"stoch": stoch, "deter": prior["deter"], **stats}
        return post, prior

    def observe(self, embed: torch.Tensor, action: torch.Tensor,
                reset: torch.Tensor | None = None) -> tuple[dict, dict]:
        """整段后验推演。

        Args:
            embed:  (B, T, E) 编码后的观测
            action: (B, T, A) 动作。约定 action[t] 作用于 t→t+1
            reset:  (B, T) bool，True 表示该时刻重置状态。离线数据通常全 False

        Returns:
            post, prior：各为 {stoch:(B,T,...), deter:(B,T,D), ...}
        """
        B, T = embed.shape[:2]
        device = embed.device

        state = self.initial(B, device)
        posts, priors = [], []
        for t in range(T):
            if reset is not None:
                r = reset[:, t].view(B, 1)
                if self._discrete:
                    rs = r.unsqueeze(-1)
                else:
                    rs = r
                state = {
                    "stoch": state["stoch"] * (1 - rs),
                    "deter": state["deter"] * (1 - r),
                }
            post, prior = self.obs_step(state, action[:, t], embed[:, t])
            posts.append(post)
            priors.append(prior)
            state = post

        def stack(lst):
            return {k: torch.stack([d[k] for d in lst], 1) for k in lst[0]}

        return stack(posts), stack(priors)

    def imagine_with_action(self, action: torch.Tensor, state: dict) -> dict:
        """给定未来动作序列，先验自由推演（不看任何观测）。

        **这是世界模型的核心能力**：反事实推演、多步预测都走这里。

        Args:
            action: (B, K, A) 未来 K 步动作
            state:  初始状态，每个张量 (B, ...)

        Returns:
            prior 轨迹 {stoch:(B,K,...), deter:(B,K,D), ...}
        """
        B, K = action.shape[:2]
        out = {k: [] for k in self.initial(B, action.device)}
        s = state
        for t in range(K):
            s = self.img_step(s, action[:, t])
            for k in out:
                out[k].append(s[k])
        return {k: torch.stack(v, 1) for k, v in out.items()}

    # ---- 损失 ----

    def kl_loss(self, post: dict, prior: dict) -> tuple:
        """KL 损失（含 free bits 与 dyn/rep 平衡）。

        对齐原实现 `kl_loss`：
            rep_loss = KL(post || sg(prior))   # 训练后验去追先验
            dyn_loss = KL(sg(post) || prior)   # 训练先验去追后验
            两者都做 clip(min=free) 的下界裁剪（free bits）
            loss = dyn_scale * dyn_loss + rep_scale * rep_loss

        为什么 free bits：
            若潜变量后验与先验几乎相同（KL≈0），说明该维没携带信息，
            不应继续施加压力。free bits 让每维 KL 至少保持 `free` nat 的"预算"。
        """
        cfg = self.cfg
        kld = torchd.kl.kl_divergence
        sg = lambda x: {k: v.detach() for k, v in x.items()}

        def _kl(a, b):
            da, db = self.get_dist(a), self.get_dist(b)
            if self._discrete:
                return kld(da, db)
            return kld(da._dist, db._dist)

        rep_loss = _kl(post, sg(prior))
        dyn_loss = _kl(sg(post), prior)
        value = rep_loss.detach()

        rep_loss = torch.clip(rep_loss, min=cfg.kl_free)
        dyn_loss = torch.clip(dyn_loss, min=cfg.kl_free)
        loss = cfg.dyn_scale * dyn_loss + cfg.rep_scale * rep_loss

        return loss, value, dyn_loss, rep_loss


# --------------------------------------------------------------------------
# Encoder / Decoder（简化：纯 MLP）
# --------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """数值观测编码器。

    相对原实现的改动：
        - 去掉 CNN 路径（观测是 ~10 维数值）
        - 去掉 symlog_inputs 的自动判断，改为显式参数
        - 保留 LayerNorm（与官方 norm=rms 等价作用；PyTorch 无 RMSNorm 时的近似）
    """

    def __init__(self, obs_dim: int, embed_dim: int = 256,
                 layers: int = 3, units: int = 512,
                 act: str = "SiLU", norm: bool = True,
                 symlog_inputs: bool = True):
        super().__init__()
        self.symlog_inputs = symlog_inputs
        act_fn = _act(act)
        mods: list[nn.Module] = []
        d = obs_dim
        for _ in range(layers):
            mods.append(nn.Linear(d, units))
            if norm:
                mods.append(nn.LayerNorm(units))
            mods.append(act_fn())
            d = units
        mods.append(nn.Linear(d, embed_dim))
        self.net = nn.Sequential(*mods)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.symlog_inputs:
            obs = torch.sign(obs) * torch.log(torch.abs(obs) + 1.0)
        return self.net(obs)


class MLPDecoder(nn.Module):
    """数值输出解码器（symlog_mse 头，与官方 rssm.py:299 一致）。"""

    def __init__(self, feat_dim: int, obs_dim: int,
                 layers: int = 3, units: int = 512,
                 act: str = "SiLU", norm: bool = True):
        super().__init__()
        act_fn = _act(act)
        mods: list[nn.Module] = []
        d = feat_dim
        for _ in range(layers):
            mods.append(nn.Linear(d, units))
            if norm:
                mods.append(nn.LayerNorm(units))
            mods.append(act_fn())
            d = units
        mods.append(nn.Linear(d, obs_dim))
        self.net = nn.Sequential(*mods)
        self.obs_dim = obs_dim

    def forward(self, feat: torch.Tensor) -> _SymlogDist:
        """返回 SymlogDist，调用方用 .mode() 取值、.log_prob() 算损失。"""
        return _SymlogDist(self.net(feat))

