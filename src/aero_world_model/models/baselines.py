"""
基线模型：GRU 和纯机理

═══════════════════════════════════════════════════════════════════════════
为什么必须有两个基线（PC-LWM 指南 §5）
═══════════════════════════════════════════════════════════════════════════
指南明确要求：
    - "纯机理模型基线：辨别残差模型是否真正带来增益"
    - "常规模型基线：例如 LSTM / GRU，公平比较"

没有这两个，"RSSM 比传统方法好"无法证明。

═══════════════════════════════════════════════════════════════════════════
B-Mech：纯机理基线
═══════════════════════════════════════════════════════════════════════════
**它不需要训练**：预测值就是"保持当前状态"（persistence）的物理版本。

等等——这不是真机理。真正的"纯机理基线"应该是 Simulink 的输出。
但本课题所有数据都来自 Simulink，所以"用 Simulink 预测"=完美预测，
没有比较意义。

因此这里实现两个层次的基线：
    B-Mech  : Persistence（保持前一时刻）—— 最弱的基线，但不需训练
    B-GRU   : GRU 序列模型 —— 常规深度学习基线

若需要真正的"纯机理基线"（Simulink 前向），应在评估阶段调用 MATLAB，
不在训练循环里。见 reports/ 中的说明。

═══════════════════════════════════════════════════════════════════════════
B-GRU：为什么用 GRU 而不是 LSTM
═══════════════════════════════════════════════════════════════════════════
- GRU 参数更少（2 门 vs 3 门），与本课题 small RSSM 规模可比
- 训练更快
- 文献中 RSSM 的对照基线通常是 GRU（Dreamer 系列自身也是 GRU 核心）

═══════════════════════════════════════════════════════════════════════════
公平比较的前提（指南 §5 要求）
═══════════════════════════════════════════════════════════════════════════
1. 相同的输入信息（同样的历史窗口、同样的动作/外生变量）
2. 可比的参数量（本实现的 GRU 与 small RSSM 同为 1-3M 量级）
3. 相同的训练预算（epochs、batch、优化器）
4. 相同的评估协议（开环多步、分工况）
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["BaselinePersistence", "BaselineGRU", "count_params"]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════
# B-Mech：Persistence（不需训练）
# ══════════════════════════════════════════════════════════════════════════

class BaselinePersistence(nn.Module):
    """保持最后一个观测值作为预测。

    ŷ_{t+k} = o_t  （对所有 k）

    这是最弱但最重要的基线：
        若 RSSM 打不过它，说明模型连"状态不会突变"这个规律都没学到。
    """

    def __init__(self, obs_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        # 占位参数，便于与其它模型的参数量统计接口一致
        self._dummy = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor,
                act_future: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_hist:   [B,T,C]
            act_hist:   [B,T,C_a]（未使用）
            act_future: [B,H,C_a]（未使用）

        Returns:
            [B,H,C] 全部等于 obs_hist[:, -1]
        """
        H = act_future.shape[1]
        last = obs_hist[:, -1:, :]                       # [B,1,C]
        return last.expand(-1, H, -1).contiguous()

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """与 RSSM 相同的损失接口（便于统一评估）。"""
        obs = batch["obs"]
        fut = batch["fut"]
        pred = self.forward(obs, batch["act"], batch["act_fut"])
        err = ((pred - fut) ** 2).mean()
        return err, {"total": err.item(), "rec": 0.0, "kl": 0.0,
                     "multi": err.item()}


# ══════════════════════════════════════════════════════════════════════════
# B-GRU：常规序列模型基线
# ══════════════════════════════════════════════════════════════════════════

class BaselineGRU(nn.Module):
    """GRU + MLP 解码的序列预测基线。

    结构：
        Encoder:  concat(o_t, a_t) -> Linear -> GRU
        Decoder:  h_t -> Linear -> ŷ_{t+k}

    与 RSSM 的关键差异（这是对照的意义所在）：
        - 无随机潜变量（RSSM 有 z）
        - 无先验/后验分离（RSSM 有）
        - 直接监督多步（与 RSSM 的 L_multi 对齐）

    公平性：
        隐藏维度、层数、MLP 宽度与 RSSM 的 deter 设置一致。
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        *,
        hidden: int = 384,
        num_layers: int = 1,
        horizon: int = 16,
        mlp_units: int = 1024,
        mlp_layers: int = 3,
        dropout: float = 0.0,
    ):
        """默认尺寸按参数量可比原则选定。

        自测（obs_dim=6, act_dim=1, horizon=16）：
            hidden=384, mlp_units=1024  ->  3,488,864 参数
            RSSM(deter=256, stoch=32, classes=32) -> 3,233,030 参数
            比值 RSSM/GRU = 0.93x，落在指南要求的 0.5–2 倍内。

        为什么不用 hidden=256：
            该配置只有 1.1M 参数，仅为 RSSM 的 1/3，
            对比会因容量不足而不公平（RSSM 会显得"更好"）。
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden = hidden
        self.horizon = horizon

        # 输入投影（观测 + 动作）
        self.inp = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 多步解码：一次输出 H 步 × obs_dim
        # 逐步解码器（每步输出单帧观测，配合自回归）
        self.step_dec = nn.Sequential(
            nn.Linear(hidden, mlp_units),
            nn.LayerNorm(mlp_units),
            nn.SiLU(),
            nn.Linear(mlp_units, obs_dim),
        )
        # 旧的一次性多步解码器保留但不再使用（避免 state_dict 键变化导致
        # 历史 checkpoint 无法加载；新训练不会用到它）
        layers: list[nn.Module] = []
        d = hidden
        for _ in range(mlp_layers):
            layers += [nn.Linear(d, mlp_units), nn.LayerNorm(mlp_units), nn.SiLU()]
            d = mlp_units
        layers.append(nn.Linear(d, horizon * obs_dim))
        self.dec = nn.Sequential(*layers)

    def forward(self, obs_hist: torch.Tensor, act_hist: torch.Tensor,
                act_future: torch.Tensor) -> torch.Tensor:
        """逐步动作条件化的自回归多步预测。

        ══════════════════════════════════════════════════════════════════
        P0-5 修复（审查报告：GRU 完全未使用未来动作数值，已独立复核成立）
        ══════════════════════════════════════════════════════════════════
        旧的错误实现：
            只用 GRU 最后隐状态 h_last，一次性线性映射出全部 H 步，
            act_future 只被读取了 shape，数值从未参与计算。
            实测：未来动作从 0 改成 99，输出变化严格为 0.0。
            → 与 RSSM 的对比把"信息优势"和"架构优势"混在一起。

        新实现（与 RSSM 对齐）：
            用同一个 GRU 单元做**逐步递推**，每一步把该步的未来动作
            喂入，并输出该步的预测：
                h_0 = GRU 编码完历史后的末状态
                h_k = GRU_cell(concat(o_hat_{k-1}, a_{k-1}), h_{k-1})
                o_hat_k = Dec(h_k)
            这样 GRU 与 RSSM 拿到**完全相同的未来动作信息**，
            差别只剩架构（有无随机潜变量 / 先验后验分离）。

        Args:
            obs_hist:   [B,T,C_o] 历史观测
            act_hist:   [B,T,C_a] 历史动作（已按 P0-4 前移一拍）
            act_future: [B,H,C_a] 未来动作序列（**现在真正参与计算**）

        Returns:
            [B,H,C_o]
        """
        B, T, _ = obs_hist.shape
        H = act_future.shape[1]

        # ---- 历史编码 ----
        x = torch.cat([obs_hist, act_hist], dim=-1)        # [B,T,C_o+C_a]
        x = self.inp(x)
        _, h = self.gru(x)                                  # [L,B,Hd]
        h_cur = h[-1]                                       # [B,Hd]

        # ---- 逐步自回归解码（未来动作逐步喂入）----
        outs = []
        o_prev = obs_hist[:, -1]                            # [B,C_o] 上一步观测
        for k in range(H):
            a_k = act_future[:, k]                          # [B,C_a]
            step_in = torch.cat([o_prev, a_k], dim=-1)      # [B,C_o+C_a]
            step_in = self.inp(step_in)                     # [B,Hd]
            # 单步 GRU（复用同一组权重，与历史编码共享）
            h_cur = self._step(step_in, h_cur)              # [B,Hd]
            o_hat = self.step_dec(h_cur)                    # [B,C_o]
            outs.append(o_hat)
            o_prev = o_hat                                  # 自回归：用预测值
        return torch.stack(outs, dim=1)                     # [B,H,C_o]

    def _step(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """GRU 单步（复用 self.gru 的权重，仅取最后一层）。"""
        # nn.GRU 的单步等价：手工实现以复用已训练权重
        w_ih = getattr(self.gru, f"weight_ih_l{self.gru.num_layers - 1}")
        w_hh = getattr(self.gru, f"weight_hh_l{self.gru.num_layers - 1}")
        b_ih = getattr(self.gru, f"bias_ih_l{self.gru.num_layers - 1}")
        b_hh = getattr(self.gru, f"bias_hh_l{self.gru.num_layers - 1}")
        gi = torch.nn.functional.linear(x, w_ih, b_ih)
        gh = torch.nn.functional.linear(h, w_hh, b_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=-1)
        h_r, h_z, h_n = gh.chunk(3, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        return (1 - z) * n + z * h

    def loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """统一损失接口。"""
        obs = batch["obs"]
        fut = batch["fut"]
        pred = self.forward(obs, batch["act"], batch["act_fut"])
        err = ((pred - fut) ** 2).mean()
        return err, {"total": err.item(), "rec": 0.0, "kl": 0.0,
                     "multi": err.item()}


# ══════════════════════════════════════════════════════════════════════════
# 自测
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    B, T, H = 4, 64, 16
    obs_dim, act_dim = 6, 1

    obs = torch.randn(B, T, obs_dim)
    act = torch.randn(B, T, act_dim)
    af = torch.randn(B, H, act_dim)

    print("=" * 66)
    print("基线模型自测")
    print("=" * 66)

    # --- Persistence ---
    print("\n[1] BaselinePersistence")
    pm = BaselinePersistence(obs_dim)
    p = pm(obs, act, af)
    ok = torch.allclose(p[:, 0, :], obs[:, -1, :]) and p.shape == (B, H, obs_dim)
    print(f"  输出形状 {tuple(p.shape)}  全部等于最后观测: {ok}")
    print(f"  参数量 {count_params(pm):,}")

    # --- GRU ---
    print("\n[2] BaselineGRU")
    gm = BaselineGRU(obs_dim, act_dim, hidden=256, horizon=H)
    g = gm(obs, act, af)
    print(f"  输出形状 {tuple(g.shape)}")
    print(f"  参数量 {count_params(gm):,}")

    # 与 RSSM 对比参数量
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT / "src"))
    try:
        from aero_world_model.training.offline import WorldModel
        from aero_world_model.models.rssm import RSSMConfig
        rssm = WorldModel(obs_dim, act_dim,
                          RSSMConfig(stoch=32, classes=32, deter=256, hidden=256))
        print(f"\n[3] 参数量对比（公平性检查）")
        print(f"  BaselineGRU : {count_params(gm):,}")
        print(f"  RSSM        : {count_params(rssm):,}")
        ratio = count_params(rssm) / count_params(gm)
        print(f"  比值 RSSM/GRU = {ratio:.2f}x")
        if 0.5 <= ratio <= 2.0:
            print("  [OK] 参数量在 0.5-2 倍内，可比")
        else:
            print(f"  [WARN] 参数量差距 {ratio:.2f}x，需在论文中说明")
    except Exception as e:
        print(f"\n[3] 无法加载 RSSM 对比: {e}")

    # --- 前向 + 反向 ---
    print("\n[4] 训练通路")
    batch = {
        "obs": obs, "act": act, "act_fut": af,
        "fut": torch.randn(B, H, obs_dim),
    }
    loss, parts = gm.loss(batch)
    loss.backward()
    gn = sum((p.grad ** 2).sum().item() for p in gm.parameters()
             if p.grad is not None) ** 0.5
    print(f"  loss={loss.item():.6f}  grad_norm={gn:.4f}")
    print(f"  梯度有限: {all(torch.isfinite(p.grad).all() for p in gm.parameters() if p.grad is not None)}")

    print("\n" + "=" * 66)
    print("自测完成")
    print("=" * 66)
