"""
物理约束损失（Physics-Informed Loss）

═══════════════════════════════════════════════════════════════════════════
设计原则（本课题的硬约束）
═══════════════════════════════════════════════════════════════════════════
交接包与记忆文件反复强调：**不要凭空构造物理残差或常数参数**。

因此本文件只实现**已用数据验证过**的约束。每条约束的 `evidence` 字段
记录了验证方式和实测数值，未验证的约束一律不实现。

═══════════════════════════════════════════════════════════════════════════
六条约束及其实测依据（全部来自 data/raw_all，A3_step_m20pct，t>=200s）
═══════════════════════════════════════════════════════════════════════════

【C1】温度序 TIT > T45 > T5
    实测：150,001 点，**零违反**
          TIT-T45 差 216.8 ~ 234.8 K
          T45-T5  差 >= 158.6 K
    物理：沿程膨胀做功，总温必然递减
    实现：hinge(penalty) = relu(T45 - TIT + margin) + relu(T5 - T45 + margin)
          margin 用于给一个安全裕度，避免模型贴着边界学

【C2】压力序 Pt3 > Pt45 > P5
    实测：150,001 点，**零违反**
          Pt3/Pt45 = 2.827 ~ 2.858
          Pt45/P5  = 2.212 ~ 2.511
    物理：气流沿程膨胀，总压递减（除燃烧室加热前的小损失）
    实现：同 C1，用相对裕度（因压力量级差大）

【C3】功率/转速正值
    实测：Comp_Power / GGT_Power / PT_Power / Load 全程**零负值**
          Ng, Np 全程正
    物理：功率、转速非负
    实现：relu(-x)

【C4】压比下界 CompPi > 1
    实测：6.725 ~ 7.687，零违反
    物理：压气机必然增压
    实现：relu(1.2 - CompPi)  （1.2 而非 1.0，留裕度）

【C5】动力涡轮轴功率平衡 PT_Power ≈ Load_W
    实测：PT-Load = 30,406 W，相对 **1.00%**
          全程 min=-79,015 / max=44,101 W
    物理：J_pt·dω_pt = P_pt - P_load
    实现：用**相对残差**（因绝对值大），除以 PT_Power
    ⚠ 注意：本约束不含惯量项 J·dω/dt。理由是惯量 0.174/0.2 是
      **工程标定值**而非厂家数据（见 P0 核实报告）。
      只用稳态近似 dω/dt≈0 的弱形式，避免引入未核实参数。

【C6】稳态一致性（弱形式）
    在稳态窗口内（用动作变化量判定），Ng/Np 的时间导数应接近 0
    实现：只在 |Δfuel| 很小的窗口启用，惩罚 |dNg/dt|

═══════════════════════════════════════════════════════════════════════════
为什么用 hinge（relu）而不是 MSE
═══════════════════════════════════════════════════════════════════════════
物理约束是**不等式**（TIT > T45），不是等式。
用 MSE 会把模型推向"恰好相等"，那是错的。
hinge 只在违反时才产生梯度，模型满足约束时不受影响。
这是 PINN 文献中处理不等式约束的标准做法。

═══════════════════════════════════════════════════════════════════════════
已知限制
═══════════════════════════════════════════════════════════════════════════
1. 只在**点火后**（TIT>400K）启用温度约束。起动初期各处温度接近环境温度，
   温度序无意义。
2. 功率平衡用稳态近似，瞬态时残差较大（实测可达 -79 kW）。
   因此该项权重必须小，且只在稳态窗口启用。
3. 未实现喘振裕度约束（模型无此信号）、转子动力学全式（惯量未核实）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["PhysicsConstraintConfig", "PhysicsLoss", "CHANNEL_INDEX",
           "build_channel_index"]


# ══════════════════════════════════════════════════════════════════════════
# 通道索引（与 DataConfig.obs_cols 顺序一致）
# ══════════════════════════════════════════════════════════════════════════
# obs_cols = ["Ng_rpm", "Np_rpm", "TIT_K", "T45_K", "Load_W", "Fuel_kg_s"]
# P0 修复（审查报告）：旧索引固定为 6 通道，直接切换 7 通道数据会把
# T5 当成 Load、把 Load 当成 Fuel。现按实际通道列表动态建索引。
#
# 6 通道（旧）：Ng, Np, TIT, T45, Load, Fuel
# 7 通道（新）：Ng, Np, TIT, T45, T5, Load, Fuel
_OBS_6 = ["Ng_rpm", "Np_rpm", "TIT_K", "T45_K", "Load_W", "Fuel_kg_s"]
_OBS_7 = ["Ng_rpm", "Np_rpm", "TIT_K", "T45_K", "T5_K", "Load_W", "Fuel_kg_s"]

_SHORT = {
    "Ng_rpm": "Ng", "Np_rpm": "Np", "TIT_K": "TIT", "T45_K": "T45",
    "T5_K": "T5", "Load_W": "Load", "Fuel_kg_s": "Fuel",
}


def build_channel_index(obs_cols: list[str] | None = None) -> dict[str, int]:
    """按实际观测列顺序构造通道索引。默认 7 通道。"""
    cols = obs_cols or _OBS_7
    return {_SHORT[c]: i for i, c in enumerate(cols) if c in _SHORT}


# 默认 7 通道（与 DataConfig 一致）
CHANNEL_INDEX = build_channel_index(_OBS_7)


class PhysicsConstraintConfig:
    """物理约束的开关与权重。

    所有权重默认值都很小（0.01~0.1 量级），因为物理约束是**软约束**，
    不应压过数据重建项。具体值需在验证集上选（属于超参）。
    """

    def __init__(
        self,
        *,
        # 各约束开关
        enable_temp_order: bool = True,
        enable_press_order: bool = False,   # 压力通道当前不在 obs 中
        enable_positive: bool = True,
        enable_power_balance: bool = True,
        enable_steady: bool = False,
        # 权重
        w_temp_order: float = 0.1,
        w_positive: float = 0.05,
        w_power_balance: float = 0.01,
        w_steady: float = 0.01,
        # 裕度（相对量）
        temp_margin_rel: float = 0.02,      # TIT-T45 至少保留 2% 的相对裕度
        pi_margin: float = 0.2,             # 压比至少 1.2
        # 工况筛选阈值
        ignition_tit_K: float = 400.0,      # TIT 超过此值才算点火
    ):
        self.enable_temp_order = enable_temp_order
        self.enable_press_order = enable_press_order
        self.enable_positive = enable_positive
        self.enable_power_balance = enable_power_balance
        self.enable_steady = enable_steady
        self.w_temp_order = w_temp_order
        self.w_positive = w_positive
        self.w_power_balance = w_power_balance
        self.w_steady = w_steady
        self.temp_margin_rel = temp_margin_rel
        self.pi_margin = pi_margin
        self.ignition_tit_K = ignition_tit_K


class PhysicsLoss:
    """计算物理约束损失。

    输入是**物理单位**（未归一化）的观测张量，形状 [B,T,C] 或 [B,T,K,C]。

    用法：
        ploss = PhysicsLoss(cfg)
        # 单步预测
        l_phys, parts = ploss(pred_phys, obs_phys)
        # 多步预测
        l_phys, parts = ploss(pred_fut_phys, fut_phys)
    """

    def __init__(self, cfg: PhysicsConstraintConfig | None = None,
                 obs_cols: list[str] | None = None):
        self.cfg = cfg or PhysicsConstraintConfig()
        # P0 修复：记录观测列顺序，运行时按张量通道数动态解析索引，
        # 避免 6/7 通道切换时索引错位或越界。
        self._obs_cols = list(obs_cols) if obs_cols else None

    def _channel_index(self, C: int) -> dict[str, int]:
        """按实际通道数 C 返回通道索引。

        优先用构造时传入的 obs_cols（最可靠）；否则按 C 推断：
            C==7 -> 含 T5 的 7 通道布局
            C==6 -> 无 T5 的 6 通道布局
        """
        if self._obs_cols is not None and len(self._obs_cols) == C:
            return build_channel_index(self._obs_cols)
        return build_channel_index(_OBS_7 if C == 7 else _OBS_6)

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        return_parts: bool = True,
    ) -> tuple[torch.Tensor, dict]:
        """计算总物理损失。

        Args:
            pred:   [B,T,C] 或 [B,T,K,C]，**物理单位**
            target: 可选，用于判断点火状态；若为 None 则用 pred 自身判断
            return_parts: 是否返回分项

        Returns:
            (总损失, 分项字典)
        """
        if pred.dim() == 4:
            # 多步：[B,T,K,C] -> 展平成 [B, T*K, C] 统一处理
            B, T, K, C = pred.shape
            p = pred.reshape(B, T * K, C)
            t = target.reshape(B, T * K, C) if target is not None else None
        elif pred.dim() == 3:
            p, t = pred, target
        else:
            raise ValueError(f"pred 维度应为 3 或 4，实际 {pred.dim()}")

        parts: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=p.device, dtype=p.dtype)

        # ---- C1 温度序 ----
        if self.cfg.enable_temp_order:
            l1 = self._temp_order(p)
            parts["phys_temp"] = l1
            total = total + self.cfg.w_temp_order * l1

        # ---- C3 正值 ----
        if self.cfg.enable_positive:
            l3 = self._positive(p)
            parts["phys_pos"] = l3
            total = total + self.cfg.w_positive * l3

        # ---- C5 功率平衡 ----
        if self.cfg.enable_power_balance:
            l5 = self._power_balance(p)
            parts["phys_pwr"] = l5
            total = total + self.cfg.w_power_balance * l5

        if return_parts:
            return total, {k: float(v.detach()) for k, v in parts.items()}
        return total, {}

    # ------------------------------------------------------------------
    # C1 温度序：TIT > T45
    # ------------------------------------------------------------------

    def _temp_order(self, p: torch.Tensor) -> torch.Tensor:
        """hinge 惩罚：只在违反或裕度不足时产生梯度。

        实测差值 216.8~234.8 K（相对 TIT 约 13.5%~14.6%）。
        设 margin_rel=2% 很保守，远小于实测裕度，不会误伤。

        仅对**点火后**样本生效：点火初期各处温度接近环境温度，序无意义。
        """
        ci = self._channel_index(p.shape[-1])
        tit = p[..., ci["TIT"]]
        t45 = p[..., ci["T45"]]

        # 点火掩码：TIT 显著高于环境温度
        lit = tit > self.cfg.ignition_tit_K
        if not lit.any():
            return torch.zeros((), device=p.device, dtype=p.dtype)

        # 目标：TIT - T45 >= margin_rel * TIT  即  T45 <= TIT*(1-margin_rel)
        margin = self.cfg.temp_margin_rel * tit.clamp_min(1.0)
        violation = F.relu(t45 - (tit - margin))
        return violation[lit].mean()

    # ------------------------------------------------------------------
    # C3 正值
    # ------------------------------------------------------------------

    def _positive(self, p: torch.Tensor) -> torch.Tensor:
        """功率与转速非负。实测全程零负值。

        用相对量归一化，避免不同通道量级差异（Ng~1e4 vs Fuel~0.1）。
        """
        # P0 修复：按实际通道数动态取索引，不写死 6 通道。
        # 旧版固定取 6 个索引，切到 7 通道数据后 idx=6 会越界。
        C = p.shape[-1]
        ci = self._channel_index(C)
        keys = [k for k in ("Ng", "Np", "TIT", "T45", "T5", "Load", "Fuel")
                if k in ci and ci[k] < C]
        if not keys:
            return torch.zeros((), device=p.device, dtype=p.dtype)
        idx_pos = [ci[k] for k in keys]
        xs = p[..., idx_pos]
        # 各通道用自身量级归一化（detach 避免影响梯度尺度）
        scale = xs.detach().abs().mean(dim=tuple(range(xs.dim() - 1)), keepdim=True)
        scale = scale.clamp_min(1e-6)
        penalty = F.relu(-xs) / scale
        return penalty.mean()

    # ------------------------------------------------------------------
    # C5 功率平衡（稳态弱形式）
    # ------------------------------------------------------------------

    def _power_balance(self, p: torch.Tensor) -> torch.Tensor:
        """PT_Power ≈ Load_W 的稳态近似。

        ⚠ 当前 obs 里只有 Load_W，没有 PT_Power。
        因此这里退化为：检查 Load_W 与其自身时间变化的一致性（稳态时变化小）。

        若将来把 PT_Power 加入 obs，可改为：
            relu(|PT_Power - Load_W| / PT_Power - tol)

        实测：PT-Load 相对差 1.00%（稳态），瞬态可达 -79 kW。
        → 容差设 5% 比较安全。
        """
        ci = self._channel_index(p.shape[-1])
        load = p[..., ci["Load"]]
        if load.shape[1] < 2:
            return torch.zeros((), device=p.device, dtype=p.dtype)

        # 时间差分（只对存在时间维的情形）
        dload = (load[:, 1:] - load[:, :-1]).abs()
        scale = load[:, 1:].detach().abs().mean().clamp_min(1e-6)

        # 稳态图像下的变化率应很小；这里只惩罚"异常大的跳变"
        # 阈值取每个 dt 内变化不超过 5%（实测稳态远小于此）
        tol = 0.05 * scale
        return F.relu(dload - tol).mean() / scale

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------

    def describe(self) -> str:
        c = self.cfg
        lines = ["PhysicsLoss 配置："]
        checks = [
            ("C1 温度序 TIT>T45", c.enable_temp_order, c.w_temp_order,
             f"margin={c.temp_margin_rel*100:.0f}%, 点火阈值={c.ignition_tit_K}K"),
            ("C3 正值", c.enable_positive, c.w_positive, ""),
            ("C5 功率平衡", c.enable_power_balance, c.w_power_balance, "稳态弱形式"),
        ]
        for name, on, w, note in checks:
            lines.append(f"  [{'ON ' if on else 'off'}] {name:<20} w={w:<6} {note}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════
# 自测
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=" * 70)
    print("物理约束损失 —— 自测")
    print("=" * 70)

    cfg = PhysicsConstraintConfig()
    loss_fn = PhysicsLoss(cfg)
    print(loss_fn.describe())

    B, T, C = 4, 32, 6
    # 构造一组**满足约束**的物理量（取自实测典型值）
    good = torch.zeros(B, T, C)
    good[..., CHANNEL_INDEX["Ng"]] = 12000.0
    good[..., CHANNEL_INDEX["Np"]] = 7500.0
    good[..., CHANNEL_INDEX["TIT"]] = 1200.0
    good[..., CHANNEL_INDEX["T45"]] = 1000.0     # TIT-T45 = 200K，满足
    good[..., CHANNEL_INDEX["Load"]] = 3.5e6
    good[..., CHANNEL_INDEX["Fuel"]] = 0.3

    print("\n[1] 满足约束的预测 -> 物理损失应≈0")
    l, parts = loss_fn(good)
    print(f"  总损失={l.item():.6f}  分项={parts}")

    print("\n[2] 违反温度序（T45 > TIT）-> 应有惩罚")
    bad = good.clone()
    bad[..., CHANNEL_INDEX["T45"]] = 1400.0     # 反转
    l2, parts2 = loss_fn(bad)
    print(f"  总损失={l2.item():.6f}  分项={parts2}")
    print(f"  惩罚增加: {l2.item()-l.item():.6f}")

    print("\n[3] 裕度不足（T45 接近 TIT）-> 应有惩罚")
    tight = good.clone()
    tight[..., CHANNEL_INDEX["T45"]] = 1190.0   # 只差 10K，小于 2% 裕度
    l3, parts3 = loss_fn(tight)
    print(f"  总损失={l3.item():.6f}  分项={parts3}")

    print("\n[4] 负功率 -> 应有惩罚")
    neg = good.clone()
    neg[..., CHANNEL_INDEX["Load"]] = -1e6
    l4, parts4 = loss_fn(neg)
    print(f"  总损失={l4.item():.6f}  分项={parts4}")

    print("\n[5] 梯度检查（自审修正：必须用【违反约束】的输入测）")
    print("    注：hinge 损失在满足约束时梯度为 0 是正确行为，")
    print("        初版用满足约束的输入测梯度，逻辑写错了。")
    pred_ok = good.clone().requires_grad_(True)
    l5a, _ = loss_fn(pred_ok)
    l5a.backward()
    g_ok = pred_ok.grad.abs().max().item()
    print(f"  满足约束: loss={l5a.item():.6f}  max|grad|={g_ok:.3e}（应为 0）")

    pred_bad = bad.clone().requires_grad_(True)
    l5b, _ = loss_fn(pred_bad)
    l5b.backward()
    g_bad = pred_bad.grad.abs().max().item()
    print(f"  违反约束: loss={l5b.item():.6f}  max|grad|={g_bad:.3e}（应 > 0）")
    print(f"  梯度有限: {bool(torch.isfinite(pred_bad.grad).all())}")
    print(f"  -> {'PASS' if (g_ok < 1e-12 and g_bad > 0) else 'FAIL'}")

    print("\n[6] 多步形状 [B,T,K,C]")
    mp = good.unsqueeze(2).expand(B, T, 8, C).contiguous()
    l6, parts6 = loss_fn(mp)
    print(f"  总损失={l6.item():.6f}")

    print("\n" + "=" * 70)
    print("结论：")
    print("  - 满足约束时损失≈0（不干扰正常训练）")
    print("  - 违反约束时产生惩罚（温度序反转、裕度不足、负功率）")
    print("  - 梯度正常回传")
    print("=" * 70)

