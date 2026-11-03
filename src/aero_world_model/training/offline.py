"""
离线训练循环（RSSM + 多时间尺度加权）

═══════════════════════════════════════════════════════════════════════════
本文件实现的内容
═══════════════════════════════════════════════════════════════════════════
1. 落地的模型组合：MLPEncoder → RSSM → MLPDecoder（symlog_mse）
2. 损失：
      L = λ_rec · L_rec^MTW + λ_multi · L_multi^MTW + λ_kl · L_kl
   其中 L_rec/ L_multi 用 MultiTimescaleWeighter（见 losses/mtw.py）
3. 训练/验证循环，早停，checkpoint
4. 记录「训练集大小 vs 验证误差」的学习曲线（回答"数据够不够"）

═══════════════════════════════════════════════════════════════════════════
与官方 DreamerV3 的对应关系（便于论文说明）
═══════════════════════════════════════════════════════════════════════════
- Encoder / RSSM / Decoder 结构：对齐官方（见 reports/G1.1_RSSM等价性核对）
- symlog_mse 输出头：对齐官方 rssm.py:299
- KL 平衡（dyn_scale=0.5, rep_scale=0.1, kl_free=1.0）：对齐官方
- **本课题改动**：
    (a) 移除 RL 相关（reward/value/policy/ImagBehavior）
    (b) 加入多步开环损失 L_multi（官方主要靠想象训练，我们用离线数据）
    (c) 加入多时间尺度加权（MTW，见 mtw.py）

═══════════════════════════════════════════════════════════════════════════
已知限制
═══════════════════════════════════════════════════════════════════════════
- 未实现 GPLD 平滑正则（第二阶段）
- 未实现物理约束损失（第二阶段）
- 未实现集成（第二阶段）
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aero_world_model.models.rssm import (  # noqa: E402
    MLPDecoder, MLPEncoder, RSSM, RSSMConfig,
)
from aero_world_model.losses.mtw import (  # noqa: E402
    MultiTimescaleWeighter, TimescaleProfile,
)

__all__ = ["TrainConfig", "WorldModel", "Trainer"]


# ══════════════════════════════════════════════════════════════════════════
# 模型封装
# ══════════════════════════════════════════════════════════════════════════

class WorldModel(nn.Module):
    """Encoder + RSSM + Decoder 的完整世界模型。"""

    def __init__(self, obs_dim: int, act_dim: int, cfg: RSSMConfig,
                 embed_dim: int = 256, mlp_units: int = 512, mlp_layers: int = 3):
        super().__init__()
        self.cfg = cfg
        self.encoder = MLPEncoder(obs_dim, embed_dim=embed_dim,
                                  layers=mlp_layers, units=mlp_units)
        self.dynamics = RSSM(act_dim, cfg)
        self.dynamics.build(embed_dim)
        self.decoder = MLPDecoder(cfg.feat_dim, obs_dim,
                                  layers=mlp_layers, units=mlp_units)

    def forward(self, obs: torch.Tensor, act: torch.Tensor):
        """返回 (post, prior, recon_mode, recon_dist)。"""
        embed = self.encoder(obs)
        post, prior = self.dynamics.observe(embed, act)
        feat = self.dynamics.get_feat(post)
        recon_dist = self.decoder(feat)
        return post, prior, recon_dist.mode(), recon_dist

    def imagine(self, obs_hist: torch.Tensor, act_hist: torch.Tensor,
                act_future: torch.Tensor):
        """给定历史观测/动作与未来动作，先验自由推演未来观测。

        Args:
            obs_hist:   [B, T, C_o] 历史观测（用来建立后验初始状态）
            act_hist:   [B, T, C_a] 历史动作（**必须传入真实值**，
                        用 zeros 会让后验状态失真 —— 自审修正）
            act_future: [B, K, C_a] 未来动作序列

        Returns:
            pred: [B, K, C_o] 预测的未来观测（归一化空间）
        """
        embed = self.encoder(obs_hist)
        # 用真实历史动作建立后验（自审修正：原实现传 zeros，错误）
        post, _ = self.dynamics.observe(embed, act_hist)
        # 取最后一个后验状态作为推演起点
        state = {k: v[:, -1] for k, v in post.items()}
        trail = self.dynamics.imagine_with_action(act_future, state)
        feat = self.dynamics.get_feat(trail)          # [B,K,F]
        pred = self.decoder(feat).mode()               # [B,K,C]
        return pred


# ══════════════════════════════════════════════════════════════════════════
# 训练配置
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainConfig:
    # 优化
    lr: float = 4e-4              # 官方 4e-5，我们数据少且 batch 小，放大 10x
    weight_decay: float = 0.0
    grad_clip: float = 100.0      # 官方 grad_clip=100
    batch_size: int = 16
    epochs: int = 200
    patience: int = 30
    # 损失权重
    lam_rec: float = 1.0
    lam_multi: float = 1.0
    lam_kl: float = 1.0
    # MTW
    mtw_alpha: float = 0.5        # 0=等权（消融）
    mtw_beta: float = 3.0
    # 物理约束（指南 §3.3：只用有实测依据的约束）
    #   lam_phys=0 时完全关闭，作为消融的对照组 A/B
    lam_phys: float = 0.0
    enable_temp_order: bool = True      # TIT > T45（实测零违反）
    enable_positive: bool = True        # 功率/转速为正（实测零负值）
    # 多步损失的模式
    #   "true_future": 用真实未来动作（指南要求，默认）
    #   "copy_last"  : 复制最后一个动作（旧实现，仅用于复现历史 bug）
    multistep_mode: str = "true_future"
    # 其他
    seed: int = 0
    device: str = "cuda"
    out_dir: str = "experiments/run0"
    log_every: int = 10


# ══════════════════════════════════════════════════════════════════════════
# Trainer
# ══════════════════════════════════════════════════════════════════════════

class Trainer:
    def __init__(self, model: WorldModel, cfg: TrainConfig,
                 profile: TimescaleProfile, scaler):
        self.model = model.to(cfg.device)
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.out = Path(cfg.out_dir)
        self.out.mkdir(parents=True, exist_ok=True)

        self.opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                                    weight_decay=cfg.weight_decay)
        self.weighter = MultiTimescaleWeighter(
            profile, alpha=cfg.mtw_alpha, beta=cfg.mtw_beta,
            weights_are_symlog=False,      # 数据已归一化，无需再 symlog
            device=self.device,
        )
        self.scaler = scaler
        self.history: list[dict] = []
        self.best_val = float("inf")
        self.best_epoch = -1

        # 反归一化用的张量常量（P0-2 修复：保持计算图，不用 numpy 中转）
        # scaler 的均值/标准差原本是 numpy 数组，这里转成 device 上的张量。
        self._obs_mean_t = torch.as_tensor(
            np.asarray(scaler.obs_mean, dtype=np.float32), device=self.device)
        self._obs_std_t = torch.as_tensor(
            np.asarray(scaler.obs_std, dtype=np.float32), device=self.device)
        self._phys_warned = False

        # 物理约束（仅当 lam_phys > 0 时构建）
        self.phys = None
        if getattr(cfg, "lam_phys", 0.0) > 0.0:
            try:
                from aero_world_model.losses.physics import (
                    PhysicsConstraintConfig, PhysicsLoss,
                )
                pcfg = PhysicsConstraintConfig(
                    enable_temp_order=cfg.enable_temp_order,
                    enable_positive=cfg.enable_positive,
                    enable_press_order=False,   # 压力通道不在 obs 中
                )
                # 注意：PhysicsLoss 只接收配置，输入需是物理单位
                # 反归一化由 Trainer 用 scaler 完成
                self.phys = PhysicsLoss(pcfg)
            except Exception as e:      # noqa: BLE001
                print(f"[warn] 物理约束初始化失败，将禁用: {e}")
                self.phys = None

    # ---------------- 反归一化（保持梯度）----------------

    def _inverse_obs_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """在物理单位下反归一化，**全程保持 torch 张量以保留计算图**。

        P0-2 修复：旧实现经 detach().cpu().numpy() 中转，
        导致物理项对模型无梯度（已独立复核确认：loss 数值相同但
        requires_grad=False）。

        Args:
            x: [..., C] 归一化空间的预测

        Returns:
            [..., C] 物理单位，requires_grad 与输入一致
        """
        return x * self._obs_std_t + self._obs_mean_t

    # ---------------- 单步 ----------------

    def _compute_loss(self, batch: dict) -> tuple[torch.Tensor, dict]:
        obs = batch["obs"].to(self.device)       # [B,T,C_o]
        act = batch["act"].to(self.device)       # [B,T,C_a]
        fut = batch["fut"].to(self.device)       # [B,H,C_o]

        post, prior, recon_mode, _ = self.model(obs, act)

        # --- 重建（MTW 加权）---
        l_rec = self.weighter.reconstruction_loss(recon_mode, obs)

        # --- KL ---
        kl, kl_val, dyn_l, rep_l = self.model.dynamics.kl_loss(post, prior)
        l_kl = kl.mean()

        # --- 多步开环（MTW 加权）---
        # ⚠ 关键修正（指南 §3.1、§4.2）：
        #   必须用【真实未来动作序列】 act_fut，不能复制最后一个动作。
        #   复制会把阶跃/斜坡抹成常量，模型学不到「动作→未来」的因果。
        B, T, _ = obs.shape
        H = fut.shape[1]
        state = {k: v[:, -1] for k, v in post.items()}

        mode = getattr(self.cfg, "multistep_mode", "true_future")
        if mode == "copy_last":
            # 保留旧实现，仅用于复现历史 bug / 消融对照
            act_fut = act[:, -1:, :].expand(B, H, act.shape[-1])
        else:
            act_fut = batch["act_fut"].to(self.device)  # [B,H,C_a] 真实未来动作

        trail = self.model.dynamics.imagine_with_action(act_fut, state)
        feat = self.model.dynamics.get_feat(trail)      # [B,H,F]
        pred_fut = self.model.decoder(feat).mode()      # [B,H,C]
        # MTW 需要 [B,1,K,C] 形状 -> 加 T=1 维
        l_multi = self.weighter.multistep_loss(
            pred_fut.unsqueeze(1), fut.unsqueeze(1))

        # ══════════════════════════════════════════════════════════════════
        # P0-2 修复（审查报告：物理损失已断开梯度，已独立复核成立）
        # ══════════════════════════════════════════════════════════════════
        # 旧实现：pred_fut.detach().cpu().numpy() -> inverse -> from_numpy
        #   实测：loss 数值相同（11.80），但 requires_grad=False，
        #         物理项对整个模型没有任何梯度贡献。
        #
        # 新实现：全程保持 torch 张量，用 *Tensor 形式的均值/标准差反归一化，
        #         计算图完整保留，物理项真正参与优化。
        # ══════════════════════════════════════════════════════════════════
        l_phys_val = 0.0
        l_phys = torch.zeros((), device=self.device)
        if self.phys is not None and getattr(self.cfg, "lam_phys", 0.0) > 0.0:
            try:
                pred_phys = self._inverse_obs_tensor(pred_fut)
                phys_out = self.phys(pred_phys)
                l_phys = phys_out[0] if isinstance(phys_out, tuple) else phys_out
                if not torch.isfinite(l_phys):
                    l_phys = torch.zeros((), device=self.device)
                    l_phys_val = -1.0
                else:
                    l_phys_val = float(l_phys.detach().item())
            except Exception as e:      # noqa: BLE001
                l_phys = torch.zeros((), device=self.device)
                l_phys_val = -1.0
                if not getattr(self, "_phys_warned", False):
                    print(f"[warn] 物理项计算异常（将按 0 处理）: {e}")
                    self._phys_warned = True

        total = (self.cfg.lam_rec * l_rec
                 + self.cfg.lam_kl * l_kl
                 + self.cfg.lam_multi * l_multi
                 + getattr(self.cfg, "lam_phys", 0.0) * l_phys)
        parts = {
            "rec": l_rec.item(),
            "kl": l_kl.item(),
            "multi": l_multi.item(),
            "phys": l_phys_val,
            "kl_raw": kl_val.mean().item(),
            "dyn": dyn_l.mean().item(),
            "rep": rep_l.mean().item(),
            "total": total.item(),
        }
        return total, parts

    # ---------------- 评估 ----------------

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        agg: dict[str, list[float]] = {}
        for batch in loader:
            _, parts = self._compute_loss(batch)
            for k, v in parts.items():
                agg.setdefault(k, []).append(v)
        self.model.train()
        return {k: float(np.mean(v)) for k, v in agg.items()}

    # ---------------- 训练 ----------------

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        t0 = time.time()
        no_improve = 0

        for ep in range(1, cfg.epochs + 1):
            self.model.train()
            ep_parts: dict[str, list[float]] = {}
            for batch in train_loader:
                self.opt.zero_grad(set_to_none=True)
                loss, parts = self._compute_loss(batch)
                loss.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                   cfg.grad_clip)
                self.opt.step()
                for k, v in parts.items():
                    ep_parts.setdefault(k, []).append(v)

            tr = {f"train_{k}": float(np.mean(v)) for k, v in ep_parts.items()}
            va = {f"val_{k}": v for k, v in self.evaluate(val_loader).items()}
            rec = {"epoch": ep, "elapsed_s": time.time() - t0, **tr, **va}
            self.history.append(rec)

            # 早停
            if va["val_total"] < self.best_val - 1e-6:
                self.best_val = va["val_total"]
                self.best_epoch = ep
                no_improve = 0
                torch.save({
                    "model": self.model.state_dict(),
                    "epoch": ep,
                    "val_total": self.best_val,
                    "cfg": cfg.__dict__,
                }, self.out / "best.pt")
            else:
                no_improve += 1

            if ep % cfg.log_every == 0 or ep == 1:
                print(f"ep {ep:>4} | "
                      f"train rec={tr['train_rec']:.6f} kl={tr['train_kl']:.5f} "
                      f"multi={tr['train_multi']:.6f} | "
                      f"val rec={va['val_rec']:.6f} multi={va['val_multi']:.6f} "
                      f"total={va['val_total']:.6f} | "
                      f"best@{self.best_epoch}")

            if no_improve >= cfg.patience:
                print(f"早停于 epoch {ep}（best={self.best_epoch}）")
                break

        (self.out / "history.json").write_text(
            json.dumps(self.history, indent=1), encoding="utf-8")
        return {
            "best_epoch": self.best_epoch,
            "best_val_total": self.best_val,
            "epochs_run": len(self.history),
            "elapsed_s": time.time() - t0,
        }

