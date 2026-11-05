"""PC-RSSM：物理锚定 RSSM（第一轮）

设计（按用户 2026-11-05 审核意见）：
  静态监督: h(128) -> Linear(128,2) -> 标准化(N2dot, N3dot)
  动态监督: 多跨度 k∈{1,4,8,16}，目标 N(t+k)-N(t)
            未来潜状态必须通过【先验自主递推】获得
            **禁止读取未来真实观测**

轴对应（已核实）：
  N2dot = 低压轴（LP：风扇+LPC+LPT）角加速度，rpm/s
  N3dot = 高压轴（HP：HPC+HPT）角加速度，rpm/s

四个消融：
  E0: baseline（无监督）
  E1: 仅静态
  E2: 仅动态
  E3: 静态 + 动态
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))

from mask_impl_v3 import MaskRSSM


class PCRSSM(MaskRSSM):
    """物理锚定 RSSM

    在 MaskRSSM 基础上增加：
      - phys_head: 静态监督头 Linear(deter, 2)
      - dyn_head:  动态监督头 Linear(deter, 2)
    """

    def __init__(self, c_obs, c_act, **kw):
        super().__init__(c_obs, c_act, **kw)
        self.phys_head = nn.Linear(self.deter, 2)
        self.dyn_head = nn.Linear(self.deter, 2)
        # 初始化小权重，避免初期扰动
        nn.init.zeros_(self.phys_head.weight)
        nn.init.zeros_(self.phys_head.bias)
        nn.init.zeros_(self.dyn_head.weight)
        nn.init.zeros_(self.dyn_head.bias)

    def static_loss(self, stats):
        """静态监督：从每个时刻的 h 预测 [N2dot, N3dot]

        stats: list of dict 含 "h" [B,H] 与 "phys" [B,2]
        返回 (loss, 预测, 目标)
        """
        hs = torch.stack([s["h"] for s in stats], 1)        # [B,T,H]
        ys = torch.stack([s["phys"] for s in stats], 1)     # [B,T,2]
        pred = self.phys_head(hs)                            # [B,T,2]
        loss = F.mse_loss(pred, ys)
        return loss, pred, ys

    def dynamic_loss(self, stats_future, phys_future, ks=(1, 4, 8, 16)):
        """动态监督：多跨度 N(t+k) - N(t)

        stats_future: list of dict，来自【先验递推】（不看未来观测）
        phys_future:  [B, H, 2] 未来真实物理量（只作监督目标）
        ks: 跨度列表

        返回 dict: {k: loss}
        """
        hs = torch.stack([s["h"] for s in stats_future], 1)  # [B,H,Hd]
        B, H, _ = hs.shape
        losses = {}
        for k in ks:
            if k >= H:
                continue
            # Δh_phys = dyn_head(h_{t+k}) - dyn_head(h_t)
            d_pred = self.dyn_head(hs[:, k, :]) - self.dyn_head(hs[:, 0, :])
            # 目标：N(t+k) - N(t)
            d_true = phys_future[:, k, :] - phys_future[:, 0, :]
            losses[k] = F.mse_loss(d_pred, d_true)
        return losses

    def prior_rollout(self, state, act_future, mode="gumbel"):
        """★ 先验自主递推（不看任何未来观测），返回每步 h

        state: 历史编码后的状态
        act_future: [B,H,A]
        返回 list of {"h": [B,Hd]}
        """
        H = act_future.shape[1]
        out = []
        st = {"h": state["h"].clone(), "z": state["z"].clone()}
        for k in range(H):
            st = self._step_prior(st, act_future[:, k], mode)
            out.append({"h": st["h"]})
        return out, st


class PCWorldModel(nn.Module):
    """PC-RSSM 训练包装

    损失：
      loss = rec + 0.1*kl + multi
             + λ_static * L_static + λ_dynamic * L_dynamic
    """

    def __init__(self, c_obs, c_act, lam_static=0.0, lam_dynamic=0.0,
                 **kw):
        super().__init__()
        self.rssm = PCRSSM(c_obs, c_act, **kw)
        self.lam_s = lam_static
        self.lam_d = lam_dynamic

    def forward(self, obs, mask, act, af, fut_true, obs_true=None,
                phys_now=None, phys_future=None, mode="gumbel"):
        """phys_now:    [B,T,2]  历史段的真实物理量（静态监督）
           phys_future: [B,H,2]  未来段的真实物理量（动态监督目标）
        """
        rssm = self.rssm
        B, T, _ = obs.shape
        H = af.shape[1]
        target = obs if obs_true is None else obs_true
        # ★ 统一：mask=None 视为全部可见
        if mask is None:
            mask = torch.ones(B, T, self.rssm.c_obs, device=obs.device)
        post, prior, recon, hs_list = rssm.observe_full(
            obs, mask, act, mode)
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - target[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        # ★ 三项分别记录：
        #   kl_raw  = 原始 KL 均值
        #   kl_clip = free_bits 裁剪后的值（实际进入损失）
        #   kl_frac = 超过 free_bits 的比例（判断 KL 是否真的有效）
        kl_raw = kl_all.mean()
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        kl_frac = (kl_all > rssm.free_bits).float().mean()

        # ---- 多步预测（原有）----
        pred = rssm.imagine(obs, mask, act, af, mode)
        multi = ((pred - fut_true) ** 2).mean()

        out = {"rec": rec, "kl": kl_clip, "kl_raw": kl_raw,
               "kl_frac": kl_frac, "multi": multi,
               "pred": pred, "l_static": torch.tensor(0.0, device=obs.device),
               "l_dynamic": torch.tensor(0.0, device=obs.device),
               "diag": {}}

        # ---- 静态物理监督 ----
        if self.lam_s > 0 and phys_now is not None:
            hs = torch.stack(hs_list, 1)                   # [B,T,Hd]
            p_pred = rssm.phys_head(hs)                     # [B,T,2]
            l_s = F.mse_loss(p_pred[:, burn:], phys_now[:, burn:])
            out["l_static"] = l_s
            out["diag"]["l_static"] = float(l_s.detach())

        # ---- 动态物理监督（★ 先验自主递推）----
        if self.lam_d > 0 and phys_future is not None:
            # 历史用后验编码，未来只用先验（不看未来观测）
            st_list, _ = rssm.prior_rollout_from_hist(
                obs, mask, act, af, mode)
            hs_f = torch.stack([s["h"] for s in st_list], 1)  # [B,H,Hd]
            losses = {}
            for k in (1, 4, 8, 16):
                if k >= H:
                    continue
                d_pred = rssm.dyn_head(hs_f[:, k, :]) - \
                    rssm.dyn_head(hs_f[:, 0, :])
                d_true = phys_future[:, k, :] - phys_future[:, 0, :]
                losses[k] = F.mse_loss(d_pred, d_true)
            if losses:
                l_d = sum(losses.values()) / len(losses)
                out["l_dynamic"] = l_d
                out["diag"]["l_dynamic"] = float(l_d.detach())
                out["diag"]["dyn_each"] = {k: float(v.detach())
                                          for k, v in losses.items()}

        out["loss"] = (out["rec"] + 0.1*out["kl"] + out["multi"]
                       + self.lam_s * out["l_static"]
                       + self.lam_d * out["l_dynamic"])
        return out

    def imagine(self, obs, mask, act, af, mode="gumbel"):
        if mask is None:
            B, T, _ = obs.shape
            mask = torch.ones(B, T, self.rssm.c_obs, device=obs.device)
        return self.rssm.imagine(obs, mask, act, af, mode)


def add_helper_methods():
    """给 MaskRSSM 打补丁：observe_full / prior_rollout_from_hist"""
    def observe_full(self, obs_masked, mask, act, mode="gumbel"):
        """与 observe 相同，但额外返回每步 h"""
        B, T, _ = obs_masked.shape
        if mask is None:
            mask = torch.ones(B, T, self.c_obs, device=obs_masked.device)
        obs_m = obs_masked * mask
        st = self.init_state(B, obs_masked.device)
        posts, priors, recons, hs = [], [], [], []
        for t in range(T):
            z_prev = st["z"].reshape(B, -1)
            h = self.cell(torch.cat([z_prev, act[:, t]], -1), st["h"])
            prior_logits = self.prior(
                torch.cat([h, z_prev], -1)).reshape(
                B, self.stoch, self.classes)
            embed = self.enc(torch.cat([obs_m[:, t], mask[:, t]], -1))
            post_logits = self.post(
                torch.cat([torch.cat([h, z_prev], -1), embed], -1)
            ).reshape(B, self.stoch, self.classes)
            z = self.sample(post_logits, mode)
            st = {"h": h, "z": z}
            posts.append(post_logits)
            priors.append(prior_logits)
            recons.append(self.dec(self.feat(st)))
            hs.append(h)
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1), hs)

    def prior_rollout_from_hist(self, obs, mask, act, af, mode="gumbel"):
        """历史用后验编码，未来只用先验递推（不看未来观测）"""
        B, T, _ = obs.shape
        H = af.shape[1]
        if mask is None:
            mask = torch.ones(B, T, self.c_obs, device=obs.device)
        obs_m = obs * mask
        st = self.init_state(B, obs.device)
        for t in range(T):
            z_prev = st["z"].reshape(B, -1)
            h = self.cell(torch.cat([z_prev, act[:, t]], -1), st["h"])
            embed = self.enc(torch.cat([obs_m[:, t], mask[:, t]], -1))
            post_logits = self.post(
                torch.cat([torch.cat([h, z_prev], -1), embed], -1)
            ).reshape(B, self.stoch, self.classes)
            z = self.sample(post_logits, mode)
            st = {"h": h, "z": z}
        # ★ 先验递推
        out = []
        for k in range(H):
            z_prev = st["z"].reshape(B, -1)
            h = self.cell(torch.cat([z_prev, af[:, k]], -1), st["h"])
            prior_logits = self.prior(
                torch.cat([h, z_prev], -1)).reshape(
                B, self.stoch, self.classes)
            z = self.sample(prior_logits, mode)
            st = {"h": h, "z": z}
            out.append({"h": h})
        return out, st

    MaskRSSM.observe_full = observe_full
    MaskRSSM.prior_rollout_from_hist = prior_rollout_from_hist


add_helper_methods()





