"""PSVA：物理监督诱导的快慢潜状态表征分离

设计（v4 定稿）：
  h = [h_fast(64); h_slow(64)]    ← 索引切分，不改结构，梯度全通

监督：
  L_fast    = MSE(H_f(h_f), norm(N2dot, N3dot))         静态：加速度
  L_slow    = MSE(H_s(h_s), norm(N2, N3))               静态：转速
  L_dynamic = PSG-JEPA 形式的多跨度 ΔN
              L_dyn = (1/|K|) Σ_k (1/|I_k|) Σ_i
                      MSE(H_Δ(h^f_{t+i}, h^f_{t+i+k}), ΔN_{t+i,k})
              K = {1,4,8,16}, ΔN_{t,k} = N_{t+k} - N_t

重要：
  - 保留 ΔN（转速变化），不是 ΔNdot
  - L_dynamic 输入是【状态对】(h^f_t, h^f_{t+k})，与 PSG-JEPA 一致
  - 不加入预测空间一致性（第一版）
  - 不加入功率约束（第一版）

与 PC-RSSM E3 的关系：
  E3 = 全 h 静态锚定(Ndot,N3dot) + 全 h 动态锚定(ΔNdot)
  PSVA = h_f 锚定 Ndot + h_s 锚定 N + h_f 对锚定 ΔN
  → PSVA 是 E3 的"分工版"，不是全新方法
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from mask_impl_v3 import MaskRSSM
# ★ 必须导入 pc_rssm 以激活 observe_full / prior_rollout_from_hist 补丁
from pc_rssm import PCWorldModel  # noqa: F401  (side-effect: patches MaskRSSM)


class PSVA_RSSM(MaskRSSM):
    """PSVA：在 MaskRSSM 基础上增加快慢切分与三个监督头

    参数：
      d_fast : 快状态维数（默认 64）
      d_slow : 慢状态维数（默认 64）
      d_fast + d_slow 必须等于 deter
    """

    def __init__(self, c_obs, c_act, d_fast=64, d_slow=64,
                 unsplit=False, **kw):
        """unsplit=True 时，所有监督头都作用于【完整 h】(128维)
        这是"不切分但相同监督"的对照（P_ctrl）
        """
        deter = kw.pop("deter", d_fast + d_slow)
        assert d_fast + d_slow == deter, \
            f"d_fast({d_fast}) + d_slow({d_slow}) != deter({deter})"
        super().__init__(c_obs, c_act, deter=deter, **kw)
        self.d_fast = d_fast
        self.d_slow = d_slow
        self.unsplit = unsplit

        # 静态监督头（unsplit: 全部输入完整 h）
        in_f = deter if unsplit else d_fast
        in_s = deter if unsplit else d_slow
        in_d = deter if unsplit else d_fast
        self.head_fast = nn.Linear(in_f, 2)     # N2dot, N3dot
        self.head_slow = nn.Linear(in_s, 2)     # N2, N3
        self.head_dyn = nn.Linear(in_d, 2)      # ΔN2, ΔN3

        # 零初始化（避免初期扰动基础损失）
        for h in [self.head_fast, self.head_slow, self.head_dyn]:
            nn.init.zeros_(h.weight)
            nn.init.zeros_(h.bias)

    # ---------- 切分 ----------
    def split_h(self, h):
        """[B, D] 或 [B,T,D] -> (h_fast, h_slow)

        unsplit=True 时返回 (h, h)，即所有头都看完整状态
        """
        if self.unsplit:
            return h, h
        return h[..., :self.d_fast], h[..., self.d_fast:]

    # ---------- 静态损失 ----------
    def static_loss(self, hs_seq, y_fast, y_slow):
        """hs_seq : [B,T,D] 逐时刻 h
           y_fast : [B,T,2] 归一化 (N2dot, N3dot)
           y_slow : [B,T,2] 归一化 (N2, N3)
        """
        hf, hs = self.split_h(hs_seq)
        l_f = F.mse_loss(self.head_fast(hf), y_fast)
        l_s = F.mse_loss(self.head_slow(hs), y_slow)
        return l_f, l_s

    # ---------- 动态损失（PSG-JEPA 形式）----------
    def dynamic_loss(self, hf_seq, y_delta, ks=(1, 4, 8, 16)):
        """hf_seq  : [B,H,d_fast] 未来快状态（先验递推得到）
           y_delta : dict{k: [B, 2]} 目标 ΔN
                     其中 y_delta[k][:, i] 对应起始位置 i 的 N(t+k)-N(t)
                     （这里只取 t=0 这一对，简化）

        严格按 PSG-JEPA：
          L_dyn = (1/|K|) Σ_k MSE(H_Δ(h^f_{t}, h^f_{t+k}), ΔN_{t,k})
          每个跨度等权（不是每对等权）
        """
        losses = {}
        h0 = hf_seq[:, 0, :]                     # [B, d_fast]
        for k in ks:
            if k >= hf_seq.shape[1]:
                continue
            hk = hf_seq[:, k, :]                 # [B, d_fast]
            # ★ 状态对输入：H_Δ([h_t, h_{t+k}])
            # 这里用差分形式：H_Δ(h_k) - H_Δ(h_0)
            # 等价于对"状态变化"建线性映射，参数量更省
            pred = self.head_dyn(hk) - self.head_dyn(h0)
            if k in y_delta:
                losses[k] = F.mse_loss(pred, y_delta[k])
        if not losses:
            return torch.tensor(0.0, device=hf_seq.device), {}
        # 每跨度等权
        l = sum(losses.values()) / len(losses)
        return l, {kk: float(v.detach()) for kk, v in losses.items()}

    # ---------- 先验递推（返回每步 h）----------
    def prior_rollout_h(self, state, act_future, mode="gumbel"):
        """与 PC-RSSM 的 prior_rollout_from_hist 相同，但返回每步 h"""
        out = []
        st = {"h": state["h"].clone(), "z": state["z"].clone()}
        for k in range(act_future.shape[1]):
            st = self._step_prior(st, act_future[:, k], mode)
            out.append(st["h"])
        return torch.stack(out, 1)               # [B, H, D]


class PSVA_WorldModel(nn.Module):
    """PSVA 训练包装

    损失：
      loss = rec + 0.1*kl + multi
             + λ_fast*L_fast + λ_slow*L_slow + λ_dyn*L_dyn
    """

    def __init__(self, c_obs, c_act, d_fast=64, d_slow=64,
                 lam_fast=0.0, lam_slow=0.0, lam_dyn=0.0,
                 unsplit=False, **kw):
        super().__init__()
        self.rssm = PSVA_RSSM(c_obs, c_act, d_fast=d_fast,
                              d_slow=d_slow, unsplit=unsplit, **kw)
        self.lam_f = lam_fast
        self.lam_s = lam_slow
        self.lam_d = lam_dyn

    def forward(self, obs, mask, act, af, fut_true, obs_true=None,
                y_fast=None, y_slow=None, y_delta=None, mode="gumbel"):
        """y_fast  : [B,T,2] 归一化 (N2dot, N3dot)
           y_slow  : [B,T,2] 归一化 (N2, N3)
           y_delta : dict{k: [B,2]}  ΔN 目标
        """
        rssm = self.rssm
        B, T, _ = obs.shape
        if mask is None:
            mask = torch.ones(B, T, rssm.c_obs, device=obs.device)
        target = obs if obs_true is None else obs_true

        # ---- 基础 ----
        post, prior, recon, hs_list = rssm.observe_full(obs, mask, act, mode)
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - target[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        kl_raw = kl_all.mean()
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        kl_frac = (kl_all > rssm.free_bits).float().mean()

        pred = rssm.imagine(obs, mask, act, af, mode)
        multi = ((pred - fut_true) ** 2).mean()

        out = {"rec": rec, "kl": kl_clip, "kl_raw": kl_raw,
               "kl_frac": kl_frac, "multi": multi, "pred": pred,
               "l_fast": torch.tensor(0.0, device=obs.device),
               "l_slow": torch.tensor(0.0, device=obs.device),
               "l_dyn": torch.tensor(0.0, device=obs.device),
               "diag": {}}

        # ---- 静态监督 ----
        if (self.lam_f > 0 or self.lam_s > 0) and \
                (y_fast is not None or y_slow is not None):
            hs_seq = torch.stack(hs_list, 1)          # [B,T,D]
            if self.lam_f > 0 and y_fast is not None:
                # 需要先清零再算
                hf_seq, _ = rssm.split_h(hs_seq)
                l_f = F.mse_loss(rssm.head_fast(hf_seq), y_fast)
                out["l_fast"] = l_f
                out["diag"]["l_fast"] = float(l_f.detach())
            if self.lam_s > 0 and y_slow is not None:
                _, hs_seq_s = rssm.split_h(hs_seq)
                l_s = F.mse_loss(rssm.head_slow(hs_seq_s), y_slow)
                out["l_slow"] = l_s
                out["diag"]["l_slow"] = float(l_s.detach())

        # ---- 动态监督（先验递推）----
        if self.lam_d > 0 and y_delta is not None:
            # 历史用后验编码 → 未来只用先验
            st_list, _ = rssm.prior_rollout_from_hist(obs, mask, act, af, mode)
            hf_seq = torch.stack([s["h"] for s in st_list], 1)  # [B,H,D]
            if not rssm.unsplit:
                hf_seq = hf_seq[..., :rssm.d_fast]              # 只要快部分
            l_d, each = rssm.dynamic_loss(hf_seq, y_delta)
            out["l_dyn"] = l_d
            out["diag"]["l_dyn"] = float(l_d.detach())
            out["diag"]["dyn_each"] = each

        out["loss"] = (out["rec"] + 0.1 * out["kl"] + out["multi"]
                       + self.lam_f * out["l_fast"]
                       + self.lam_s * out["l_slow"]
                       + self.lam_d * out["l_dyn"])
        return out

    def imagine(self, obs, mask, act, af, mode="gumbel"):
        if mask is None:
            B, T, _ = obs.shape
            mask = torch.ones(B, T, self.rssm.c_obs, device=obs.device)
        return self.rssm.imagine(obs, mask, act, af, mode)

    def params_count(self):
        return sum(p.numel() for p in self.parameters())



