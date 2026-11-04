"""DT-RSSM：双时间尺度 RSSM

设计（第一版，简洁优先）：
  - h_fast : 每步更新的确定性状态
  - h_slow : 每 K 步更新一次的确定性状态
  - z      : 保留原 RSSM 的离散随机潜状态
  - 融合   : feat = [h_fast, h_slow, z]，共同参与 prior/post/decoder

关键约束（按用户要求）：
  1. 复用现有数据/评价代码，不改基线权重
  2. Decoder 输出与控制输入与原 RSSM 一致
  3. 先只改潜状态演化结构，不加物理损失
  4. 显式指定 Gumbel/Prob，不依赖 train/eval 自动切换
  5. K 是待验证的工程超参，不是物理时间常数

时间尺度验证结论（timescale_check.py）：
  - ACF 衰减时间 7.7-8.7 s，比值仅 1.1x
  - 上升时间 1.05-2.31 s，比值 2.2x
  - 分离不明显，K>1 的提升预期有限，需靠消融验证
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DTRSSM(nn.Module):
    """双时间尺度 RSSM

    参数
    ----
    k_slow : 慢状态更新周期（每 K 步更新一次）
             K=1 退化为两路并行但同步更新
    """

    def __init__(self, c_obs, c_act, deter_fast=96, deter_slow=96,
                 stoch=32, classes=32, hidden=128, k_slow=4,
                 free_bits=1.0, unimix=0.01):
        super().__init__()
        self.c_obs = c_obs
        self.c_act = c_act
        self.deter_fast = deter_fast
        self.deter_slow = deter_slow
        self.stoch = stoch
        self.classes = classes
        self.k_slow = k_slow
        self.free_bits = free_bits
        self.unimix = unimix

        z_dim = stoch * classes
        feat_dim = deter_fast + deter_slow + z_dim

        # Encoder
        self.enc = nn.Sequential(
            nn.Linear(c_obs, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())

        # 快状态：每步更新，输入含慢状态
        self.cell_fast = nn.GRUCell(z_dim + c_act + deter_slow, deter_fast)
        # 慢状态：每 K 步更新，输入含快状态
        self.cell_slow = nn.GRUCell(deter_fast + z_dim, deter_slow)

        # 先验：从 [h_fast, h_slow] 预测
        self.prior = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim))
        # 后验：从 [h_fast, h_slow, embed]
        self.post = nn.Sequential(
            nn.Linear(feat_dim + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim))
        # 解码
        self.dec = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, c_obs))

        self._feat_dim = feat_dim

    # ---------- 状态 ----------
    def initial(self, B, device):
        return {
            "hf": torch.zeros(B, self.deter_fast, device=device),
            "hs": torch.zeros(B, self.deter_slow, device=device),
            "z": torch.zeros(B, self.stoch, self.classes, device=device),
            "step": 0,
        }

    def feat(self, state):
        B = state["hf"].shape[0]
        z = state["z"].reshape(B, -1)
        return torch.cat([state["hf"], state["hs"], z], -1)

    # ---------- 采样（显式指定，绝不依赖 self.training） ----------
    def sample(self, logits, mode="gumbel"):
        if mode == "prob":
            return F.softmax(logits, -1)
        if self.unimix > 0:
            p = F.softmax(logits, -1)
            p = p * (1 - self.unimix) + self.unimix / self.classes
            logits = torch.log(p.clamp_min(1e-9))
        if mode == "gumbel":
            return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)
        if mode == "argmax":
            idx = logits.argmax(-1, keepdim=True)
            return torch.zeros_like(logits).scatter_(-1, idx, 1.0)
        raise ValueError(f"unknown mode: {mode}")

    # ---------- 单步推进 ----------
    def step(self, state, act_t, update_slow):
        """推进一个仿真步

        update_slow: bool 或 [B] 的 bool 张量，表示该步是否更新慢状态
        """
        B = state["hf"].shape[0]
        z_prev = state["z"].reshape(B, -1)

        # 1) 快状态：每步更新
        hf = self.cell_fast(
            torch.cat([z_prev, act_t, state["hs"]], -1), state["hf"])

        # 2) 慢状态：按周期更新
        if update_slow:
            hs = self.cell_slow(
                torch.cat([hf, z_prev], -1), state["hs"])
        else:
            hs = state["hs"]

        # 3) 先验（基于新的 [hf, hs]）
        z_prev2 = state["z"].reshape(B, -1)
        prior_in = torch.cat([hf, hs, z_prev2], -1)
        prior_logits = self.prior(prior_in).reshape(
            B, self.stoch, self.classes)

        return {"hf": hf, "hs": hs, "z": state["z"],
                "prior_logits": prior_logits,
                "prior_in": prior_in}

    def posterior(self, prior_in, obs_t):
        """后验：用当前有效观测"""
        embed = self.enc(obs_t)
        logits = self.post(torch.cat([prior_in, embed], -1)).reshape(
            prior_in.shape[0], self.stoch, self.classes)
        return logits

    # ---------- 训练路径：observe（用后验） ----------
    def observe(self, obs, act, mode="gumbel"):
        B, T, _ = obs.shape
        state = self.initial(B, obs.device)
        posts, priors, recons = [], [], []
        for t in range(T):
            need_slow = (t % self.k_slow == 0) or (t == 0)
            s = self.step(state, act[:, t], need_slow)
            post_logits = self.posterior(s["prior_in"], obs[:, t])
            z = self.sample(post_logits, mode)
            state = {"hf": s["hf"], "hs": s["hs"], "z": z,
                     "step": t}
            posts.append(post_logits)
            priors.append(s["prior_logits"])
            recons.append(self.dec(self.feat(state)))
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1))

    # ---------- 推理路径：imagine（未来只用先验） ----------
    def imagine(self, obs_hist, act_hist, act_future, mode="gumbel"):
        B, T, _ = obs_hist.shape
        H = act_future.shape[1]
        state = self.initial(B, obs_hist.device)

        # 历史：用后验编码
        for t in range(T):
            need_slow = (t % self.k_slow == 0) or (t == 0)
            s = self.step(state, act_hist[:, t], need_slow)
            post_logits = self.posterior(s["prior_in"], obs_hist[:, t])
            z = self.sample(post_logits, mode)
            state = {"hf": s["hf"], "hs": s["hs"], "z": z, "step": t}

        # 未来：只用先验
        outs = []
        for k in range(H):
            t_abs = T + k
            need_slow = (t_abs % self.k_slow == 0) or (t_abs == 0)
            s = self.step(state, act_future[:, k], need_slow)
            z = self.sample(s["prior_logits"], mode)
            state = {"hf": s["hf"], "hs": s["hs"], "z": z,
                     "step": t_abs}
            outs.append(self.dec(self.feat(state)))
        return torch.stack(outs, 1)

    # ---------- KL ----------
    def kl_terms(self, post_logits, prior_logits):
        logp = F.log_softmax(post_logits, -1)
        logq = F.log_softmax(prior_logits, -1)
        p = logp.exp()
        kl_dyn = (p.detach() * (logp - logq)).sum(-1)
        kl_rep = (p * (logp - logq.detach())).sum(-1)
        return kl_dyn, kl_rep


class DTWorldModel(nn.Module):
    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.rssm = DTRSSM(c_obs, c_act, **kw)

    def imagine(self, obs, act, act_fut, mode="gumbel"):
        return self.rssm.imagine(obs, act, act_fut, mode)

    def forward(self, obs, act, act_fut, fut, mode="gumbel"):
        """返回 rec / kl / multi 三部分（与基线 RSSM 口径一致）

        注意：外层应传 fut 真值，用于计算 multi 损失。
        """
        rssm = self.rssm
        B, T, _ = obs.shape
        post, prior, recon = rssm.observe(obs, act, mode)
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - obs[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        pred = rssm.imagine(obs, act, act_fut, mode)
        multi = ((pred - fut) ** 2).mean()
        return {"rec": rec, "kl": kl_clip, "kl_raw": kl_all.mean(),
                "multi": multi, "pred": pred, "post": post, "prior": prior}


def load_dt(ckpt, c_obs, c_act, **kw):
    m = DTWorldModel(c_obs, c_act, **kw)
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    return m
