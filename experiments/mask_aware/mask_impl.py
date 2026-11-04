"""Mask-aware 观测机制

核心设计（按用户要求）：
  1. 掩码 m_t 与观测通道一一对应（m=1 有效，m=0 缺失）
  2. 显式 Mask-aware Encoder：输入 = [有效观测(中性填充), mask]
  3. 中性填充值 = 训练集均值（归一化空间为 0），必须与 mask 配合
  4. 严格信息隔离：故障后的 T4 真值绝不进入模型

数学形式：
  x̃_t = m_t ⊙ x_t + (1 - m_t) ⊙ x_fill     (x_fill = 0 在归一化空间)
  e_t = Enc([x̃_t, m_t])
  z_t ~ q(z | h_t, e_t)                    (后验只用有效信息)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(obs, mask, fill=0.0):
    """应用掩码

    obs  : [B,T,C] 归一化观测
    mask : [B,T,C] 1=有效, 0=缺失
    fill : 中性填充值（归一化空间用 0 = train 均值）
    """
    return obs * mask + (1.0 - mask) * fill


def make_mask(B, T, C, mode="none", ratio=0.0, channels=None,
              burst_len=0, t0=None, generator=None, device="cpu"):
    """生成掩码

    mode:
      none   全 1
      fixed  指定通道全程 0
      random 每步随机丢 ratio 比例通道（但至少留 1 个）
      burst  指定通道从 t0 起连续缺失 burst_len 步
    """
    if mode == "none":
        return torch.ones(B, T, C, device=device)

    if mode == "fixed":
        m = torch.ones(B, T, C, device=device)
        if channels:
            m[:, :, channels] = 0.0
        return m

    if mode == "random":
        m = torch.ones(B, T, C, device=device)
        n_drop = max(1, int(round(C * ratio)))
        n_drop = min(n_drop, C - 1)     # 至少留 1 个有效通道
        if n_drop <= 0:
            return m
        for b in range(B):
            for t in range(T):
                idx = torch.randperm(C, device=device)[:n_drop]
                m[b, t, idx] = 0.0
        return m

    if mode == "burst":
        m = torch.ones(B, T, C, device=device)
        ch = channels if channels else [0]
        st = 0 if t0 is None else max(0, t0)
        en = min(T, st + burst_len) if burst_len > 0 else T
        for c in ch:
            m[:, st:en, c] = 0.0
        return m

    raise ValueError(f"unknown mask mode: {mode}")


def augment_mask_batch(batch_obs, batch_act, batch_fut, batch_af,
                       cfg, device):
    """训练时随机增强：混合多种缺失模式

    cfg: dict 含 p_none/p_fixed/p_random/p_burst 概率与参数
    """
    B, T, C = batch_obs.shape
    device = batch_obs.device
    obs = batch_obs.clone()
    fut = batch_fut.clone()

    m_hist = torch.ones(B, T, C, device=device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    for b in range(B):
        r = torch.rand(1, generator=g).item()
        if r < cfg.get("p_none", 0.25):
            continue
        elif r < cfg.get("p_none", 0.25) + cfg.get("p_fixed", 0.25):
            ch = cfg.get("fixed_channels") or [2]
            for c in ch:
                m_hist[b, :, c] = 0.0
        elif r < (cfg.get("p_none", 0.25) + cfg.get("p_fixed", 0.25)
                  + cfg.get("p_random", 0.25)):
            ratio = cfg.get("random_ratio", 0.3)
            n_drop = max(1, min(int(round(C * ratio)), C - 1))
            for t in range(T):
                idx = torch.randperm(C, generator=g)[:n_drop]
                m_hist[b, t, idx] = 0.0
        else:
            ch = cfg.get("burst_channels") or [2]
            bl = int(torch.randint(
                cfg.get("burst_min", 8),
                cfg.get("burst_max", min(T, 32)) + 1,
                (1,), generator=g).item())
            st = int(torch.randint(0, max(1, T - bl + 1), (1,),
                                   generator=g).item())
            for c in ch:
                m_hist[b, st:st+bl, c] = 0.0

    # 未来段的 mask：延续历史末尾的状态（不泄漏未来）
    m_fut = m_hist[:, -1:, :].expand(-1, fut.shape[1], -1).clone()

    obs_m = apply_mask(obs, m_hist)
    fut_m = apply_mask(fut, m_fut)
    return obs_m, m_hist, fut_m, m_fut


# ══════════════════════════════════════════════════════
# Mask-aware GRU（公平基线）
# ══════════════════════════════════════════════════════
class MaskGRU(nn.Module):
    """把 [观测, mask, 动作] 送入 GRU，训练重构完整传感器状态"""

    def __init__(self, c_obs, c_act, hidden=128, horizon=32, mlp=256):
        super().__init__()
        self.horizon = horizon
        self.c_obs = c_obs
        self.inp = nn.Sequential(
            nn.Linear(c_obs*2 + c_act, hidden), nn.LayerNorm(hidden),
            nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        self.o2h = nn.Linear(c_obs*2 + c_act, hidden)
        self.dec = nn.Sequential(
            nn.Linear(hidden, mlp), nn.SiLU(),
            nn.Linear(mlp, mlp), nn.SiLU(),
            nn.Linear(mlp, c_obs))

    def forward(self, obs, mask, act, af, af_mask=None):
        """obs:[B,T,C], mask:[B,T,C], act:[B,T,A], af:[B,H,A]"""
        B, T, C = obs.shape
        H = af.shape[1]
        x = self.inp(torch.cat([obs, mask, act], -1))
        _, h = self.gru(x)
        h = h[-1]

        # 未来：mask 固定为历史末尾值（不偷看未来）
        m_last = mask[:, -1]
        obs_prev = apply_mask(obs[:, -1], m_last)
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([obs_prev, m_last, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            # 自回归时用预测值填充有效通道，缺失通道保持中性
            obs_prev = apply_mask(o_hat, m_last)
        return torch.stack(outs, 1)


# ══════════════════════════════════════════════════════
# Mask-aware RSSM
# ══════════════════════════════════════════════════════
class MaskRSSM(nn.Module):
    """在 RSSM 上接 Mask-aware Encoder

    关键：后验只在有效观测上更新
      - 观测先 mask 填充
      - Encoder 输入 [obs_filled, mask]
      - 这样缺失通道不提供信息，但 mask 告诉网络"这个通道丢了"
    """

    def __init__(self, c_obs, c_act, deter=128, stoch=32, classes=32,
                 hidden=128, free_bits=1.0, unimix=0.01, use_dt=False,
                 deter_slow=96, k_slow=4):
        super().__init__()
        self.c_obs = c_obs
        self.c_act = c_act
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.free_bits = free_bits
        self.unimix = unimix
        self.use_dt = use_dt
        self.deter_slow = deter_slow if use_dt else 0
        self.k_slow = k_slow

        z_dim = stoch * classes
        feat_dim = deter + self.deter_slow + z_dim

        # Encoder 输入 = obs + mask
        self.enc = nn.Sequential(
            nn.Linear(c_obs*2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())

        if use_dt:
            self.cell_fast = nn.GRUCell(
                z_dim + c_act + self.deter_slow, deter)
            self.cell_slow = nn.GRUCell(deter + z_dim, self.deter_slow)
        else:
            self.cell = nn.GRUCell(z_dim + c_act, deter)

        self.prior = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim))
        self.post = nn.Sequential(
            nn.Linear(feat_dim + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, z_dim))
        self.dec = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, c_obs))

    def initial(self, B, device):
        st = {"z": torch.zeros(B, self.stoch, self.classes, device=device),
              "step": 0}
        if self.use_dt:
            st["hf"] = torch.zeros(B, self.deter, device=device)
            st["hs"] = torch.zeros(B, self.deter_slow, device=device)
        else:
            st["h"] = torch.zeros(B, self.deter, device=device)
        return st

    def feat(self, state):
        B = state["z"].shape[0]
        z = state["z"].reshape(B, -1)
        if self.use_dt:
            return torch.cat([state["hf"], state["hs"], z], -1)
        return torch.cat([state["h"], z], -1)

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
        raise ValueError(mode)

    def _advance(self, state, act_t, t_abs):
        B = state["z"].shape[0]
        z_prev = state["z"].reshape(B, -1)
        if self.use_dt:
            need_slow = (t_abs % self.k_slow == 0) or (t_abs == 0)
            hf = self.cell_fast(
                torch.cat([z_prev, act_t, state["hs"]], -1), state["hf"])
            if need_slow:
                hs = self.cell_slow(torch.cat([hf, z_prev], -1),
                                    state["hs"])
            else:
                hs = state["hs"]
            new = {"hf": hf, "hs": hs}
            prior_in = torch.cat([hf, hs, z_prev], -1)
        else:
            h = self.cell(torch.cat([z_prev, act_t], -1), state["h"])
            new = {"h": h}
            prior_in = torch.cat([h, z_prev], -1)
        prior_logits = self.prior(prior_in).reshape(
            B, self.stoch, self.classes)
        return new, prior_in, prior_logits

    def observe(self, obs, mask, act, mode="gumbel"):
        B, T, _ = obs.shape
        obs_f = apply_mask(obs, mask)
        state = self.initial(B, obs.device)
        posts, priors, recons = [], [], []
        for t in range(T):
            new, prior_in, prior_logits = self._advance(
                state, act[:, t], t)
            embed = self.enc(torch.cat([obs_f[:, t], mask[:, t]], -1))
            post_logits = self.post(
                torch.cat([prior_in, embed], -1)).reshape(
                B, self.stoch, self.classes)
            z = self.sample(post_logits, mode)
            state = {**new, "z": z, "step": t}
            posts.append(post_logits)
            priors.append(prior_logits)
            recons.append(self.dec(self.feat(state)))
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1))

    def imagine(self, obs, mask, act, af, mode="gumbel"):
        B, T, _ = obs.shape
        H = af.shape[1]
        obs_f = apply_mask(obs, mask)
        state = self.initial(B, obs.device)
        for t in range(T):
            new, prior_in, _ = self._advance(state, act[:, t], t)
            embed = self.enc(torch.cat([obs_f[:, t], mask[:, t]], -1))
            post_logits = self.post(
                torch.cat([prior_in, embed], -1)).reshape(
                B, self.stoch, self.classes)
            z = self.sample(post_logits, mode)
            state = {**new, "z": z, "step": t}

        # 未来：mask 固定为历史末尾
        m_last = mask[:, -1]
        outs = []
        for k in range(H):
            t_abs = T + k
            new, _, prior_logits = self._advance(state, af[:, k], t_abs)
            z = self.sample(prior_logits, mode)
            state = {**new, "z": z, "step": t_abs}
            outs.append(self.dec(self.feat(state)))
        return torch.stack(outs, 1)

    def kl_terms(self, post_logits, prior_logits):
        logp = F.log_softmax(post_logits, -1)
        logq = F.log_softmax(prior_logits, -1)
        p = logp.exp()
        kl_dyn = (p.detach() * (logp - logq)).sum(-1)
        kl_rep = (p * (logp - logq.detach())).sum(-1)
        return kl_dyn, kl_rep


class MaskWorldModel(nn.Module):
    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.rssm = MaskRSSM(c_obs, c_act, **kw)

    def forward(self, obs, mask, act, af, fut, mode="gumbel"):
        rssm = self.rssm
        T = obs.shape[1]
        post, prior, recon = rssm.observe(obs, mask, act, mode)
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - obs[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        pred = rssm.imagine(obs, mask, act, af, mode)
        multi = ((pred - fut) ** 2).mean()
        return {"rec": rec, "kl": kl_clip, "kl_raw": kl_all.mean(),
                "multi": multi, "pred": pred}

    def imagine(self, obs, mask, act, af, mode="gumbel"):
        return self.rssm.imagine(obs, mask, act, af, mode)
