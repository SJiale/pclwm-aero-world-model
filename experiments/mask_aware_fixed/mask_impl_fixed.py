"""Mask-aware 观测机制（修复版）

修复两个关键问题：
  1. **重建损失监督目标**：Encoder 输入用遮蔽观测 + mask，
     但重建 Loss 必须用**完整真实观测**作监督。
     原实现把填充值 0 当重建目标，导致模型学到"遮蔽通道输出 0"。
  2. **逐步在线状态估计**：新增 step() 方法，每 0.105 s 用正常传感器
     更新隐状态。原实现每 32 步才更新一次。

数学形式：
  x̃_t = m_t ⊙ x_t + (1-m_t) ⊙ 0        (归一化空间填充 0)
  e_t = Enc([x̃_t, m_t])
  z_t ~ q(z | h_t, e_t)
  ô_t = Dec(feat)                       ← 监督目标是真实 x_t（全 5 通道）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(obs, mask, fill=0.0):
    """应用掩码：obs [B,T,C] 或 [B,C]，mask 同形状"""
    return obs * mask + (1.0 - mask) * fill


def make_mask(B, T, C, mode="none", ratio=0.0, channels=None,
              burst_len=0, t0=None, device="cpu"):
    if mode == "none":
        return torch.ones(B, T, C, device=device)
    if mode == "fixed":
        m = torch.ones(B, T, C, device=device)
        if channels:
            m[:, :, channels] = 0.0
        return m
    if mode == "random":
        m = torch.ones(B, T, C, device=device)
        n_drop = max(1, min(int(round(C * ratio)), C - 1))
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
    """训练时随机缺失增强

    返回 (obs_masked, mask_hist, obs_true, fut_true)
      注意：不再返回 fut_masked —— 重建/预测监督都用真实值
    """
    B, T, C = batch_obs.shape
    device = batch_obs.device
    obs = batch_obs.clone()

    m_hist = torch.ones(B, T, C, device=device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    for b in range(B):
        r = torch.rand(1, generator=g).item()
        p_none = cfg.get("p_none", 0.25)
        p_fixed = cfg.get("p_fixed", 0.25)
        p_rand = cfg.get("p_random", 0.25)
        if r < p_none:
            continue
        elif r < p_none + p_fixed:
            for c in (cfg.get("fixed_channels") or [2]):
                m_hist[b, :, c] = 0.0
        elif r < p_none + p_fixed + p_rand:
            ratio = cfg.get("random_ratio", 0.3)
            n_drop = max(1, min(int(round(C * ratio)), C - 1))
            for t in range(T):
                idx = torch.randperm(C, generator=g)[:n_drop]
                m_hist[b, t, idx] = 0.0
        else:
            ch = cfg.get("burst_channels") or [2]
            bl = int(torch.randint(cfg.get("burst_min", 8),
                                   cfg.get("burst_max", min(T, 32)) + 1,
                                   (1,), generator=g).item())
            st = int(torch.randint(0, max(1, T - bl + 1), (1,),
                                   generator=g).item())
            for c in ch:
                m_hist[b, st:st+bl, c] = 0.0

    obs_m = apply_mask(obs, m_hist)
    # ★ 监督目标：完整真实观测（不是遮蔽后的）
    return obs_m, m_hist, obs, batch_fut


# ══════════════════════════════════════════════════════
# Mask-aware GRU（含逐步在线估计）
# ══════════════════════════════════════════════════════
class MaskGRU(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=32, mlp=256):
        super().__init__()
        self.horizon = horizon
        self.c_obs = c_obs
        self.c_act = c_act
        self.hidden = hidden
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
        B, T, C = obs.shape
        H = af.shape[1]
        x = self.inp(torch.cat([obs, mask, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        m_last = mask[:, -1]
        obs_prev = apply_mask(obs[:, -1], m_last)
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([obs_prev, m_last, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            obs_prev = apply_mask(o_hat, m_last)
        return torch.stack(outs, 1)

    # ---- 逐步在线估计 ----
    def init_hidden(self, B, device):
        return torch.zeros(1, B, self.hidden, device=device)

    def step(self, obs_t, mask_t, act_t, h):
        """单步更新：用当前（可能遮蔽的）观测更新隐状态

        obs_t [B,C], mask_t [B,C], act_t [B,A], h [1,B,hidden]
        返回 (o_hat [B,C], h_new)
        """
        obs_m = apply_mask(obs_t, mask_t)
        x = self.inp(torch.cat([obs_m, mask_t, act_t], -1)).unsqueeze(1)
        out, h_new = self.gru(x, h)
        o_hat = self.dec(out[:, 0])
        return o_hat, h_new


# ══════════════════════════════════════════════════════
# Mask-aware RSSM（含逐步在线估计）
# ══════════════════════════════════════════════════════
class MaskRSSM(nn.Module):
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

        self.enc = nn.Sequential(
            nn.Linear(c_obs*2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())
        if use_dt:
            self.cell_fast = nn.GRUCell(z_dim + c_act + self.deter_slow,
                                        deter)
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

    def observe(self, obs_masked, mask, act, mode="gumbel"):
        """训练路径：Encoder 用遮蔽观测，返回的是全部通道的重建"""
        B, T, _ = obs_masked.shape
        state = self.initial(B, obs_masked.device)
        posts, priors, recons = [], [], []
        for t in range(T):
            new, prior_in, prior_logits = self._advance(state, act[:, t], t)
            embed = self.enc(torch.cat([obs_masked[:, t], mask[:, t]], -1))
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

    def imagine(self, obs_masked, mask, act, af, mode="gumbel"):
        B, T, _ = obs_masked.shape
        H = af.shape[1]
        state = self.initial(B, obs_masked.device)
        for t in range(T):
            new, prior_in, _ = self._advance(state, act[:, t], t)
            embed = self.enc(torch.cat([obs_masked[:, t], mask[:, t]], -1))
            post_logits = self.post(
                torch.cat([prior_in, embed], -1)).reshape(
                B, self.stoch, self.classes)
            z = self.sample(post_logits, mode)
            state = {**new, "z": z, "step": t}
        outs = []
        for k in range(H):
            t_abs = T + k
            new, _, prior_logits = self._advance(state, af[:, k], t_abs)
            z = self.sample(prior_logits, mode)
            state = {**new, "z": z, "step": t_abs}
            outs.append(self.dec(self.feat(state)))
        return torch.stack(outs, 1)

    # ---- 逐步在线估计 ----
    def step(self, obs_t, mask_t, act_t, state, t_abs, mode="gumbel"):
        """单步：用当前（可能遮蔽的）观测更新后验并重构全部通道

        obs_t [B,C], mask_t [B,C], act_t [B,A]
        返回 (o_hat [B,C], state_new)
        """
        obs_m = apply_mask(obs_t, mask_t)
        new, prior_in, _ = self._advance(state, act_t, t_abs)
        embed = self.enc(torch.cat([obs_m, mask_t], -1))
        post_logits = self.post(
            torch.cat([prior_in, embed], -1)).reshape(
            obs_t.shape[0], self.stoch, self.classes)
        z = self.sample(post_logits, mode)
        state_new = {**new, "z": z, "step": t_abs}
        o_hat = self.dec(self.feat(state_new))
        return o_hat, state_new

    def kl_terms(self, post_logits, prior_logits):
        logp = F.log_softmax(post_logits, -1)
        logq = F.log_softmax(prior_logits, -1)
        p = logp.exp()
        kl_dyn = (p.detach() * (logp - logq)).sum(-1)
        kl_rep = (p * (logp - logq.detach())).sum(-1)
        return kl_dyn, kl_rep


class MaskWorldModel(nn.Module):
    """★ 修复：forward 同时接收遮蔽观测（给 Encoder）和真实观测（作监督）"""

    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.rssm = MaskRSSM(c_obs, c_act, **kw)

    def forward(self, obs_masked, mask, act, af, fut_true,
                obs_true=None, mode="gumbel"):
        """obs_masked: Encoder 输入
           obs_true:   重建监督目标（None 时回退到 obs_masked，仅为兼容）
           fut_true:   多步预测监督目标（始终是真实值）
        """
        rssm = self.rssm
        T = obs_masked.shape[1]
        target = obs_masked if obs_true is None else obs_true
        post, prior, recon = rssm.observe(obs_masked, mask, act, mode)
        burn = max(1, T // 4)
        # ★ 关键修复：与真实观测比较，不是与遮蔽观测
        rec = ((recon[:, burn:] - target[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        pred = rssm.imagine(obs_masked, mask, act, af, mode)
        multi = ((pred - fut_true) ** 2).mean()
        return {"rec": rec, "kl": kl_clip, "kl_raw": kl_all.mean(),
                "multi": multi, "pred": pred,
                "recon": recon}

    def imagine(self, obs, mask, act, af, mode="gumbel"):
        return self.rssm.imagine(obs, mask, act, af, mode)
