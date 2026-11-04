"""Mask-aware 实现 v3：统一状态更新接口

修复 v2 的问题：
  MaskGRU 训练用 [self.gru 编码历史 + self.cell 递推未来]，
  在线 step 却用 self.gru 直接更新状态，绕过 self.cell，
  导致 Decoder 在训练/推理时收到不同分布的隐状态。

v3 设计：所有路径共用同一套更新原语
  enc_hist(obs, mask, act) -> h0       历史编码（用 self.gru）
  step(h, obs, mask, act) -> (o_hat, h) 单步更新（用 self.cell）
  rollout(h, af, m_last, H) -> obs_seq  未来递推（用 self.cell）

训练、在线估计、自由递推都调用同样的函数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_mask(obs, mask, fill=0.0):
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
    """返回 (obs_masked, mask_hist, obs_true, fut_true)

    ★ 关键：obs_true / fut_true 是完整真实观测，作监督目标
    """
    B, T, C = batch_obs.shape
    obs = batch_obs.clone()
    m_hist = torch.ones(B, T, C, device=device)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(torch.randint(0, 2**31 - 1, (1,)).item()))

    for b in range(B):
        r = torch.rand(1, generator=g).item()
        p1 = cfg.get("p_none", 0.25)
        p2 = p1 + cfg.get("p_fixed", 0.25)
        p3 = p2 + cfg.get("p_random", 0.25)
        if r < p1:
            continue
        elif r < p2:
            for c in (cfg.get("fixed_channels") or [2]):
                m_hist[b, :, c] = 0.0
        elif r < p3:
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
    return obs_m, m_hist, obs, batch_fut


# ══════════════════════════════════════════════════════
# Mask-aware GRU（统一状态更新）
# ══════════════════════════════════════════════════════
class MaskGRU(nn.Module):
    """所有路径共用 self.cell 更新状态

    历史编码：self.gru 一次前向得到 h0
    状态更新：self.cell(h, z) 其中 z = cell_in([o_prev, m, a])
    解码：self.dec(h)

    训练 / 在线估计 / 自由递推全部走同一套 cell 更新。
    """

    def __init__(self, c_obs, c_act, hidden=128, horizon=32, mlp=256):
        super().__init__()
        self.horizon = horizon
        self.c_obs = c_obs
        self.c_act = c_act
        self.hidden = hidden

        # 历史编码器（只在开头跑一次）
        self.inp = nn.Sequential(
            nn.Linear(c_obs * 2 + c_act, hidden), nn.LayerNorm(hidden),
            nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)

        # 统一的状态更新原语
        self.cell_in = nn.Linear(c_obs * 2 + c_act, hidden)
        self.cell = nn.GRUCell(hidden, hidden)

        # 解码器
        self.dec = nn.Sequential(
            nn.Linear(hidden, mlp), nn.SiLU(),
            nn.Linear(mlp, mlp), nn.SiLU(),
            nn.Linear(mlp, c_obs))

    # ---------- 统一原语 ----------
    def init_state(self, B, device):
        """初始隐状态：[B, H]（不是 [1,B,H]）"""
        return torch.zeros(B, self.hidden, device=device)

    def enc_hist(self, obs, mask, act):
        """编码历史观测 -> h0 [B,H]

        obs/mask [B,T,C], act [B,T,A]
        """
        x = self.inp(torch.cat([obs, mask, act], -1))
        _, hn = self.gru(x)
        return hn[-1]                        # [B,H]

    def update(self, h, obs_prev, mask_prev, act_prev):
        """★ 唯一的状态更新原语（训练/推理共用）

        h        [B,H]
        obs_prev [B,C] 上一步观测（可能被遮蔽）
        mask_prev[B,C]
        act_prev [B,A]
        返回 (o_hat [B,C], h_new [B,H])
        """
        o_m = apply_mask(obs_prev, mask_prev)
        z = self.cell_in(torch.cat([o_m, mask_prev, act_prev], -1))
        h_new = self.cell(z, h)
        o_hat = self.dec(h_new)
        return o_hat, h_new

    # ---------- 训练路径 ----------
    def forward(self, obs, mask, act, af, obs_true=None):
        """obs/mask/act: 历史 [B,T,*]；af: 未来动作 [B,H,A]

        返回未来 H 步的预测 [B,H,C]
        """
        B, T, C = obs.shape
        H = af.shape[1]
        h = self.enc_hist(obs, mask, act)        # [B,H]
        m_last = mask[:, -1]                     # [B,C]
        o_prev = obs[:, -1]                      # [B,C]
        outs = []
        for k in range(H):
            o_hat, h = self.update(h, o_prev, m_last, af[:, k])
            outs.append(o_hat)
            # 自回归：下一步用预测值作观测（遮蔽通道保持 0）
            o_prev = o_hat
        return torch.stack(outs, 1)

    # ---------- 逐步在线估计 ----------
    def online_init(self, hist_obs, hist_mask, hist_act):
        """用故障前历史初始化状态"""
        return self.enc_hist(hist_obs, hist_mask, hist_act)

    def online_step(self, h, obs_t, mask_t, act_t):
        """★ 与 forward 完全相同的 update 原语"""
        return self.update(h, obs_t, mask_t, act_t)

    # ---------- 自由递推（不看任何新观测） ----------
    def rollout_open(self, h, af, mask_last):
        """严格开环：只用控制输入，不接收新观测"""
        B, H = af.shape[0], af.shape[1]
        o_prev = torch.zeros(B, self.c_obs, device=af.device)
        outs = []
        for k in range(H):
            o_hat, h = self.update(h, o_prev, mask_last, af[:, k])
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


# ══════════════════════════════════════════════════════
# Mask-aware RSSM（统一状态更新）
# ══════════════════════════════════════════════════════
class MaskRSSM(nn.Module):
    """同样把所有路径收敛到 cell + posterior 这一套原语"""

    def __init__(self, c_obs, c_act, deter=128, stoch=32, classes=32,
                 hidden=128, free_bits=1.0, unimix=0.01):
        super().__init__()
        self.c_obs = c_obs
        self.c_act = c_act
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.free_bits = free_bits
        self.unimix = unimix
        z_dim = stoch * classes
        feat_dim = deter + z_dim

        self.enc = nn.Sequential(
            nn.Linear(c_obs * 2, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())
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
        self._feat_dim = feat_dim

    # ---------- 统一原语 ----------
    def init_state(self, B, device):
        return {"h": torch.zeros(B, self.deter, device=device),
                "z": torch.zeros(B, self.stoch, self.classes,
                                 device=device)}

    def feat(self, st):
        B = st["z"].shape[0]
        return torch.cat([st["h"], st["z"].reshape(B, -1)], -1)

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

    def enc_hist(self, obs, mask, act, mode="gumbel"):
        """编码历史（用后验）"""
        B, T, _ = obs.shape
        obs_m = apply_mask(obs, mask)
        st = self.init_state(B, obs.device)
        for t in range(T):
            st = self._step_post(st, obs_m[:, t], mask[:, t], act[:, t],
                                 mode)
        return st

    def _step_post(self, st, obs_m_t, mask_t, act_t, mode):
        """一步：先验 -> 后验 -> 采样（唯一的状态更新实现）"""
        B = st["z"].shape[0]
        z_prev = st["z"].reshape(B, -1)
        h = self.cell(torch.cat([z_prev, act_t], -1), st["h"])
        prior_logits = self.prior(
            torch.cat([h, z_prev], -1)).reshape(B, self.stoch,
                                                self.classes)
        embed = self.enc(torch.cat([obs_m_t, mask_t], -1))
        post_logits = self.post(
            torch.cat([torch.cat([h, z_prev], -1), embed], -1)).reshape(
            B, self.stoch, self.classes)
        z = self.sample(post_logits, mode)
        return {"h": h, "z": z}

    def _step_prior(self, st, act_t, mode):
        """一步：只有先验（不看观测）"""
        B = st["z"].shape[0]
        z_prev = st["z"].reshape(B, -1)
        h = self.cell(torch.cat([z_prev, act_t], -1), st["h"])
        prior_logits = self.prior(
            torch.cat([h, z_prev], -1)).reshape(B, self.stoch,
                                                self.classes)
        z = self.sample(prior_logits, mode)
        return {"h": h, "z": z}

    # ---------- 训练 ----------
    def observe(self, obs_masked, mask, act, mode="gumbel"):
        """训练路径：Encoder 用遮蔽观测，返回全部通道重建"""
        B, T, _ = obs_masked.shape
        obs_m = apply_mask(obs_masked, mask)
        st = self.init_state(B, obs_masked.device)
        posts, priors, recons = [], [], []
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
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1))

    def imagine(self, obs_masked, mask, act, af, mode="gumbel"):
        B, T, _ = obs_masked.shape
        H = af.shape[1]
        obs_m = apply_mask(obs_masked, mask)
        st = self.init_state(B, obs_masked.device)
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
        outs = []
        for k in range(H):
            st = self._step_prior(st, af[:, k], mode)
            outs.append(self.dec(self.feat(st)))
        return torch.stack(outs, 1)

    # ---------- 在线估计 ----------
    def online_init(self, hist_obs, hist_mask, hist_act, mode="gumbel"):
        return self.enc_hist(hist_obs, hist_mask, hist_act, mode)

    def online_step(self, st, obs_t, mask_t, act_t, mode="gumbel"):
        """★ 与训练相同的后验更新"""
        obs_m = apply_mask(obs_t, mask_t)
        st2 = self._step_post(st, obs_m, mask_t, act_t, mode)
        return self.dec(self.feat(st2)), st2

    # ---------- 严格开环 ----------
    def rollout_open(self, st, af, mode="gumbel"):
        H = af.shape[1]
        outs = []
        for k in range(H):
            st = self._step_prior(st, af[:, k], mode)
            outs.append(self.dec(self.feat(st)))
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

    def forward(self, obs_masked, mask, act, af, fut_true,
                obs_true=None, mode="gumbel"):
        rssm = self.rssm
        T = obs_masked.shape[1]
        target = obs_masked if obs_true is None else obs_true
        post, prior, recon = rssm.observe(obs_masked, mask, act, mode)
        burn = max(1, T // 4)
        rec = ((recon[:, burn:] - target[:, burn:]) ** 2).mean()
        kl_dyn, kl_rep = rssm.kl_terms(post, prior)
        kl_all = kl_dyn + kl_rep
        kl_clip = torch.clamp(kl_all, min=rssm.free_bits).mean()
        pred = rssm.imagine(obs_masked, mask, act, af, mode)
        multi = ((pred - fut_true) ** 2).mean()
        return {"rec": rec, "kl": kl_clip, "kl_raw": kl_all.mean(),
                "multi": multi, "pred": pred, "recon": recon}

    def imagine(self, obs_masked, mask, act, af, mode="gumbel"):
        return self.rssm.imagine(obs_masked, mask, act, af, mode)


class MaskGRUModel(nn.Module):
    """MaskGRU 的训练包装（与 MaskWorldModel 接口一致）"""
    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.gru = MaskGRU(c_obs, c_act, **kw)

    def forward(self, obs_masked, mask, act, af, fut_true,
                obs_true=None, mode=None):
        pred = self.gru(obs_masked, mask, act, af)
        multi = ((pred - fut_true) ** 2).mean()
        return {"rec": torch.tensor(0.0), "kl": torch.tensor(0.0),
                "kl_raw": torch.tensor(0.0), "multi": multi,
                "pred": pred, "recon": None}
