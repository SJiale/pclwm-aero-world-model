"""RSSM 实现（独立文件，无 import 副作用）

与 run_rssm.py 中的定义**完全一致**（保证能加载同一检查点），
但把维度作为参数传入，不在模块级依赖数据。

诊断用：可安全 import，不会触发训练。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(self, c_obs, c_act, deter=192, stoch=32, classes=32,
                 hidden=192, free_bits=1.0, dyn_scale=0.5,
                 rep_scale=0.1, unimix=0.01):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.classes = classes
        self.free_bits = free_bits
        self.dyn_scale = dyn_scale
        self.rep_scale = rep_scale
        self.unimix = unimix
        self.c_obs = c_obs
        self.c_act = c_act

        self.enc = nn.Sequential(
            nn.Linear(c_obs, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())
        self.post = nn.Sequential(
            nn.Linear(deter + 128, hidden), nn.SiLU(),
            nn.Linear(hidden, stoch * classes))
        self.prior = nn.Sequential(
            nn.Linear(deter, hidden), nn.SiLU(),
            nn.Linear(hidden, stoch * classes))
        self.cell = nn.GRUCell(stoch * classes + c_act, deter)
        self.dec = nn.Sequential(
            nn.Linear(deter + stoch * classes, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, c_obs))

    def initial(self, B, device):
        return {"h": torch.zeros(B, self.deter, device=device),
                "z": torch.zeros(B, self.stoch, self.classes,
                                 device=device)}

    def feat(self, state):
        z = state["z"].reshape(state["z"].shape[0], -1)
        return torch.cat([state["h"], z], -1)

    # ---- 采样：可显式指定模式，不再依赖 self.training ----
    def sample(self, logits, mode="gumbel"):
        """mode: gumbel | argmax | prob(返回 softmax 期望, 非 one-hot)"""
        if mode == "prob":
            return F.softmax(logits, -1)
        if self.unimix > 0:
            probs = F.softmax(logits, -1)
            probs = probs * (1 - self.unimix) + self.unimix / self.classes
            logits2 = torch.log(probs.clamp_min(1e-9))
        else:
            logits2 = logits
        if mode == "gumbel":
            return F.gumbel_softmax(logits2, tau=1.0, hard=True, dim=-1)
        if mode == "argmax":
            idx = logits2.argmax(-1, keepdim=True)
            return torch.zeros_like(logits2).scatter_(-1, idx, 1.0)
        raise ValueError(mode)

    def _sample(self, logits):
        """兼容原实现：训练时 gumbel，eval 时 argmax"""
        return self.sample(logits, "gumbel" if self.training else "argmax")

    def observe(self, obs, act, sample_mode=None):
        B, T, _ = obs.shape
        dev = obs.device
        state = self.initial(B, dev)
        posts, priors, recons, zs = [], [], [], []
        for t in range(T):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act[:, t]], -1), state["h"])
            prior_logits = self.prior(h).reshape(B, self.stoch, self.classes)
            embed = self.enc(obs[:, t])
            post_logits = self.post(torch.cat([h, embed], -1)) \
                .reshape(B, self.stoch, self.classes)
            sm = sample_mode or ("gumbel" if self.training else "argmax")
            z = self.sample(post_logits, sm)
            state = {"h": h, "z": z}
            posts.append(post_logits)
            priors.append(prior_logits)
            recons.append(self.dec(self.feat(state)))
            zs.append(z)
        return (torch.stack(posts, 1), torch.stack(priors, 1),
                torch.stack(recons, 1), torch.stack(zs, 1))

    def imagine(self, obs_hist, act_hist, act_future, mode=None):
        """推理：未来只用先验"""
        B, T, _ = obs_hist.shape
        H = act_future.shape[1]
        dev = obs_hist.device
        sm = mode or ("gumbel" if self.training else "argmax")
        state = self.initial(B, dev)
        for t in range(T):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act_hist[:, t]], -1),
                          state["h"])
            embed = self.enc(obs_hist[:, t])
            post_logits = self.post(torch.cat([h, embed], -1)) \
                .reshape(B, self.stoch, self.classes)
            z = self.sample(post_logits, sm)
            state = {"h": h, "z": z}
        outs = []
        for k in range(H):
            prev_z = state["z"].reshape(B, -1)
            h = self.cell(torch.cat([prev_z, act_future[:, k]], -1),
                          state["h"])
            prior_logits = self.prior(h).reshape(B, self.stoch,
                                                 self.classes)
            z = self.sample(prior_logits, sm)
            state = {"h": h, "z": z}
            outs.append(self.dec(self.feat(state)))
        return torch.stack(outs, 1)

    def kl_terms(self, post_logits, prior_logits, mode="both"):
        """分解 KL

        dynamics KL   = KL(sg(post) || prior)   -> 训练 **先验**
        representation KL = KL(post || sg(prior)) -> 训练 **后验/编码器**
        """
        logp = F.log_softmax(post_logits, -1)
        logq = F.log_softmax(prior_logits, -1)
        p = logp.exp()
        q = logq.exp()
        # per (B,T,stoch) 维度
        kl_dyn = (p.detach() * (logp - logq)).sum(-1)
        kl_rep = (p * (logp - logq.detach())).sum(-1)
        return kl_dyn, kl_rep


class WorldModel(nn.Module):
    def __init__(self, c_obs, c_act, **kw):
        super().__init__()
        self.rssm = RSSM(c_obs, c_act, **kw)

    def imagine(self, obs, act, act_fut, mode=None):
        return self.rssm.imagine(obs, act, act_fut, mode)


def load_rssm(ckpt, c_obs, c_act, deter=128, hidden=128, stoch=32,
              classes=32, device="cuda"):
    m = WorldModel(c_obs, c_act, deter=deter, stoch=stoch,
                   classes=classes, hidden=hidden, free_bits=1.0)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    m.load_state_dict(sd)
    m.to(device)
    m.eval()
    return m
