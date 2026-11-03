"""GRU 公平重跑（WINDOW=32，与 RSSM 完全一致）

修正：原 GRU 在 WINDOW=64 下训练，RSSM 在 WINDOW=32，
     历史信息不同，比较不公平。
     本脚本用当前 WINDOW=32 重跑 GRU。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import (UnifiedData, UnifiedEval, rollout_far, WINDOW,
                      HORIZON, LONG_TARGETS)

OUT = Path(r"C:\pclwm\pclwm_experiments")
CKPT = OUT / "checkpoints"
DIAG = OUT / "diagnosis"
DIAG.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt

tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)

print("=" * 84)
print("GRU 公平重跑（WINDOW=%d，历史 %.2f s）" % (WINDOW, WINDOW*DT))
print("=" * 84)
print(f"  窗口: train={len(tr_W)} val={len(va_W)} "
      f"test_id={len(te_W)} test_ood={len(ood_W)}")
print(f"  >>> 与 RSSM 完全相同的窗口数，确保公平")


class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=HORIZON,
                 mlp=256, layers=2):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Sequential(
            nn.Linear(c_obs + c_act, hidden), nn.LayerNorm(hidden),
            nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        mods = [nn.Linear(hidden, mlp), nn.SiLU()]
        for _ in range(layers-1):
            mods += [nn.Linear(mlp, mlp), nn.SiLU()]
        mods += [nn.Linear(mlp, c_obs)]
        self.dec = nn.Sequential(*mods)
        self.o2h = nn.Linear(c_obs + c_act, hidden)

    def forward(self, obs, act, af):
        B, T, _ = obs.shape
        H = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


def iterate(ws, bs, shuffle=True):
    n = len(ws)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i+bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c]))
               .float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut"]}


@torch.no_grad()
def predict(m, ws, bs=128):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        out.append(m(b["obs"], b["act"], b["act_fut"]).cpu().numpy())
    return np.concatenate(out, 0)


def train_one(name, hidden, mlp, max_ep=120, patience=25, lr=1e-3,
              bs=64):
    torch.manual_seed(0); np.random.seed(0)
    m = GRUModel(C_O, C_A, hidden=hidden, mlp=mlp).to(DEV)
    npar = sum(p.numel() for p in m.parameters())
    print(f"\n  [{name}] hidden={hidden} mlp={mlp} 参数={npar:,}")
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()
    for ep in range(1, max_ep+1):
        m.train()
        tl, n = 0.0, 0
        for b in iterate(tr_W, bs, True):
            opt.zero_grad(set_to_none=True)
            p = m(b["obs"], b["act"], b["act_fut"])
            loss = ((p - b["fut"])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 100.)
            opt.step()
            tl += float(loss.detach())*len(b["obs"]); n += len(b["obs"])
        sched.step()
        m.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in iterate(va_W, 128, False):
                p = m(b["obs"], b["act"], b["act_fut"])
                vs += float(((p - b["fut"])**2).mean())*len(b["obs"])
                vn += len(b["obs"])
        v = vs/vn
        hist["train"].append(tl/n); hist["val"].append(v)
        if v < best - 1e-7:
            best, bep, bad = v, ep, 0
            torch.save(m.state_dict(), CKPT/f"{name}_w32.pt")
        else:
            bad += 1
        if ep % 25 == 0 or ep == 1:
            print(f"    ep {ep:>3} train={tl/n:.6f} val={v:.6f} "
                  f"best={best:.6f}@{bep} ({time.time()-t0:.0f}s)",
                  flush=True)
        if bad >= patience:
            print(f"    早停 ep {ep}")
            break
    m.load_state_dict(torch.load(CKPT/f"{name}_w32.pt",
                                map_location=DEV, weights_only=True))
    m.eval()
    return m, hist, best, bep, time.time()-t0, npar


RES = {}
for name, hid, mlp in [("GRU96", 96, 192), ("GRU128", 128, 256),
                       ("GRU192", 192, 384)]:
    m, hist, bv, bep, el, npar = train_one(name, hid, mlp)
    p_te = predict(m, te_W)
    p_ood = predict(m, ood_W)
    r_te = ev.evaluate(p_te, te_W)
    r_ood = ev.evaluate(p_ood, ood_W)

    def mfn(obs, act, af, _m=m):
        with torch.no_grad():
            return _m(torch.from_numpy(obs).float().to(DEV),
                      torch.from_numpy(act).float().to(DEV),
                      torch.from_numpy(af).float().to(DEV)).cpu().numpy()

    lng = {}
    for split in ["train", "val", "test_id"]:
        for tgt in LONG_TARGETS:
            rs = []
            for r in data.splits[split]:
                x = rollout_far(mfn, data, r, tgt)
                if x:
                    rs.append(x)
            if rs:
                lng[f"{split}_{tgt}s"] = float(
                    np.mean([x["final_rel"] for x in rs]))

    RES[name] = {
        "params": int(npar), "best_val": float(bv),
        "best_epoch": int(bep), "train_time_s": el,
        "window": WINDOW,
        "hist": {k: [float(x) for x in v] for k, v in hist.items()},
        "test_id": r_te, "test_ood": r_ood, "long": lng,
        "config": {"hidden": hid, "mlp": mlp},
    }
    print(f"    test_id={r_te['overall_rel']:.4f} "
          f"test_ood={r_ood['overall_rel']:.4f}")
    for k, v in lng.items():
        print(f"    长时 {k}: {v:.4f}")

(DIAG / "gru_fair_w32.json").write_text(
    json.dumps(RES, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {DIAG/'gru_fair_w32.json'}")
