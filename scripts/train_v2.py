"""改进版训练 v2：修 KL、防过拟合、增强评估"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(r"D:\涡轴WM\_backup\workspace_20261031_182906")
sys.path.insert(0, str(ROOT / "src"))
from aero_world_model.models.rssm import RSSMConfig
from aero_world_model.models.baselines import BaselineGRU
from aero_world_model.training.offline import WorldModel

DATA = Path(r"C:\pclwm\agtf30\prepared2")
OUT = Path(r"C:\pclwm\agtf30\train2")
OUT.mkdir(parents=True, exist_ok=True)

STEP, WINDOW, HORIZON = 7, 64, 32      # horizon 16 -> 32 (3.4 s)
STRIDE_TR, STRIDE_EV = 4, 8            # stride 8 -> 4（窗口翻倍）
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)

d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = list(d["obs_cols"])
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
runs = {str(r): {"obs": d[f"{r}_obs"][::STEP].astype(np.float32),
                 "act": d[f"{r}_act"][::STEP].astype(np.float32)}
        for r in d["run_ids"]}

DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]
KEEP = [c for c in OBS_ALL if c not in DROP]
KIDX = [OBS_ALL.index(c) for c in KEEP]

tr_obs = np.concatenate([runs[r]["obs"] for r in splits["train"]], 0)
tr_act = np.concatenate([runs[r]["act"] for r in splits["train"]], 0)
o_mu = tr_obs.mean(0).astype(np.float64)
o_sd = tr_obs.std(0).astype(np.float64) + 1e-8
a_mu = tr_act.mean(0).astype(np.float64)
a_sd = tr_act.std(0).astype(np.float64) + 1e-8
C_O, C_A = len(OBS_ALL), 1

print(f"训练轨迹 {len(splits['train'])}, 总点 {len(tr_obs):,}")
print(f"WINDOW={WINDOW} HORIZON={HORIZON} (={HORIZON*STEP*0.015:.2f}s)")
print(f"通道 {C_O} 个，评估用 {len(KEEP)} 个")


def mkw(ids, stride):
    W = []
    for r in ids:
        o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float32)
        a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float32)
        for st in range(1, len(o) - WINDOW - HORIZON, stride):
            W.append({"obs": o[st:st + WINDOW],
                      "act": a[st - 1:st - 1 + WINDOW],
                      "fut": o[st + WINDOW:st + WINDOW + HORIZON],
                      "act_fut": a[st - 1 + WINDOW:st - 1 + WINDOW + HORIZON]})
    return W


tr_W = mkw(splits["train"], STRIDE_TR)
va_W = mkw(splits["val"], STRIDE_EV)
te_W = mkw(splits["test"], STRIDE_EV)
print(f"窗口: train={len(tr_W)} val={len(va_W)} test={len(te_W)}")


def batchify(ws, bs, shuffle=True):
    idx = np.random.permutation(len(ws)) if shuffle else np.arange(len(ws))
    for i in range(0, len(ws), bs):
        c = [ws[j] for j in idx[i:i + bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c])).to(DEV)
               for k in ["obs", "act", "fut", "act_fut"]}


def rel_norm(preds, ws):
    e = np.stack([p - w["fut"] for p, w in zip(preds, ws)], 0)
    return np.sqrt((e ** 2).mean(0))[:, KIDX].mean(1)


@torch.no_grad()
def nn_predict(model, ws, kind, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i + bs]
        o = torch.from_numpy(np.stack([w["obs"] for w in c])).to(DEV)
        a = torch.from_numpy(np.stack([w["act"] for w in c])).to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).to(DEV)
        p = model(o, a, af) if kind == "gru" else model.imagine(o, a, af)
        out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


# ══════════════════════════════════════════════════
# GRU（改进：hidden 256->192, dropout 0.1）
# ══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("GRU v2（hidden=192, dropout=0.1, weight_decay=1e-4）")
print("=" * 70)
gru = BaselineGRU(C_O, C_A, hidden=192, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
n_gru = sum(p.numel() for p in gru.parameters())
print(f"  参数: {n_gru:,}")
opt = torch.optim.Adam(gru.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)
best, bep, ni = float("inf"), -1, 0
hg = {"train": [], "val": []}
t0 = time.time()
for ep in range(1, 121):
    gru.train()
    tl, n = 0.0, 0
    for b in batchify(tr_W, 64):
        opt.zero_grad(set_to_none=True)
        p = gru(b["obs"], b["act"], b["act_fut"])
        loss = ((p - b["fut"]) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), 100.0)
        opt.step()
        tl += float(loss.detach()) * len(b["obs"]); n += len(b["obs"])
    sched.step()
    gru.eval()
    vs, vn = 0.0, 0
    with torch.no_grad():
        for b in batchify(va_W, 128, shuffle=False):
            p = gru(b["obs"], b["act"], b["act_fut"])
            vs += float(((p - b["fut"]) ** 2).mean()) * len(b["obs"])
            vn += len(b["obs"])
    hg["train"].append(tl / n); hg["val"].append(vs / vn)
    if vs / vn < best - 1e-7:
        best, bep, ni = vs / vn, ep, 0
        torch.save(gru.state_dict(), OUT / "gru.pt")
    else:
        ni += 1
    if ep % 20 == 0 or ep == 1:
        print(f"  ep {ep:>3} train={tl/n:.5f} val={best:.5f}@{bep} "
              f"({time.time()-t0:.0f}s)")
    if ni >= 40:
        print(f"  早停 ep {ep}")
        break
gru.load_state_dict(torch.load(OUT / "gru.pt"))
gru.eval()
r_gru_te = rel_norm(nn_predict(gru, te_W, "gru"), te_W)
r_gru_tr = rel_norm(nn_predict(gru, tr_W, "gru"), tr_W)
print(f"  test: mean={r_gru_te.mean():.4f} k1={r_gru_te[0]:.4f} "
      f"kH={r_gru_te[-1]:.4f}")

# ══════════════════════════════════════════════════
# RSSM v2（关键修复：kl_free 降到 0.5, kl_scale)
# ══════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RSSM v2（kl_free=0.5, kl_scale=0.3, 更强的多步权重）")
print("=" * 70)
KL_SCALE = 0.3
rcfg = RSSMConfig(stoch=32, classes=32, deter=256, hidden=256)
rssm = WorldModel(C_O, C_A, rcfg).to(DEV)
n_rssm = sum(p.numel() for p in rssm.parameters())
print(f"  参数: {n_rssm:,}")
opt = torch.optim.Adam(rssm.parameters(), lr=4e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=120)


def rssm_parts(b):
    obs, act, fut, af = b["obs"], b["act"], b["fut"], b["act_fut"]
    post, prior, recon, _ = rssm(obs, act)
    l_rec = ((recon - obs) ** 2).mean()
    kl, _, _, _ = rssm.dynamics.kl_loss(post, prior)
    l_kl = kl.mean() * KL_SCALE
    st = {k: v[:, -1] for k, v in post.items()}
    trail = rssm.dynamics.imagine_with_action(af, st)
    pf = rssm.decoder(rssm.dynamics.get_feat(trail)).mode()
    # 多步 + 单步混合
    l_multi = ((pf - fut) ** 2).mean()
    l_one = ((pf[:, 0] - fut[:, 0]) ** 2).mean() * 2.0
    return (l_rec + l_kl + l_multi + l_one), l_rec, l_kl, l_multi


best, bep, ni = float("inf"), -1, 0
hr = {"train": [], "val": [], "rec": [], "kl": [], "multi": []}
t0 = time.time()
for ep in range(1, 121):
    rssm.train()
    acc = np.zeros(4); n = 0
    for b in batchify(tr_W, 32):
        opt.zero_grad(set_to_none=True)
        tot, r, k, m = rssm_parts(b)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
        opt.step()
        acc += np.array([float(tot.detach()), float(r.detach()),
                         float(k.detach()), float(m.detach())]) * len(b["obs"])
        n += len(b["obs"])
    sched.step()
    rssm.eval()
    va, vn = 0.0, 0
    with torch.no_grad():
        for b in batchify(va_W, 64, shuffle=False):
            tot, _, _, _ = rssm_parts(b)
            va += float(tot) * len(b["obs"]); vn += len(b["obs"])
    hr["train"].append(acc[0] / n); hr["rec"].append(acc[1] / n)
    hr["kl"].append(acc[2] / n); hr["multi"].append(acc[3] / n)
    hr["val"].append(va / vn)
    if va / vn < best - 1e-7:
        best, bep, ni = va / vn, ep, 0
        torch.save(rssm.state_dict(), OUT / "rssm.pt")
    else:
        ni += 1
    if ep % 20 == 0 or ep == 1:
        print(f"  ep {ep:>3} train={acc[0]/n:.5f} val={best:.5f}@{bep}  "
              f"rec={acc[1]/n:.5f} kl={acc[2]/n:.5f} multi={acc[3]/n:.5f} "
              f"({time.time()-t0:.0f}s)")
    if ni >= 40:
        print(f"  早停 ep {ep}")
        break
rssm.load_state_dict(torch.load(OUT / "rssm.pt"))
rssm.eval()
r_rssm_te = rel_norm(nn_predict(rssm, te_W, "rssm"), te_W)
r_rssm_tr = rel_norm(nn_predict(rssm, tr_W, "rssm"), tr_W)
print(f"  test: mean={r_rssm_te.mean():.4f} k1={r_rssm_te[0]:.4f} "
      f"kH={r_rssm_te[-1]:.4f}")

# ══════════════════════════════════════════════════
# 基线
# ══════════════════════════════════════════════════
def baselines(ws):
    out = {}
    out["Persistence"] = rel_norm(
        [np.repeat(w["obs"][-1:], HORIZON, 0) for w in ws], ws)
    Xa, Ya = [], []
    for r in splits["train"]:
        o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float64)
        a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float64)
        Xa.append(np.c_[o[:-1], a[:-1]]); Ya.append(o[1:])
    X = np.concatenate(Xa, 0); Y = np.concatenate(Ya, 0)
    Wm = np.linalg.lstsq(np.c_[X, np.ones((len(X), 1))], Y, rcond=1e-6)[0]
    A, B, c0 = Wm[:C_O].T, Wm[C_O:C_O + 1].T, Wm[-1]
    preds = []
    for w in ws:
        y = w["obs"][-1].astype(np.float64).copy()
        oo = []
        for k in range(HORIZON):
            y = y @ A.T + w["act_fut"][k].astype(np.float64) @ B.T + c0
            oo.append(y.copy())
        preds.append(np.stack(oo, 0))
    out["ARX"] = rel_norm(preds, ws)
    return out


b_te = baselines(te_W); b_tr = baselines(tr_W)

print("\n" + "=" * 78)
print("最终结果 v2")
print("=" * 78)
print(f"{'模型':<16}{'训练':>10}{'测试':>10}{'泛化比':>10}"
      f"{'k=1':>9}{'k=H':>9}{'增长':>8}")
print("-" * 74)
rows = {}
allm = {"Persistence": (b_tr["Persistence"], b_te["Persistence"]),
        "ARX": (b_tr["ARX"], b_te["ARX"]),
        "GRU": (r_gru_tr, r_gru_te),
        "RSSM": (r_rssm_tr, r_rssm_te)}
for nm, (a, b) in allm.items():
    rows[nm] = {"train": a.mean(), "test": b.mean(), "k1": b[0],
                "kH": b[-1], "curve": b.tolist()}
    print(f"{nm:<16}{a.mean():>10.4f}{b.mean():>10.4f}"
          f"{b.mean()/a.mean():>10.1f}x{b[0]:>9.4f}{b[-1]:>9.4f}"
          f"{b[-1]/max(b[0],1e-12):>8.2f}x")

json.dump({"results": rows,
           "config": {"window": WINDOW, "horizon": HORIZON, "step": STEP,
                      "dt": STEP * 0.015, "keep": [str(k) for k in KEEP]},
           "hist_gru": hg, "hist_rssm": hr,
           "params": {"gru": n_gru, "rssm": n_rssm}},
          open(OUT / "results2.json", "w"), ensure_ascii=False, indent=1)
print(f"\n保存 -> {OUT/'results2.json'}")
