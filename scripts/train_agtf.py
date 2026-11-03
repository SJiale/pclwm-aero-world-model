"""AGTF30e 世界模型训练：RSSM vs ARX vs GRU

用现有 PyTorch RSSM 代码，接 AGTF30e 数据。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(r"D:\涡轴WM\_backup\workspace_20261031_182906")
sys.path.insert(0, str(ROOT / "src"))

from aero_world_model.models.rssm import RSSMConfig
from aero_world_model.models.baselines import BaselineGRU
from aero_world_model.training.offline import WorldModel

DATA = Path(r"C:\pclwm\agtf30\prepared")
OUT = Path(r"C:\pclwm\agtf30\train")
OUT.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)

STEP = 7            # 降采样: 0.015 -> 0.105 s
WINDOW = 64         # 6.7 s 历史
HORIZON = 16        # 1.7 s 预测
STRIDE = 8

# ---------- 数据 ----------
d = np.load(DATA / "data.npz", allow_pickle=True)
OBS = list(d["obs_cols"])
ACT = list(d["act_cols"])
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))

runs = {}
for r in d["run_ids"]:
    r = str(r)
    o = d[f"{r}_obs"][::STEP].astype(np.float32)
    a = d[f"{r}_act"][::STEP].astype(np.float32)
    runs[r] = {"obs": o, "act": a}

C_O = len(OBS)
C_A = len(ACT)

# 归一化（只用 train 拟合）
tr_obs = np.concatenate([runs[r]["obs"] for r in splits["train"]], 0)
tr_act = np.concatenate([runs[r]["act"] for r in splits["train"]], 0)
o_mu, o_sd = tr_obs.mean(0), tr_obs.std(0) + 1e-8
a_mu, a_sd = tr_act.mean(0), tr_act.std(0) + 1e-8

print(f"obs {C_O} 维, act {C_A} 维")
print(f"train {len(splits['train'])} 条, val {len(splits['val'])}, test {len(splits['test'])}")
print(f"总训练点: {sum(len(runs[r]['obs']) for r in splits['train']):,}")


class WinDS(Dataset):
    def __init__(self, ids):
        self.s = []
        self.data = {}
        for rid in ids:
            o = (runs[rid]["obs"] - o_mu) / o_sd
            a = (runs[rid]["act"] - a_mu) / a_sd
            self.data[rid] = (o.astype(np.float32), a.astype(np.float32))
            n = len(o)
            need = WINDOW + HORIZON
            for st in range(1, n - need):
                if st % STRIDE == 0:
                    self.s.append((rid, st))

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        rid, st = self.s[i]
        o, a = self.data[rid]
        T, H = WINDOW, HORIZON
        # 动作前移一拍：act[t] 产生 obs[t+1]
        return {
            "obs": torch.from_numpy(o[st:st + T]),
            "act": torch.from_numpy(a[st - 1:st - 1 + T]),
            "fut": torch.from_numpy(o[st + T:st + T + H]),
            "act_fut": torch.from_numpy(a[st - 1 + T:st - 1 + T + H]),
        }


ds_tr = WinDS(splits["train"])
ds_va = WinDS(splits["val"])
ds_te = WinDS(splits["test"])
print(f"窗口: train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")

tl = DataLoader(ds_tr, batch_size=32, shuffle=True, drop_last=True)
vl = DataLoader(ds_va, batch_size=32, shuffle=False)
tel = DataLoader(ds_te, batch_size=64, shuffle=False)


# ---------- 评估 ----------
@torch.no_grad()
def evaluate_rollout(predict_fn, loader, H=HORIZON):
    """返回逐通道物理单位 RMSE (k=1..H)"""
    errs = []
    for b in loader:
        pred = predict_fn(b)                    # [B,H,C] 归一化
        tgt = b["fut"].to(DEV)
        if pred.device != tgt.device:
            pred = pred.to(tgt.device)
        e = (pred - tgt).detach().cpu().numpy()
        errs.append(e)
    e = np.concatenate(errs, 0)                 # [N,H,C]
    # 反归一化
    e_phys = e * o_sd[None, None, :]
    rmse = np.sqrt((e_phys ** 2).mean(0))       # [H,C]
    rel = rmse / o_sd[None, :]
    return {"rel_rmse_by_channel": rel.tolist(),
            "rel_mean": float(rel.mean()),
            "rel_k1": float(rel[0].mean()),
            "rel_kH": float(rel[-1].mean()),
            "rel_by_k": rel.mean(1).tolist()}


results = {}

# ---------- 1. ARX ----------
print("\n" + "=" * 70)
print("[1/3] ARX 线性基线")
print("=" * 70)

X_all, Y_all = [], []
for rid in splits["train"]:
    o, a = runs[rid]["obs"], runs[rid]["act"]
    o = (o - o_mu) / o_sd
    a = (a - a_mu) / a_sd
    X_all.append(np.c_[o[:-1], a[:-1]])
    Y_all.append(o[1:])
X = np.concatenate(X_all, 0).astype(np.float64)
Y = np.concatenate(Y_all, 0).astype(np.float64)
W = np.linalg.lstsq(np.c_[X, np.ones((len(X), 1))], Y, rcond=1e-6)[0]
A = W[:C_O].T
B = W[C_O:C_O + C_A].T
c = W[-1]
n_arx = C_O * (C_O + C_A + 1)
print(f"  参数量: {n_arx}")


@torch.no_grad()
def arx_pred(b):
    o = b["obs"].to(DEV).cpu().numpy()
    af = b["act_fut"].to(DEV).cpu().numpy()
    n, H, _ = af.shape
    y = o[:, -1].copy()
    outs = []
    for k in range(H):
        y = y @ A.T + af[:, k] @ B.T + c
        outs.append(y.copy())
    return torch.from_numpy(np.stack(outs, 1)).float().to(DEV)


t0 = time.time()
results["arx"] = evaluate_rollout(arx_pred, tel)
results["arx"]["time"] = time.time() - t0
results["arx"]["params"] = n_arx
print(f"  相对RMSE: {results['arx']['rel_mean']:.4f}  "
      f"(k=1: {results['arx']['rel_k1']:.4f}, k=H: {results['arx']['rel_kH']:.4f})")

# ---------- 2. GRU ----------
print("\n" + "=" * 70)
print("[2/3] GRU 基线")
print("=" * 70)
gru = BaselineGRU(C_O, C_A, hidden=256, horizon=HORIZON,
                  mlp_units=512, mlp_layers=2).to(DEV)
n_gru = sum(p.numel() for p in gru.parameters())
print(f"  参数量: {n_gru:,}")
opt = torch.optim.Adam(gru.parameters(), lr=1e-3)
t0 = time.time()
best, bep, ni = float("inf"), -1, 0
for ep in range(1, 61):
    gru.train()
    for b in tl:
        opt.zero_grad(set_to_none=True)
        p = gru(b["obs"].to(DEV), b["act"].to(DEV), b["act_fut"].to(DEV))
        loss = ((p - b["fut"].to(DEV)) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gru.parameters(), 100.0)
        opt.step()
    gru.eval()
    vs, vn = 0.0, 0
    with torch.no_grad():
        for b in vl:
            p = gru(b["obs"].to(DEV), b["act"].to(DEV), b["act_fut"].to(DEV))
            vs += float(((p - b["fut"].to(DEV)) ** 2).mean()) * len(b["obs"])
            vn += len(b["obs"])
    v = vs / max(vn, 1)
    if v < best - 1e-7:
        best, bep, ni = v, ep, 0
        torch.save(gru.state_dict(), OUT / "gru.pt")
    else:
        ni += 1
    if ep % 10 == 0 or ep == 1:
        print(f"    ep {ep:>2} val={v:.6f} best={best:.6f}@{bep} ({time.time()-t0:.0f}s)")
    if ni >= 15:
        print(f"    早停 ep {ep}")
        break
gru.load_state_dict(torch.load(OUT / "gru.pt"))
gru.eval()
results["gru"] = evaluate_rollout(
    lambda b: gru(b["obs"].to(DEV), b["act"].to(DEV), b["act_fut"].to(DEV)),
    tel)
results["gru"]["time"] = time.time() - t0
results["gru"]["params"] = n_gru
print(f"  相对RMSE: {results['gru']['rel_mean']:.4f}  "
      f"(k=1: {results['gru']['rel_k1']:.4f}, k=H: {results['gru']['rel_kH']:.4f})")

# ---------- 3. RSSM ----------
print("\n" + "=" * 70)
print("[3/3] RSSM 世界模型")
print("=" * 70)
rcfg = RSSMConfig(stoch=32, classes=32, deter=256, hidden=256)
rssm = WorldModel(C_O, C_A, rcfg).to(DEV)
n_rssm = sum(p.numel() for p in rssm.parameters())
print(f"  参数量: {n_rssm:,}  (GRU 的 {n_rssm/n_gru:.2f}x)")

opt = torch.optim.Adam(rssm.parameters(), lr=4e-4)


def rssm_loss(b):
    obs = b["obs"].to(DEV)
    act = b["act"].to(DEV)
    fut = b["fut"].to(DEV)
    af = b["act_fut"].to(DEV)

    post, prior, recon, _ = rssm(obs, act)
    # 重建
    l_rec = ((recon - obs) ** 2).mean()
    # KL
    kl, kl_raw, dyn, rep = rssm.dynamics.kl_loss(post, prior)
    l_kl = kl.mean()
    # 多步
    state = {k: v[:, -1] for k, v in post.items()}
    trail = rssm.dynamics.imagine_with_action(af, state)
    feat = rssm.dynamics.get_feat(trail)
    pf = rssm.decoder(feat).mode()
    l_multi = ((pf - fut) ** 2).mean()
    return l_rec + l_kl + l_multi, {
        "rec": l_rec.item(), "kl": l_kl.item(), "multi": l_multi.item()}


t0 = time.time()
best, bep, ni = float("inf"), -1, 0
for ep in range(1, 81):
    rssm.train()
    agg = {}
    for b in tl:
        opt.zero_grad(set_to_none=True)
        loss, parts = rssm_loss(b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rssm.parameters(), 100.0)
        opt.step()
        for k, v in parts.items():
            agg.setdefault(k, []).append(v)
    rssm.eval()
    vs, vn = 0.0, 0
    with torch.no_grad():
        for b in vl:
            l, _ = rssm_loss(b)
            vs += float(l) * len(b["obs"])
            vn += len(b["obs"])
    v = vs / max(vn, 1)
    if v < best - 1e-7:
        best, bep, ni = v, ep, 0
        torch.save(rssm.state_dict(), OUT / "rssm.pt")
    else:
        ni += 1
    if ep % 10 == 0 or ep == 1:
        tr = {k: np.mean(v_) for k, v_ in agg.items()}
        print(f"    ep {ep:>2} val={v:.6f} best={best:.6f}@{bep}  "
              f"rec={tr['rec']:.5f} kl={tr['kl']:.5f} multi={tr['multi']:.5f} "
              f"({time.time()-t0:.0f}s)")
    if ni >= 20:
        print(f"    早停 ep {ep}")
        break
rssm.load_state_dict(torch.load(OUT / "rssm.pt"))
rssm.eval()


@torch.no_grad()
def rssm_pred(b):
    o = b["obs"].to(DEV)
    a = b["act"].to(DEV)
    af = b["act_fut"].to(DEV)
    return rssm.imagine(o, a, af)


results["rssm"] = evaluate_rollout(rssm_pred, tel)
results["rssm"]["time"] = time.time() - t0
results["rssm"]["params"] = n_rssm
print(f"  相对RMSE: {results['rssm']['rel_mean']:.4f}  "
      f"(k=1: {results['rssm']['rel_k1']:.4f}, k=H: {results['rssm']['rel_kH']:.4f})")

# ---------- 汇总 ----------
print("\n" + "=" * 78)
print("最终结果（test 集）")
print("=" * 78)
print(f"{'模型':<10}{'参数':>12}{'相对RMSE':>12}{'k=1':>9}{'k=H':>9}{'耗时s':>9}")
print("-" * 62)
for k in ["arx", "gru", "rssm"]:
    r = results[k]
    print(f"{k:<10}{r['params']:>12,}{r['rel_mean']:>12.4f}"
          f"{r['rel_k1']:>9.4f}{r['rel_kH']:>9.4f}{r['time']:>9.0f}")

# 保存
(OUT / "results.json").write_text(
    json.dumps({"config": {"window": WINDOW, "horizon": HORIZON,
                           "step": STEP, "dt": 0.015 * STEP,
                           "obs_cols": OBS, "act_cols": ACT},
                "results": results,
                "o_sd": o_sd.tolist()},
               ensure_ascii=False, indent=1), encoding="utf-8")

# 预测存盘（画图用）
preds = {}
for name, fn in [("arx", arx_pred), ("gru",
                  lambda b: gru(b["obs"].to(DEV), b["act"].to(DEV), b["act_fut"].to(DEV))),
                 ("rssm", rssm_pred)]:
    ps, ts = [], []
    with torch.no_grad():
        for b in tel:
            ps.append(fn(b).cpu().numpy())
            ts.append(b["fut"].numpy())
    preds[name] = np.concatenate(ps, 0)
np.savez_compressed(OUT / "preds.npz",
                    target=np.concatenate(ts, 0),
                    **preds, o_sd=o_sd, o_mu=o_mu,
                    obs_cols=np.array(OBS))

print(f"\n结果 -> {OUT / 'results.json'}")

