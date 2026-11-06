"""PSVA 最小验证实验训练脚本

三个配置：
  E3      : 复用现有检查点（不重新训练）
  P_ctrl  : unsplit=True，全 h + 相同监督（Ndot/N/ΔN）
  PSVA    : h 切分 64/64，h_f 锚定 Ndot，h_s 锚定 N，h_f 对锚定 ΔN

统一协议（与 PC-RSSM E0-E3 完全一致）：
  60 epoch, patience=15, bs=64, lr=4e-4
  λ_fast=0.01, λ_slow=0.01, λ_dyn=0.003
  k={1,4,8,16}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from evaluate import UnifiedEval, WINDOW, HORIZON, DT, LONG_TARGETS
from psva_data import PSVAData
from psva import PSVA_WorldModel

OUT = Path(r"C:\pclwm\pclwm_psva")
CKPT = OUT / "checkpoints"
RES = OUT / "results"
CKPT.mkdir(exist_ok=True, parents=True)
RES.mkdir(exist_ok=True, parents=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EP = 60
PATIENCE = 15
BS = 64
LR = 4e-4
LAM_F = 0.01
LAM_S = 0.01
LAM_D = 0.003
TRAIN_MODE = "gumbel"
EVAL_SEED_BASE = 777000
N_REP = 3

pcd = PSVAData()
data = pcd.d
ev = UnifiedEval(data)
C_O = data.n_obs
C_A = data.n_act

tr_W = pcd.windows("train", stride=4)
va_W = pcd.windows("val", stride=8)
te_W = pcd.windows("test_id", stride=8)
ood_W = pcd.windows("test_ood", stride=8)

print("=" * 96)
print("PSVA 最小验证实验")
print("=" * 96)
print(f"  设备={DEV}")
print(f"  WINDOW={WINDOW}  HORIZON={HORIZON}")
print(f"  λ_fast={LAM_F}  λ_slow={LAM_S}  λ_dyn={LAM_D}")
print(f"  窗口: train={len(tr_W)} val={len(va_W)} te={len(te_W)} ood={len(ood_W)}")
print(f"  训练: {MAX_EP} ep, patience={PATIENCE}, bs={BS}, lr={LR}")


def batch_iter(ws, bs, shuffle=True, seed=None):
    n = len(ws)
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n) if shuffle else np.arange(n)
    else:
        idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i + bs]]
        # y_delta 是 dict，单独处理
        out = {k: torch.from_numpy(np.stack([w[k] for w in c])).float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut", "y_fast", "y_slow"]}
        out["y_delta"] = {
            k: torch.from_numpy(np.stack([w["y_delta"][k] for w in c]))
            .float().to(DEV)
            for k in [1, 4, 8, 16]
        }
        yield out


def train_one(name, unsplit, seed):
    torch.manual_seed(seed * 10007 + 1)
    np.random.seed(seed * 10007 + 1)

    model = PSVA_WorldModel(
        C_O, C_A, d_fast=64, d_slow=64, unsplit=unsplit,
        lam_fast=LAM_F, lam_slow=LAM_S, lam_dyn=LAM_D,
    ).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*96}")
    print(f"[{name}] unsplit={unsplit} 参数={npar:,}")
    print(f"{'='*96}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP)

    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()
    hist = {"train": [], "val": [], "l_fast": [], "l_slow": [], "l_dyn": []}

    for ep in range(1, MAX_EP + 1):
        model.train()
        acc = {k: 0.0 for k in hist}
        n = 0
        nan_hit = False
        for b in batch_iter(tr_W, BS, True, seed=seed * 31 + ep):
            opt.zero_grad(set_to_none=True)
            mask = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                              device=DEV)
            out = model(b["obs"], mask, b["act"], b["act_fut"], b["fut"],
                        obs_true=b["obs"], y_fast=b["y_fast"],
                        y_slow=b["y_slow"], y_delta=b["y_delta"],
                        mode=TRAIN_MODE)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nan_hit = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            nb = len(b["obs"])
            acc["train"] += float(loss.detach()) * nb
            acc["l_fast"] += float(out["l_fast"].detach()) * nb
            acc["l_slow"] += float(out["l_slow"].detach()) * nb
            acc["l_dyn"] += float(out["l_dyn"].detach()) * nb
            n += nb
        if nan_hit or n == 0:
            print("    ⚠ NaN，中止")
            return None
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in batch_iter(va_W, 128, False):
                mask = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                                  device=DEV)
                out = model(b["obs"], mask, b["act"], b["act_fut"], b["fut"],
                            obs_true=b["obs"], y_fast=b["y_fast"],
                            y_slow=b["y_slow"], y_delta=b["y_delta"],
                            mode=TRAIN_MODE)
                vs += float(out["loss"]) * len(b["obs"])
                vn += len(b["obs"])
        v = vs / vn
        for k in hist:
            hist[k].append(acc[k] / n if n > 0 else float("nan"))
        hist["val"].append(v)

        improved = v < best - 1e-7
        if improved:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), CKPT / f"{name}_seed{seed}.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:>3} train={acc['train']/n:.5f} val={v:.5f} "
                  f"Lf={acc['l_fast']/n:.4f} Ls={acc['l_slow']/n:.4f} "
                  f"Ld={acc['l_dyn']/n:.6f} "
                  f"{'*' if improved else ' '} ({time.time()-t0:.0f}s)",
                  flush=True)
        if bad >= PATIENCE:
            print(f"  早停 ep {ep}")
            break

    model.load_state_dict(torch.load(CKPT / f"{name}_seed{seed}.pt",
                                     map_location=DEV, weights_only=True))
    model.eval()
    return {"model": model, "hist": hist, "best_val": float(best),
            "best_epoch": int(bep), "train_time_s": time.time() - t0,
            "params": npar}


# ══════════════════════════════════════════════════
# 评价
# ══════════════════════════════════════════════════
@torch.no_grad()
def predict_short(model, ws, seed, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i + bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        mask = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                          device=DEV)
        torch.manual_seed(seed + i)
        out.append(model.imagine(b["obs"], mask, b["act"], b["act_fut"],
                                 mode=TRAIN_MODE).cpu().numpy())
    return np.concatenate(out, 0)


def eval_short(model, ws):
    vals, per_traj = [], {}
    rids = np.array([w["rid"] for w in ws])
    for r in range(N_REP):
        p = predict_short(model, ws, seed=EVAL_SEED_BASE + r)
        e = ev.evaluate(p, ws)
        vals.append(e["overall_rel"])
        for rid in np.unique(rids):
            per_traj.setdefault(rid, []).append(
                e["per_traj"].get(rid, {}).get("rel", np.nan))
    return {
        "mean": float(np.mean(vals)), "std": float(np.std(vals)),
        "per_traj": {k: {"mean": float(np.nanmean(v)),
                         "std": float(np.nanstd(v))}
                     for k, v in per_traj.items()},
        "n_rep": N_REP,
    }


def rollout_continuous(model, rid, tgt_s, seed):
    """连续自主推演（返回完整曲线供画图）"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n_steps = int(round(tgt_s / DT))
    start = WINDOW + 1
    if len(o) <= start + n_steps + 2:
        n_steps = len(o) - start - 2
        if n_steps < 10:
            return None
    rssm = model.rssm
    torch.manual_seed(seed)
    with torch.no_grad():
        obs_h = torch.from_numpy(o[start - WINDOW:start][None]).float().to(DEV)
        act_h = torch.from_numpy(
            a[start - WINDOW - 1:start - 1][None]).float().to(DEV)
        st = rssm.init_state(1, DEV)
        for t in range(WINDOW):
            z_prev = st["z"].reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev, act_h[:, t]], -1), st["h"])
            emb = rssm.enc(torch.cat([
                obs_h[:, t], torch.ones(1, C_O, device=DEV)], -1))
            pl = rssm.post(torch.cat([torch.cat([h, z_prev], -1), emb],
                                     -1)).reshape(1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, TRAIN_MODE)
            st = {"h": h, "z": z}
        errs = []
        for k in range(n_steps):
            z_prev = st["z"].reshape(1, -1)
            a_t = torch.from_numpy(
                a[start - 1 + k:start + k][None]).float().to(DEV).reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev, a_t], -1), st["h"])
            pl = rssm.prior(torch.cat([h, z_prev], -1)).reshape(
                1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, TRAIN_MODE)
            st = {"h": h, "z": z}
            pred = rssm.dec(rssm.feat(st))
            gt = torch.from_numpy(o[start + k:start + k + 1][None]).float().to(DEV)
            errs.append((pred - gt[:, 0]).cpu().numpy()[0])
    E = np.stack(errs, 0).astype(np.float64)
    E = E * np.asarray(data.obs_sd).ravel()[None, :]
    ps = np.sqrt((E ** 2).mean(axis=1))
    ref = float(np.asarray(data.obs_sd).mean())
    return {"final_rel": float(ps[-1]) / ref,
            "mean_rel": float(ps.mean()) / ref,
            "curve": (ps / ref).tolist()}


def eval_continuous(model, split, tgt_s):
    finals, means, curves = [], [], []
    for r in range(N_REP):
        for rid in data.splits[split]:
            x = rollout_continuous(model, rid, tgt_s,
                                   seed=EVAL_SEED_BASE + r * 1000 +
                                   hash(rid) % 1000)
            if x:
                finals.append(x["final_rel"])
                means.append(x["mean_rel"])
                curves.append(x["curve"])
    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)),
            "mean_rel_std": float(np.std(means)),
            "n": len(finals),
            "curves": curves}


# ══════════════════════════════════════════════════
# 交叉 Probe（快慢分离度）
# ══════════════════════════════════════════════════
def ridge_r2(X_tr, Y_tr, X_te, Y_te, lams=(0.01, 0.1, 1.0, 10.0, 100.0)):
    Xm, Xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xn = (X_tr - Xm) / Xs
    Ym, Ys = Y_tr.mean(), Y_tr.std() + 1e-9
    Yn = (Y_tr - Ym) / Ys
    D = Xn.shape[1]
    best = -1e9
    for lam in lams:
        W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(D), Xn.T @ Yn)
        pt = ((X_te - Xm) / Xs) @ W * Ys + Ym
        r2 = 1 - ((pt - Y_te) ** 2).sum() / ((Y_te - Y_te.mean()) ** 2).sum()
        best = max(best, r2)
    return float(best)


def extract_h_all(model, data_obj, ws):
    """提取每个窗口最后时刻的完整 h"""
    rssm = model.rssm
    H = []
    with torch.no_grad():
        for w in ws:
            o = torch.from_numpy(w["obs"][None]).float().to(DEV)
            a = torch.from_numpy(w["act"][None]).float().to(DEV)
            st = rssm.init_state(1, DEV)
            for t in range(WINDOW):
                z_prev = st["z"].reshape(1, -1)
                h = rssm.cell(torch.cat([z_prev, a[:, t]], -1), st["h"])
                emb = rssm.enc(torch.cat([
                    o[:, t], torch.ones(1, C_O, device=DEV)], -1))
                pl = rssm.post(torch.cat([
                    torch.cat([h, z_prev], -1), emb], -1)
                ).reshape(1, rssm.stoch, rssm.classes)
                z = rssm.sample(pl, "argmax")
                st = {"h": h, "z": z}
            H.append(st["h"][0].cpu().numpy())
    return np.array(H)


def cross_probe(model, split_test=("test_id", "test_ood")):
    """交叉 Probe：h_f / h_s 对 Ndot / N 的可解码性"""
    rssm = model.rssm
    Htr = extract_h_all(model, data, tr_W)
    Hte = extract_h_all(model, data, te_W + ood_W)

    # 目标（每个元素 shape=(2,)，用 stack 得到 [N,2]）
    Ytr_dot = np.stack([w["y_fast"][-1] for w in tr_W])
    Yte_dot = np.stack([w["y_fast"][-1] for w in (te_W + ood_W)])
    Ytr_n = np.stack([w["y_slow"][-1] for w in tr_W])
    Yte_n = np.stack([w["y_slow"][-1] for w in (te_W + ood_W)])

    df, ds = rssm.d_fast, rssm.d_slow
    Htr_f, Htr_s = Htr[:, :df], Htr[:, df:df + ds]
    Hte_f, Hte_s = Hte[:, :df], Hte[:, df:df + ds]

    res = {}
    for tgt_name, Ytr, Yte in [("Ndot", Ytr_dot, Yte_dot),
                                ("N", Ytr_n, Yte_n)]:
        for ch, ch_name in enumerate(["N2", "N3"]):
            r_ff = ridge_r2(Htr_f, Ytr[:, ch], Hte_f, Yte[:, ch])
            r_ss = ridge_r2(Htr_s, Ytr[:, ch], Hte_s, Yte[:, ch])
            r_all = ridge_r2(Htr, Ytr[:, ch], Hte, Yte[:, ch])
            res[f"{tgt_name}_{ch_name}"] = {
                "h_f": r_ff, "h_s": r_ss, "h_all": r_all,
            }

    sep_fast = np.mean([res[f"Ndot_{c}"]["h_f"] - res[f"Ndot_{c}"]["h_s"]
                        for c in ["N2", "N3"]])
    sep_slow = np.mean([res[f"N_{c}"]["h_s"] - res[f"N_{c}"]["h_f"]
                        for c in ["N2", "N3"]])
    res["Separation"] = {
        "sep_fast": float(sep_fast),
        "sep_slow": float(sep_slow),
        "sep_total": float((sep_fast + sep_slow) / 2),
    }
    return res


# ══════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════
import argparse

_p = argparse.ArgumentParser()
_p.add_argument("--configs", default="P_ctrl,PSVA")
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--epochs", type=int, default=None)
_args, _ = _p.parse_known_args()

CONFIGS = [("P_ctrl", True), ("PSVA", False)]
to_run = [c for c in CONFIGS if c[0] in _args.configs.split(",")]

RESFILE = RES / "psva_results.json"
ALL = json.loads(RESFILE.read_text(encoding="utf-8")) if RESFILE.exists() else {}

for name, unsplit in to_run:
    if _args.epochs:
        MAX_EP = _args.epochs
    r = train_one(name, unsplit, _args.seed)
    if r is None:
        continue
    model = r["model"]

    print(f"    best_val={r['best_val']:.5f}@{r['best_epoch']} "
          f"({r['train_time_s']:.0f}s)")

    r_te = eval_short(model, te_W)
    r_ood = eval_short(model, ood_W)
    cont = {}
    for split in ["test_id", "test_ood"]:
        for tgt in LONG_TARGETS:
            x = eval_continuous(model, split, tgt)
            if x:
                cont[f"{split}_{tgt}s"] = x
    probe = cross_probe(model)

    ALL[f"{name}_seed{_args.seed}"] = {
        "name": name, "unsplit": unsplit, "seed": _args.seed,
        "params": r["params"], "best_val": r["best_val"],
        "best_epoch": r["best_epoch"], "train_time_s": r["train_time_s"],
        "test_id": r_te, "test_ood": r_ood,
        "continuous": cont, "probe": probe,
        "hist": {k: [float(x) for x in v] for k, v in r["hist"].items()},
    }
    RESFILE.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    print(f"    test_id={r_te['mean']:.4f}  ood={r_ood['mean']:.4f}")
    for tgt in ["10s", "30s", "60s"]:
        k = f"test_id_{tgt}"
        if k in cont:
            print(f"    {tgt}: mean={cont[k]['mean_rel']:.4f}")
    print(f"    Separation: {probe['Separation']['sep_total']:+.4f} "
          f"(fast={probe['Separation']['sep_fast']:+.4f}, "
          f"slow={probe['Separation']['sep_slow']:+.4f})")
    print(f"    -> 已保存 {name}_seed{_args.seed}")

print(f"\n{'='*96}")
print("完成")
print(f"{'='*96}")
print(f"  结果: {RESFILE}")

