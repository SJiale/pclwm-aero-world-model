"""PC-RSSM 多种子配对实验

关键设计（按用户 2026-11-05 要求）：
  1. seeds = {0,1,2,3,4}，seed=0 复用已有结果
  2. 每 seed 下四组配置共享【相同的 RSSM 主干初始权重】
     → 控制无关差异，配对比较
  3. 固定推理随机数（每个 seed 用相同推理 seed 序列）
  4. 模型选择只用 val
  5. 全程记录 KL / 梯度 / probe

配对设计：
  for seed in seeds:
      torch.manual_seed(seed)
      init_state = deepcopy(model.rssm.state_dict())  # 主干初始权重
      for cfg in [E0,E1,E2,E3]:
          model = new PCWorldModel(...)
          model.rssm.load_state_dict(init_state)      # ★ 共享主干初始化
          train(model, cfg, seed)
"""
import copy
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from evaluate import UnifiedEval, WINDOW, HORIZON, DT, LONG_TARGETS
from pc_data import PCData, PHYS_NAMES
from pc_rssm import PCWorldModel

OUT = Path(r"C:\pclwm\pclwm_pcrssm")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

SEEDS = [0, 1, 2, 3, 4]
MAX_EP = 60
PATIENCE = 15
BS = 64
LR = 4e-4
LAM_S = 0.01
LAM_D = 0.003
TRAIN_MODE = "gumbel"
EVAL_SEED_BASE = 777000        # ★ 固定推理 seed 序列（与训练种子解耦）
N_EVAL_REP = 3

pcd = PCData()
data = pcd.d
ev = UnifiedEval(data)
C_O = data.n_obs
C_A = data.n_act

tr_W = pcd.windows("train", stride=4)
va_W = pcd.windows("val", stride=8)
te_W = pcd.windows("test_id", stride=8)
ood_W = pcd.windows("test_ood", stride=8)

print("=" * 100)
print("PC-RSSM 多种子配对实验")
print("=" * 100)
print(f"  设备={DEV}")
print(f"  种子: {SEEDS}")
print(f"  λ_s={LAM_S}  λ_d={LAM_D}")
print(f"  窗口: train={len(tr_W)} val={len(va_W)} te={len(te_W)} ood={len(ood_W)}")
print(f"  训练: {MAX_EP} epoch, patience={PATIENCE}, bs={BS}, lr={LR}")
print(f"  推理: {N_EVAL_REP} 次重复, seed_base={EVAL_SEED_BASE}")


def batch_iter(ws, bs, shuffle=True, seed=None):
    n = len(ws)
    if seed is not None:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n) if shuffle else np.arange(n)
    else:
        idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i+bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c]))
               .float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut",
                         "phys_now", "phys_fut"]}


# ══════════════════════════════════════════════════
# 评价（不读物理标签）
# ══════════════════════════════════════════════════
@torch.no_grad()
def predict_short(model, ws, seed, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                        device=DEV)
        torch.manual_seed(seed + i)
        out.append(model.imagine(b["obs"], mm, b["act"], b["act_fut"],
                                 mode=TRAIN_MODE).cpu().numpy())
    return np.concatenate(out, 0)


def eval_short(model, ws):
    vals, chs, per_traj = [], [], {}
    rids = np.array([w["rid"] for w in ws])
    for r in range(N_EVAL_REP):
        p = predict_short(model, ws, seed=EVAL_SEED_BASE + r)
        e = ev.evaluate(p, ws)
        vals.append(e["overall_rel"])
        chs.append(e["per_channel"])
        for rid in np.unique(rids):
            per_traj.setdefault(rid, []).append(
                e["per_traj"].get(rid, {}).get("rel", np.nan))
    ch = {c: {"mean": float(np.mean([x[c]["rmse_rel"] for x in chs])),
              "std": float(np.std([x[c]["rmse_rel"] for x in chs]))}
          for c in data.obs_cols}
    pt = {k: {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v))}
          for k, v in per_traj.items()}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "per_channel": ch, "per_traj": pt, "n_rep": N_EVAL_REP}


def rollout_continuous(model, rid, tgt_s, seed):
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
        obs_h = torch.from_numpy(o[start-WINDOW:start][None]).float().to(DEV)
        act_h = torch.from_numpy(
            a[start-WINDOW-1:start-1][None]).float().to(DEV)
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
                a[start-1+k:start+k][None]).float().to(DEV).reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev, a_t], -1), st["h"])
            pl = rssm.prior(torch.cat([h, z_prev], -1)).reshape(
                1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, TRAIN_MODE)
            st = {"h": h, "z": z}
            pred = rssm.dec(rssm.feat(st))
            gt = torch.from_numpy(o[start+k:start+k+1][None]).float().to(DEV)
            errs.append((pred - gt[:, 0]).cpu().numpy()[0])
    E = np.stack(errs, 0).astype(np.float64)
    E = E * np.asarray(data.obs_sd).ravel()[None, :]
    ps = np.sqrt((E**2).mean(axis=1))
    ref = float(np.asarray(data.obs_sd).mean())
    return {"final_rel": float(ps[-1])/ref, "mean_rel": float(ps.mean())/ref,
            "curve": (ps/ref).tolist(), "n_steps": n_steps}


def eval_continuous(model, split, tgt_s):
    finals, means, curves = [], [], []
    for r in range(N_EVAL_REP):
        for rid in data.splits[split]:
            x = rollout_continuous(model, rid, tgt_s,
                                   seed=EVAL_SEED_BASE + r*1000 +
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
            "curve": (np.mean(curves, 0).tolist() if curves else [])}


# ══════════════════════════════════════════════════
# 物理 Probe（独立岭回归）
# ══════════════════════════════════════════════════
def probe_physics(model):
    rssm = model.rssm

    def extract(ws):
        H, P = [], []
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
                P.append(w["phys_now"][-1])
        return np.array(H), np.array(P)

    Htr, Ptr = extract(tr_W)
    Hva, Pva = extract(va_W)
    Hte, Pte = extract(te_W)

    res = {}
    for i, nm in enumerate(PHYS_NAMES):
        best = None
        for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
            Xm, Xs = Htr.mean(0), Htr.std(0) + 1e-8
            Xn = (Htr - Xm) / Xs
            Ym, Ys = Ptr[:, i].mean(), Ptr[:, i].std() + 1e-8
            Yn = (Ptr[:, i] - Ym) / Ys
            W = np.linalg.solve(Xn.T@Xn + lam*np.eye(Xn.shape[1]),
                                Xn.T@Yn)
            pv = ((Hva - Xm) / Xs) @ W * Ys + Ym
            r2v = 1 - ((pv - Pva[:, i])**2).sum() / \
                ((Pva[:, i]-Pva[:, i].mean())**2).sum()
            if best is None or r2v > best[0]:
                best = (r2v, lam, W, Xm, Xs, Ym, Ys)
        r2v, lam, W, Xm, Xs, Ym, Ys = best
        pt = ((Hte - Xm) / Xs) @ W * Ys + Ym
        r2t = 1 - ((pt - Pte[:, i])**2).sum() / \
            ((Pte[:, i]-Pte[:, i].mean())**2).sum()
        rmse_phys = float(np.sqrt(((pt - Pte[:, i])**2).mean())*pcd.phys_sd[i])
        res[nm] = {"r2_test": float(r2t), "r2_val": float(r2v),
                   "rmse_phys": rmse_phys, "lam": lam}
    return res


# ══════════════════════════════════════════════════
# KL 诊断（每模型）
# ══════════════════════════════════════════════════
def kl_diagnose(model, lam_s, lam_d):
    rssm = model.rssm
    b = {k: torch.from_numpy(np.stack([w[k] for w in va_W[:64]]))
         .float().to(DEV) for k in ["obs", "act", "fut", "act_fut"]}
    mm = torch.ones(64, b["obs"].shape[1], C_O, device=DEV)
    model.train()
    out = model(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
                obs_true=b["obs"], phys_now=None, phys_future=None,
                mode=TRAIN_MODE)
    # 仅测 KL（无物理监督的 forward）
    kl_raw = float(out["kl_raw"].detach())
    kl_clip = float(out["kl"].detach())
    kl_frac = float(out["kl_frac"].detach())

    with torch.no_grad():
        post, prior, _, _ = rssm.observe_full(b["obs"], mm, b["act"],
                                              TRAIN_MODE)
        kd, kr = rssm.kl_terms(post, prior)
        kl_dim = (kd + kr).mean(0).mean(0).cpu().numpy()
    above = int((kl_dim > rssm.free_bits).sum())

    model.eval()
    return {"kl_raw": kl_raw, "kl_clip": kl_clip, "kl_frac": kl_frac,
            "kl_dim_above": above, "kl_dim_total": len(kl_dim),
            "kl_dim_median": float(np.median(kl_dim)),
            "free_bits": float(rssm.free_bits)}


# ══════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════
def train_one(name, lam_s, lam_d, seed, init_rssm_state):
    torch.manual_seed(seed * 10007 + 1)
    np.random.seed(seed * 10007 + 1)

    model = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                         lam_static=lam_s, lam_dynamic=lam_d).to(DEV)
    # ★ 共享主干初始权重（四组配置配对）
    if init_rssm_state is not None:
        model.rssm.load_state_dict(init_rssm_state)

    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP)

    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()
    hist = {"train": [], "val": [], "rec": [], "kl_raw": [], "kl_clip": [],
            "multi": [], "l_static": [], "l_dynamic": []}
    nan_flag = False

    for ep in range(1, MAX_EP+1):
        model.train()
        acc = {k: 0.0 for k in hist}
        n = 0
        for b in batch_iter(tr_W, BS, True, seed=seed*31+ep):
            opt.zero_grad(set_to_none=True)
            mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                            device=DEV)
            out = model(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
                        obs_true=b["obs"], phys_now=b["phys_now"],
                        phys_future=b["phys_fut"], mode=TRAIN_MODE)
            loss = out["loss"]
            if not torch.isfinite(loss):
                nan_flag = True
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = len(b["obs"])
            acc["train"] += float(loss.detach())*nb
            acc["rec"] += float(out["rec"].detach())*nb
            acc["kl_raw"] += float(out["kl_raw"].detach())*nb
            acc["kl_clip"] += float(out["kl"].detach())*nb
            acc["multi"] += float(out["multi"].detach())*nb
            if lam_s > 0:
                acc["l_static"] += float(out["l_static"].detach())*nb
            if lam_d > 0:
                acc["l_dynamic"] += float(out["l_dynamic"].detach())*nb
            n += nb
        if nan_flag:
            print(f"    ⚠ NaN，跳过")
            return None
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in batch_iter(va_W, 128, False):
                mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O,
                                device=DEV)
                out = model(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
                            obs_true=b["obs"], phys_now=b["phys_now"],
                            phys_future=b["phys_fut"], mode=TRAIN_MODE)
                vs += float(out["loss"])*len(b["obs"])
                vn += len(b["obs"])
        v = vs/vn
        for k in hist:
            hist[k].append(acc[k]/n if n > 0 else float("nan"))

        if v < best - 1e-7:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(),
                       OUT/"checkpoints"/f"seed{seed}_{name}.pt")
        else:
            bad += 1
        if bad >= PATIENCE:
            break

        if ep % 20 == 0:
            print(f"    ep {ep:>3} val={v:.5f} best={best:.5f}@{bep} "
                  f"({time.time()-t0:.0f}s)", flush=True)

    model.load_state_dict(torch.load(
        OUT/"checkpoints"/f"seed{seed}_{name}.pt",
        map_location=DEV, weights_only=True))
    model.eval()
    return {"model": model, "hist": hist, "best_val": float(best),
            "best_epoch": int(bep), "train_time_s": time.time()-t0,
            "params": int(npar)}


# ══════════════════════════════════════════════════
# 主循环
# ══════════════════════════════════════════════════
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--seeds", default="0,1,2,3,4")
_p.add_argument("--resume", action="store_true")
_args, _ = _p.parse_known_args()
SEEDS = [int(x) for x in _args.seeds.split(",")]

CFGS = [("E0_baseline", 0.0, 0.0), ("E1_static", LAM_S, 0.0),
        ("E2_dynamic", 0.0, LAM_D), ("E3_both", LAM_S, LAM_D)]

RESFILE = OUT / "results" / "multiseed_results.json"
(OUT/"checkpoints").mkdir(exist_ok=True)
(OUT/"results").mkdir(exist_ok=True)

ALL = {}
if RESFILE.exists() and _args.resume:
    ALL = json.loads(RESFILE.read_text(encoding="utf-8"))
    print(f"  已加载 {len(ALL)} 个 seed 的结果")

for seed in SEEDS:
    key = f"seed{seed}"
    # ★ 单配置粒度跳过：若该 seed 的 4 个配置都已存在，跳过整个 seed
    if key in ALL and _args.resume:
        _have = set(ALL[key].get("cfgs", {}).keys())
        _need = {c[0] for c in CFGS}
        if _need <= _have:
            print(f"\n  跳过 {key}（4/4 配置已完成）")
            continue
        print(f"\n  续跑 {key}（已完成 {len(_have)}/4: {sorted(_have)}）")

    print(f"\n{'='*100}")
    print(f"SEED {seed}")
    print(f"{'='*100}")

    # ★ 生成共享主干初始权重
    torch.manual_seed(seed * 10007 + 1)
    np.random.seed(seed * 10007 + 1)
    ref = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                       lam_static=0.0, lam_dynamic=0.0)
    init_state = copy.deepcopy(ref.rssm.state_dict())
    del ref

    seed_rec = {"cfgs": {}}
    for name, ls, ld in CFGS:
        print(f"\n  [{name}] λ_s={ls} λ_d={ld} seed={seed}")
        r = train_one(name, ls, ld, seed, init_state)
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
        probe = probe_physics(model)
        kld = kl_diagnose(model, ls, ld)

        seed_rec["cfgs"][name] = {
            "lam_s": ls, "lam_d": ld,
            "best_val": r["best_val"], "best_epoch": r["best_epoch"],
            "train_time_s": r["train_time_s"], "params": r["params"],
            "test_id": r_te, "test_ood": r_ood,
            "continuous": cont, "probe": probe, "kl": kld,
            "hist": {k: [float(x) for x in v]
                     for k, v in r["hist"].items()},
        }
        print(f"    test_id={r_te['mean']:.4f} ood={r_ood['mean']:.4f}")
        for tgt in ["10s", "30s", "60s"]:
            k = f"test_id_{tgt}"
            if k in cont:
                print(f"    {tgt}: mean={cont[k]['mean_rel']:.4f} "
                      f"final={cont[k]['final_mean']:.4f}")
        for pn, v in probe.items():
            print(f"    probe {pn}: R²={v['r2_test']:.4f}")

        # ★ 每配置完成后立即落盘（防止中途停止丢失）
        ALL[key] = seed_rec
        RESFILE.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                           encoding="utf-8")
        print(f"    -> 已增量保存 {key}/{name}")

    ALL[key] = seed_rec
    RESFILE.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  -> 已保存 {key}（完整）")

print(f"\n{'='*100}")
print("完成")
print(f"{'='*100}")
print(f"  结果: {RESFILE}")


