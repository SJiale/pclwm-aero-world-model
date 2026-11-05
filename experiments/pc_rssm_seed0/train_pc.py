"""PC-RSSM 正式消融：E0 / E1 / E2 / E3

配置（用户 2026-11-05 确认方案 A）：
  λ_static  = 0.01
  λ_dynamic = 0.003
  动态跨度 k = {1, 4, 8, 16} → 0.105, 0.42, 0.84, 1.68 s

四组：
  E0: 无监督
  E1: 仅静态
  E2: 仅动态
  E3: 静态 + 动态

严格约束：
  - 物理标签只在训练损失中使用
  - 推理/评价不读物理标签
  - 因果性：动态监督用先验递推，不看未来观测
  - 逐轨迹报告（test_id 仅 2 条）
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from evaluate import UnifiedEval, WINDOW, HORIZON, DT, LONG_TARGETS
from pc_data import PCData, PHYS_NAMES
from pc_rssm import PCWorldModel

OUT = Path(r"C:\pclwm\pclwm_pcrssm")
CKPT = OUT / "checkpoints"; CKPT.mkdir(exist_ok=True)
RES = OUT / "results"; RES.mkdir(exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 0
MODE = "gumbel"
MAX_EP = 60
PATIENCE = 15
BS = 64
LR = 4e-4
N_REP = 3

LAM_S = 0.01
LAM_D = 0.003

pcd = PCData()
data = pcd.d
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")

tr_W = pcd.windows("train", stride=4)
va_W = pcd.windows("val", stride=8)
te_W = pcd.windows("test_id", stride=8)
ood_W = pcd.windows("test_ood", stride=8)

print("=" * 94)
print("PC-RSSM 正式消融 E0/E1/E2/E3")
print("=" * 94)
print(f"  设备={DEV}  WINDOW={WINDOW}  HORIZON={HORIZON}")
print(f"  λ_static={LAM_S}  λ_dynamic={LAM_D}")
print(f"  物理目标: {PHYS_NAMES}")
print(f"  窗口: train={len(tr_W)} val={len(va_W)} te={len(te_W)} ood={len(ood_W)}")
print(f"  训练: {MAX_EP} ep, patience={PATIENCE}, bs={BS}, lr={LR}")


def batch_iter(ws, bs, shuffle=True):
    n = len(ws)
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, bs):
        c = [ws[j] for j in idx[i:i+bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c]))
               .float().to(DEV)
               for k in ["obs", "act", "fut", "act_fut",
                         "phys_now", "phys_fut"]}


def batchify(ws):
    """单批全部（用于评价）"""
    return {k: torch.from_numpy(np.stack([w[k] for w in ws]))
            .float().to(DEV)
            for k in ["obs", "act", "fut", "act_fut"]}


# ══════════════════════════════════════════════════
# 训练
# ══════════════════════════════════════════════════
def train(name, lam_s, lam_d, max_ep=None):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                         lam_static=lam_s, lam_dynamic=lam_d).to(DEV)
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*94}")
    print(f"[{name}] λ_s={lam_s} λ_d={lam_d} 参数={npar:,}")
    print(f"{'='*94}")

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    if max_ep is None:
        max_ep = MAX_EP
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    hist = {"train": [], "val": [], "rec": [], "kl": [], "kl_raw": [],
            "kl_frac": [], "multi": [], "l_static": [], "l_dynamic": []}
    gnorm_hist = {"rec": [], "multi": [], "l_static": [], "l_dynamic": [],
                  "kl": []}
    best, bep, bad = float("inf"), -1, 0
    t0 = time.time()

    for ep in range(1, max_ep+1):
        model.train()
        acc = {k: 0.0 for k in hist}
        gacc = {k: 0.0 for k in gnorm_hist}
        n, ng = 0, 0
        for b in batch_iter(tr_W, BS, True):
            opt.zero_grad(set_to_none=True)
            mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O, device=DEV)
            out = model(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
                        obs_true=b["obs"], phys_now=b["phys_now"],
                        phys_future=b["phys_fut"], mode=MODE)
            loss = out["loss"]
            loss.backward()

            # ★ 记录各分量的梯度范数（ep 1, 10, 20... 时）
            if ep in (1, 10, 20, 40, 60) and ng < 4:
                # 分离各分量的梯度（需 retain_graph）
                for comp in ["rec", "multi", "l_static", "l_dynamic"]:
                    term = out[comp]
                    if comp == "l_static":
                        term = lam_s * term
                    elif comp == "l_dynamic":
                        term = lam_d * term
                    opt.zero_grad(set_to_none=True)
                    if float(term.detach()) != 0.0 and term.requires_grad:
                        term.backward(retain_graph=True)
                        g = model.rssm.cell.weight_hh.grad
                        if g is not None:
                            gacc[comp] += float(g.norm())
                # KL 的梯度（裁剪后）
                opt.zero_grad(set_to_none=True)
                kl_term = 0.1 * out["kl"]
                kl_term.backward(retain_graph=True)
                g = model.rssm.cell.weight_hh.grad
                gacc["kl"] += float(g.norm()) if g is not None else 0.0
                # 恢复完整梯度
                opt.zero_grad(set_to_none=True)
                loss.backward()
                ng += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.)
            opt.step()
            nb = len(b["obs"])
            for k in ["loss", "rec", "kl", "kl_raw", "kl_frac", "multi",
                      "l_static", "l_dynamic"]:
                v = out[k] if k != "loss" else loss
                acc[k if k in hist else "train"] += float(v.detach())*nb
            n += nb
        sched.step()

        model.eval()
        vs, vn = 0.0, 0
        with torch.no_grad():
            for b in batch_iter(va_W, 128, False):
                mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O, device=DEV)
                out = model(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
                            obs_true=b["obs"], phys_now=b["phys_now"],
                            phys_future=b["phys_fut"], mode=MODE)
                vs += float(out["loss"])*len(b["obs"])
                vn += len(b["obs"])
        v = vs/vn

        hist["train"].append(acc["train"]/n)
        for k in ["rec", "kl", "kl_raw", "kl_frac", "multi",
                  "l_static", "l_dynamic"]:
            hist[k].append(acc[k]/n)
        hist["val"].append(v)
        if ng > 0:
            for k in gnorm_hist:
                gnorm_hist[k].append(gacc[k]/ng)

        improved = v < best - 1e-7
        if improved:
            best, bep, bad = v, ep, 0
            torch.save(model.state_dict(), CKPT/f"{name}.pt")
        else:
            bad += 1
        if ep % 10 == 0 or ep == 1:
            print(f"  ep {ep:>3} loss={acc['train']/n:.5f} val={v:.5f} "
                  f"rec={acc['rec']/n:.5f} "
                  f"kl_clip={acc['kl']/n:.4f} kl_raw={acc['kl_raw']/n:.4f} "
                  f"kl_frac={acc['kl_frac']/n:.3f} "
                  f"multi={acc['multi']/n:.5f} "
                  f"Ls={acc['l_static']/n:.4f} Ld={acc['l_dynamic']/n:.4f} "
                  f"{'*' if improved else ' '} ({time.time()-t0:.0f}s)",
                  flush=True)
        if bad >= PATIENCE:
            print(f"  早停 ep {ep}")
            break

    model.load_state_dict(torch.load(CKPT/f"{name}.pt",
                                    map_location=DEV, weights_only=True))
    model.eval()
    return model, hist, gnorm_hist, best, bep, time.time()-t0, npar


# ══════════════════════════════════════════════════
# 评价（★ 不读物理标签）
# ══════════════════════════════════════════════════
@torch.no_grad()
def predict_short(model, ws, seed=None, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        if seed is not None:
            torch.manual_seed(seed + i)
        m = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O, device=DEV)
        out.append(model.imagine(b["obs"], m, b["act"], b["act_fut"],
                                 mode=MODE).cpu().numpy())
    return np.concatenate(out, 0)


def eval_short(model, ws, n_rep=N_REP):
    vals, chs, per_traj = [], [], {}
    rids = np.array([w["rid"] for w in ws])
    for r in range(n_rep):
        p = predict_short(model, ws, seed=1000+r*777)
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
            "per_channel": ch, "per_traj": pt, "n_rep": n_rep}


def rollout_continuous(model, rid, tgt_s, seed=42):
    """★ 连续自主递推（只初始化一次，之后完全自主）"""
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
        act_h = torch.from_numpy(a[start-WINDOW-1:start-1][None]).float().to(DEV)
        st = rssm.init_state(1, DEV)
        for t in range(WINDOW):
            z_prev = st["z"].reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev, act_h[:, t]], -1), st["h"])
            emb = rssm.enc(torch.cat([
                obs_h[:, t],
                torch.ones(1, C_O, device=DEV)], -1))
            pl = rssm.post(torch.cat([torch.cat([h, z_prev], -1), emb],
                                     -1)).reshape(1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, MODE)
            st = {"h": h, "z": z}

        errs = []
        for k in range(n_steps):
            z_prev = st["z"].reshape(1, -1)
            a_t = torch.from_numpy(
                a[start-1+k:start+k][None]).float().to(DEV).reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev, a_t], -1), st["h"])
            pl = rssm.prior(torch.cat([h, z_prev], -1)).reshape(
                1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, MODE)
            st = {"h": h, "z": z}
            pred = rssm.dec(rssm.feat(st))
            gt = torch.from_numpy(o[start+k:start+k+1][None]).float().to(DEV)
            errs.append((pred - gt[:, 0]).cpu().numpy()[0])

    E = np.stack(errs, 0).astype(np.float64)
    E = E * np.asarray(data.obs_sd).ravel()[None, :]
    ps = np.sqrt((E**2).mean(axis=1))
    ref = float(np.asarray(data.obs_sd).mean())
    return {"final_rel": float(ps[-1])/ref, "mean_rel": float(ps.mean())/ref,
            "n_steps": n_steps, "curve": (ps/ref).tolist()}


def eval_continuous(model, split, tgt_s, n_rep=N_REP):
    finals, means, curves = [], [], []
    for r in range(n_rep):
        for rid in data.splits[split]:
            x = rollout_continuous(model, rid, tgt_s, seed=1000+r*777)
            if x:
                finals.append(x["final_rel"])
                means.append(x["mean_rel"])
                curves.append(x["curve"])
    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)),
            "n": len(finals),
            "curve": (np.mean(curves, 0).tolist() if curves else [])}


# ══════════════════════════════════════════════════
# 潜状态物理可解码性（★ 用训练好但冻结的模型）
# ══════════════════════════════════════════════════
def probe_physics(model, n_lag=1):
    """用 h 线性解码 N2dot/N3dot（train 拟合，test 报告）"""
    rssm = model.rssm

    def extract(ws):
        """提取每个窗口历史最后时刻的 h 与物理真值"""
        H, P = [], []
        with torch.no_grad():
            for w in ws:
                o = torch.from_numpy(w["obs"][None]).float().to(DEV)
                a = torch.from_numpy(w["act"][None]).float().to(DEV)
                B = 1
                st = rssm.init_state(B, DEV)
                for t in range(WINDOW):
                    z_prev = st["z"].reshape(B, -1)
                    h = rssm.cell(torch.cat([z_prev, a[:, t]], -1), st["h"])
                    emb = rssm.enc(torch.cat([
                        o[:, t], torch.ones(B, C_O, device=DEV)], -1))
                    pl = rssm.post(torch.cat([
                        torch.cat([h, z_prev], -1), emb], -1)
                    ).reshape(B, rssm.stoch, rssm.classes)
                    z = rssm.sample(pl, "argmax")
                    st = {"h": h, "z": z}
                H.append(st["h"][0].cpu().numpy())
                P.append(w["phys_now"][-1])
        return np.array(H), np.array(P)

    Htr, Ptr = extract(tr_W)
    Hte, Pte = extract(te_W)
    Hva, Pva = extract(va_W)

    res = {}
    for i, nm in enumerate(PHYS_NAMES):
        # 岭回归，val 选 lambda
        best = None
        for lam in [1e-2, 1e-1, 1.0, 10.0]:
            Xm, Xs = Htr.mean(0), Htr.std(0)+1e-8
            Xn = (Htr-Xm)/Xs
            Ym, Ys = Ptr[:, i].mean(), Ptr[:, i].std()+1e-8
            Yn = (Ptr[:, i]-Ym)/Ys
            W = np.linalg.solve(Xn.T@Xn + lam*np.eye(Xn.shape[1]),
                                Xn.T@Yn)
            pv = ((Hva-Xm)/Xs) @ W * Ys + Ym
            r2v = 1 - ((pv-Pva[:, i])**2).sum() / \
                ((Pva[:, i]-Pva[:, i].mean())**2).sum()
            if best is None or r2v > best[0]:
                best = (r2v, lam, W, Xm, Xs, Ym, Ys)
        r2v, lam, W, Xm, Xs, Ym, Ys = best
        pt = ((Hte-Xm)/Xs) @ W * Ys + Ym
        r2t = 1 - ((pt-Pte[:, i])**2).sum() / \
            ((Pte[:, i]-Pte[:, i].mean())**2).sum()
        # 转回物理单位
        rmse_phys = float(np.sqrt(((pt-Pte[:, i])**2).mean()) * pcd.phys_sd[i])
        res[nm] = {"r2_test": float(r2t), "r2_val": float(r2v),
                   "rmse_phys": rmse_phys, "lam": lam}
    return res


# ══════════════════════════════════════════════════
CONFIGS = [
    ("E0_baseline", 0.0, 0.0),
    ("E1_static",   LAM_S, 0.0),
    ("E2_dynamic",  0.0, LAM_D),
    ("E3_both",     LAM_S, LAM_D),
]

ALL = {}
import argparse
_p = argparse.ArgumentParser()
_p.add_argument("--runs", default="all")
_p.add_argument("--epochs", type=int, default=None)
_args, _ = _p.parse_known_args()
to_run = CONFIGS if _args.runs == "all" else [
    c for c in CONFIGS if c[0] in _args.runs.split(",")]

resfile = RES / "pc_results.json"
if resfile.exists():
    ALL = json.loads(resfile.read_text(encoding="utf-8"))

for name, ls, ld in to_run:
    model, hist, ghist, bv, bep, el, npar = train(name, ls, ld, _args.epochs)

    # 推理耗时
    b = batchify(te_W[:8])
    with torch.no_grad():
        for _ in range(3):
            mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O, device=DEV)
            model.imagine(b["obs"], mm, b["act"], b["act_fut"], mode=MODE)
        if DEV == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            mm = torch.ones(b["obs"].shape[0], b["obs"].shape[1], C_O, device=DEV)
            model.imagine(b["obs"], mm, b["act"], b["act_fut"], mode=MODE)
        if DEV == "cuda":
            torch.cuda.synchronize()
        infer_ms = (time.time()-t0)/10*1000

    r_te = eval_short(model, te_W)
    r_ood = eval_short(model, ood_W)
    cont = {}
    for split in ["test_id", "test_ood"]:
        for tgt in LONG_TARGETS:
            x = eval_continuous(model, split, tgt)
            if x:
                cont[f"{split}_{tgt}s"] = x

    probe = probe_physics(model)

    ALL[name] = {
        "params": int(npar), "lam_s": ls, "lam_d": ld,
        "best_val": float(bv), "best_epoch": int(bep),
        "train_time_s": el, "infer_ms": float(infer_ms),
        "test_id": r_te, "test_ood": r_ood,
        "continuous": cont, "probe": probe,
        "hist": {k: [float(x) for x in v] for k, v in hist.items()},
        "gnorm": {k: [float(x) for x in v] for k, v in ghist.items()},
    }
    print(f"\n  [{name}] test_id={r_te['mean']:.4f}  "
          f"ood={r_ood['mean']:.4f}")
    for k, v in cont.items():
        if "test_id" in k:
            print(f"    连续 {k}: {v['final_mean']:.4f}")
    for nm, v in probe.items():
        print(f"    probe {nm}: R²={v['r2_test']:.4f}  "
              f"RMSE={v['rmse_phys']:.2f}")
    resfile.write_text(json.dumps(ALL, ensure_ascii=False, indent=1),
                       encoding="utf-8")

print("\n" + "=" * 94)
print("汇总")
print("=" * 94)
print(f"  {'配置':<14}{'参数':>10}{'test_id':>10}{'test_ood':>10}"
      f"{'连续10s':>10}{'连续30s':>10}{'连续60s':>10}")
print("  " + "-" * 76)
for nm, r in ALL.items():
    print(f"  {nm:<14}{r['params']:>10,}{r['test_id']['mean']:>10.4f}"
          f"{r['test_ood']['mean']:>10.4f}"
          f"{r['continuous'].get('test_id_10s',{}).get('final_mean',0):>10.4f}"
          f"{r['continuous'].get('test_id_30s',{}).get('final_mean',0):>10.4f}"
          f"{r['continuous'].get('test_id_60s',{}).get('final_mean',0):>10.4f}")

print("\n  物理 Probe (R², test_id):")
print(f"  {'配置':<14}" + "".join(f"{n:>14}" for n in PHYS_NAMES))
print("  " + "-" * 44)
for nm, r in ALL.items():
    line = f"  {nm:<14}"
    for pn in PHYS_NAMES:
        line += f"{r['probe'][pn]['r2_test']:>14.4f}"
    print(line)

print(f"\n  -> {resfile}")



