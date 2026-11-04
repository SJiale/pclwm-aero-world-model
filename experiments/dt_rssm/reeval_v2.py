"""重评脚本（修复 broadcast bug 后）

用途：训练已完成并保存检查点后，用修复的代码重新计算所有指标。

背景：
  原 train_dt_v2.py 的 long_continuous 有广播 bug：
    pred [1,C] - gt [1,1,C] -> [1,1,C]
    取 [0] 得 [1,C]，mean(axis=1) 得 [C]（逐通道）而非标量
    再取 [-1] 变成"最后一个通道的误差"
  -> 连续递推数字全部错误（偏小 7 倍）

本脚本修正后重算，检查点不变。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_dt")))

from evaluate import UnifiedData, UnifiedEval, rollout_far, WINDOW, DT, \
    LONG_TARGETS
from dt_rssm import DTWorldModel
from rssm_impl import WorldModel as BaseRSSM

CKPT = Path(r"C:\pclwm\pclwm_dt\checkpoints")
RES = Path(r"C:\pclwm\pclwm_dt\results_v2")
RES.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "gumbel"
N_REP = 3

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
REF = float(data.obs_sd.mean())

te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)

MATRIX = {
    "RSSM-d128": ("base", dict(deter=128, hidden=128)),
    "RSSM-wide": ("base", dict(deter=136, hidden=256)),
    "DT-K1": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=1)),
    "DT-K4": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=4)),
    "DT-K8": ("dt", dict(deter_fast=96, deter_slow=96, k_slow=8)),
    "DT-K4-matched": ("dt", dict(deter_fast=48, deter_slow=48, k_slow=4)),
}

print("=" * 88)
print("重评（修复 broadcast bug）")
print("=" * 88)
print(f"  设备={DEV}  ref_std={REF:.2f}  模式={MODE}  重复={N_REP}")


def build(name):
    kind, kw = MATRIX[name]
    ck = CKPT / f"{name}.pt"
    if not ck.exists():
        return None, kind
    if kind == "base":
        m = BaseRSSM(C_O, C_A, **kw).to(DEV)
    else:
        m = DTWorldModel(C_O, C_A, **kw).to(DEV)
    m.load_state_dict(torch.load(ck, map_location=DEV,
                                 weights_only=True))
    m.eval()
    return m, kind


@torch.no_grad()
def predict(model, ws, seed=None, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
             .float().to(DEV) for k in ["obs", "act", "act_fut"]}
        if seed is not None:
            torch.manual_seed(seed + i)
        out.append(model.imagine(b["obs"], b["act"], b["act_fut"],
                                 mode=MODE).cpu().numpy())
    return np.concatenate(out, 0)


def eval_short(model, ws, n_rep=N_REP):
    vals, chs = [], []
    for r in range(n_rep):
        p = predict(model, ws, seed=1000 + r*777)
        e = ev.evaluate(p, ws)
        vals.append(e["overall_rel"])
        chs.append(e["per_channel"])
    ch = {c: {"mean": float(np.mean([x[c]["rmse_rel"] for x in chs])),
              "std": float(np.std([x[c]["rmse_rel"] for x in chs]))}
          for c in data.obs_cols}
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
            "min": float(np.min(vals)), "max": float(np.max(vals)),
            "per_channel": ch, "n_rep": n_rep}


def eval_segmented(model, split, tgt, n_rep=N_REP):
    finals, means = [], []
    for r in range(n_rep):
        def mfn(obs, act, af, _sd=1000+r*777):
            with torch.no_grad():
                torch.manual_seed(_sd)
                return model.imagine(
                    torch.from_numpy(obs).float().to(DEV),
                    torch.from_numpy(act).float().to(DEV),
                    torch.from_numpy(af).float().to(DEV),
                    mode=MODE).cpu().numpy()
        rs = []
        for rid in data.splits[split]:
            x = rollout_far(mfn, data, rid, tgt)
            if x:
                rs.append(x)
        if rs:
            finals.append(np.mean([x["final_rel"] for x in rs]))
            means.append(np.mean([x["mean_rel"] for x in rs]))
    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)), "n_rep": n_rep}


def eval_continuous(model, split, tgt, seed=42):
    """★ 修复版：一次历史初始化后连续潜状态递推

    关键修正：
      pred 形状 [1,C]，gt 形状 [1,C]（不是 [1,1,C]）
      每步误差标量 = sqrt(mean((pred-gt)^2 * sd^2))
    """
    is_dt = isinstance(model, DTWorldModel)
    rssm = model.rssm
    finals, means, curves = [], [], []
    for rid in data.splits[split]:
        o = data.norm_obs(rid).astype(np.float32)
        a = data.norm_act(rid).astype(np.float32)
        n_steps = int(round(tgt / DT))
        start = WINDOW + 1
        if len(o) <= start + n_steps + 2:
            n_steps = len(o) - start - 2
            if n_steps < 10:
                continue
        torch.manual_seed(seed)
        with torch.no_grad():
            obs_h = torch.from_numpy(o[start-WINDOW:start][None]) \
                .float().to(DEV)
            act_h = torch.from_numpy(a[start-WINDOW-1:start-1][None]) \
                .float().to(DEV)
            state = rssm.initial(1, DEV)

            # 一次性编码历史
            for t in range(WINDOW):
                if is_dt:
                    need = (t % rssm.k_slow == 0) or (t == 0)
                    s = rssm.step(state, act_h[:, t], need)
                    post = rssm.posterior(s["prior_in"], obs_h[:, t])
                    z = rssm.sample(post, MODE)
                    state = {"hf": s["hf"], "hs": s["hs"], "z": z,
                             "step": t}
                else:
                    pz = state["z"].reshape(1, -1)
                    h = rssm.cell(torch.cat([pz, act_h[:, t]], -1),
                                  state["h"])
                    emb = rssm.enc(obs_h[:, t])
                    pl = rssm.post(torch.cat([h, emb], -1)).reshape(
                        1, rssm.stoch, rssm.classes)
                    z = rssm.sample(pl, MODE)
                    state = {"h": h, "z": z}

            # 连续自由递推（不看未来真值）
            errs = []
            for k in range(n_steps):
                t_abs = WINDOW + k
                a_t = torch.from_numpy(
                    a[start-1+k:start+k][None]).float().to(DEV).reshape(1, -1)
                if is_dt:
                    need = (t_abs % rssm.k_slow == 0) or (t_abs == 0)
                    s = rssm.step(state, a_t, need)
                    z = rssm.sample(s["prior_logits"], MODE)
                    state = {"hf": s["hf"], "hs": s["hs"], "z": z,
                             "step": t_abs}
                    pred = rssm.dec(rssm.feat(state))          # [1,C]
                else:
                    pz = state["z"].reshape(1, -1)
                    h = rssm.cell(torch.cat([pz, a_t], -1), state["h"])
                    pl = rssm.prior(h).reshape(1, rssm.stoch,
                                               rssm.classes)
                    z = rssm.sample(pl, MODE)
                    state = {"h": h, "z": z}
                    pred = rssm.dec(rssm.feat(state))          # [1,C]

                gt = torch.from_numpy(
                    o[start+k:start+k+1][None]).float().to(DEV)  # [1,1,C]
                # ★ 关键修正：pred [1,C] vs gt[:,0] [1,C]
                d = (pred - gt[:, 0]).cpu().numpy()[0]           # [C]
                assert d.shape == (C_O,), f"d.shape={d.shape}"
                errs.append(d)

        E = np.stack(errs, 0).astype(np.float64)                 # [n,C]
        E = E * np.asarray(data.obs_sd).ravel()[None, :]
        ps = np.sqrt((E**2).mean(axis=1))                        # [n]
        finals.append(float(ps[-1]) / REF)
        means.append(float(ps.mean()) / REF)
        curves.append((ps / REF).tolist())

    if not finals:
        return None
    return {"final_mean": float(np.mean(finals)),
            "final_std": float(np.std(finals)),
            "mean_rel": float(np.mean(means)),
            "n_traj": len(finals),
            "curve_mean": (np.mean(curves, 0).tolist() if curves else [])}


# ══════════════════════════════════════════════════
OUT = {}
old = json.loads((RES / "results_v2.json").read_text(encoding="utf-8")) \
    if (RES / "results_v2.json").exists() else {}

for name in MATRIX:
    model, kind = build(name)
    if model is None:
        print(f"  跳过 {name}（无检查点）")
        continue
    npar = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] kind={kind} 参数={npar:,}")

    t_id = eval_short(model, te_W)
    t_ood = eval_short(model, ood_W)
    print(f"  test_id={t_id['mean']:.4f}  ood={t_ood['mean']:.4f}")

    seg, cont = {}, {}
    for split in ["test_id", "test_ood"]:
        for tgt in LONG_TARGETS:
            x = eval_segmented(model, split, tgt)
            if x:
                seg[f"{split}_{tgt}s"] = x
            y = eval_continuous(model, split, tgt)
            if y:
                cont[f"{split}_{tgt}s"] = y
    for t in LONG_TARGETS:
        sk = f"test_id_{t}s"
        s = seg.get(sk, {}).get("final_mean")
        c = cont.get(sk, {}).get("final_mean")
        print(f"  {t:>2}s: 分段={s if s else 0:.4f}  连续={c if c else 0:.4f}")

    prev = old.get(name, {})
    OUT[name] = {
        "kind": kind, "config": MATRIX[name][1], "params": int(npar),
        "best_val": prev.get("best_val"),
        "best_epoch": prev.get("best_epoch"),
        "train_time_s": prev.get("train_time_s"),
        "test_id": t_id, "test_ood": t_ood,
        "long_segmented": seg, "long_continuous": cont,
        "hist": prev.get("hist", {}),
        "eval_bugfix": "continuous rollout broadcast bug fixed "
                       "2026-11-04",
    }
    (RES / "results_v2_fixed.json").write_text(
        json.dumps(OUT, ensure_ascii=False, indent=1), encoding="utf-8")

print("\n" + "=" * 88)
print("汇总（修复后）")
print("=" * 88)
print(f"  {'模型':<18}{'参数':>12}{'test_id':>11}{'test_ood':>11}"
      f"{'分段60s':>11}{'连续60s':>11}")
print("  " + "-" * 74)
for nm, r in OUT.items():
    s60 = r["long_segmented"].get("test_id_60s", {}).get("final_mean", 0)
    c60 = r["long_continuous"].get("test_id_60s", {}).get("final_mean", 0)
    print(f"  {nm:<18}{r['params']:>12,}{r['test_id']['mean']:>11.4f}"
          f"{r['test_ood']['mean']:>11.4f}{s60:>11.4f}{c60:>11.4f}")
print(f"\n  -> {RES/'results_v2_fixed.json'}")
