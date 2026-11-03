"""容量消融实验：GRU hidden=96/128/192 × 3 种子

严格约束：
  - 只用验证集选检查点，**绝不使用测试集**
  - 逐轨迹输出 val/test RMSE
  - 阶跃 vs 斜坡 瞬态误差分别统计
  - 完整记录 loss 曲线
  - 所有 print 加 flush=True
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(r"C:\pclwm\handoff\PC-LWM_handoff_20261102")
sys.path.insert(0, str(PKG / "src"))
from aero_world_model.models.baselines import BaselineGRU

DATA = PKG / "data" / "prepared"
OUT = PKG / "results" / "capacity_ablation"
OUT.mkdir(parents=True, exist_ok=True)
CKPT_DIR = PKG / "checkpoints" / "capacity"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
STEP, WINDOW, HORIZON = 7, 64, 32
STRIDE_TR, STRIDE_EV = 4, 8
DROP = ["S4_FAR", "S0_Tt", "S0_Pt", "S2_Tt", "S2_Pt"]

HIDDENS = [96, 128, 192]
SEEDS = [0, 1, 2]
MAX_EPOCHS = 100
PATIENCE = 25

# ───────────── 数据 ─────────────
d = np.load(DATA / "data.npz", allow_pickle=True)
OBS_ALL = [str(c) for c in d["obs_cols"]]
splits = json.loads((DATA / "splits.json").read_text(encoding="utf-8"))
meta = {m["run_id"]: m for m in json.loads((DATA / "meta.json").read_text(encoding="utf-8"))}
runs = {str(r): {"obs": d[f"{r}_obs"][::STEP].astype(np.float32),
                 "act": d[f"{r}_act"][::STEP].astype(np.float32)}
        for r in d["run_ids"]}
KEEP = [c for c in OBS_ALL if c not in DROP]
KIDX = [OBS_ALL.index(c) for c in KEEP]
C_O = len(OBS_ALL)

tr_obs = np.concatenate([runs[r]["obs"] for r in splits["train"]], 0)
tr_act = np.concatenate([runs[r]["act"] for r in splits["train"]], 0)
o_mu = tr_obs.mean(0).astype(np.float64)
o_sd = tr_obs.std(0).astype(np.float64) + 1e-8
a_mu = tr_act.mean(0).astype(np.float64)
a_sd = tr_act.std(0).astype(np.float64) + 1e-8

# 瞬态区间定义（原始 120 s 时间轴）
# PLA 阶跃发生在 20 s；斜坡 20-80 s 上升，80-110 s 下降
TRANS_STEP = (20.0, 50.0)      # 阶跃后瞬态
TRANS_RAMP = (20.0, 110.0)     # 斜坡全过程


def windows(ids, stride):
    W = []
    for r in ids:
        o = ((runs[r]["obs"] - o_mu) / o_sd).astype(np.float32)
        a = ((runs[r]["act"] - a_mu) / a_sd).astype(np.float32)
        t = np.arange(len(o)) * STEP * 0.015
        for st in range(1, len(o) - WINDOW - HORIZON, stride):
            W.append({"obs": o[st:st + WINDOW],
                      "act": a[st - 1:st - 1 + WINDOW],
                      "fut": o[st + WINDOW:st + WINDOW + HORIZON],
                      "act_fut": a[st - 1 + WINDOW:st - 1 + WINDOW + HORIZON],
                      "rid": r,
                      "t0": float(t[st + WINDOW])})
    return W


tr_W = windows(splits["train"], STRIDE_TR)
va_W = windows(splits["val"], STRIDE_EV)
te_W = windows(splits["test"], STRIDE_EV)
print(f"窗口: train={len(tr_W)} val={len(va_W)} test={len(te_W)}", flush=True)


def batchify(ws, bs, shuffle):
    idx = np.random.permutation(len(ws)) if shuffle else np.arange(len(ws))
    for i in range(0, len(ws), bs):
        c = [ws[j] for j in idx[i:i + bs]]
        yield {k: torch.from_numpy(np.stack([w[k] for w in c])).to(DEV)
               for k in ["obs", "act", "fut", "act_fut"]}


@torch.no_grad()
def predict(model, ws, bs=128):
    model.eval()
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i + bs]
        o = torch.from_numpy(np.stack([w["obs"] for w in c])).to(DEV)
        a = torch.from_numpy(np.stack([w["act"] for w in c])).to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])).to(DEV)
        out.append(model(o, a, af).cpu().numpy())
    return np.concatenate(out, 0)


def eval_detail(model, ws, name):
    """逐轨迹 + 阶跃/斜坡 分层"""
    P = predict(model, ws)
    F = np.stack([w["fut"] for w in ws], 0)
    E = (P - F)[:, :, KIDX]
    step_rmse = np.sqrt((E ** 2).mean(2)).mean(1)      # [N]

    # 总体
    res = {"overall_mean": float(step_rmse.mean()),
           "k1": float(np.sqrt((E[:, 0] ** 2).mean(1)).mean()),
           "kH": float(np.sqrt((E[:, -1] ** 2).mean(1)).mean())}

    # 逐轨迹
    per = {}
    for rid in sorted(set(w["rid"] for w in ws)):
        m = np.array([w["rid"] == rid for w in ws])
        if m.sum() == 0:
            continue
        per[rid] = {"n_win": int(m.sum()),
                    "mean": float(step_rmse[m].mean()),
                    "k1": float(np.sqrt((E[m][:, 0] ** 2).mean(1)).mean()),
                    "kH": float(np.sqrt((E[m][:, -1] ** 2).mean(1)).mean())}
    res["per_traj"] = per

    # 按激励类型分层
    prof = np.array([meta[w["rid"]]["profile"] for w in ws])
    by_prof = {}
    for p in sorted(set(prof)):
        m = prof == p
        by_prof[p] = {"n": int(m.sum()),
                      "mean": float(step_rmse[m].mean())}
    res["by_profile"] = by_prof

    # 阶跃组 vs 斜坡组
    is_step = np.isin(prof, ["step80", "step55", "step95"])
    is_ramp = prof == "ramp"
    res["step_group"] = float(step_rmse[is_step].mean()) if is_step.any() else None
    res["ramp_group"] = float(step_rmse[is_ramp].mean()) if is_ramp.any() else None

    # 瞬态窗口（按 t0 落在阶跃/斜坡区间）
    t0 = np.array([w["t0"] for w in ws])
    # 预测起点时间在 20-50 s 视为阶跃瞬态区
    in_step_tr = is_step & (t0 >= TRANS_STEP[0]) & (t0 <= TRANS_STEP[1])
    in_ramp_tr = is_ramp & (t0 >= TRANS_RAMP[0]) & (t0 <= TRANS_RAMP[1])
    res["step_transient"] = (float(step_rmse[in_step_tr].mean())
                             if in_step_tr.any() else None)
    res["step_transient_n"] = int(in_step_tr.sum())
    res["ramp_transient"] = (float(step_rmse[in_ramp_tr].mean())
                             if in_ramp_tr.any() else None)
    res["ramp_transient_n"] = int(in_ramp_tr.sum())

    # 稳态窗口（t0 > 90 s，且是阶跃族）
    in_steady = is_step & (t0 > 90)
    res["steady"] = float(step_rmse[in_steady].mean()) if in_steady.any() else None
    res["steady_n"] = int(in_steady.sum())
    return res


# ───────────── 训练 ─────────────
ALL = {}

for hidden in HIDDENS:
    for seed in SEEDS:
        tag = f"h{hidden}_s{seed}"
        print(f"\n{'='*70}", flush=True)
        print(f"[{tag}] hidden={hidden} seed={seed}", flush=True)
        print("=" * 70, flush=True)

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = BaselineGRU(C_O, 1, hidden=hidden, horizon=HORIZON,
                            mlp_units=512, mlp_layers=2).to(DEV)
        n_par = sum(p.numel() for p in model.parameters())
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)

        hist = {"train": [], "val": []}
        best_val, best_ep, bad = float("inf"), -1, 0
        t0 = time.time()

        for ep in range(1, MAX_EPOCHS + 1):
            model.train()
            tl, n = 0.0, 0
            for b in batchify(tr_W, 64, True):
                opt.zero_grad(set_to_none=True)
                p = model(b["obs"], b["act"], b["act_fut"])
                loss = ((p - b["fut"]) ** 2).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
                opt.step()
                tl += float(loss.detach()) * len(b["obs"])
                n += len(b["obs"])
            sched.step()

            model.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for b in batchify(va_W, 128, False):
                    p = model(b["obs"], b["act"], b["act_fut"])
                    vs += float(((p - b["fut"]) ** 2).mean()) * len(b["obs"])
                    vn += len(b["obs"])
            v = vs / vn
            hist["train"].append(tl / n)
            hist["val"].append(v)

            if v < best_val - 1e-7:
                best_val, best_ep, bad = v, ep, 0
                torch.save(model.state_dict(), CKPT_DIR / f"{tag}.pt")
            else:
                bad += 1
            if ep % 10 == 0 or ep == 1:
                print(f"  ep {ep:>3} train={tl/n:.5f} val={v:.5f} "
                      f"best={best_val:.5f}@{best_ep} ({time.time()-t0:.0f}s)",
                      flush=True)
            if bad >= PATIENCE:
                print(f"  早停 ep {ep}（best={best_val:.5f}@{best_ep}）",
                      flush=True)
                break

        # 加载最佳（验证集选的）检查点
        model.load_state_dict(torch.load(CKPT_DIR / f"{tag}.pt",
                                         map_location=DEV, weights_only=True))
        model.eval()

        res = {"hidden": hidden, "seed": seed, "params": n_par,
               "best_val": best_val, "best_epoch": best_ep,
               "epochs_run": len(hist["train"]),
               "train_time_s": time.time() - t0,
               "hist": hist,
               "val": eval_detail(model, va_W, "val"),
               "test": eval_detail(model, te_W, "test")}

        print(f"  -> val={res['val']['overall_mean']:.4f} "
              f"test={res['test']['overall_mean']:.4f} "
              f"kH_test={res['test']['kH']:.4f}", flush=True)
        ALL[tag] = res

        # 增量保存，防丢失
        (OUT / "results.json").write_text(
            json.dumps(ALL, ensure_ascii=False, indent=1), encoding="utf-8")

print("\n" + "=" * 78, flush=True)
print("容量消融汇总", flush=True)
print("=" * 78, flush=True)
print(f"{'config':<14}{'params':>10}{'best_ep':>8}{'val':>9}"
      f"{'test':>9}{'step':>9}{'ramp':>9}{'steady':>9}", flush=True)
print("-" * 78, flush=True)
for tag, r in ALL.items():
    v, t = r["val"], r["test"]
    def f(x):
        return f"{x:.4f}" if x is not None else "  -  "
    print(f"{tag:<14}{r['params']:>10,}{r['best_epoch']:>8}"
          f"{v['overall_mean']:>9.4f}{t['overall_mean']:>9.4f}"
          f"{f(t['step_transient']):>9}{f(t['ramp_transient']):>9}"
          f"{f(t['steady']):>9}", flush=True)

# 按 hidden 聚合（跨种子）
print("\n按容量聚合（3 种子均值 ± 标准差）", flush=True)
print(f"{'hidden':<10}{'params':>10}{'val':>20}{'test':>20}", flush=True)
print("-" * 62, flush=True)
for h in HIDDENS:
    ts = [r for r in ALL.values() if r["hidden"] == h]
    if not ts:
        continue
    vs = [r["val"]["overall_mean"] for r in ts]
    tes = [r["test"]["overall_mean"] for r in ts]
    print(f"{h:<10}{ts[0]['params']:>10,}"
          f"{np.mean(vs):>11.4f} ± {np.std(vs):<6.4f}"
          f"{np.mean(tes):>11.4f} ± {np.std(tes):<6.4f}", flush=True)

print(f"\n结果 -> {OUT/'results.json'}", flush=True)
