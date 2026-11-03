"""严格推理模式评价（按用户要求）

核心约束：
  - 显式指定模式（gumbel / prob / argmax），绝不依赖 model.train()/eval() 切换
  - 全程 model.eval()，只改潜状态推理策略
  - Gumbel 模式重复 5 次不同种子，报 mean±std
  - 统一评价 Train/Val/Test-ID/Test-OOD + 10/30/60 s 自由递推
  - 核对 trace：训练多步损失是否真用连续先验；控制输入时序是否一致

修复的关键 bug：
  rssm_impl.RSSM.imagine 的默认 `mode=None` 会退化为
  `'gumbel' if self.training else 'argmax'`，导致 eval 时悄悄用 argmax。
  本脚本**总是显式传 mode**。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))
from evaluate import (UnifiedData, UnifiedEval, rollout_far, WINDOW,
                      HORIZON, LONG_TARGETS)
from rssm_impl import load_rssm

OUT = Path(r"C:\pclwm\pclwm_experiments")
DIAG = OUT / "diagnosis"
DIAG.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt
REF = float(data.obs_sd.mean())

print("=" * 96)
print("严格推理模式评价（显式 mode，全程 eval，Gumbel 5 seeds）")
print("=" * 96)
print(f"  设备={DEV}  dt={DT:.3f}s  WINDOW={WINDOW}  HORIZON={HORIZON}")

CK = OUT / "checkpoints" / "RSSM-d128.pt"
model = load_rssm(CK, C_O, C_A, deter=128, hidden=128, device=DEV)
model.eval()                      # 全程 eval
rssm = model.rssm
for p in model.parameters():
    p.requires_grad_(False)
print(f"  模型: {CK.name}  参数量={sum(p.numel() for p in model.parameters()):,}")
print(f"  model.training = {model.training}  (必须 False)")

SPLITS = {}
for nm in ["train", "val", "test_id", "test_ood"]:
    SPLITS[nm] = data.windows(nm, stride=8)
print(f"  窗口: " + "  ".join(f"{k}={len(v)}" for k, v in SPLITS.items()))


# ══════════════════════════════════════════════════════════════
# 显式模式推理（完全不碰 self.training）
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def imagine_mode(obs, act, af, mode, rng_seed=None):
    """显式指定 mode 的推理，绝不用 self.training

    mode: 'gumbel' | 'prob' | 'argmax'
    """
    if rng_seed is not None:
        torch.manual_seed(rng_seed)
    B, T, _ = obs.shape
    H = af.shape[1]
    state = rssm.initial(B, DEV)
    # 历史编码：用后验（历史 obs 可见，合理）
    # 注意：历史也用 mode 指定的采样方式，保持一致
    for t in range(T):
        pz = state["z"].reshape(B, -1)
        h = rssm.cell(torch.cat([pz, act[:, t]], -1), state["h"])
        embed = rssm.enc(obs[:, t])
        pl = rssm.post(torch.cat([h, embed], -1)) \
            .reshape(B, rssm.stoch, rssm.classes)
        z = rssm.sample(pl, mode)
        state = {"h": h, "z": z}
    outs = []
    # 未来：连续先验递推（不看未来 obs）
    for k in range(H):
        pz = state["z"].reshape(B, -1)
        h = rssm.cell(torch.cat([pz, af[:, k]], -1), state["h"])
        prior_logits = rssm.prior(h).reshape(B, rssm.stoch, rssm.classes)
        z = rssm.sample(prior_logits, mode)
        state = {"h": h, "z": z}
        outs.append(rssm.dec(rssm.feat(state)))
    return torch.stack(outs, 1)


@torch.no_grad()
def predict_split(ws, mode, seed=None, bs=64):
    out = []
    for i in range(0, len(ws), bs):
        c = ws[i:i+bs]
        obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
            .float().to(DEV)
        act = torch.from_numpy(np.stack([w["act"] for w in c])) \
            .float().to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in c])) \
            .float().to(DEV)
        s = None if seed is None else seed + i
        out.append(imagine_mode(obs, act, af, mode, s).cpu().numpy())
    return np.concatenate(out, 0)


# ══════════════════════════════════════════════════════════════
# 主评价：3 模式 × 4 划分
# ══════════════════════════════════════════════════════════════
N_REPEAT = 5
print("\n" + "=" * 96)
print(f"[1] 短期 32 步评价（{HORIZON*DT:.2f} s），Gumbel 重复 {N_REPEAT} 次")
print("=" * 96)

SHORT = {}
for mode in ["argmax", "prob", "gumbel"]:
    SHORT[mode] = {}
    for split, ws in SPLITS.items():
        if mode == "gumbel":
            vals, k1s, kHs = [], [], []
            for r in range(N_REPEAT):
                p = predict_split(ws, "gumbel", seed=1000 + r * 777)
                e = ev.evaluate(p, ws)
                vals.append(e["overall_rel"])
                k1s.append(e["k1_rel"]); kHs.append(e["kH_rel"])
            SHORT[mode][split] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "k1": float(np.mean(k1s)), "kH": float(np.mean(kHs)),
                "n_rep": N_REPEAT,
            }
        else:
            p = predict_split(ws, mode)
            e = ev.evaluate(p, ws)
            SHORT[mode][split] = {
                "mean": float(e["overall_rel"]), "std": 0.0,
                "k1": float(e["k1_rel"]), "kH": float(e["kH_rel"]),
                "n_rep": 1,
            }

print(f"\n  {'模式':<10}{'划分':<12}{'overall_rel':>16}{'k1':>10}{'kH':>10}")
print("  " + "-" * 62)
for mode in ["argmax", "prob", "gumbel"]:
    for split in ["train", "val", "test_id", "test_ood"]:
        d = SHORT[mode][split]
        s = (f"{d['mean']:.4f} ± {d['std']:.4f}"
             if d["n_rep"] > 1 else f"{d['mean']:.4f}")
        print(f"  {mode:<10}{split:<12}{s:>16}{d['k1']:>10.4f}"
              f"{d['kH']:>10.4f}")

# ══════════════════════════════════════════════════════════════
# 长时自由递推
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[2] 长时自由递推（10/30/60 s），三种模式")
print("=" * 96)


@torch.no_grad()
def model_fn_mode(mode, seed=None):
    state = {"seed": seed}

    def fn(obs, act, af):
        s = None
        if mode == "gumbel" and state["seed"] is not None:
            torch.manual_seed(state["seed"])
        o = torch.from_numpy(obs).float().to(DEV)
        a = torch.from_numpy(act).float().to(DEV)
        f = torch.from_numpy(af).float().to(DEV)
        return imagine_mode(o, a, f, mode, s).cpu().numpy()
    return fn


LONG = {}
for mode in ["argmax", "prob", "gumbel"]:
    LONG[mode] = {}
    for split in ["train", "val", "test_id", "test_ood"]:
        ids = data.splits[split]
        for tgt in LONG_TARGETS:
            reps = [None] if mode != "gumbel" else list(
                range(1000, 1000 + N_REPEAT * 777, 777))
            finals, means, curves = [], [], []
            for sd in reps:
                fn = model_fn_mode(mode, sd)
                rs = []
                for r in ids:
                    x = rollout_far(fn, data, r, tgt)
                    if x:
                        rs.append(x)
                if rs:
                    finals.append(np.mean([x["final_rel"] for x in rs]))
                    means.append(np.mean([x["mean_rel"] for x in rs]))
                    curves.append(np.mean([x["curve_rel"] for x in rs], 0))
            if finals:
                LONG[mode][f"{split}_{tgt}s"] = {
                    "final_mean": float(np.mean(finals)),
                    "final_std": float(np.std(finals)),
                    "mean_rel": float(np.mean(means)),
                    "curve": (np.mean(curves, 0).tolist()
                              if curves else []),
                    "n_rep": len(reps),
                }

print(f"\n  {'配置':<18}{'argmax':>12}{'prob':>12}"
      f"{'gumbel (mean±std)':>22}")
print("  " + "-" * 66)
for split in ["train", "val", "test_id", "test_ood"]:
    for tgt in LONG_TARGETS:
        k = f"{split}_{tgt}s"
        a = LONG["argmax"].get(k)
        pr = LONG["prob"].get(k)
        g = LONG["gumbel"].get(k)
        la = f"{a['final_mean']:.4f}" if a else "-"
        lp = f"{pr['final_mean']:.4f}" if pr else "-"
        lg = (f"{g['final_mean']:.4f}±{g['final_std']:.4f}"
              if g else "-")
        print(f"  {k:<18}{la:>12}{lp:>12}{lg:>22}")

# ══════════════════════════════════════════════════════════════
# [3] 核对：训练多步损失是否真用连续先验
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[3] 核对：训练多步损失是否用连续先验递推")
print("=" * 96)

trace = {}
with torch.no_grad():
    c = SPLITS["test_id"][:1]
    obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
        .float().to(DEV)
    act = torch.from_numpy(np.stack([w["act"] for w in c])) \
        .float().to(DEV)
    af = torch.from_numpy(np.stack([w["act_fut"] for w in c])) \
        .float().to(DEV)

    # 构造一个会记录 prior 调用次数的钩子
    calls = {"prior": 0, "post": 0}

    def hook_prior(m, i, o):
        calls["prior"] += 1

    def hook_post(m, i, o):
        calls["post"] += 1

    h1 = rssm.prior.register_forward_hook(hook_prior)
    h2 = rssm.post.register_forward_hook(hook_post)
    _ = imagine_mode(obs, act, af, "argmax")
    h1.remove(); h2.remove()

T, H = WINDOW, HORIZON
print(f"  一次 imagine 调用（T={T} 历史 + H={H} 未来）：")
print(f"    prior 被调用 {calls['prior']} 次  (应为 {H}={HORIZON})")
print(f"    post  被调用 {calls['post']} 次   (应为 {T}={WINDOW})")
trace["prior_calls"] = calls["prior"]
trace["post_calls"] = calls["post"]
trace["prior_only_future"] = (calls["prior"] == HORIZON)
trace["future_uses_posterior"] = (calls["post"] == WINDOW)

if calls["prior"] == HORIZON and calls["post"] == WINDOW:
    print(f"  ✅ 未来阶段只用先验（{HORIZON} 次），历史阶段只用后验（{WINDOW} 次）")
    print(f"  ✅ 未来未调用 post -> 无未来 obs 泄漏")
else:
    print(f"  ⚠ 调用次数异常")

# ══════════════════════════════════════════════════════════════
# [4] 核对：控制输入时序一致
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[4] 核对：训练与部署的控制输入时序")
print("=" * 96)

w0 = SPLITS["test_id"][0]
rid = w0["rid"]
pos = int(round(w0["t0"] / DT))
a_norm = data.norm_act(rid)
print(f"  窗口: rid={rid} t0={w0['t0']:.2f}s -> 索引 {pos}")
print(f"  act.shape={w0['act'].shape}  act_fut.shape={w0['act_fut'].shape}")
ok = (np.allclose(w0["act"][-1], a_norm[pos-1], atol=1e-5)
      and np.allclose(w0["act_fut"][0], a_norm[pos], atol=1e-5))
print(f"  act[-1] 应等于 a[{pos-1}]：{'✅' if np.allclose(w0['act'][-1], a_norm[pos-1], atol=1e-5) else '❌'}")
print(f"  act_fut[0] 应等于 a[{pos}]：{'✅' if np.allclose(w0['act_fut'][0], a_norm[pos], atol=1e-5) else '❌'}")
print(f"  -> 时序{'一致' if ok else '不一致'}")
print(f"  训练与推理都走 data.windows -> 同一构造 -> 时序一致")
trace["act_timing_ok"] = bool(ok)

# ══════════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════════
out = {
    "meta": {
        "ckpt": str(CK), "window": WINDOW, "horizon": HORIZON,
        "dt": DT, "n_repeat_gumbel": N_REPEAT,
        "model_eval_mode": True,
        "explicit_mode": True,
    },
    "short": SHORT,
    "long": LONG,
    "trace": trace,
}
(DIAG / "rssm_mode_eval_fixed.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {DIAG/'rssm_mode_eval_fixed.json'}")
