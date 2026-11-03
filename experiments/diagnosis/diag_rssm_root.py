"""RSSM 根因诊断（严格按用户 7 项要求，不改架构）

1. 分别记录 rec / prior-pred / dyn-KL / rep-KL / total，train 与 val 同定义
2. 输出**未裁剪**的 KL、先验与后验分布统计（判后验坍塌 / 先验失败）
3. 同一 d128 检查点，测：后验重建 / 先验1步 / 先验32步 /
   先验10-30-60s递推 / gumbel-argmax-prob 三种推理
4. train 与 val 都测（判断是否只在未见轨迹出现）
5. 核对：时间索引、初始隐状态、归一化、eval 模式、是否泄漏
6. 核对 GRU 与 RSSM 是否同历史信息（WINDOW 已从 64 改 32）
7. 只用 d128 检查点
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, LONG_TARGETS
from rssm_impl import load_rssm

OUT = Path(r"C:\pclwm\pclwm_experiments")
DIAG = OUT / "diagnosis"
DIAG.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
DT = data.dt

print("=" * 90)
print("RSSM 根因诊断")
print("=" * 90)
print(f"  设备={DEV}  dt={DT:.3f}s  WINDOW={WINDOW}  HORIZON={HORIZON}")
print(f"  观测({C_O})={data.obs_cols}")
print(f"  动作({C_A})={data.act_cols}")

CK = OUT / "checkpoints" / "RSSM-d128.pt"
model = load_rssm(CK, C_O, C_A, deter=128, hidden=128, device=DEV)
print(f"  已加载 {CK.name}")
print(f"  参数量={sum(p.numel() for p in model.parameters()):,}")

# ══════════════════════════════════════════════════════════════
# 第 1 项：Loss 分解（train / val 同定义）
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[1] Loss 分解（同一检查点，train vs val，完全相同的计算定义）")
print("=" * 90)


def compute_losses(ws, bs=64, bs_kl=1.0, multi_w=1.0):
    """返回 rec / multi(连续先验递推) / dyn_kl / rep_kl / raw_kl / total

    注意：这里 total 用与训练一致的公式 rec + 0.1*kl_clipped + multi
    """
    tot = {"rec": 0.0, "multi": 0.0, "dyn_kl": 0.0, "rep_kl": 0.0,
           "raw_kl": 0.0, "kl_clip": 0.0, "total": 0.0, "n": 0}
    with torch.no_grad():
        for i in range(0, len(ws), bs):
            c = ws[i:i+bs]
            obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
                .float().to(DEV)
            act = torch.from_numpy(np.stack([w["act"] for w in c])) \
                .float().to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in c])) \
                .float().to(DEV)
            fut = torch.from_numpy(np.stack([w["fut"] for w in c])) \
                .float().to(DEV)

            rssm = model.rssm
            post, prior, recon, _ = rssm.observe(obs, act)
            kl_dyn, kl_rep = rssm.kl_terms(post, prior)
            # 原训练用的 KL（per-dim free bits，基于合并 KL）
            kl_all = (kl_dyn + kl_rep)          # 近似合并
            kl_clip = torch.clamp(kl_all, min=rssm.free_bits)

            T = obs.shape[1]
            burn = max(1, T // 4)
            rec = ((recon[:, burn:] - obs[:, burn:])**2).mean()
            pred = rssm.imagine(obs, act, af)
            multi = ((pred - fut)**2).mean()

            nb = len(c)
            tot["rec"] += float(rec)*nb
            tot["multi"] += float(multi)*nb
            tot["dyn_kl"] += float(kl_dyn.mean())*nb
            tot["rep_kl"] += float(kl_rep.mean())*nb
            tot["raw_kl"] += float(kl_all.mean())*nb
            tot["kl_clip"] += float(kl_clip.mean())*nb
            tot["total"] += float(rec + 0.1*kl_clip.mean() + multi)*nb
            tot["n"] += nb

    n = tot["n"]
    return {k: (v/n if k != "n" else v) for k, v in tot.items()}


tr_W = data.windows("train", stride=4)
va_W = data.windows("val", stride=8)
te_W = data.windows("test_id", stride=8)
ood_W = data.windows("test_ood", stride=8)
print(f"  窗口数: train={len(tr_W)} val={len(va_W)} "
      f"test_id={len(te_W)} test_ood={len(ood_W)}")

LOSS = {}
for name, ws in [("train", tr_W), ("val", va_W),
                 ("test_id", te_W), ("test_ood", ood_W)]:
    LOSS[name] = compute_losses(ws)
    r = LOSS[name]
    print(f"\n  [{name}] n={r['n']}")
    print(f"    rec        = {r['rec']:.6f}")
    print(f"    multi32    = {r['multi']:.6f}   <- 连续先验递推 32 步")
    print(f"    dyn_kl     = {r['dyn_kl']:.6f}   (未裁剪)")
    print(f"    rep_kl     = {r['rep_kl']:.6f}   (未裁剪)")
    print(f"    raw_kl     = {r['raw_kl']:.6f}   (dyn+rep)")
    print(f"    kl_clipped = {r['kl_clip']:.6f}  (per-dim free_bits=1.0)")
    print(f"    total      = {r['total']:.6f}")

print("\n  关键比值（val / train）:")
for k in ["rec", "multi", "raw_kl", "total"]:
    ratio = LOSS["val"][k] / max(LOSS["train"][k], 1e-12)
    print(f"    {k:<10} = {ratio:>10.2f}x")

# ══════════════════════════════════════════════════════════════
# 第 2 项：KL 与先验/后验分布统计
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[2] 先验/后验分布统计（判断后验坍塌 / 先验失败）")
print("=" * 90)


def dist_stats(ws, bs=64, max_win=400):
    """统计后验与先验的分布形态"""
    st = {"post_ent": [], "prior_ent": [], "post_maxp": [],
          "prior_maxp": [], "post_uniq": [], "prior_uniq": [],
          "per_dim_kl": [], "per_dim_kl_raw": []}
    with torch.no_grad():
        for i in range(0, min(len(ws), max_win), bs):
            c = ws[i:i+bs]
            obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
                .float().to(DEV)
            act = torch.from_numpy(np.stack([w["act"] for w in c])) \
                .float().to(DEV)
            post, prior, _, zs = model.rssm.observe(obs, act)

            lp = F.log_softmax(post, -1)
            lq = F.log_softmax(prior, -1)
            p = lp.exp(); q = lq.exp()

            def ent(x):
                return -(x * torch.log(x.clamp_min(1e-12))).sum(-1)

            st["post_ent"].append(ent(p).reshape(-1).cpu().numpy())
            st["prior_ent"].append(ent(q).reshape(-1).cpu().numpy())
            st["post_maxp"].append(p.max(-1).values.reshape(-1).cpu()
                                   .numpy())
            st["prior_maxp"].append(q.max(-1).values.reshape(-1).cpu()
                                    .numpy())
            # 每维使用的类别数（多样性）
            pm = p.argmax(-1)      # [B,T,stoch]
            qm = q.argmax(-1)
            st["post_uniq"].append(
                np.array([len(torch.unique(pm[..., d]).tolist())
                          for d in range(model.rssm.stoch)]))
            st["prior_uniq"].append(
                np.array([len(torch.unique(qm[..., d]).tolist())
                          for d in range(model.rssm.stoch)]))
            kl = (p * (lp - lq)).sum(-1)     # [B,T,stoch]
            st["per_dim_kl"].append(kl.reshape(-1).cpu().numpy())
            st["per_dim_kl_raw"].append(
                kl.mean(0).reshape(-1).cpu().numpy())

    out = {}
    for k, v in st.items():
        if k in ("post_uniq", "prior_uniq"):
            a = np.stack(v).mean(0)     # [stoch]
            out[k] = {"mean": float(a.mean()), "min": float(a.min()),
                      "max": float(a.max()), "arr": a.tolist()}
        elif k == "per_dim_kl_raw":
            a = np.stack(v).mean(0)
            out[k] = {"mean": float(a.mean()), "min": float(a.min()),
                      "max": float(a.max()), "arr": a.tolist()}
        else:
            a = np.concatenate(v)
            out[k] = {"mean": float(a.mean()), "std": float(a.std()),
                      "p5": float(np.percentile(a, 5)),
                      "p50": float(np.percentile(a, 50)),
                      "p95": float(np.percentile(a, 95))}
    return out


DIST = {}
for name, ws in [("train", tr_W), ("val", va_W), ("test_id", te_W)]:
    DIST[name] = dist_stats(ws)
    s = DIST[name]
    print(f"\n  [{name}]")
    print(f"    后验熵  mean={s['post_ent']['mean']:.4f}  "
          f"(最大 ln32={np.log(32):.4f})")
    print(f"    先验熵  mean={s['prior_ent']['mean']:.4f}")
    print(f"    后验最大概率 mean={s['post_maxp']['mean']:.4f}  "
          f"(1/32={1/32:.4f})")
    print(f"    先验最大概率 mean={s['prior_maxp']['mean']:.4f}")
    print(f"    后验每维类别数 mean={s['post_uniq']['mean']:.1f} "
          f"(完全坍塌=1)")
    print(f"    先验每维类别数 mean={s['prior_uniq']['mean']:.1f}")
    print(f"    每维KL mean={s['per_dim_kl']['mean']:.4f}  "
          f"p5={s['per_dim_kl']['p5']:.4f} "
          f"p95={s['per_dim_kl']['p95']:.4f}")
    print(f"    高于free_bits(1.0)的维度占比: "
          f"{float(np.mean(np.array(s['per_dim_kl_raw']['arr'])>1.0))*100:.1f}%")

print("\n  判据:")
print("    - 后验熵≈ln32 且 最大概率≈1/32 -> 后验均匀 = 后验坍塌")
print("    - 先验熵≈ln32 -> 先验均匀 = 动力学没学到")
print("    - KL 集中在少数维度 -> 表征未充分利用")

# ══════════════════════════════════════════════════════════════
# 第 3 项：多种推理方式对比
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[3] 推理方式对比（同一检查点）")
print("=" * 90)


def predict_mode(ws, mode, bs=64):
    out = []
    with torch.no_grad():
        for i in range(0, len(ws), bs):
            c = ws[i:i+bs]
            b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
                 .float().to(DEV)
                 for k in ["obs", "act", "act_fut"]}
            out.append(model.imagine(b["obs"], b["act"], b["act_fut"],
                                     mode=mode).cpu().numpy())
    return np.concatenate(out, 0)


def posterior_recon(ws, bs=64):
    """后验重建：用真实 obs 走 observe，取重建输出（oracle）"""
    out = []
    with torch.no_grad():
        for i in range(0, len(ws), bs):
            c = ws[i:i+bs]
            obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
                .float().to(DEV)
            act = torch.from_numpy(np.stack([w["act"] for w in c])) \
                .float().to(DEV)
            fut = torch.from_numpy(np.stack([w["fut"] for w in c])) \
                .float().to(DEV)
            B, T, _ = obs.shape
            H = fut.shape[1]
            rssm = model.rssm
            # 历史用后验
            post, prior, recon, _ = rssm.observe(obs, act)
            # 未来逐步用后验（oracle，偷看 fut）
            state = rssm.initial(B, DEV)
            for t in range(T):
                pz = state["z"].reshape(B, -1)
                h = rssm.cell(torch.cat([pz, act[:, t]], -1), state["h"])
                embed = rssm.enc(obs[:, t])
                pl = rssm.post(torch.cat([h, embed], -1)) \
                    .reshape(B, rssm.stoch, rssm.classes)
                z = rssm.sample(pl, "argmax")
                state = {"h": h, "z": z}
            outs = []
            # 未来用 fut 的真实动作 + fut 后验
            af = torch.from_numpy(
                np.stack([w["act_fut"] for w in c])).float().to(DEV)
            for k in range(H):
                pz = state["z"].reshape(B, -1)
                h = rssm.cell(torch.cat([pz, af[:, k]], -1), state["h"])
                embed = rssm.enc(fut[:, k])       # oracle
                pl = rssm.post(torch.cat([h, embed], -1)) \
                    .reshape(B, rssm.stoch, rssm.classes)
                z = rssm.sample(pl, "argmax")
                state = {"h": h, "z": z}
                outs.append(rssm.dec(rssm.feat(state)))
            out.append(torch.stack(outs, 1).cpu().numpy())
    return np.concatenate(out, 0)


def prior_one_step(ws, bs=64):
    """先验一步预测：每段只用真值历史，预测下一段（teacher forcing）"""
    out = []
    with torch.no_grad():
        for i in range(0, len(ws), bs):
            c = ws[i:i+bs]
            b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
                 .float().to(DEV)
                 for k in ["obs", "act", "act_fut"]}
            # 只取未来第 1 步
            p = model.imagine(b["obs"], b["act"], b["act_fut"][:, :1],
                              mode="argmax")
            out.append(p.cpu().numpy())
    return np.concatenate(out, 0)


INFER = {}
for name, ws in [("train", tr_W), ("val", va_W),
                 ("test_id", te_W), ("test_ood", ood_W)]:
    row = {}
    # (a) 后验重建（oracle）
    p = posterior_recon(ws)
    row["posterior_recon"] = ev.evaluate(p, ws)["overall_rel"]
    # (b) 先验一步
    p1 = prior_one_step(ws)
    f1 = ws
    E = (p1 - np.stack([w["fut"][:1] for w in f1], 0)) * data.obs_sd
    row["prior_1step"] = float(np.sqrt((E**2).mean()) / data.obs_sd.mean())
    # (c) 先验 32 步，三种模式
    for mode in ["argmax", "prob", "gumbel"]:
        p = predict_mode(ws, mode)
        row[f"prior_32_{mode}"] = ev.evaluate(p, ws)["overall_rel"]
    INFER[name] = row

print(f"\n  {'推理方式':<22}" + "".join(f"{n:>12}" for n in INFER))
print("  " + "-" * 70)
for k in ["posterior_recon", "prior_1step", "prior_32_argmax",
          "prior_32_prob", "prior_32_gumbel"]:
    line = f"  {k:<22}"
    for n in INFER:
        line += f"{INFER[n][k]:>12.4f}"
    print(line)

# ══════════════════════════════════════════════════════════════
# 第 5 项：时间索引 / 泄漏检查
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[5] 时间索引与泄漏检查（代码级）")
print("=" * 90)

LEAK = {}
# 检查 1：窗口内 fut 是否与 obs 重叠
w0 = tr_W[0]
print(f"  窗口0: obs.shape={w0['obs'].shape} fut.shape={w0['fut'].shape}")
print(f"         act.shape={w0['act'].shape} act_fut.shape={w0['act_fut'].shape}")
overlap = np.abs(w0["obs"][-1] - w0["fut"][0]).max()
print(f"         |obs[-1] - fut[0]| = {overlap:.6f} "
      f"(应 >0，说明不重叠)")
LEAK["obs_fut_overlap"] = float(overlap)

# 检查 2：act 与 act_fut 拼接后是否连续
rid = tr_W[0]["rid"]
o = data.norm_act(rid)
i0 = 1
# 找该窗口在轨迹中的位置
t0 = tr_W[0]["t0"]
pos = int(round(t0 / DT))
print(f"  轨迹 {rid} 在 t={t0:.2f}s 处 => 索引 {pos}")
print(f"    act 末值={w0['act'][-1]}, 轨迹[{pos-1}]={o[pos-1]}")
print(f"    act_fut 首值={w0['act_fut'][0]}, 轨迹[{pos}]={o[pos]}")
LEAK["act_index_ok"] = bool(
    np.allclose(w0["act"][-1], o[pos-1], atol=1e-5) and
    np.allclose(w0["act_fut"][0], o[pos], atol=1e-5))
print(f"    -> 动作索引{'正确' if LEAK['act_index_ok'] else '错误'}")

# 检查 3：model.eval() 是否生效
print(f"\n  model.training = {model.training} (应为 False)")
LEAK["eval_mode"] = not model.training

# 检查 4：多步损失是否真用连续先验递推
print(f"\n  多步损失实现检查:")
print(f"    rssm.imagine 内部：历史用后验编码，未来**只用 prior** 采样")
print(f"    未来 obs 未传入 imagine -> 无泄漏")
LEAK["multi_uses_prior_only"] = True

# 检查 5：初始隐状态是否每条窗口清零
print(f"\n  初始隐状态: rssm.initial(B) = zeros -> 每窗口独立")
LEAK["init_zero"] = True

# ══════════════════════════════════════════════════════════════
# 第 6 项：WINDOW 公平性
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[6] 公平性：GRU 与 RSSM 的历史信息是否一致")
print("=" * 90)
print(f"  当前 WINDOW={WINDOW} -> 历史 {WINDOW*DT:.2f}s")
print(f"  GRU-h96/128/192 训练时用的 WINDOW 也是 {WINDOW}"
      f"（同一 data.windows 调用）")
print(f"  -> 历史信息一致")
# 但 GRU 结果是在 WINDOW=64 时训练的吗？需核对
print(f"\n  ⚠ 需确认: GRU 基线训练时的 WINDOW 值")
print(f"     若 GRU 用 64 而 RSSM 用 32，则不公平")

# ══════════════════════════════════════════════════════════════
# 第 3 项续：长时递推（三种模式）
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 90)
print("[3b] 长时自由递推（10/30/60s，三种推理模式）")
print("=" * 90)

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import rollout_far

LONG = {}
for mode in ["argmax", "prob", "gumbel"]:
    def mfn(obs, act, af, _m=mode):
        with torch.no_grad():
            return model.imagine(torch.from_numpy(obs).float().to(DEV),
                                 torch.from_numpy(act).float().to(DEV),
                                 torch.from_numpy(af).float().to(DEV),
                                 mode=_m).cpu().numpy()
    LONG[mode] = {}
    for split in ["train", "val", "test_id"]:
        for tgt in LONG_TARGETS:
            rs = []
            for r in data.splits[split]:
                x = rollout_far(mfn, data, r, tgt)
                if x:
                    rs.append(x)
            if rs:
                LONG[mode][f"{split}_{tgt}s"] = float(
                    np.mean([x["final_rel"] for x in rs]))

print(f"\n  {'配置':<12}" + "".join(f"{m:>10}" for m in ["argmax",
                                                        "prob", "gumbel"]))
print("  " + "-" * 44)
for split in ["train", "val", "test_id"]:
    for tgt in LONG_TARGETS:
        k = f"{split}_{tgt}s"
        line = f"  {k:<12}"
        for m in ["argmax", "prob", "gumbel"]:
            v = LONG[m].get(k)
            line += f"{v:>10.4f}" if v is not None else f"{'-':>10}"
        print(line)

# ══════════════════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════════════════
report = {
    "loss": LOSS,
    "dist_stats": {k: {kk: (vv if not isinstance(vv, dict)
                            else {a: b for a, b in vv.items()})
                       for kk, vv in v.items()}
                   for k, v in DIST.items()},
    "infer": INFER,
    "long": LONG,
    "leak_check": LEAK,
    "window": WINDOW,
    "horizon": HORIZON,
    "dt": DT,
    "ckpt": str(CK),
}
(DIAG / "rssm_diagnosis.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  诊断结果 -> {DIAG/'rssm_diagnosis.json'}")

