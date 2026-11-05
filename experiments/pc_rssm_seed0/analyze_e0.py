"""E0 潜状态结构分析

问题：
  1. h 的 128 维都做了什么？有效秩是多少？
  2. z（32×32 离散）实际用到了多少类别？
  3. h 和 z 各自编码什么物理量？
  4. T4 信息在 h 中如何分布？
  5. 时间尺度：h 各维的 ACF
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from evaluate import UnifiedData, WINDOW, DT
from pc_data import PCData, PHYS_NAMES
from pc_rssm import PCWorldModel

OUT = Path(r"C:\pclwm\pclwm_pcrssm\analysis")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"

pcd = PCData()
data = pcd.d
C_O = data.n_obs

# 载入 E0
wm = PCWorldModel(C_O, 3, deter=128, hidden=128,
                  lam_static=0.0, lam_dynamic=0.0).to(DEV)
wm.load_state_dict(torch.load(
    r"C:\pclwm\pclwm_pcrssm\checkpoints\E0_baseline.pt",
    map_location=DEV, weights_only=True))
wm.eval()
rssm = wm.rssm

print("=" * 92)
print("E0 潜状态结构分析（无物理监督的基线）")
print("=" * 92)
print(f"  deter={rssm.deter}  stoch={rssm.stoch}x{rssm.classes}")
print(f"  参数={sum(p.numel() for p in wm.parameters()):,}")


# ══════════════════════════════════════════════════
# 提取潜状态（用后验，argmax 保证确定性）
# ══════════════════════════════════════════════════
def extract_traj(rid):
    """逐时刻提取 h / z_logits / z_prob / z_argmax"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    mask = np.ones((n, C_O), np.float32)

    hs, zl, zp, za = [], [], [], []
    with torch.no_grad():
        st = rssm.init_state(1, DEV)
        for t in range(n):
            z_prev = st["z"].reshape(1, -1)
            at = torch.from_numpy(a[t:t+1]).float()
            h = rssm.cell(torch.cat([z_prev, at], -1), st["h"])
            emb = rssm.enc(torch.cat([
                torch.from_numpy(o[t:t+1]).float(),
                torch.from_numpy(mask[t:t+1]).float()], -1))
            pre = torch.cat([h, z_prev], -1)
            post = rssm.post(torch.cat([pre, emb], -1)).reshape(
                1, rssm.stoch, rssm.classes)
            zp_t = F.softmax(post, -1)
            z_hard = rssm.sample(post, "argmax")
            st = {"h": h, "z": z_hard}
            hs.append(h[0].numpy())
            zl.append(post[0].numpy())
            zp.append(zp_t[0].numpy())
            za.append(z_hard[0].argmax(-1).numpy())
    return (np.array(hs), np.array(zl), np.array(zp), np.array(za),
            o, a)


print("\n  提取潜状态...")
REP = {}
for split in ["train", "val", "test_id", "test_ood"]:
    for rid in data.splits[split]:
        REP[rid] = {"split": split, **dict(zip(
            ["h", "z_logits", "z_prob", "z_argmax", "obs", "act"],
            extract_traj(rid)))}
print(f"  完成 {len(REP)} 条")

# ══════════════════════════════════════════════════
# 1. h 的有效维度
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[1] h 的有效维度（PCA 谱）")
print("=" * 92)

Htr = np.concatenate([REP[r]["h"] for r in data.splits["train"]], 0)
Hte = np.concatenate([REP[r]["h"] for r in data.splits["test_id"]], 0)
print(f"  train h: {Htr.shape}")

mu = Htr.mean(0)
Xc = Htr - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = S**2 / (S**2).sum()
cum = np.cumsum(ev)

print(f"\n  {'秩':>4}{'单维方差占比':>16}{'累积':>10}")
print("  " + "-" * 32)
for i in [0, 1, 2, 4, 9, 19, 31, 47, 63, 95, 127]:
    if i < len(ev):
        print(f"  {i+1:>4}{ev[i]:>15.4f}{cum[i]:>10.4f}")

for thresh in [0.90, 0.95, 0.99]:
    k = int(np.searchsorted(cum, thresh) + 1)
    print(f"\n  达到 {thresh:.0%} 方差需要 {k} 维 "
          f"（共 128 维，占 {k/128:.1%}）")

# 参与比（有效维度数）
participation = (S**2).sum()**2 / (S**4).sum()
print(f"  参与比（有效维度）: {participation:.1f} / 128")

# 各维标准差
sd = Htr.std(0)
print(f"\n  h 各维标准差: min={sd.min():.4f} max={sd.max():.4f} "
      f"median={np.median(sd):.4f}")
n_dead = int((sd < 0.01).sum())
print(f"  近死维度 (sd<0.01): {n_dead} / 128")

# ══════════════════════════════════════════════════
# 2. z 的类别使用情况
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[2] z（32 组 × 32 类）的类别使用情况")
print("=" * 92)

Ztr = np.concatenate([REP[r]["z_argmax"] for r in data.splits["train"]], 0)
Zp = np.concatenate([REP[r]["z_prob"] for r in data.splits["train"]], 0)
print(f"  train z_argmax: {Ztr.shape}  z_prob: {Zp.shape}")

print(f"\n  {'组':>4}{'用到的类别数':>14}{'最常见类别':>12}{'占比':>8}"
      f"{'熵':>10}{'最大概率均值':>14}")
print("  " + "-" * 66)
n_used_list = []
for g in range(rssm.stoch):
    vals, counts = np.unique(Ztr[:, g], return_counts=True)
    n_used = len(vals)
    n_used_list.append(n_used)
    top = counts.argmax()
    frac = counts.max() / counts.sum()
    p = Zp[:, g, :].mean(0)
    ent = float(-(p * np.log(p + 1e-12)).sum())
    maxp = float(Zp[:, g, :].max(-1).mean())
    if g < 12 or g == rssm.stoch - 1:
        print(f"  {g:>4}{n_used:>14}{vals[top]:>12}{frac:>7.1%}"
              f"{ent:>10.3f}{maxp:>14.4f}")

n_used_arr = np.array(n_used_list)
print(f"\n  32 组统计: mean={n_used_arr.mean():.1f} "
      f"min={n_used_arr.min()} max={n_used_arr.max()}")
print(f"  只用 1 类的组: {int((n_used_arr<=1).sum())} / 32")
print(f"  用 <5 类的组: {int((n_used_arr<5).sum())} / 32")
print(f"  理论类别数: {rssm.classes}")

# 全局类别分布
all_ent = []
for g in range(rssm.stoch):
    p = Zp[:, g, :].mean(0)
    all_ent.append(float(-(p * np.log(p + 1e-12)).sum()))
print(f"\n  各组平均熵: {np.mean(all_ent):.3f} "
      f"（均匀分布熵 = ln32 = {np.log(32):.3f}）")
print(f"  熵 < 0.5 的组: {sum(1 for e in all_ent if e < 0.5)} / 32")

# ══════════════════════════════════════════════════
# 3. h / z 对物理量的可解码性
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[3] h / z 对物理量的线性可解码性（train 拟合，val 选 λ，test 报告）")
print("=" * 92)

# 目标：观测 5 通道 + 物理 2 量
TARGETS = {}
for i, c in enumerate(data.obs_cols):
    TARGETS[c] = "obs"
TARGETS["N2dot"] = "phys"
TARGETS["N3dot"] = "phys"

# 构造目标值
def get_target(rid, name):
    r = REP[rid]
    if name in data.obs_cols:
        return r["obs"][:, data.obs_cols.index(name)]
    if name == "N2dot":
        return pcd.phys_raw[rid][:, 0]
    if name == "N3dot":
        return pcd.phys_raw[rid][:, 1]
    return None


FEATS = {
    "h": lambda r: r["h"],
    "z_argmax": lambda r: r["z_argmax"],
    "z_prob": lambda r: r["z_prob"].reshape(len(r["z_prob"]), -1),
    "h+z_prob": lambda r: np.concatenate(
        [r["h"], r["z_prob"].reshape(len(r["z_prob"]), -1)], 1),
}


def ridge(Xtr, Ytr, Xva, Yva, Xte, Yte):
    best = None
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0]:
        Xm, Xs = Xtr.mean(0), Xtr.std(0) + 1e-8
        Xn = (Xtr - Xm) / Xs
        Ym, Ys = Ytr.mean(), Ytr.std() + 1e-8
        Yn = (Ytr - Ym) / Ys
        W = np.linalg.solve(Xn.T @ Xn + lam*np.eye(Xn.shape[1]),
                            Xn.T @ Yn)
        pv = ((Xva - Xm) / Xs) @ W * Ys + Ym
        r2v = 1 - ((pv - Yva)**2).sum() / ((Yva - Yva.mean())**2).sum()
        if best is None or r2v > best[0]:
            best = (r2v, lam, W, Xm, Xs, Ym, Ys)
    r2v, lam, W, Xm, Xs, Ym, Ys = best
    pt = ((Xte - Xm) / Xs) @ W * Ys + Ym
    r2t = 1 - ((pt - Yte)**2).sum() / ((Yte - Yte.mean())**2).sum()
    return float(r2t), float(r2v), lam


skip = WINDOW
RES = {}
for fname, ffn in FEATS.items():
    RES[fname] = {}
    for tg in TARGETS:
        Xtr = np.concatenate([ffn(REP[r])[skip:] for r in data.splits["train"]], 0)
        Ytr = np.concatenate([get_target(r, tg)[skip:] for r in data.splits["train"]], 0)
        Xva = np.concatenate([ffn(REP[r])[skip:] for r in data.splits["val"]], 0)
        Yva = np.concatenate([get_target(r, tg)[skip:] for r in data.splits["val"]], 0)
        Xte = np.concatenate([ffn(REP[r])[skip:] for r in data.splits["test_id"]], 0)
        Yte = np.concatenate([get_target(r, tg)[skip:] for r in data.splits["test_id"]], 0)
        r2t, r2v, lam = ridge(Xtr, Ytr, Xva, Yva, Xte, Yte)
        RES[fname][tg] = {"r2_test": r2t, "r2_val": r2v, "lam": lam}

print(f"\n  {'目标':<14}{'h':>12}{'z_argmax':>12}{'z_prob':>12}{'h+z_prob':>12}")
print("  " + "-" * 62)
for tg in TARGETS:
    print(f"  {tg:<14}" + "".join(
        f"{RES[f][tg]['r2_test']:>12.4f}" for f in FEATS))

# ══════════════════════════════════════════════════
# 4. h 各维与物理量的相关性
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[4] h 各维与物理目标的最大 |r|")
print("=" * 92)

Htr2 = np.concatenate([REP[r]["h"] for r in data.splits["train"]], 0)
print(f"  {'目标':<14}{'最大|r|':>10}{'对应维度':>10}")
print("  " + "-" * 36)
for tg in ["N2dot", "N3dot", "S4_Tt", "N2int"]:
    Ytr2 = np.concatenate([get_target(r, tg) for r in data.splits["train"]], 0)
    # 标准化
    Hn = (Htr2 - Htr2.mean(0)) / (Htr2.std(0) + 1e-8)
    Yn = (Ytr2 - Ytr2.mean()) / (Ytr2.std() + 1e-8)
    rs = np.abs(Hn.T @ Yn / len(Yn))
    print(f"  {tg:<14}{rs.max():>10.4f}{int(rs.argmax()):>10}")

# ══════════════════════════════════════════════════
# 5. 时间尺度：h 主成分的 ACF
# ══════════════════════════════════════════════════
print("\n" + "=" * 92)
print("[5] h 主成分的时间尺度（自相关衰减）")
print("=" * 92)

# 用 PCA 投影
P = (Htr - mu) @ Vt[:8].T     # [N, 8]
# 但跨轨迹拼接，需分轨迹算 ACF
acf_lags = []
for rid in data.splits["train"][:6]:
    H = REP[rid]["h"]
    Pp = (H - mu) @ Vt[:8].T
    Pp = (Pp - Pp.mean(0)) / (Pp.std(0) + 1e-8)
    ac = []
    for lag in [1, 2, 4, 8, 16, 32]:
        c = np.mean([np.corrcoef(Pp[:-lag, i], Pp[lag:, i])[0, 1]
                     for i in range(8)])
        ac.append(c)
    acf_lags.append(ac)
acf_lags = np.array(acf_lags).mean(0)

print(f"  {'滞后[步]':>10}{'滞后[s]':>10}{'PC1-8 平均ACF':>16}")
print("  " + "-" * 38)
for lag, c in zip([1, 2, 4, 8, 16, 32], acf_lags):
    print(f"  {lag:>10}{lag*DT:>10.3f}{c:>16.4f}")

# 找 ACF 降到 0.5 的滞后
for i, c in enumerate(acf_lags):
    if c < 0.5:
        print(f"\n  ACF 降到 0.5 约需 {[1,2,4,8,16,32][i]} 步 "
              f"= {[1,2,4,8,16,32][i]*DT:.2f} s")
        break
else:
    print(f"\n  ACF 在 32 步 ({32*DT:.2f}s) 内未降到 0.5，"
          f"说明潜状态变化很慢")

print("\n  对比：")
print(f"    N2dot  lag1 ACF ≈ 0.99  （物理量本身很快）")
print(f"    h 主成分 lag1 ACF = {acf_lags[0]:.4f}")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
out = {
    "h_spectrum": {"eig_ratio": ev.tolist()[:20],
                   "cum": cum.tolist()[:20],
                   "participation": float(participation),
                   "dim_90": int(np.searchsorted(cum, 0.90)+1),
                   "dim_95": int(np.searchsorted(cum, 0.95)+1),
                   "dim_99": int(np.searchsorted(cum, 0.99)+1),
                   "n_dead": n_dead},
    "z_usage": {"n_used_per_group": n_used_arr.tolist(),
                "mean_used": float(n_used_arr.mean()),
                "entropy_per_group": all_ent,
                "mean_entropy": float(np.mean(all_ent)),
                "uniform_entropy": float(np.log(32))},
    "decodability": RES,
    "acf": {"lags": [1, 2, 4, 8, 16, 32],
            "acf": acf_lags.tolist()},
}
(OUT / "e0_latent.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'e0_latent.json'}")
