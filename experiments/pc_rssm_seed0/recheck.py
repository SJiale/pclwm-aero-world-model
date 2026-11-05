"""潜空间分析复核（回应三点质疑）

问题 1: PCA 参与比 2.1 不代表只有 2.1 维有用
        → 做 PCA 前 k 维对 N2dot/N3dot 的 Probe R² 对照
        → 检查动态信息是否集中在低方差维度

问题 2: z 的类别利用
        → 区分 argmax 次数 / softmax 平均概率 / 熵 / 原始KL
        → ★ 关键：z_argmax 之前被当作连续数值用，这是错的
        → 增加正确的 one-hot(1024) 对照

问题 3: h 的 ACF 非单调（lag1=0.712 < lag8=0.859）
        → 按轨迹独立计算，检查是否由拼接导致
        → 检查逐主成分的 ACF（可能的正负交替）
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
from pc_data import PCData
from pc_rssm import PCWorldModel

OUT = Path(r"C:\pclwm\pclwm_pcrssm\analysis")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cpu"

pcd = PCData()
data = pcd.d
C_O = data.n_obs

wm = PCWorldModel(C_O, 3, deter=128, hidden=128).to(DEV)
wm.load_state_dict(torch.load(
    r"C:\pclwm\pclwm_pcrssm\checkpoints\E0_baseline.pt",
    map_location=DEV, weights_only=True))
wm.eval()
rssm = wm.rssm

print("=" * 96)
print("潜空间分析复核")
print("=" * 96)


# ══════════════════════════════════════════════════
# 提取（三种 z 表示都保留）
# ══════════════════════════════════════════════════
def extract_traj(rid):
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
            # 同时保留三种表示
            zp_t = F.softmax(post, -1)                    # 概率
            za_t = post.argmax(-1)                        # 索引
            hs.append(h[0].numpy())
            zl.append(post[0].numpy())
            zp.append(zp_t[0].numpy())
            za.append(za_t[0].numpy())
            # ★ 状态用 argmax（与推理一致）
            z_hard = rssm.sample(post, "argmax")
            st = {"h": h, "z": z_hard}
    return (np.array(hs), np.array(zl), np.array(zp), np.array(za),
            o, a)


print("\n  提取潜状态...")
REP = {}
for split in ["train", "val", "test_id", "test_ood"]:
    for rid in data.splits[split]:
        hs, zl, zp, za, o, a = extract_traj(rid)
        REP[rid] = {"split": split, "h": hs, "z_logits": zl,
                    "z_prob": zp, "z_argmax": za, "obs": o, "act": a}
print(f"  完成 {len(REP)} 条")


def get_target(rid, name):
    r = REP[rid]
    if name in data.obs_cols:
        return r["obs"][:, data.obs_cols.index(name)]
    if name == "N2dot":
        return pcd.phys_raw[rid][:, 0]
    if name == "N3dot":
        return pcd.phys_raw[rid][:, 1]
    return None


def ridge(Xtr, Ytr, Xva, Yva, Xte, Yte):
    best = None
    for lam in [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]:
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


def probe_with(Xtr, Ytr, Xva, Yva, Xte, Yte):
    return ridge(Xtr, Ytr, Xva, Yva, Xte, Yte)


# ══════════════════════════════════════════════════
# 问题 1：PCA 前 k 维的 Probe
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[复核1] PCA 前 k 维对 N2dot/N3dot 的 Probe R²")
print("=" * 96)

Htr = np.concatenate([REP[r]["h"] for r in data.splits["train"]], 0)
mu = Htr.mean(0)
Xc = Htr - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
ev = S**2 / (S**2).sum()

print(f"  投影矩阵: {Vt.shape}")
print(f"\n  {'k维':>5}{'方差累积':>10}{'N2dot R²':>12}{'N3dot R²':>12}"
      f"{'T4 R²':>10}")
print("  " + "-" * 52)

res_pca = {}
for k in [2, 4, 8, 16, 32, 64, 128]:
    Pk = Vt[:k].T        # [128, k]
    def feat(r, Pk=Pk, mu=mu):
        return (r["h"] - mu) @ Pk

    line = {}
    for tg in ["N2dot", "N3dot", "S4_Tt"]:
        Xtr = np.concatenate([feat(REP[r])[skip:]
                              for r in data.splits["train"]], 0)
        Ytr = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["train"]], 0)
        Xva = np.concatenate([feat(REP[r])[skip:]
                              for r in data.splits["val"]], 0)
        Yva = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["val"]], 0)
        Xte = np.concatenate([feat(REP[r])[skip:]
                              for r in data.splits["test_id"]], 0)
        Yte = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["test_id"]], 0)
        r2t, _, _ = probe_with(Xtr, Ytr, Xva, Yva, Xte, Yte)
        line[tg] = r2t
    res_pca[k] = line
    print(f"  {k:>5}{ev[:k].sum():>10.4f}{line['N2dot']:>12.4f}"
          f"{line['N3dot']:>12.4f}{line['S4_Tt']:>10.4f}")

print(f"\n  全 128 维（原始 h）:")
for tg in ["N2dot", "N3dot", "S4_Tt"]:
    Xtr = np.concatenate([REP[r]["h"][skip:]
                          for r in data.splits["train"]], 0)
    Ytr = np.concatenate([get_target(r, tg)[skip:]
                          for r in data.splits["train"]], 0)
    Xva = np.concatenate([REP[r]["h"][skip:]
                          for r in data.splits["val"]], 0)
    Yva = np.concatenate([get_target(r, tg)[skip:]
                          for r in data.splits["val"]], 0)
    Xte = np.concatenate([REP[r]["h"][skip:]
                          for r in data.splits["test_id"]], 0)
    Yte = np.concatenate([get_target(r, tg)[skip:]
                          for r in data.splits["test_id"]], 0)
    r2t, _, _ = probe_with(Xtr, Ytr, Xva, Yva, Xte, Yte)
    print(f"    {tg:<10} R²={r2t:.4f}")

# 关键：与"只用前 2 维"对比
r2_2 = res_pca[2]["N2dot"]
r2_128 = res_pca[128]["N2dot"]
print(f"\n  ★ 关键判读:")
print(f"    前 2 维 N2dot R² = {r2_2:.4f}")
print(f"    前 128 维 N2dot R² = {r2_128:.4f}")
print(f"    提升 = {(r2_128-r2_2)*100:.1f} 个百分点")

# ══════════════════════════════════════════════════
# 问题 2：z 的类别利用（四种视角）
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[复核2] z 的类别利用（四视角）")
print("=" * 96)

Zarg = np.concatenate([REP[r]["z_argmax"] for r in data.splits["train"]], 0)
Zprob = np.concatenate([REP[r]["z_prob"] for r in data.splits["train"]], 0)
Zlog = np.concatenate([REP[r]["z_logits"] for r in data.splits["train"]], 0)
N = len(Zarg)

print(f"  样本数: {N}")
print(f"\n  {'组':>4}{'argmax类别数':>14}{'平均概率(最大)':>16}"
      f"{'平均概率(最小)':>16}{'熵':>8}{'原始KL(近似)':>14}")
print("  " + "-" * 74)

z_stats = []
for g in range(rssm.stoch):
    vals, cnts = np.unique(Zarg[:, g], return_counts=True)
    n_used = len(vals)
    p = Zprob[:, g, :].mean(0)          # 平均 softmax 概率
    ent = float(-(p * np.log(p + 1e-12)).sum())
    # 原始 KL：post vs prior 的 KL（这里用 post 自身的熵近似不可靠）
    # 更好：用 logits 的 softmax 与均匀分布的距离
    # 实际 KL = sum p*log(p/u) 其中 u 是 prior（未知），这里用均匀近似
    kl_uni = float((p * np.log(p * rssm.classes + 1e-12)).sum())
    z_stats.append({"g": g, "n_used": n_used, "entropy": ent,
                    "p_max": float(p.max()), "p_min": float(p.min()),
                    "kl_from_uniform": kl_uni})
    if g < 10 or g == rssm.stoch - 1:
        print(f"  {g:>4}{n_used:>14}{p.max():>16.4f}{p.min():>16.6f}"
              f"{ent:>8.3f}{kl_uni:>14.6f}")

print(f"\n  32 组汇总:")
print(f"    argmax 类别数: mean={np.mean([z['n_used'] for z in z_stats]):.1f}")
print(f"    平均最大概率:  mean={np.mean([z['p_max'] for z in z_stats]):.4f}")
print(f"    平均熵:        mean={np.mean([z['entropy'] for z in z_stats]):.4f}"
      f"  (均匀={np.log(32):.4f})")
print(f"    离均匀的KL:    mean={np.mean([z['kl_from_uniform'] for z in z_stats]):.6f}")

# ★ 关键：argmax 与 prob 的一致性
print(f"\n  ★ argmax 是否与概率峰值一致:")
match = 0
for g in range(rssm.stoch):
    p = Zprob[:, g, :].mean(0)
    vals, cnts = np.unique(Zarg[:, g], return_counts=True)
    top_arg = vals[cnts.argmax()]
    top_prob = int(p.argmax())
    if top_arg == top_prob:
        match += 1
print(f"    一致: {match}/{rssm.stoch}")
print(f"    说明: {'argmax 采样与平均概率峰值一致' if match>28 else '不一致，argmax 受瞬时噪声影响大'}")

# ══════════════════════════════════════════════════
# ★ z 的三种表示对比（修正之前的错误）
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[复核2b] z 三种表示的 Probe 对比（★ 修正：之前 z_argmax 被误用为连续数值）")
print("=" * 96)

FEATS_Z = {
    "z_argmax(错误:当连续值)": lambda r: r["z_argmax"],
    "z_argmax(正确:one-hot)": lambda r: np.concatenate([
        np.eye(rssm.classes)[r["z_argmax"][:, g]]
        for g in range(rssm.stoch)], 1),
    "z_prob(1024)": lambda r: r["z_prob"].reshape(len(r["z_prob"]), -1),
    "z_prob(每维argmax的onehot)": lambda r: np.concatenate([
        np.eye(rssm.classes)[r["z_prob"][:, g].argmax(-1)]
        for g in range(rssm.stoch)], 1),
}

print(f"\n  {'目标':<10}" + "".join(f"{k:>26}" for k in FEATS_Z))
print("  " + "-" * (10 + 26*len(FEATS_Z)))
res_z = {}
for tg in ["N2dot", "N3dot", "S4_Tt"]:
    res_z[tg] = {}
    line = f"  {tg:<10}"
    for fn, ff in FEATS_Z.items():
        Xtr = np.concatenate([ff(REP[r])[skip:]
                              for r in data.splits["train"]], 0)
        Ytr = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["train"]], 0)
        Xva = np.concatenate([ff(REP[r])[skip:]
                              for r in data.splits["val"]], 0)
        Yva = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["val"]], 0)
        Xte = np.concatenate([ff(REP[r])[skip:]
                              for r in data.splits["test_id"]], 0)
        Yte = np.concatenate([get_target(r, tg)[skip:]
                              for r in data.splits["test_id"]], 0)
        r2t, _, _ = probe_with(Xtr, Ytr, Xva, Yva, Xte, Yte)
        res_z[tg][fn] = r2t
        line += f"{r2t:>26.4f}"
    print(line)

# ══════════════════════════════════════════════════
# 问题 3：h 的 ACF（按轨迹独立计算）
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[复核3] h 的 ACF（按轨迹独立计算，不拼接）")
print("=" * 96)

lags = [1, 2, 4, 8, 16, 32]

# 3a. 逐主成分的 ACF（找非单调原因）
print(f"\n  3a. 逐主成分 ACF（train 前 6 条均值）")
print(f"  {'PC':>4}" + "".join(f"{f'lag{l}':>9}" for l in lags))
print("  " + "-" * 58)

per_pc_acf = {}
for pc in range(8):
    acs_lag = []
    for lag in lags:
        vals = []
        for rid in data.splits["train"][:6]:
            H = REP[rid]["h"]
            p = (H - mu) @ Vt[pc]      # [T]
            if p.std() < 1e-9:
                continue
            p = (p - p.mean()) / p.std()
            if len(p) > lag + 2:
                c = np.corrcoef(p[:-lag], p[lag:])[0, 1]
                if np.isfinite(c):
                    vals.append(c)
        acs_lag.append(float(np.mean(vals)) if vals else np.nan)
    per_pc_acf[pc] = acs_lag
    print(f"  {pc+1:>4}" + "".join(f"{c:>9.4f}" for c in acs_lag))

# 3b. 全体 h 标准化后的 ACF（逐轨迹）
print(f"\n  3b. 全体 128 维逐维 ACF 再平均（逐轨迹，不拼接）")
print(f"  {'轨迹':<18}{'lag1':>9}{'lag2':>9}{'lag4':>9}"
      f"{'lag8':>9}{'lag16':>9}{'lag32':>9}")
print("  " + "-" * 72)

traj_acf = {}
for rid in data.splits["train"]:
    H = REP[rid]["h"]
    Hn = (H - mu) / (Htr.std(0) + 1e-8)
    acs = []
    for lag in lags:
        if len(Hn) <= lag + 2:
            acs.append(np.nan); continue
        cs = []
        for d in range(Hn.shape[1]):
            c = np.corrcoef(Hn[:-lag, d], Hn[lag:, d])[0, 1]
            if np.isfinite(c):
                cs.append(c)
        acs.append(float(np.mean(cs)) if cs else np.nan)
    traj_acf[rid] = acs
    if rid in data.splits["train"][:6]:
        print(f"  {rid:<18}" + "".join(f"{c:>9.4f}" for c in acs))

all_acf = np.array([v for v in traj_acf.values()])
print(f"  {'均值':<18}" + "".join(f"{np.nanmean(all_acf[:,i]):>9.4f}"
                                  for i in range(len(lags))))
print(f"  {'标准差':<18}" + "".join(f"{np.nanstd(all_acf[:,i]):>9.4f}"
                                  for i in range(len(lags))))
print(f"  {'轨迹数':<18}{len(all_acf)}")

# 3c. 检查非单调
mean_acf = [float(np.nanmean(all_acf[:, i])) for i in range(len(lags))]
print(f"\n  3c. 单调性检查:")
mono = all(mean_acf[i] >= mean_acf[i+1] for i in range(len(mean_acf)-1))
print(f"    序列: {[f'{c:.4f}' for c in mean_acf]}")
print(f"    单调递减: {mono}")
if not mono:
    print(f"    ⚠ 非单调，检查各轨迹:")
    for rid, ac in list(traj_acf.items())[:6]:
        m = all(ac[i] >= ac[i+1] for i in range(len(ac)-1))
        print(f"      {rid:<18} 单调={m}  {[f'{c:.3f}' for c in ac]}")

# 3d. 与之前拼接方式的对比
print(f"\n  3d. 之前拼接方式的错误来源:")
print(f"    之前把 17 条轨迹的 h 直接 concat 后算 ACF")
print(f"    → 轨迹边界处出现跳变，污染 lag1")
Hcat = np.concatenate([REP[r]["h"] for r in data.splits["train"]], 0)
Hcat_n = (Hcat - mu) / (Htr.std(0) + 1e-8)
cat_acf = []
for lag in lags:
    cs = [np.corrcoef(Hcat_n[:-lag, d], Hcat_n[lag:, d])[0, 1]
          for d in range(Hcat_n.shape[1])]
    cat_acf.append(float(np.nanmean(cs)))
print(f"    拼接方式 ACF: {[f'{c:.4f}' for c in cat_acf]}")
print(f"    逐轨迹方式 ACF: {[f'{c:.4f}' for c in mean_acf]}")
print(f"    lag1 差异: {cat_acf[0]:.4f} vs {mean_acf[0]:.4f}")

# ══════════════════════════════════════════════════
# 额外：采样策略一致性核查
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[额外] z 采样策略一致性核查")
print("=" * 96)

print(f"""
  训练时: rssm.observe(...) → sample(post_logits, mode="gumbel")
  推理时: rssm.imagine(...) → sample(prior_logits, mode="gumbel")
  分析时: 上面用 argmax 提取（与训练/推理不一致！）

  ★ 这是一个需要澄清的点：
    - 训练/推理：gumbel(hard=True) → one-hot
    - 本分析：argmax → one-hot
    两者都产生 one-hot，但 gumbel 有随机性，argmax 是确定性的。

    对于【评价】应该用 argmax（确定性，可复现）
    对于【训练】用 gumbel（可微）
""")

# 检查 gumbel 与 argmax 的一致率
print("  gumbel vs argmax 一致率测试:")
o = data.norm_obs(data.splits["test_id"][0]).astype(np.float32)
a = data.norm_act(data.splits["test_id"][0]).astype(np.float32)
n_t = min(200, len(o))
mask = np.ones((n_t, C_O), np.float32)
agree = []
with torch.no_grad():
    st = rssm.init_state(1, DEV)
    for t in range(n_t):
        z_prev = st["z"].reshape(1, -1)
        at = torch.from_numpy(a[t:t+1]).float()
        h = rssm.cell(torch.cat([z_prev, at], -1), st["h"])
        emb = rssm.enc(torch.cat([
            torch.from_numpy(o[t:t+1]).float(),
            torch.from_numpy(mask[t:t+1]).float()], -1))
        post = rssm.post(torch.cat([torch.cat([h, z_prev], -1), emb],
                                   -1)).reshape(1, rssm.stoch, rssm.classes)
        z_gum = rssm.sample(post, "gumbel")
        z_arg = rssm.sample(post, "argmax")
        g_idx = z_gum[0].argmax(-1).numpy()
        a_idx = z_arg[0].argmax(-1).numpy()
        agree.append((g_idx == a_idx).mean())
        st = {"h": h, "z": z_arg}
print(f"    单组一致率均值: {np.mean(agree):.4f}")
print(f"    说明: gumbel 在分布接近均匀时随机性大")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
out = {
    "pca_probe": {str(k): v for k, v in res_pca.items()},
    "z_stats": z_stats,
    "z_probe": res_z,
    "acf_per_pc": {str(k): v for k, v in per_pc_acf.items()},
    "acf_per_traj": {r: v for r, v in traj_acf.items()},
    "acf_mean": mean_acf,
    "acf_std": [float(np.nanstd(all_acf[:, i])) for i in range(len(lags))],
    "acf_concat": cat_acf,
    "z_sample_agreement": float(np.mean(agree)),
}
(OUT / "e0_latent_v2.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'e0_latent_v2.json'}")
