"""KL 与梯度诊断（补测，E0-E3）

回答用户三个问题：
  1. KL=1.0000 是否只是 free_bits 裁剪结果
     → 分别记录 raw KL / clipped KL / KL 梯度范数
  2. KL 项的实际有效梯度比例
  3. 区分重复推理 vs 独立训练种子
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

from evaluate import UnifiedEval, WINDOW, HORIZON
from pc_data import PCData
from pc_rssm import PCWorldModel

OUT = Path(r"C:\pclwm\pclwm_pcrssm")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

pcd = PCData()
data = pcd.d
ev = UnifiedEval(data)
C_O = data.n_obs
C_A = data.n_act

va_W = pcd.windows("val", stride=8)

print("=" * 96)
print("KL 与梯度诊断")
print("=" * 96)

CFGS = [("E0_baseline", 0.0, 0.0), ("E1_static", 0.01, 0.0),
        ("E2_dynamic", 0.0, 0.003), ("E3_both", 0.01, 0.003)]

ALL = {}
for name, ls, ld in CFGS:
    ck = OUT / "checkpoints" / f"{name}.pt"
    if not ck.exists():
        print(f"  跳过 {name}（检查点不存在）")
        continue
    wm = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                      lam_static=ls, lam_dynamic=ld).to(DEV)
    wm.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
    wm.eval()
    rssm = wm.rssm

    print(f"\n{'='*96}")
    print(f"[{name}] λ_s={ls} λ_d={ld}")
    print(f"{'='*96}")

    # ══════════════════════════════════════════════
    # 1. 在一个 val batch 上测 KL 三项
    # ══════════════════════════════════════════════
    b = {k: torch.from_numpy(np.stack([w[k] for w in va_W[:64]])).float().to(DEV)
         for k in ["obs", "act", "fut", "act_fut", "phys_now", "phys_fut"]}
    mm = torch.ones(64, b["obs"].shape[1], C_O, device=DEV)

    wm.train()   # 需要梯度
    out = wm(b["obs"], mm, b["act"], b["act_fut"], b["fut"],
             obs_true=b["obs"], phys_now=b["phys_now"],
             phys_future=b["phys_fut"], mode="gumbel")

    kl_raw = float(out["kl_raw"].detach())
    kl_clip = float(out["kl"].detach())
    kl_frac = float(out["kl_frac"].detach()) if "kl_frac" in out else float("nan")
    free_bits = rssm.free_bits

    print(f"\n  [KL 三项]  free_bits = {free_bits}")
    print(f"    kl_raw  (原始 KL 均值)      = {kl_raw:.6f}")
    print(f"    kl_clip (裁剪后，进损失)    = {kl_clip:.6f}")
    print(f"    kl_frac (超阈比例)          = {kl_frac:.4f}")
    print(f"    差值 kl_clip - free_bits    = {kl_clip - free_bits:.6f}")
    print(f"    判断: {'KL 完全被裁剪' if abs(kl_clip-free_bits)<1e-4 else 'KL 部分有效'}")

    # 逐维 KL（看有多少维真的被裁剪）
    with torch.no_grad():
        post, prior, recon, hs = rssm.observe_full(b["obs"], mm, b["act"],
                                                   "gumbel")
        kd, kr = rssm.kl_terms(post, prior)
        kl_dim = (kd + kr).mean(0).mean(0)   # [stoch] 每维的 KL
        kl_dims = kl_dim.cpu().numpy()

    above = (kl_dims > free_bits).sum()
    print(f"\n  [逐维 KL] 共 {len(kl_dims)} 维（stoch 组）")
    print(f"    KL > free_bits 的维度: {above} / {len(kl_dims)}")
    print(f"    KL 范围: [{kl_dims.min():.4f}, {kl_dims.max():.4f}]")
    print(f"    KL 中位数: {np.median(kl_dims):.4f}")
    if above == 0:
        print(f"    ★ 全部 {len(kl_dims)} 维都低于 free_bits → KL 梯度完全为 0")

    # ══════════════════════════════════════════════
    # 2. 各分量的梯度范数（在共享 cell 上）
    # ══════════════════════════════════════════════
    print(f"\n  [梯度范数] 在 rssm.cell（共享动力学参数）上")

    comps = [("rec", out["rec"], 1.0),
             ("kl_clip", out["kl"], 0.1),
             ("multi", out["multi"], 1.0)]
    if ls > 0:
        comps.append(("l_static", out["l_static"], ls))
    if ld > 0:
        comps.append(("l_dynamic", out["l_dynamic"], ld))

    gnorms = {}
    for cname, term, w in comps:
        wm.zero_grad(set_to_none=True)
        if float(term.detach()) == 0.0 or not term.requires_grad:
            continue
        (w * term).backward(retain_graph=True)
        g = rssm.cell.weight_hh.grad
        gnorms[cname] = float(g.norm()) if g is not None else 0.0

    base = gnorms.get("multi", 1e-9)
    print(f"    {'分量':<14}{'|∇|':>14}{'相对 multi':>14}")
    print("    " + "-" * 42)
    for k, v in gnorms.items():
        print(f"    {k:<14}{v:>14.4e}{v/base:>14.3f}x")

    # KL 梯度占比
    kl_g = gnorms.get("kl_clip", 0.0)
    tot = sum(gnorms.values())
    print(f"\n    KL 梯度占总梯度: {kl_g/tot*100 if tot>0 else 0:.2f}%")

    # ══════════════════════════════════════════════
    # 3. 验证 KL 是否真的"无效"
    # ══════════════════════════════════════════════
    print(f"\n  [KL 有效性验证]")
    # 计算 before/after：如果 KL 项为零，prior 是否接近均匀？
    with torch.no_grad():
        prior_logits = prior
        p = F.softmax(prior_logits, -1)
        ent = -(p * torch.log(p + 1e-12)).sum(-1).mean().item()
        print(f"    prior 平均熵 = {ent:.4f}  (均匀 = {np.log(32):.4f})")
        print(f"    差值 = {np.log(32)-ent:.4f}")
        if np.log(32) - ent < 0.15:
            print(f"    ★ prior 接近均匀 → KL 未起到约束作用")

    ALL[name] = {
        "kl_raw": kl_raw, "kl_clip": kl_clip, "kl_frac": kl_frac,
        "free_bits": free_bits,
        "kl_dim_above": int(above), "kl_dim_total": len(kl_dims),
        "kl_dim_min": float(kl_dims.min()),
        "kl_dim_max": float(kl_dims.max()),
        "kl_dim_median": float(np.median(kl_dims)),
        "gnorms": {k: float(v) for k, v in gnorms.items()},
        "kl_grad_ratio": float(kl_g / tot) if tot > 0 else 0.0,
        "prior_entropy": float(ent),
        "uniform_entropy": float(np.log(32)),
    }

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("KL 诊断汇总")
print("=" * 96)
print(f"\n  {'配置':<14}{'kl_raw':>11}{'kl_clip':>11}{'超阈维数':>10}"
      f"{'prior熵':>10}{'KL梯度占比':>12}")
print("  " + "-" * 70)
for nm, r in ALL.items():
    print(f"  {nm:<14}{r['kl_raw']:>11.6f}{r['kl_clip']:>11.6f}"
          f"{r['kl_dim_above']:>7}/{r['kl_dim_total']:<2}"
          f"{r['prior_entropy']:>10.4f}{r['kl_grad_ratio']*100:>11.2f}%")

print("\n  梯度占比（相对 multi=1.0）：")
print(f"  {'配置':<14}" + "".join(f"{k:>13}" for k in
                                  ["rec", "kl_clip", "multi",
                                   "l_static", "l_dynamic"]))
print("  " + "-" * 80)
for nm, r in ALL.items():
    b = r["gnorms"].get("multi", 1e-9)
    line = f"  {nm:<14}"
    for k in ["rec", "kl_clip", "multi", "l_static", "l_dynamic"]:
        v = r["gnorms"].get(k)
        line += f"{(v/b if v else 0):>13.3f}" if v else f"{'-':>13}"
    print(line)

(OUT / "results" / "kl_diag.json").write_text(
    json.dumps(ALL, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'results'/'kl_diag.json'}")

