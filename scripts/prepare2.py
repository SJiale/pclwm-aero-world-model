"""组装 dataset2 训练数据，检查多样性"""
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path

D = Path(r"C:\pclwm\agtf30\dataset2")
OUT = Path(r"C:\pclwm\agtf30\prepared2")
OUT.mkdir(parents=True, exist_ok=True)

OBS = ["S0_W", "S0_Tt", "S0_Pt",
       "S2_Tt", "S2_Pt",
       "S36_Tt", "S36_Pt", "S36_Ts", "S36_Ps",
       "S4_Tt", "S4_Pt", "S4_FAR",
       "S45_Tt", "S45_Pt",
       "S5_Tt", "S5_Pt"]

dirs = sorted([p for p in D.iterdir() if p.is_dir() and p.name.startswith("d")])
runs, meta = {}, []
for d in dirs:
    fd, fa = d / "data.csv", d / "action.csv"
    if not (fd.exists() and fa.exists()):
        continue
    df, da = pd.read_csv(fd), pd.read_csv(fa)
    if len(df) < 5000:
        continue

    t = df["time_s"].to_numpy()
    act = da["Wfact_lb_s"].to_numpy() if len(da) == len(t) else np.interp(
        t, da["time_s"].to_numpy(), da["Wfact_lb_s"].to_numpy())

    obs = np.stack([df[c].to_numpy() for c in OBS], axis=1)
    runs[d.name] = {"obs": obs.astype(np.float32),
                    "act": act.reshape(-1, 1).astype(np.float32),
                    "time": t.astype(np.float64)}

    m = re.search(r"d(\d+)_(\w+)_a(\d+)_m(\d+)_g(\d+)", d.name)
    meta.append({"run_id": d.name,
                 "profile": m.group(2) if m else "",
                 "alt": int(m.group(3)) if m else 0,
                 "mn": int(m.group(4)) / 10 if m else 0,
                 "gs": int(m.group(5)) / 10 if m else 1,
                 "n": len(t)})

print(f"有效轨迹: {len(runs)} 条")
print(f"观测 {len(OBS)} 通道 + 1 动作")
print(f"总点数: {sum(len(v['time']) for v in runs.values()):,}")

# ── 轨迹形状多样性 ──
print()
print("=" * 88)
print("轨迹形状多样性（T4 归一化曲线两两相关）")
print("=" * 88)
N = 200
names, curves = [], []
for rid, v in runs.items():
    t4 = v["obs"][:, OBS.index("S4_Tt")]
    idx = np.linspace(0, len(t4) - 1, N).astype(int)
    c = t4[idx]
    rng = c.max() - c.min()
    if rng < 1e-6:
        continue
    curves.append((c - c.min()) / rng)
    names.append(rid)
C = np.corrcoef(np.array(curves))
iu = np.triu_indices(len(curves), 1)
vals = C[iu]
print(f"  轨迹数: {len(curves)}")
print(f"  相关系数: 中位 {np.median(vals):.3f}  范围 [{vals.min():.3f}, {vals.max():.3f}]")
print(f"  <0.9 的对: {(vals<0.9).sum()}/{len(vals)}  "
      f"({(vals<0.9).sum()/len(vals)*100:.0f}%)")
print(f"  <0.5 的对: {(vals<0.5).sum()}/{len(vals)}")
print(f"  对比 dataset1: 中位 0.993, <0.9 占 0%")

# 按 profile 分组看
print()
print("  各 profile 的轨迹数:")
from collections import Counter
cnt = Counter(m["profile"] for m in meta)
for k, v in cnt.items():
    print(f"    {k:<10} {v}")

# ── 划分：按 profile 分层，确保每种 profile 都在 train/test ──
print()
print("=" * 88)
print("数据划分")
print("=" * 88)
rng = np.random.RandomState(42)
ids = sorted(runs.keys())
by_prof = {}
for rid in ids:
    m = re.search(r"d(\d+)_(\w+)_a", rid)
    p = m.group(2) if m else "x"
    by_prof.setdefault(p, []).append(rid)

tr, va, te = [], [], []
for p, lst in by_prof.items():
    lst = sorted(lst)
    perm = rng.permutation(len(lst))
    n = len(lst)
    n_te = max(1, int(round(n * 0.25)))
    n_va = max(1, int(round(n * 0.2)))
    te += [lst[i] for i in perm[:n_te]]
    va += [lst[i] for i in perm[n_te:n_te + n_va]]
    tr += [lst[i] for i in perm[n_te + n_va:]]
    print(f"  {p:<10} train={len(lst)-n_te-n_va} val={n_va} test={n_te}")

tr, va, te = sorted(tr), sorted(va), sorted(te)
print(f"\n  合计: train={len(tr)} val={len(va)} test={len(te)}")

splits = {"train": tr, "val": va, "test": te}
np.savez_compressed(OUT / "data.npz",
                    **{f"{k}_obs": v["obs"] for k, v in runs.items()},
                    **{f"{k}_act": v["act"] for k, v in runs.items()},
                    **{f"{k}_time": v["time"] for k, v in runs.items()},
                    obs_cols=np.array(OBS), act_cols=np.array(["Wfact_lb_s"]),
                    run_ids=np.array(list(runs.keys())))
(OUT / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
(OUT / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

print(f"\n保存 -> {OUT}")
print(f"  训练点: {sum(len(runs[r]['time']) for r in tr):,}")
