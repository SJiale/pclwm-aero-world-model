"""组装 AGTF30e 训练数据：obs + action + 划分"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import re

D = Path(r"C:\pclwm\agtf30\dataset")
OUT = Path(r"C:\pclwm\agtf30\prepared")
OUT.mkdir(parents=True, exist_ok=True)

# 选观测通道：覆盖各截面关键物理量
OBS = [
    "S0_W", "S0_Tt", "S0_Pt",
    "S2_Tt", "S2_Pt",
    "S36_Tt", "S36_Pt", "S36_Ts", "S36_Ps",
    "S4_Tt", "S4_Pt", "S4_FAR",
    "S45_Tt", "S45_Pt",
    "S5_Tt", "S5_Pt",
]
ACT = ["Wfact_lb_s"]

dirs = sorted([p for p in D.iterdir() if p.is_dir() and p.name.startswith("r")])

runs = {}
meta = []
for d in dirs:
    fd = d / "data.csv"
    fa = d / "action.csv"
    if not (fd.exists() and fa.exists()):
        continue
    df = pd.read_csv(fd)
    da = pd.read_csv(fa)

    # 只保留完整轨迹（>=100 s）
    if df["time_s"].iloc[-1] < 100:
        print(f"  SKIP {d.name}: 仅 {df['time_s'].iloc[-1]:.1f} s")
        continue

    t = df["time_s"].to_numpy()
    # 对齐 action（同一时间轴）
    if len(da) != len(t) or not np.allclose(da["time_s"].to_numpy(), t):
        ta = da["time_s"].to_numpy()
        act = np.interp(t, ta, da["Wfact_lb_s"].to_numpy())
    else:
        act = da["Wfact_lb_s"].to_numpy()

    cols = []
    miss = []
    for c in OBS:
        if c in df.columns:
            cols.append(df[c].to_numpy())
        else:
            miss.append(c)
    if miss:
        print(f"  WARN {d.name}: 缺 {miss}")
    obs = np.stack(cols, axis=1)

    runs[d.name] = {"obs": obs.astype(np.float32),
                    "act": act.reshape(-1, 1).astype(np.float32),
                    "time": t.astype(np.float64)}
    m = re.search(r"a(\d+)_m(\d+)_p(\d+)_g(\d+)", d.name)
    meta.append({"run_id": d.name,
                 "alt": int(m.group(1)) if m else 0,
                 "mn": int(m.group(2)) / 10 if m else 0,
                 "pla": int(m.group(3)) if m else 0,
                 "gs": int(m.group(4)) / 10 if m else 1,
                 "n": len(t), "t_end": float(t[-1])})

print(f"\n有效轨迹: {len(runs)} 条")
print(f"观测通道: {len(OBS)} 个")
print(f"动作: {ACT}")

# 划分：按工况分层（不同 Alt/MN/PLA）
# 目标：train 覆盖主要工况，val/test 留出不同工况
ids = sorted(runs.keys())
rng = np.random.RandomState(42)
perm = rng.permutation(len(ids))
n_tr = int(len(ids) * 0.6)
n_va = int(len(ids) * 0.2)
train_ids = sorted([ids[i] for i in perm[:n_tr]])
val_ids = sorted([ids[i] for i in perm[n_tr:n_tr + n_va]])
test_ids = sorted([ids[i] for i in perm[n_tr + n_va:]])

splits = {"train": train_ids, "val": val_ids, "test": test_ids}
print(f"\n划分: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

# 保存
np.savez_compressed(OUT / "data.npz",
                    **{f"{k}_obs": v["obs"] for k, v in runs.items()},
                    **{f"{k}_act": v["act"] for k, v in runs.items()},
                    **{f"{k}_time": v["time"] for k, v in runs.items()},
                    obs_cols=np.array(OBS), act_cols=np.array(ACT),
                    run_ids=np.array(list(runs.keys())))

(OUT / "splits.json").write_text(json.dumps(splits, indent=1), encoding="utf-8")
(OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                               encoding="utf-8")

print(f"\n保存 -> {OUT}")
print(f"  总点数: {sum(len(v['time']) for v in runs.values()):,}")
print(f"  每点: {len(OBS)} obs + {len(ACT)} act")

# 快速检查各通道范围
print(f"\n=== 各通道范围 ===")
allobs = np.concatenate([v["obs"] for v in runs.values()], 0)
print(f"{'通道':<12}{'min':>12}{'max':>12}{'std':>12}")
print("-" * 48)
for i, c in enumerate(OBS):
    x = allobs[:, i]
    print(f"{c:<12}{x.min():>12.4g}{x.max():>12.4g}{x.std():>12.4g}")

# 线性度检验（关键）
print(f"\n=== 一步线性回归检验 ===")
N, C = allobs.shape
X_list, Y_list = [], []
for rid, v in runs.items():
    o = v["obs"]
    a = v["act"]
    # 归一化后回归
    X_list.append(np.c_[o[:-1], a[:-1]])
    Y_list.append(o[1:])
X = np.concatenate(X_list, 0)
Y = np.concatenate(Y_list, 0)
mu = X.mean(0); sd = X.std(0) + 1e-12
Xn = (X - mu) / sd
Yn = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)
W = np.linalg.lstsq(np.c_[Xn, np.ones(len(Xn))],
                    Yn, rcond=None)[0]
pred = np.c_[Xn, np.ones(len(Xn))] @ W
r2 = 1 - ((Yn - pred) ** 2).sum() / ((Yn - Yn.mean(0)) ** 2).sum()
print(f"  一步预测 R2 = {r2:.6f}")
print(f"  对比 Model4MW: 0.999977（几乎线性）")
print(f"  >>> {'AGTF30e 非线性更强' if r2 < 0.999 else '仍然线性'}")
