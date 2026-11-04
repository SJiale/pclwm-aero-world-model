"""v3 补充实验：多种子 + 传感器噪声 + 图表数据

用户要求：
  - RSSM 多随机种子推理均值与标准差
  - 检查传感器噪声对结果的影响
"""
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
from evaluate import UnifiedData, WINDOW, DT
from mask_impl_v3 import MaskGRU, MaskRSSM, MaskWorldModel, MaskGRUModel

OUT = Path(r"C:\pclwm\pclwm_mask")
RES = OUT / "results_v3"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODE = "gumbel"
N_SEED = 5

data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
SD_T4 = float(data.obs_sd[I_T4])
MU_T4 = float(data.obs_mu[I_T4])

print("=" * 88)
print("v3 补充实验")
print("=" * 88)

# 加载模型
gm = MaskGRUModel(C_O, C_A, hidden=128, mlp=256).to(DEV)
gm.load_state_dict(torch.load(OUT/"checkpoints_v3"/"MaskGRU.pt",
                              map_location=DEV, weights_only=True))
gm.eval()
wm = MaskWorldModel(C_O, C_A, deter=128, hidden=128).to(DEV)
wm.load_state_dict(torch.load(OUT/"checkpoints_v3"/"MaskRSSM.pt",
                              map_location=DEV, weights_only=True))
wm.eval()


def run_stepwise(model, kind, rid, t_fail, dur=None, seed=42,
                 noise_sd=0.0):
    """逐步在线，可加观测噪声"""
    o = data.norm_obs(rid).astype(np.float32).copy()
    a = data.norm_act(rid).astype(np.float32)
    n = len(o)
    i_fail = int(round(t_fail / DT))
    i_end = n if dur is None else min(n, int(round((t_fail+dur)/DT)))
    if i_fail < WINDOW or i_fail >= n - 5:
        return None

    # 加噪声（归一化空间）
    if noise_sd > 0:
        rng = np.random.RandomState(seed + 1000)
        o = o + rng.randn(*o.shape).astype(np.float32) * noise_sd

    mask = np.ones((n, C_O), np.float32)
    mask[i_fail:, I_T4] = 0.0

    torch.manual_seed(seed)
    errs = []
    rssm = model.rssm if kind == "rssm" else None
    with torch.no_grad():
        h_obs = torch.from_numpy(o[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        h_mask = torch.from_numpy(mask[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        h_act = torch.from_numpy(a[i_fail-WINDOW:i_fail][None]).float().to(DEV)
        if kind == "gru":
            st = model.gru.online_init(h_obs, h_mask, h_act)
        else:
            st = rssm.online_init(h_obs, h_mask, h_act, MODE)

        for t in range(i_fail, i_end):
            ot = torch.from_numpy(o[t:t+1][None]).float().to(DEV)
            mt = torch.from_numpy(mask[t:t+1][None]).float().to(DEV)
            at = torch.from_numpy(a[t:t+1][None]).float().to(DEV).reshape(1, -1)
            if kind == "gru":
                o_hat, st = model.gru.online_step(st, ot[:, 0], mt[:, 0], at)
            else:
                o_hat, st = rssm.online_step(st, ot[:, 0], mt[:, 0], at, MODE)
            # 误差用原始（无噪声）真值
            o_clean = data.norm_obs(rid)
            e = float(o_hat[0, I_T4]) - float(o_clean[t, I_T4])
            errs.append(e * SD_T4)
    return np.array(errs)


# ══════════════════════════════════════════════════
# 1. 多种子（RSSM 的 gumbel 采样）
# ══════════════════════════════════════════════════
print("\n[1] 多种子重复（逐步在线，永久失效 @20s）")
print(f"  {'模型':<12}{'划分':<10}{'mean':>10}{'std':>9}{'min':>9}{'max':>9}")
print("  " + "-" * 60)

multi = {}
for nm, model, kind in [("MaskGRU", gm, "gru"), ("MaskRSSM", wm, "rssm")]:
    multi[nm] = {}
    for split in ["test_id", "test_ood"]:
        vals = []
        for sd in range(N_SEED):
            rs = [run_stepwise(model, kind, rid, 20.0, None, seed=1000+sd*777)
                  for rid in data.splits[split]]
            rs = [r for r in rs if r is not None]
            if rs:
                allerr = np.concatenate(rs)
                vals.append(float(np.sqrt((allerr**2).mean())))
        if vals:
            multi[nm][split] = {
                "mean": float(np.mean(vals)), "std": float(np.std(vals)),
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "n_seed": len(vals)}
            v = multi[nm][split]
            print(f"  {nm:<12}{split:<10}{v['mean']:>10.2f}{v['std']:>9.3f}"
                  f"{v['min']:>9.2f}{v['max']:>9.2f}")

# ══════════════════════════════════════════════════
# 2. 传感器噪声
# ══════════════════════════════════════════════════
print("\n[2] 传感器噪声影响（逐步在线，永久失效 @20s）")
print(f"  {'噪声sd':<10}{'MaskGRU':>12}{'MaskRSSM':>12}")
print("  " + "-" * 36)

noise_res = {}
for nsd in [0.0, 0.01, 0.05, 0.10, 0.20]:
    row = {}
    for nm, model, kind in [("MaskGRU", gm, "gru"), ("MaskRSSM", wm, "rssm")]:
        rs = [run_stepwise(model, kind, rid, 20.0, None, seed=42,
                           noise_sd=nsd)
              for rid in data.splits["test_id"]]
        rs = [r for r in rs if r is not None]
        if rs:
            allerr = np.concatenate(rs)
            row[nm] = float(np.sqrt((allerr**2).mean()))
    if row:
        noise_res[str(nsd)] = row
        print(f"  {nsd:<10.2f}{row.get('MaskGRU',0):>12.2f}"
              f"{row.get('MaskRSSM',0):>12.2f}")

# ══════════════════════════════════════════════════
# 3. 保存
# ══════════════════════════════════════════════════
out = {"multi_seed": multi, "noise": noise_res,
       "config": {"n_seed": N_SEED, "mode": MODE, "t_fail": 20.0}}
(RES / "supplementary.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {RES/'supplementary.json'}")