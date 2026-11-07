"""实验 A：用真实执行动作序列回放 GRU
实验 C：计划 vs 实际执行偏差分析
"""
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

CD = Path(r"C:\pclwm\pclwm_stage3a\causal")
OUT = CD
DT = 0.105
DEV = "cpu"

# ══ 载入 ══
S = pd.read_csv(CD / "steps.csv")
PS = pd.read_csv(CD / "pla_seq_matrix.csv", header=None).to_numpy()
P = pd.read_csv(CD / "pred_n2_matrix.csv", header=None).to_numpy()

cfg = {}
for ln in (CD / "cfg.txt").read_text().split("\n"):
    if "=" in ln:
        k, v = ln.split("=")
        cfg[k] = float(v)
N2_REF = cfg["n2_ref"]
H = int(cfg["H"])
N = len(S)

print("=" * 96)
print("实验 A + C：GRU 真实动作回放 + 计划执行偏差")
print("=" * 96)
print(f"  N={N}, H={H}, N2_ref={N2_REF:.2f}")

# ══ GRU 模型 ══
import scipy.io as sio
G = sio.loadmat(r"C:\pclwm\pclwm_stage3a\gru_pack2.mat")
obs_mu = G["obs_mu"].ravel()
obs_sd = G["obs_sd"].ravel()
act_mu = G["act_mu"].ravel()
act_sd = G["act_sd"].ravel()


class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=H, mlp=256, layers=2):
        super().__init__()
        self.horizon = horizon
        self.inp = nn.Sequential(
            nn.Linear(c_obs + c_act, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.gru = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.cell = nn.GRUCell(hidden, hidden)
        mods = [nn.Linear(hidden, mlp), nn.SiLU()]
        for _ in range(layers - 1):
            mods += [nn.Linear(mlp, mlp), nn.SiLU()]
        mods += [nn.Linear(mlp, c_obs)]
        self.dec = nn.Sequential(*mods)
        self.o2h = nn.Linear(c_obs + c_act, hidden)

    def forward(self, obs, act, af):
        B, T, _ = obs.shape
        Hh = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(Hh):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


sd = torch.load(r"C:\pclwm\pclwm_experiments\checkpoints\GRU-h128.pt",
                map_location=DEV, weights_only=True)
if isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
GRU = GRUModel(5, 3).to(DEV)
GRU.load_state_dict(sd, strict=True)
GRU.eval()
print(f"  ✅ GRU strict=True, {sum(p.numel() for p in GRU.parameters()):,} 参数")

# ══ 重建历史观测（5通道）══
OBS_ALL = S[["obs_wf", "obs_n2", "obs_t4", "obs_t45", "obs_p36"]].to_numpy()
# obs_all[k-1] = 第 k 步 MPC 计算【前】的观测
# 历史窗口：obs_all[k-1-WINDOW+1 : k-1+1]，动作同理

WINDOW = 32
ACT_ALL = np.zeros((N, 3))
ACT_ALL[:, 0] = S["pla_apply"].to_numpy()
ACT_ALL[:, 1] = 0.0
ACT_ALL[:, 2] = 0.0

# 稳态初值（用于 k<WINDOW 时的填充）
obs_steady = OBS_ALL[0]
act_steady = np.array([55.0, 0.0, 0.0])

print("\n" + "=" * 96)
print("实验 A：用【真实执行动作序列】回放 GRU")
print("=" * 96)
print(f"  协议：历史=真实观测；未来动作=实际执行的 PLA（不是计划）")
print(f"  预测窗口按剩余真实数据截断")

# 关键时刻
KEY_K = [9, 10, 14, 17, 20, 25]

resA = []
for k in KEY_K:
    i = k - 1
    # 历史窗口
    lo = i - WINDOW + 1
    if lo < 0:
        pad = -lo
        obs_h = np.vstack([np.tile(obs_steady, (pad, 1)), OBS_ALL[:i+1]])
        act_h = np.vstack([np.tile(act_steady, (pad, 1)), ACT_ALL[:i+1]])
    else:
        obs_h = OBS_ALL[lo:i+1]
        act_h = ACT_ALL[lo:i+1]

    # 未来动作 = 实际执行（从 k+1 步开始，因为我们已执行了第 k 步）
    # 注意：第 k 步的 pla_apply 是刚决定的，实际执行发生在本步
    # 历史窗口末端的动作应该是 pla_prev（第 k-1 步执行的）
    # 所以历史 act 用 ACT_ALL[..i] 的最后一项应该是 pla_prev
    # 简化：用 ACT_ALL[:i+1]，最后一项是第 k 步的 pla_apply（已执行）
    # 未来从 i+1 开始的真实动作
    n_fut = min(H, N - (i + 1))
    act_f = ACT_ALL[i+1:i+1+n_fut]
    true_n2 = OBS_ALL[i+1:i+1+n_fut, 1]

    if n_fut < 3:
        print(f"  k={k} 剩余数据不足，跳过")
        continue

    o = torch.from_numpy(((obs_h - obs_mu) / obs_sd)[None]).float()
    a = torch.from_numpy(((act_h - act_mu) / act_sd)[None]).float()
    af = torch.from_numpy(((act_f - act_mu) / act_sd)[None]).float()

    with torch.no_grad():
        torch.manual_seed(777)
        p = GRU(o, a, af)
    pred_n2 = (p[0].numpy() * obs_sd + obs_mu)[:, 1]

    e = pred_n2 - true_n2
    dn2_pred = pred_n2[-1] - obs_h[-1, 1]
    dn2_true = true_n2[-1] - obs_h[-1, 1]

    print(f"\n  ── k={k}, t={S['t'].iloc[i]:.2f}s, PLA_apply={S['pla_apply'].iloc[i]:.2f} ──")
    print(f"    历史末 N2 = {obs_h[-1,1]:.1f}")
    print(f"    可用未来步数 = {n_fut}")
    print(f"    {'步':>4}{'真实t':>8}{'预测N2':>10}{'真实N2':>10}{'误差':>10}")
    for kk in list(range(0, min(n_fut, 10), 3)) + [n_fut-1]:
        tt = S['t'].iloc[i] + (kk+1)*DT
        print(f"    {kk+1:>4}{tt:>8.2f}{pred_n2[kk]:>10.1f}{true_n2[kk]:>10.1f}"
              f"{e[kk]:>+10.1f}")

    resA.append({
        "k": k, "t": float(S["t"].iloc[i]), "n_fut": int(n_fut),
        "n2_start": float(obs_h[-1, 1]),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "err_1": float(e[0]), "err_last": float(e[-1]),
        "dn2_pred": float(dn2_pred), "dn2_true": float(dn2_true),
        "pred_peak": float(np.max(pred_n2)),
        "true_peak": float(np.max(true_n2)),
        "pred_curve": pred_n2.tolist(),
        "true_curve": true_n2.tolist(),
        "pla_curve": act_f[:, 0].tolist(),
    })

print("\n" + "=" * 96)
print("[A 汇总]")
print("=" * 96)
print(f"\n  {'k':>3}{'t':>7}{'步数':>6}{'MAE':>9}{'有符号均值':>12}"
      f"{'步1误差':>10}{'末步误差':>10}")
print("  " + "-" * 62)
for r in resA:
    print(f"  {r['k']:>3}{r['t']:>7.2f}{r['n_fut']:>6}{r['mae']:>9.2f}"
          f"{r['bias']:>+12.2f}{r['err_1']:>+10.2f}{r['err_last']:>+10.2f}")

if resA:
    all_mae = np.mean([r["mae"] for r in resA])
    all_bias = np.mean([r["bias"] for r in resA])
    print(f"\n  总体 MAE = {all_mae:.2f} rpm, 平均有符号误差 = {all_bias:+.2f} rpm")

    print(f"\n  {'k':>3}{'ΔN2预测':>12}{'ΔN2真实':>12}{'低估量':>10}")
    print("  " + "-" * 40)
    for r in resA:
        under = r["dn2_true"] - r["dn2_pred"]
        print(f"  {r['k']:>3}{r['dn2_pred']:>+12.1f}{r['dn2_true']:>+12.1f}"
              f"{under:>+10.1f}")

# ══════════════════════════════════════════════════
# 实验 C：计划 vs 实际执行
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("实验 C：计划 PLA vs 实际执行 PLA（按绝对时间对齐）")
print("=" * 96)

print(f"\n  k=10（t={S['t'].iloc[9]:.2f}s）时计划的未来 PLA 轨迹:")
plan10 = PS[9]
print(f"    {'step':>5}{'绝对t':>9}{'计划PLA':>10}{'实际PLA':>10}"
      f"{'偏差':>10}{'是否执行':>10}")
print("  " + "-" * 58)
for j in range(1, min(H, N - 9)):
    abs_k = 9 + j    # k=10 之后的第 j 步（0-based: 9+j）
    if abs_k >= N:
        break
    plan = plan10[j]
    actual = S["pla_apply"].iloc[abs_k]
    dev = actual - plan
    print(f"    {j:>5}{S['t'].iloc[abs_k]:>9.2f}{plan:>10.2f}"
          f"{actual:>10.2f}{dev:>+10.2f}{'✅' if abs(dev)<0.5 else '❌重规划':>10}")

# 统计：计划降 PLA vs 实际降
print(f"\n  ── 关键检查：k=10 计划 PLA[32]={plan10[31]:.2f}（应降），"
      f"实际到 k=30 时 PLA={S['pla_apply'].iloc[-1]:.2f} ──")

# 每步的计划"未来最小PLA"vs 实际
print(f"\n  逐步：计划的最小未来 PLA vs 后续实际最小值")
print(f"  {'k':>3}{'t':>7}{'当前PLA':>9}{'计划min':>9}{'计划末':>9}"
      f"{'实际后续min':>13}{'推迟?':>8}")
print("  " + "-" * 62)
defer_count = 0
for i in range(N):
    ps = PS[i]
    cur = S["pla_apply"].iloc[i]
    plan_min = float(np.min(ps))
    plan_end = float(ps[-1])
    # 实际后续
    fut = S["pla_apply"].iloc[i+1:].to_numpy() if i+1 < N else np.array([])
    if len(fut) == 0:
        continue
    act_min = float(np.min(fut))
    # 若计划末值 < 当前，但实际后续最小值 > 计划末值 -> 推迟
    deferred = (plan_end < cur - 0.5) and (act_min > plan_end + 0.5)
    if deferred:
        defer_count += 1
    if i % 3 == 0 or deferred:
        print(f"  {int(S['k'].iloc[i]):>3}{S['t'].iloc[i]:>7.2f}{cur:>9.2f}"
              f"{plan_min:>9.2f}{plan_end:>9.2f}{act_min:>13.2f}"
              f"{'❗是' if deferred else '':>8}")

print(f"\n  '计划降 PLA 但实际未达计划值'的步数: {defer_count} / {N}")

# ══ 保存 ══
out = {"A": resA, "C_defer_count": defer_count, "C_n_steps": N}
(OUT / "offline_ablation.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'offline_ablation.json'}")
