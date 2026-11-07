"""阶段 III-A 完整评价（RSSM 已修复）

P0-2：定位 test_ctrl 高误差
  - 三个数据集统一协议
  - 分通道误差
  - 误差与动作幅度/工况的关系
"""
import json
import sys
import time
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path[:0] = [r"C:\pclwm\pclwm_data", r"C:\pclwm\pclwm_mask",
                r"C:\pclwm\pclwm_pcrssm", r"C:\pclwm\pclwm_experiments"]

from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON, DT

OUT = Path(r"C:\pclwm\pclwm_stage3a\results")
OUT.mkdir(parents=True, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 777

print("=" * 100)
print("阶段 III-A 完整评价（RSSM 已修复）")
print("=" * 100)

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act
obs_mu = np.asarray(data.obs_mu).ravel()
obs_sd = np.asarray(data.obs_sd).ravel()
REF = float(ev.ref_std)
act_mu = np.asarray(data.act_mu).ravel()
act_sd = np.asarray(data.act_sd).ravel()
print(f"  act_mu={act_mu}")
print(f"  act_sd={act_sd}")

print(f"  DEV={DEV}  WINDOW={WINDOW}  HORIZON={HORIZON}  DT={DT}")
print(f"  n_obs={C_O} n_act={C_A} ref_std={REF:.4f}")


# ══════════════════════════════════════════════════
# 模型定义
# ══════════════════════════════════════════════════
class GRUModel(nn.Module):
    def __init__(self, c_obs, c_act, hidden=128, horizon=HORIZON,
                 mlp=256, layers=2):
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
        H = af.shape[1]
        x = self.inp(torch.cat([obs, act], -1))
        _, h = self.gru(x)
        h = h[-1]
        o_prev = obs[:, -1]
        outs = []
        for k in range(H):
            z = self.o2h(torch.cat([o_prev, af[:, k]], -1))
            h = self.cell(z, h)
            o_hat = self.dec(h)
            outs.append(o_hat)
            o_prev = o_hat
        return torch.stack(outs, 1)


def strip(sd, pref="rssm."):
    return {k[len(pref):] if k.startswith(pref) else k: v for k, v in sd.items()}


def load_sd(p):
    sd = torch.load(p, map_location=DEV, weights_only=True)
    return sd.get("state_dict", sd) if isinstance(sd, dict) else sd


# ══════════════════════════════════════════════════
# 载入（严格）
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[载入模型 - 严格模式]")
print("=" * 100)

MODELS = {}

m = GRUModel(C_O, C_A).to(DEV)
m.load_state_dict(load_sd(r"C:\pclwm\pclwm_experiments\checkpoints\GRU-h128.pt"),
                  strict=True)
m.eval()
MODELS["GRU-h128"] = {"model": m, "type": "gru",
                      "params": sum(p.numel() for p in m.parameters())}
print(f"  ✅ GRU-h128    {MODELS['GRU-h128']['params']:>10,}  (strict=True)")

from rssm_impl import RSSM
m = RSSM(C_O, C_A, deter=128, stoch=32, classes=32, hidden=128).to(DEV)
r = m.load_state_dict(strip(load_sd(
    r"C:\pclwm\pclwm_experiments\checkpoints\RSSM-d128.pt")), strict=True)
m.eval()
MODELS["RSSM-d128"] = {"model": m, "type": "rssm",
                       "params": sum(p.numel() for p in m.parameters())}
print(f"  ✅ RSSM-d128   {MODELS['RSSM-d128']['params']:>10,}  (strict=True)")

from pc_rssm import PCWorldModel
m = PCWorldModel(C_O, C_A, deter=128, hidden=128,
                 lam_static=0.01, lam_dynamic=0.003).to(DEV)
sd_pc = load_sd(r"C:\pclwm\pclwm_pcrssm\checkpoints\seed0_E3_both.pt")
if any(k.startswith("model.") for k in sd_pc):
    sd_pc = {k[len("model."):]: v for k, v in sd_pc.items()}
r = m.load_state_dict(sd_pc, strict=True)
m.eval()
MODELS["PC-LWM-E3"] = {"model": m, "type": "pclwm",
                       "params": sum(p.numel() for p in m.parameters())}
print(f"  ✅ PC-LWM-E3   {MODELS['PC-LWM-E3']['params']:>10,}  (strict=True)")


# ══════════════════════════════════════════════════
# 统一预测接口
# ══════════════════════════════════════════════════
def predict(info, obs, act, af):
    m = info["model"]
    t = info["type"]
    if t == "gru":
        return m(obs, act, af)
    elif t == "rssm":
        return m.imagine(obs, act, af, mode="gumbel")
    else:
        mask = torch.ones_like(obs)
        return m.imagine(obs, mask, act, af, mode="gumbel")


def rel_err(pred, true):
    """[B,H,C] -> [B,H] 相对误差（与 UnifiedEval 一致）"""
    diff = (pred - true) * torch.from_numpy(obs_sd).float().to(pred.device)
    return (diff ** 2).mean(-1).sqrt() / REF


def ch_err(pred, true):
    """逐通道相对误差 {ch: rmse/sd}"""
    diff = (pred - true) * torch.from_numpy(obs_sd).float().to(pred.device)
    return {c: float((diff[:, :, i] ** 2).mean().sqrt() / obs_sd[i])
            for i, c in enumerate(data.obs_cols)}


# ══════════════════════════════════════════════════
# 数据准备
# ══════════════════════════════════════════════════
CTRL_DIR = Path(r"C:\pclwm\ctrl\test_ctrl")
STEP = 7


def load_ctrl():
    out = {}
    for f in sorted(CTRL_DIR.glob("V*.csv")):
        df = pd.read_csv(f)
        d = df.iloc[::STEP].reset_index(drop=True)
        obs = (d[["Wfact_lb_s", "N2int", "S4_Tt", "S45_Tt",
                  "S36_Pt"]].to_numpy(np.float64) - obs_mu) / obs_sd
        # ★ 正确归一化：用训练集的 act_mu / act_sd
        act = (d[["PLA_cmd_deg", "Alt_ft", "MN"]].to_numpy(np.float64)
               - act_mu) / act_sd
        out[f.stem] = {
            "obs": obs.astype(np.float32), "act": act.astype(np.float32),
            "pla": d["PLA_cmd_deg"].to_numpy(np.float64),
            "alt": float(d["Alt_ft"].iloc[0]),
            "mn": float(d["MN"].iloc[0]),
            "t": d["time_s"].to_numpy(np.float64),
        }
    return out


TC = load_ctrl()
TE_ID = data.windows("test_id", stride=8)
TE_OOD = data.windows("test_ood", stride=8)

# test_ctrl 也构造窗口（同一协议）
CTRL_WINS = {}
for k, tc in TC.items():
    obs, act = tc["obs"], tc["act"]
    n = len(obs)
    ws = []
    for t0 in range(0, n - WINDOW - HORIZON, 4):
        ws.append({
            "obs": obs[t0:t0 + WINDOW],
            "act": act[t0:t0 + WINDOW],
            "fut": obs[t0 + WINDOW:t0 + WINDOW + HORIZON],
            "act_fut": act[t0 + WINDOW:t0 + WINDOW + HORIZON],
            "rid": k,
        })
    CTRL_WINS[k] = ws

print(f"\n  test_id={len(TE_ID)} 窗口, test_ood={len(TE_OOD)} 窗口")
for k, ws in CTRL_WINS.items():
    print(f"  {k:<24} {len(ws)} 窗口")


# ══════════════════════════════════════════════════
# 评价
# ══════════════════════════════════════════════════
def eval_win(info, ws, bs=64):
    e1, eH, chs = [], [], []
    for i in range(0, len(ws), bs):
        b = ws[i:i + bs]
        o = torch.from_numpy(np.stack([w["obs"] for w in b])).float().to(DEV)
        a = torch.from_numpy(np.stack([w["act"] for w in b])).float().to(DEV)
        af = torch.from_numpy(np.stack([w["act_fut"] for w in b])).float().to(DEV)
        f = torch.from_numpy(np.stack([w["fut"] for w in b])).float().to(DEV)
        with torch.no_grad():
            torch.manual_seed(SEED)
            p = predict(info, o, a, af)
        H = min(p.shape[1], f.shape[1])
        E = rel_err(p[:, :H], f[:, :H])
        e1.extend(E[:, 0].cpu().numpy())
        eH.extend(E.mean(1).cpu().numpy())
        chs.append(ch_err(p[:, :H], f[:, :H]))
    ch = {k: float(np.mean([c[k] for c in chs])) for k in chs[0]}
    return float(np.mean(e1)), float(np.mean(eH)), ch


print("\n" + "=" * 100)
print("[1] 窗口级评价（三个数据集统一协议）")
print("=" * 100)

R = {}
for ds_name, ws in [("test_id", TE_ID), ("test_ood", TE_OOD)]:
    print(f"\n  ── {ds_name} ({len(ws)} 窗口) ──")
    print(f"  {'模型':<14}{'1-step':>11}{'32-step':>11}")
    for nm, info in MODELS.items():
        e1, eH, ch = eval_win(info, ws)
        R.setdefault(nm, {}).setdefault(ds_name, {})
        R[nm][ds_name] = {"e1": e1, "e32": eH, "per_channel": ch}
        print(f"  {nm:<14}{e1:>11.5f}{eH:>11.5f}")

print(f"\n  ── test_ctrl（6 条合计）──")
print(f"  {'模型':<14}{'1-step':>11}{'32-step':>11}")
for nm, info in MODELS.items():
    all_ws = [w for ws in CTRL_WINS.values() for w in ws]
    e1, eH, ch = eval_win(info, all_ws, bs=64)
    R[nm]["test_ctrl"] = {"e1": e1, "e32": eH, "per_channel": ch}
    print(f"  {nm:<14}{e1:>11.5f}{eH:>11.5f}")

# ══════════════════════════════════════════════════
# 分通道
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[2] 分通道 32-step 误差（rmse/sd）")
print("=" * 100)

for ds in ["test_id", "test_ood", "test_ctrl"]:
    print(f"\n  ── {ds} ──")
    print(f"  {'模型':<14}" + "".join(f"{c[:12]:>14}" for c in data.obs_cols))
    for nm in MODELS:
        ch = R[nm][ds]["per_channel"]
        print(f"  {nm:<14}" + "".join(f"{ch[c]:>14.4f}" for c in data.obs_cols))


# ══════════════════════════════════════════════════
# test_ctrl 连续推演
# ══════════════════════════════════════════════════
def eval_cont(info, tc, tgt_s):
    obs_all, act_all = tc["obs"], tc["act"]
    n = len(obs_all)
    n_steps = int(round(tgt_s / DT))
    start = WINDOW
    if n < start + n_steps:
        n_steps = n - start
        if n_steps < 10:
            return None
    oh = torch.from_numpy(obs_all[start - WINDOW:start][None]).float().to(DEV)
    ah = torch.from_numpy(act_all[start - WINDOW:start][None]).float().to(DEV)
    af = torch.from_numpy(act_all[start:start + n_steps][None]).float().to(DEV)
    tf = torch.from_numpy(obs_all[start:start + n_steps][None]).float().to(DEV)
    with torch.no_grad():
        torch.manual_seed(SEED)
        p = predict(info, oh, ah, af)
    H = min(p.shape[1], tf.shape[1])
    E = rel_err(p[:, :H], tf[:, :H])[0].cpu().numpy()
    pch = ch_err(p[:, :H], tf[:, :H])
    # 分段：前10% vs 后10%
    n10 = max(3, H // 10)
    return {"mean_rel": float(E.mean()), "final_rel": float(E[-1]),
            "early": float(E[:n10].mean()), "late": float(E[-n10:].mean()),
            "curve": E.tolist(), "per_channel": pch, "n": int(H),
            "pred": p[:, :H].cpu().numpy()[0], "true": tf[:, :H].cpu().numpy()[0]}


print("\n" + "=" * 100)
print("[3] test_ctrl 连续自主推演")
print("=" * 100)

for tgt in [10, 30, 60]:
    print(f"\n  ── {tgt}s ──")
    print(f"  {'模型':<14}" + "".join(f"{k.split('_')[0]:>10}" for k in TC) + f"{'均值':>10}")
    for nm, info in MODELS.items():
        vals = []
        row = f"  {nm:<14}"
        for k, tc in TC.items():
            r = eval_cont(info, tc, tgt)
            if r:
                vals.append(r["mean_rel"])
                row += f"{r['mean_rel']:>10.4f}"
                R[nm].setdefault(f"ctrl_{tgt}s", {})[k] = r
            else:
                row += f"{'n/a':>10}"
        row += f"{np.mean(vals):>10.4f}"
        print(row)

# ══════════════════════════════════════════════════
# 误差来源分解
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[4] ★ 误差来源分解（10s 连续推演）")
print("=" * 100)
print("  区分：初始偏差（前10%）/ 长期漂移（后10%）/ 动作响应")

print(f"\n  {'模型':<14}{'轨迹':<22}{'早期':>10}{'晚期':>10}{'增长倍数':>10}")
for nm in MODELS:
    info = MODELS[nm]
    for k, tc in TC.items():
        r = eval_cont(info, tc, 10)
        if r:
            print(f"  {nm:<14}{k:<22}{r['early']:>10.4f}{r['late']:>10.4f}"
                  f"{r['late']/max(r['early'],1e-6):>9.1f}x")
    print()

# ══════════════════════════════════════════════════
# 误差与工况/动作的关系
# ══════════════════════════════════════════════════
print("=" * 100)
print("[5] 误差与轨迹属性关系（10s 连续推演）")
print("=" * 100)

print(f"\n  {'轨迹':<22}{'Alt':>8}{'MN':>6}{'PLA范围':>14}{'dPLA_max':>10}"
      + "".join(f"{n[:10]:>12}" for n in MODELS))
for k, tc in TC.items():
    pla = tc["pla"]
    dp = np.abs(np.diff(pla)).max()
    row = f"  {k:<22}{tc['alt']:>8.0f}{tc['mn']:>6.2f}" \
          f"[{pla.min():>5.1f},{pla.max():>5.1f}]{dp:>10.3f}"
    for nm in MODELS:
        r = eval_cont(MODELS[nm], tc, 10)
        row += f"{r['mean_rel']:>12.4f}" if r else f"{'n/a':>12}"
    print(row)

# ══════════════════════════════════════════════════
# 推理延迟
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[6] 推理延迟")
print("=" * 100)

w = TE_ID[0]
o1 = torch.from_numpy(w["obs"][None]).float().to(DEV)
a1 = torch.from_numpy(w["act"][None]).float().to(DEV)
af1 = torch.from_numpy(w["act_fut"][None]).float().to(DEV)

LAT = {}
for nm, info in MODELS.items():
    for _ in range(5):
        with torch.no_grad():
            predict(info, o1, a1, af1)
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(30):
        with torch.no_grad():
            predict(info, o1, a1, af1)
    if DEV == "cuda":
        torch.cuda.synchronize()
    ms = (time.time() - t0) / 30 * 1000
    LAT[nm] = {"ms_h32": ms, "params": info["params"]}
    print(f"  {nm:<14}{ms:>9.3f} ms/horizon32  {info['params']:>10,} 参数")

# ══════════════════════════════════════════════════
# 保存
# ══════════════════════════════════════════════════
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


out = {"config": {"window": WINDOW, "horizon": HORIZON, "dt": DT,
                  "ref_std": REF, "seed": SEED,
                  "obs_cols": list(data.obs_cols)},
       "models": {k: {"params": v["params"], "type": v["type"]}
                  for k, v in MODELS.items()},
       "results": R, "latency": LAT}
(OUT / "eval_final.json").write_text(
    json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'eval_final.json'}")

