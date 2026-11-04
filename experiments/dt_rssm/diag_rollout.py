"""诊断：连续递推误差为何比短期更小（异常）

异常现象（RSSM-d128）：
  test_id 短期 32 步（3.36s）: 0.0898
  连续递推 60s:               0.0128   <- 矛盾！

自由递推误差不可能小于短期误差。必有 bug。

本脚本逐步对比两种评价，打印每步误差与中间量。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))

from evaluate import UnifiedData, UnifiedEval, rollout_far, WINDOW, DT
from rssm_impl import load_rssm

DEV = "cpu"
data = UnifiedData()
C_O = data.n_obs
rid = data.splits["test_id"][0]
model = load_rssm(r"C:\pclwm\pclwm_dt\checkpoints\RSSM-d128.pt",
                  C_O, 3, deter=128, hidden=128, device=DEV)
ref = float(data.obs_sd.mean())

print("=" * 84)
print("诊断：连续递推 vs 分段递推")
print("=" * 84)
print(f"  rid={rid}  ref_std={ref:.2f}")

o = data.norm_obs(rid).astype(np.float32)
a = data.norm_act(rid).astype(np.float32)
W = WINDOW
print(f"  轨迹长度 {len(o)} 步 ({len(o)*DT:.1f}s)")

# ══════════════════════════════════════════════════
# A. 短期 32 步（用统一评价器）
# ══════════════════════════════════════════════════
ev = UnifiedEval(data)
print("\n[A] 短期 32 步（unified evaluator）")
ws = [w for w in data.windows([rid], stride=8)]
p = []
with torch.no_grad():
    for w in ws:
        ob = torch.from_numpy(w["obs"][None]).float()
        ac = torch.from_numpy(w["act"][None]).float()
        af = torch.from_numpy(w["act_fut"][None]).float()
        torch.manual_seed(1000)
        p.append(model.imagine(ob, ac, af, mode="gumbel").numpy()[0])
p = np.array(p)
r = ev.evaluate(p, ws)
print(f"  overall_rel = {r['overall_rel']:.4f}")
print(f"  k1_rel      = {r['k1_rel']:.4f}   (第 1 步)")
print(f"  kH_rel      = {r['kH_rel']:.4f}   (第 32 步)")

# ══════════════════════════════════════════════════
# B. 分段递推（rollout_far）
# ══════════════════════════════════════════════════
print("\n[B] 分段递推 rollout_far(60s)")


def mfn(obs, act, af):
    with torch.no_grad():
        torch.manual_seed(1000)
        return model.imagine(torch.from_numpy(obs).float(),
                             torch.from_numpy(act).float(),
                             torch.from_numpy(af).float(),
                             mode="gumbel").numpy()


rs = rollout_far(mfn, data, rid, 60.0)
print(f"  nseg={rs['nseg']}  steps={rs['steps']}  "
      f"seconds={rs['seconds']:.1f}")
print(f"  mean_rel={rs['mean_rel']:.4f}  final_rel={rs['final_rel']:.4f}")
c = np.array(rs["curve_rel"])
print(f"  曲线前5: {c[:5].round(4)}")
print(f"  曲线中5: {c[len(c)//2:len(c)//2+5].round(4)}")
print(f"  曲线后5: {c[-5:].round(4)}")

# ══════════════════════════════════════════════════
# C. 连续递推（手动复现 long_continuous）
# ══════════════════════════════════════════════════
print("\n[C] 连续递推（复现 long_continuous 逻辑）")
n_steps = int(round(60.0 / DT))
start = W + 1
rssm = model.rssm
torch.manual_seed(42)
errs = []
with torch.no_grad():
    obs_h = torch.from_numpy(o[start-W:start][None]).float()
    act_h = torch.from_numpy(a[start-W-1:start-1][None]).float()
    state = rssm.initial(1, DEV)
    for t in range(W):
        pz = state["z"].reshape(1, -1)
        h = rssm.cell(torch.cat([pz, act_h[:, t]], -1), state["h"])
        embed = rssm.enc(obs_h[:, t])
        pl = rssm.post(torch.cat([h, embed], -1)).reshape(
            1, rssm.stoch, rssm.classes)
        z = rssm.sample(pl, "gumbel")
        state = {"h": h, "z": z}

    for k in range(n_steps):
        pz = state["z"].reshape(1, -1)
        a_t = torch.from_numpy(a[start-1+k:start+k][None]).float()
        a_t = a_t.reshape(1, -1)
        h = rssm.cell(torch.cat([pz, a_t], -1), state["h"])
        pl = rssm.prior(h).reshape(1, rssm.stoch, rssm.classes)
        z = rssm.sample(pl, "gumbel")
        state = {"h": h, "z": z}
        pred = rssm.dec(rssm.feat(state))
        gt = torch.from_numpy(o[start+k:start+k+1][None]).float()
        errs.append((pred - gt).numpy()[0])

E = np.array(errs) * data.obs_sd[None, :]
ps = np.sqrt((E**2).mean(1))
print(f"  n_steps={n_steps}")
print(f"  mean_rel={float(ps.mean())/ref:.4f}  final_rel={float(ps[-1])/ref:.4f}")
print(f"  曲线前5: {(ps[:5]/ref).round(4)}")
print(f"  曲线中5: {(ps[len(ps)//2:len(ps)//2+5]/ref).round(4)}")
print(f"  曲线后5: {(ps[-5:]/ref).round(4)}")

# ══════════════════════════════════════════════════
# D. 关键对比：对齐同一时间点
# ══════════════════════════════════════════════════
print("\n[D] 对齐同一时间点的误差")
print(f"  {'步数':>6}{'时间[s]':>10}{'分段':>12}{'连续':>12}")
print("  " + "-" * 42)
for k in [0, 31, 63, 95, 191, 287, 383, 479, 570]:
    if k >= len(ps):
        continue
    seg_v = c[k] if k < len(c) else float("nan")
    con_v = ps[k] / ref
    print(f"  {k:>6}{k*DT:>10.1f}{seg_v:>12.4f}{con_v:>12.4f}")

print("\n" + "=" * 84)
print("诊断判据")
print("=" * 84)
print(f"""
  若连续递推的第 1 步误差就远小于分段的第 1 步 -> 初始化方式不同
  若连续递推误差不随时间增长 -> 有真值泄漏
  若两者起点相同但走势不同 -> 递推机制不同（需进一步查）

  实测：
    分段  第0步 {c[0]:.4f}
    连续  第0步 {ps[0]/ref:.4f}
    分段  末步 {c[-1]:.4f}
    连续  末步 {ps[-1]/ref:.4f}
""")

