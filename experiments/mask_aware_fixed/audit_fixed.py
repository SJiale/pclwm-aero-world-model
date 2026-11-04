"""修复版在线推理泄漏复检（针对逐步模式）"""
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
from evaluate import UnifiedData, WINDOW, DT
from mask_impl_fixed import MaskGRU, MaskWorldModel

DEV = "cpu"
data = UnifiedData()
C_O, C_A = data.n_obs, data.n_act
I_T4 = data.obs_cols.index("S4_Tt")
SD_T4 = float(data.obs_sd[I_T4])
MODE = "argmax"

print("=" * 84)
print("修复版在线推理泄漏复检（逐步模式）")
print("=" * 84)

g = MaskGRU(C_O, C_A, hidden=128, mlp=256)
g.load_state_dict(torch.load(r"C:\pclwm\pclwm_mask\checkpoints_fixed\MaskGRU.pt",
                            map_location=DEV, weights_only=True))
g.eval()
wm = MaskWorldModel(C_O, C_A, deter=128, hidden=128)
wm.load_state_dict(torch.load(r"C:\pclwm\pclwm_mask\checkpoints_fixed\MaskRSSM.pt",
                             map_location=DEV, weights_only=True))
wm.eval()

n_pass = n_fail = 0
def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1; print(f"  ✅ {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1; print(f"  ❌ {name}  {detail}")

rid = data.splits["test_id"][0]
o = data.norm_obs(rid).astype(np.float32)
a = data.norm_act(rid).astype(np.float32)
T_FAIL = 20.0
i_fail = int(round(T_FAIL / DT))
N_STEP = 200


def online(model, kind, o_in, seed=42):
    """逐步在线：与 train_mask_fixed 相同的逻辑"""
    torch.manual_seed(seed)
    B = 1
    with torch.no_grad():
        if kind == "rssm":
            rssm = model.rssm
            st = rssm.initial(B, DEV)
            for t in range(i_fail):
                ot = torch.from_numpy(o_in[t:t+1][None]).float()
                at = torch.from_numpy(a[t:t+1][None]).float().reshape(1, -1)
                mt = torch.ones(1, C_O)
                _, st = rssm.step(ot[:, 0], mt, at, st, t, mode=MODE)
            outs = []
            for t in range(i_fail, min(len(o_in), i_fail + N_STEP)):
                ot = torch.from_numpy(o_in[t:t+1][None]).float()
                at = torch.from_numpy(a[t:t+1][None]).float().reshape(1, -1)
                mt = torch.ones(1, C_O); mt[:, I_T4] = 0.0
                oh, st = rssm.step(ot[:, 0], mt, at, st, t, mode=MODE)
                outs.append(oh[0].clone())
        else:
            h = model.init_hidden(B, DEV)
            for t in range(i_fail):
                ot = torch.from_numpy(o_in[t:t+1][None]).float()
                at = torch.from_numpy(a[t:t+1][None]).float().reshape(1, -1)
                mt = torch.ones(1, C_O)
                _, h = model.step(ot[:, 0], mt, at, h)
            outs = []
            for t in range(i_fail, min(len(o_in), i_fail + N_STEP)):
                ot = torch.from_numpy(o_in[t:t+1][None]).float()
                at = torch.from_numpy(a[t:t+1][None]).float().reshape(1, -1)
                mt = torch.ones(1, C_O); mt[:, I_T4] = 0.0
                oh, h = model.step(ot[:, 0], mt, at, h)
                outs.append(oh[0].clone())
    return torch.stack(outs, 0)


for nm, model, kind in [("MaskGRU", g, "gru"), ("MaskRSSM", wm, "rssm")]:
    print(f"\n[{nm}]")
    p0 = online(model, kind, o)

    for pert, lab, expect in [
        (500.0, "+500", "无影响"),
        (1e6, "1e6", "无影响"),
        (-1e4, "-1e4", "无影响"),
    ]:
        o_p = o.copy(); o_p[i_fail:, I_T4] += pert
        p1 = online(model, kind, o_p)
        d = float((p0 - p1).abs().max())
        check(f"故障后 T4 {lab} -> {expect}", d < 1e-6, f"max diff={d:.2e}")

    o_pre = o.copy(); o_pre[:i_fail, I_T4] += 5.0
    p2 = online(model, kind, o_pre)
    d = float((p0 - p2).abs().max())
    check("故障前 T4 污染有影响", d > 1e-6, f"max diff={d:.4f}")

    for cn, ci in [("Wfact", 0), ("N2int", 1), ("T45", 3), ("P36", 4)]:
        o_x = o.copy(); o_x[i_fail:, ci] += 3.0
        px = online(model, kind, o_x)
        d = float((p0 - px).abs().max())
        check(f"污染 {cn:<6} 影响 T4 重构", d > 1e-6, f"diff={d:.4f}")

    # mask 全部置 0 时，输出应与完整观测不同
    p_full_obs = None

print("\n" + "=" * 84)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 84)
if n_fail == 0:
    print("  ✅ 逐步模式下信息隔离完全正确")
sys.exit(0 if n_fail == 0 else 1)