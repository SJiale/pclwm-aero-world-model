"""解析闭环日志并计算指标（绕过 MATLAB 脚本内函数限制）"""
import re
import json
from pathlib import Path
import numpy as np

IIIC = Path(r"C:\pclwm\pclwm_stage3a\iiic")
txt = (IIIC / "closedloop.txt").read_text(encoding="utf-8")
lines = txt.split("\n")

# 解析 MPC 段（行首为数字）和 PI 段（行首为 P数字）
mpc, pi = [], []
for ln in lines:
    s = ln.strip()
    m = re.match(r"^(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)$", s)
    if m:
        mpc.append([float(x) for x in m.groups()])
        continue
    m = re.match(r"^P(\d+)\s+([-\d.]+)\s+([-\d.]+)\s+([-+\d.]+)\s+([-\d.]+)\s+([\d.]+)$", s)
    if m:
        g = m.groups()
        pi.append([float(g[0]), float(g[1]), float(g[2]), float(g[3]), float(g[4]), float(g[5])])

M = np.array(mpc)
P = np.array(pi)
DW = 0.105

print("=" * 80)
print("闭环指标计算")
print("=" * 80)
print(f"  MPC 数据: {M.shape}")
print(f"  PI  数据: {P.shape}")

print("\n  配置与指标：")
print(f"  {'指标':<20}{'GRU-MPC':>14}{'PI(整定)':>14}{'MPC优势':>12}")
print("  " + "-" * 62)


def metrics(L, err_col, pla_col, ref_start=3):
    e = L[ref_start - 1:, err_col]
    pla = L[:, pla_col]
    ma = mean_abs = float(np.mean(np.abs(e)))
    iae = float(np.sum(np.abs(e)) * DW)
    ovs = float(max(0.0, np.max(e)))
    fin = float(abs(e[-1]))
    sdpla = float(np.sum(np.abs(np.diff(pla))))
    jit = float(np.std(np.diff(pla)))
    settle = -1.0
    for j in range(len(e)):
        if np.all(np.abs(e[j:]) < 5):
            settle = (j + ref_start) * DW
            break
    # 振荡指标：误差符号变化次数
    sign_chg = int(np.sum(np.diff(np.sign(e)) != 0))
    return dict(mae=ma, iae=iae, overshoot=ovs, final=fin, sdpla=sdpla,
                jitter=jit, settle=settle, sign_chg=sign_chg,
                nosc=float(np.ptp(e)))


Mm = metrics(M, 3, 4)
Mp = metrics(P, 3, 4)

rows = [
    ("MAE (rpm)", "mae", "%.2f", "low"),
    ("IAE (rpm·s)", "iae", "%.1f", "low"),
    ("超调 (rpm)", "overshoot", "%.2f", "low"),
    ("调节时间 (s)", "settle", "%.2f", "low"),
    ("稳态误差 (rpm)", "final", "%.2f", "low"),
    ("Σ|ΔPLA| (deg)", "sdpla", "%.2f", "low"),
    ("PLA抖动 std", "jitter", "%.4f", "low"),
    ("误差振荡幅 (rpm)", "nosc", "%.2f", "low"),
    ("误差符号变化", "sign_chg", "%d", "low"),
]

results = {}
for name, key, fmt, _ in rows:
    a, b = Mm[key], Mp[key]
    if key == "settle" and (a < 0 or b < 0):
        adv = "n/a"
    elif b > 0:
        adv = f"{(b-a)/b*100:+.1f}%"
    else:
        adv = "n/a"
    sa = fmt % a if key != "sign_chg" else str(int(a))
    sb = fmt % b if key != "sign_chg" else str(int(b))
    print(f"  {name:<20}{sa:>14}{sb:>14}{adv:>12}")
    results[key] = {"mpc": a, "pi": b}

# MPC 耗时
t_opt = M[:, 5]
t_sim = M[:, 6]
print("\n  === MPC 计算开销（60 步）===")
print(f"    中位 = {np.median(t_opt):.1f} ms")
print(f"    P95  = {np.percentile(t_opt, 95):.1f} ms")
print(f"    最大 = {np.max(t_opt):.1f} ms")
print(f"    最小 = {np.min(t_opt):.1f} ms")
print(f"    控制周期 = {DW*1000:.0f} ms")
n_ok = int(np.sum(t_opt < DW * 1000))
print(f"    满足预算: {n_ok}/{len(t_opt)} 步 ({n_ok/len(t_opt)*100:.1f}%)")

print("\n  === Simulink 单步（评估成本，不计入控制预算）===")
print(f"    中位 = {np.median(t_sim):.0f} ms")

out = {
    "mpc": results and Mm,
    "pi": Mp,
    "timing": {
        "median_ms": float(np.median(t_opt)),
        "p95_ms": float(np.percentile(t_opt, 95)),
        "max_ms": float(np.max(t_opt)),
        "min_ms": float(np.min(t_opt)),
        "n_ok": n_ok, "n_total": int(len(t_opt)),
        "sim_step_median_ms": float(np.median(t_sim)),
    },
    "n_steps": int(len(M)),
}
# 清理 numpy 类型
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (np.floating, np.float32, np.float64)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    return o

(IIIC / "metrics.json").write_text(
    json.dumps(clean(out), ensure_ascii=False, indent=1), encoding="utf-8")
np.savez(IIIC / "logs.npz", mpc=M, pi=P)
print(f"\n  -> {IIIC/'metrics.json'}")
print(f"  -> {IIIC/'logs.npz'}")

