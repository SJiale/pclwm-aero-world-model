"""PSVA 数据加载：在 PCData 基础上增加慢变量与 ΔN 目标

新增目标：
  y_fast  : [T, 2] 归一化 (N2dot, N3dot)  —— 加速度（快）
  y_slow  : [T, 2] 归一化 (N2int, N3int)  —— 转速（慢）
  y_delta : {k: [2]}  ΔN(t, t+k) = N(t+k) - N(t)

时间对齐：
  物理标签 8001 点（0.015s）→ 降采样 step=7 → 1143 点（0.105s）
  与 evaluate.py 的 obs 完全一致（已验证）
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
from evaluate import UnifiedData, WINDOW, HORIZON

PHYS_DIR = Path(r"C:\pclwm\phys_labels")
STEP = 7

# 列名
COL_FAST = ["loop_N2dot", "loop_N3dot"]      # 快：加速度
COL_SLOW = ["loop_N2int", "loop_N3int"]      # 慢：转速

PHYS_NAMES = {
    "fast": ["N2dot_LP", "N3dot_HP"],
    "slow": ["N2int_LP", "N3int_HP"],
}


def load_phys_all(run_ids):
    """加载并降采样，返回 rid -> dict(fast=[n,2], slow=[n,2])"""
    tag2file = {}
    for f in PHYS_DIR.glob("*.labels.csv"):
        stem = f.stem.replace(".labels", "")
        parts = stem.split("_", 1)
        tag = parts[1] if len(parts) == 2 and parts[0].isdigit() else stem
        tag2file[tag] = f

    out, missing = {}, []
    for rid in run_ids:
        if rid not in tag2file:
            missing.append(rid)
            continue
        df = pd.read_csv(tag2file[rid])
        out[rid] = {
            "fast": df[COL_FAST].to_numpy(np.float64)[::STEP].astype(np.float32),
            "slow": df[COL_SLOW].to_numpy(np.float64)[::STEP].astype(np.float32),
        }
    if missing:
        print(f"  ⚠ 缺失: {missing}")
    return out


def fit_norm(phys, train_ids, key):
    arrs = [phys[r][key] for r in train_ids if r in phys]
    A = np.concatenate(arrs, 0)
    return A.mean(0).astype(np.float64), (A.std(0).astype(np.float64) + 1e-8)


class PSVAData:
    """PSVA 专用数据：窗口 + 三组物理目标"""

    def __init__(self, data=None, ks=(1, 4, 8, 16)):
        self.d = data if data is not None else UnifiedData()
        self.ks = ks
        self.phys = load_phys_all(self.d.run_ids)

        tr = self.d.splits["train"]
        self.mu_f, self.sd_f = fit_norm(self.phys, tr, "fast")
        self.mu_s, self.sd_s = fit_norm(self.phys, tr, "slow")

        print("  物理标签归一化（train 拟合）:")
        for i, nm in enumerate(PHYS_NAMES["fast"]):
            print(f"    [fast] {nm:<10} mu={self.mu_f[i]:>10.3f} "
                  f"sd={self.sd_f[i]:>10.3f}")
        for i, nm in enumerate(PHYS_NAMES["slow"]):
            print(f"    [slow] {nm:<10} mu={self.mu_s[i]:>10.3f} "
                  f"sd={self.sd_s[i]:>10.3f}")

    def norm_fast(self, rid):
        return (self.phys[rid]["fast"] - self.mu_f) / self.sd_f

    def norm_slow(self, rid):
        return (self.phys[rid]["slow"] - self.mu_s) / self.sd_s

    def windows(self, split, stride=4, horizon=HORIZON):
        """构造窗口

        每个窗口:
          obs      [T, C_O]
          act      [T, C_A]
          fut      [H, C_O]
          act_fut  [H, C_A]
          y_fast   [T, 2]       历史段快变量（归一化）
          y_slow   [T, 2]       历史段慢变量（归一化）
          y_delta  dict{k: [2]} ΔN 目标（RAW 单位 rpm，未归一化！）
          rid, t0
        """
        ids = self.d.splits[split] if isinstance(split, str) else split
        W = []
        for r in ids:
            if r not in self.phys:
                continue
            o = self.d.norm_obs(r).astype(np.float32)
            a = self.d.norm_act(r).astype(np.float32)
            pf = self.norm_fast(r).astype(np.float32)   # [n,2]
            ps = self.norm_slow(r).astype(np.float32)   # [n,2]
            p_slow_raw = self.phys[r]["slow"]           # [n,2] rpm
            t = self.d.runs[r]["time"]
            n = len(o)
            n_p = len(pf)
            n_use = min(n, n_p)

            for st in range(1, n_use - WINDOW - horizon, stride):
                # ΔN 目标：以窗口末尾 t=st+WINDOW 为起点
                t0 = st + WINDOW
                y_delta = {}
                for k in self.ks:
                    if t0 + k < n_use:
                        dN = p_slow_raw[t0 + k] - p_slow_raw[t0]  # rpm
                        # 归一化：用慢变量的 sd
                        y_delta[k] = (dN / self.sd_s).astype(np.float32)
                if len(y_delta) < len(self.ks):
                    continue

                W.append({
                    "obs": o[st:st + WINDOW],
                    "act": a[st - 1:st - 1 + WINDOW],
                    "fut": o[st + WINDOW:st + WINDOW + horizon],
                    "act_fut": a[st - 1 + WINDOW:st - 1 + WINDOW + horizon],
                    "y_fast": pf[st:st + WINDOW],
                    "y_slow": ps[st:st + WINDOW],
                    "y_delta": y_delta,
                    "rid": r,
                    "t0": float(t[t0]),
                })
        return W


if __name__ == "__main__":
    print("=" * 84)
    print("PSVA 数据加载自检")
    print("=" * 84)

    pd_ = PSVAData()
    d = pd_.d

    tr = pd_.windows("train", stride=4)
    va = pd_.windows("val", stride=8)
    te = pd_.windows("test_id", stride=8)
    ood = pd_.windows("test_ood", stride=8)
    print(f"\n  窗口数: train={len(tr)} val={len(va)} "
          f"te={len(te)} ood={len(ood)}")

    w = tr[0]
    print(f"\n  窗口0: {w['rid']} t0={w['t0']:.3f}")
    for k in ["obs", "act", "fut", "act_fut", "y_fast", "y_slow"]:
        print(f"    {k:<10} shape={w[k].shape}")
    print(f"    y_delta   keys={list(w['y_delta'].keys())}")
    for k, v in w["y_delta"].items():
        print(f"      k={k}: shape={v.shape} value={v}")

    # 时间对齐验证
    print("\n  ★ 时间对齐验证")
    rid = w["rid"]
    o = d.norm_obs(rid)
    pf = pd_.norm_fast(rid)
    pos = int(round(w["t0"] / 0.105))
    checks = [
        ("obs      = o[pos-W:pos]", np.allclose(w["obs"], o[pos - WINDOW:pos], atol=1e-5)),
        ("y_fast   = pf[pos-W:pos]", np.allclose(w["y_fast"], pf[pos - WINDOW:pos], atol=1e-5)),
        ("fut      = o[pos:pos+H]", np.allclose(w["fut"], o[pos:pos + HORIZON], atol=1e-5)),
    ]
    for nm, ok in checks:
        print(f"    {nm:<30} {ok}")

    # ΔN 验证（物理单位）
    print("\n  ΔN 物理量验证（rpm）:")
    p_raw = pd_.phys[rid]["slow"]
    for k, v in w["y_delta"].items():
        dN_phys = p_raw[pos + k] - p_raw[pos]
        dN_recon = v * pd_.sd_s
        ok = np.allclose(dN_phys, dN_recon, atol=0.5)
        print(f"    k={k:>2}: 目标={dN_phys}  重建={dN_recon}  一致={ok}")

    # 统计
    F = np.concatenate([x["y_fast"] for x in tr], 0)
    S = np.concatenate([x["y_slow"] for x in tr], 0)
    print(f"\n  快变量（归一化后）: mean={F.mean(0)} sd={F.std(0)}")
    print(f"  慢变量（归一化后）: mean={S.mean(0)} sd={S.std(0)}")

    ok_all = all(v for _, v in checks)
    print(f"\n  自检: {'✅ 通过' if ok_all else '❌ 失败'}")
