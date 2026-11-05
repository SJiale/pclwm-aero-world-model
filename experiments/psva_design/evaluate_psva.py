"""PSVA 评价：交叉 Probe + Separation 指标 + E3 参考切分

修正（按用户意见 2）：
  Separation = 同一物理目标在快慢分支间的 R² 差值

  对快变量（Ndot）: Sep_fast = R²(h_f -> Ndot) - R²(h_s -> Ndot)
  对慢变量（N）   : Sep_slow = R²(h_s -> N)    - R²(h_f -> N)

  两者都 > 0 表示分工正确。
  注意：这是【差值】，不是比值（避免分母为 0 的问题）。

参照：
  - E3 任意 64/64 切分（E3 没有分工，但 h 的前/后 64 维天然有差异）
  - 全 128 维（上界）
  - 不切分 + 相同监督（对照，排除"额外监督"的收益）
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from evaluate import UnifiedData, WINDOW
from psva_data import PSVAData

DEV = "cpu"

print("=" * 96)
print("PSVA 评价工具（交叉 Probe + Separation）")
print("=" * 96)


def ridge_r2(X_tr, Y_tr, X_te, Y_te, lams=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """标准 Ridge，返回 test R²（val 选 λ 时此处简化为取最优）"""
    Xm, Xs = X_tr.mean(0), X_tr.std(0) + 1e-8
    Xn = (X_tr - Xm) / Xs
    Ym, Ys = Y_tr.mean(), Y_tr.std() + 1e-9
    Yn = (Y_tr - Ym) / Ys
    D = Xn.shape[1]
    best = -1e9
    for lam in lams:
        W = np.linalg.solve(Xn.T @ Xn + lam * np.eye(D), Xn.T @ Yn)
        pt = ((X_te - Xm) / Xs) @ W * Ys + Ym
        r2 = 1 - ((pt - Y_te)**2).sum() / ((Y_te - Y_te.mean())**2).sum()
        best = max(best, r2)
    return float(best)


def extract_h(rssm, data, rid, n_max=None):
    """逐时刻提取 h（后验，argmax 采样）"""
    o = data.norm_obs(rid).astype(np.float32)
    a = data.norm_act(rid).astype(np.float32)
    if n_max:
        o, a = o[:n_max], a[:n_max]
    C_O = data.n_obs
    n = len(o)
    hs = []
    with torch.no_grad():
        st = rssm.init_state(1, DEV)
        for t in range(n):
            z_prev = st["z"].reshape(1, -1)
            h = rssm.cell(torch.cat([z_prev,
                                     torch.from_numpy(a[t:t+1]).float()],
                                    -1), st["h"])
            emb = rssm.enc(torch.cat([
                torch.from_numpy(o[t:t+1]).float(),
                torch.ones(1, C_O, device=DEV)], -1))
            pl = rssm.post(torch.cat([torch.cat([h, z_prev], -1), emb],
                                     -1)).reshape(1, rssm.stoch, rssm.classes)
            z = rssm.sample(pl, "argmax")
            st = {"h": h, "z": z}
            hs.append(h[0].numpy())
    return np.array(hs)


def get_phys(pd_obj, rid):
    """返回 (fast, slow) 物理量，已按 step=7 降采样"""
    if rid not in pd_obj.phys:
        return None, None
    return pd_obj.phys[rid]["fast"], pd_obj.phys[rid]["slow"]


def build_xy(rssm, data, pd_obj, splits, split_keys, d_fast, d_slow,
             n_max=900):
    """构造 (h_f, h_s, h_full, Ndot, N) 的训练/测试集"""
    Xf, Xs, Xa = [], [], []
    Yd, Yn = [], []
    for split in split_keys:
        for rid in data.splits[split]:
            h = extract_h(rssm, data, rid, n_max)
            pf, ps = get_phys(pd_obj, rid)
            if pf is None:
                continue
            L = min(len(h), len(pf))
            if L < 50:
                continue
            Xf.append(h[:L, :d_fast])
            Xs.append(h[:L, d_fast:d_fast + d_slow])
            Xa.append(h[:L])
            Yd.append(pf[:L])
            Yn.append(ps[:L])
    if not Xf:
        return None
    return {
        "h_f": np.concatenate(Xf), "h_s": np.concatenate(Xs),
        "h_all": np.concatenate(Xa),
        "Ndot": np.concatenate(Yd), "N": np.concatenate(Yn),
    }


def cross_probe(rssm, data, pd_obj, d_fast=64, d_slow=64):
    """计算交叉 Probe 矩阵与 Separation 指标"""
    tr = build_xy(rssm, data, pd_obj, None, ["train"], d_fast, d_slow)
    te = build_xy(rssm, data, pd_obj, None,
                  ["test_id", "test_ood"], d_fast, d_slow)
    if tr is None or te is None:
        return None

    res = {}
    # 4 个交叉 Probe（两个物理目标各 2 个通道）
    for tgt_name in ["Ndot", "N"]:
        for ch, ch_name in enumerate(["N2", "N3"]):
            y_tr = tr[tgt_name][:, ch]
            y_te = te[tgt_name][:, ch]
            r_ff = ridge_r2(tr["h_f"], y_tr, te["h_f"], y_te)
            r_ss = ridge_r2(tr["h_s"], y_tr, te["h_s"], y_te)
            r_fs = ridge_r2(tr["h_f"], y_tr, te["h_s"], y_te)  # 形状不符，跳过
            r_all = ridge_r2(tr["h_all"], y_tr, te["h_all"], y_te)
            res[f"{tgt_name}_{ch_name}"] = {
                "h_f": r_ff, "h_s": r_ss, "h_all": r_all,
            }

    # Separation（改进版：同一目标在快慢分支间的 R² 差值）
    # 快变量：Sep_fast = R²(h_f -> Ndot) - R²(h_s -> Ndot)
    sep_fast = np.mean([
        res[f"Ndot_{c}"]["h_f"] - res[f"Ndot_{c}"]["h_s"]
        for c in ["N2", "N3"]
    ])
    # 慢变量：Sep_slow = R²(h_s -> N) - R²(h_f -> N)
    sep_slow = np.mean([
        res[f"N_{c}"]["h_s"] - res[f"N_{c}"]["h_f"]
        for c in ["N2", "N3"]
    ])
    # 总 Separation（两个方向的平均）
    res["Separation"] = {
        "sep_fast": float(sep_fast),
        "sep_slow": float(sep_slow),
        "sep_total": float((sep_fast + sep_slow) / 2),
    }
    return res


def print_table(res, title):
    print(f"\n{title}")
    print(f"  {'Probe':<12}{'R²(h_f)':>10}{'R²(h_s)':>10}"
          f"{'R²(h_all)':>12}{'差值(f-s)':>12}")
    print("  " + "-" * 58)
    for k in ["Ndot_N2", "Ndot_N3", "N_N2", "N_N3"]:
        if k not in res:
            continue
        v = res[k]
        print(f"  {k:<12}{v['h_f']:>10.4f}{v['h_s']:>10.4f}"
              f"{v['h_all']:>12.4f}{v['h_f']-v['h_s']:>12.4f}")
    s = res["Separation"]
    print()
    print(f"  ★ Separation:")
    print(f"      快变量 (h_f - h_s 对 Ndot): {s['sep_fast']:+.4f}  "
          f"{'✅ 分工正确' if s['sep_fast'] > 0.05 else '⚠ 无分工'}")
    print(f"      慢变量 (h_s - h_f 对 N):    {s['sep_slow']:+.4f}  "
          f"{'✅ 分工正确' if s['sep_slow'] > 0.05 else '⚠ 无分工'}")
    print(f"      总 Separation:              {s['sep_total']:+.4f}")


if __name__ == "__main__":
    data = UnifiedData()
    pd_obj = PSVAData(data)
    C_O = data.n_obs

    print("\n  评价工具自检：用 E3 任意 64/64 切分作参照")
    print("  （E3 没有快慢监督，应显示无分工）")

    from pc_rssm import PCWorldModel
    e3 = PCWorldModel(C_O, data.n_act, deter=128, hidden=128,
                      lam_static=0.01, lam_dynamic=0.003)
    ck = Path(r"C:\pclwm\pclwm_pcrssm\checkpoints\E3_both.pt")
    if ck.exists():
        e3.load_state_dict(torch.load(ck, map_location=DEV,
                                      weights_only=True))
        e3.eval()
        res = cross_probe(e3.rssm, data, pd_obj, d_fast=64, d_slow=64)
        if res:
            print_table(res, "  [E3_both] 任意 64/64 切分（参照）")
            out = Path(r"C:\pclwm\pclwm_psva\results")
            out.mkdir(exist_ok=True)
            (out / "e3_reference_split.json").write_text(
                json.dumps(res, ensure_ascii=False, indent=1),
                encoding="utf-8")
            print(f"\n  -> {out/'e3_reference_split.json'}")
    else:
        print(f"  ⚠ E3 检查点不存在: {ck}")
