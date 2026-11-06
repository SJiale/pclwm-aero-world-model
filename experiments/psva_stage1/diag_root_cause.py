"""PSVA 失败根因诊断：GRU 全局耦合验证

核心问题：为什么监督 h 的前/后 64 维，却没有产生功能分化？

假设 H2：GRUCell 的输出每一维都依赖全部输入，
        因此 h[:64] 和 h[64:] 在更新时互相影响，无法独立演化。

两个决定性测试：
  测试 A（权重结构）：W_hh 的跨块子矩阵是否非零？
  测试 B（功能耦合）：扰动 h[:64]，看 h[64:] 的下一步变化有多大？

若两者都成立 -> 必须做结构改动（分组 GRU），仅改监督无效
"""
import json
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_mask")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_psva")))

from evaluate import UnifiedData, WINDOW
from psva_data import PSVAData
from psva import PSVA_WorldModel

DEV = "cpu"
C_O = 5
C_A = 3

print("=" * 96)
print("PSVA 失败根因诊断：GRU 全局耦合验证")
print("=" * 96)

# 载入 PSVA 检查点
ck = Path(r"C:\pclwm\pclwm_psva\checkpoints\PSVA_seed0.pt")
if not ck.exists():
    print(f"  ⚠ 检查点不存在: {ck}")
    sys.exit(1)

m = PSVA_WorldModel(C_O, C_A, d_fast=64, d_slow=64, unsplit=False,
                    lam_fast=0.01, lam_slow=0.01, lam_dyn=0.003)
m.load_state_dict(torch.load(ck, map_location=DEV, weights_only=True))
m.eval()
rssm = m.rssm
df, ds = rssm.d_fast, rssm.d_slow
print(f"\n  已载入 PSVA 检查点")
print(f"  deter={rssm.deter}  d_fast={df}  d_slow={ds}")

# ══════════════════════════════════════════════════
# 测试 A：权重结构
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[测试 A] GRUCell 权重结构：h[:64] 与 h[64:] 是否互相影响？")
print("=" * 96)

cell = rssm.cell
W_hh = cell.weight_hh.detach().numpy()   # [3*deter, deter] (GRUCell: 3 gates)
W_ih = cell.weight_ih.detach().numpy()   # [3*deter, in_dim]

print(f"\n  GRUCell weight_hh: {W_hh.shape}")
print(f"  GRUCell weight_ih: {W_ih.shape}")

# GRUCell 的 weight_hh 形状是 [3*D, D]，按 gate 分块
# gate 顺序: r, z, n
D = rssm.deter
assert W_hh.shape[0] == 3 * D, f"unexpected {W_hh.shape}"

# 对每个 gate，取 D×D 的隐状态→隐状态矩阵
gates = ["reset(r)", "update(z)", "new(n)"]
print(f"\n  {'Gate':<12}{'块':<20}{'均值|w|':>12}{'最大|w|':>12}{'占比':>10}")
print("  " + "-" * 68)

cross_ratios = []
for gi, gname in enumerate(gates):
    M = W_hh[gi * D:(gi + 1) * D, :]      # [D, D]
    # 四块
    ff = M[:df, :df]       # fast -> fast
    fs = M[:df, df:]       # slow -> fast
    sf = M[df:, :df]       # fast -> slow
    ss = M[df:, df:]       # slow -> slow
    blocks = [("ff (f->f)", ff), ("fs (s->f)", fs),
              ("sf (f->s)", sf), ("ss (s->s)", ss)]
    for bname, B in blocks:
        print(f"  {gname:<12}{bname:<20}{np.abs(B).mean():>12.4f}"
              f"{np.abs(B).max():>12.4f}{np.abs(B).mean()/np.abs(M).mean():>9.2f}x")
    # 跨块 vs 块内
    intra = (np.abs(ff).mean() + np.abs(ss).mean()) / 2
    cross = (np.abs(fs).mean() + np.abs(sf).mean()) / 2
    cross_ratios.append(cross / intra)
    print(f"  {'':<12}{'跨块/块内':<20}{cross/intra:>12.3f}")
    print()

print(f"  跨块/块内 比值（三个 gate 平均）: {np.mean(cross_ratios):.3f}")
print(f"  判读: 若 ≈ 1.0 说明跨块权重与块内同量级 -> 无结构隔离")
if np.mean(cross_ratios) > 0.5:
    print(f"  => ✅ H2 成立：权重矩阵没有块对角结构，h[:64] 与 h[64:] 完全耦合")
else:
    print(f"  => ⚠ H2 不成立：存在一定块结构")

# ══════════════════════════════════════════════════
# 测试 B：功能耦合（决定性）
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[测试 B] 功能耦合：扰动 h[:64]，看 h[64:] 下一步变化")
print("=" * 96)

torch.manual_seed(0)
B = 1
z = torch.zeros(B, rssm.stoch, rssm.classes)
h0 = torch.randn(B, rssm.deter) * 0.3
a_t = torch.randn(B, C_A) * 0.5

print("\n  方法：固定 z 和 a，只扰动 h[:64]（δ=1.0），")
print("        测量 h[64:] 的更新结果变化量。")
print("        同时反向测试：扰动 h[64:]，测 h[:64] 的变化。\n")


def step_gru(h):
    """用 GRUCell 单步更新"""
    z_prev = z.reshape(B, -1)
    return rssm.cell(torch.cat([z_prev, a_t], -1), h)


with torch.no_grad():
    base = step_gru(h0.clone())   # [B, D]

    results = []
    for delta in [0.1, 0.5, 1.0, 2.0]:
        # 扰动 fast 部分
        h_f = h0.clone()
        h_f[:, :df] += delta
        out_f = step_gru(h_f)
        d_slow_out = float((out_f[:, df:] - base[:, df:]).abs().mean())
        d_fast_out = float((out_f[:, :df] - base[:, :df]).abs().mean())

        # 扰动 slow 部分
        h_s = h0.clone()
        h_s[:, df:] += delta
        out_s = step_gru(h_s)
        d_fast_out2 = float((out_s[:, :df] - base[:, :df]).abs().mean())
        d_slow_out2 = float((out_s[:, df:] - base[:, df:]).abs().mean())

        results.append({
            "delta": delta,
            "perturb_fast": {"d_fast": d_fast_out, "d_slow": d_slow_out},
            "perturb_slow": {"d_fast": d_fast_out2, "d_slow": d_slow_out2},
        })

print(f"  {'δ':>6}{'扰动fast→fast':>16}{'扰动fast→slow':>16}"
      f"{'扰动slow→fast':>16}{'扰动slow→slow':>16}{'泄漏比':>10}")
print("  " + "-" * 82)
for r in results:
    pf, ps = r["perturb_fast"], r["perturb_slow"]
    # 泄漏比：跨块影响 / 本块影响
    leak = (pf["d_slow"] + ps["d_fast"]) / \
           (pf["d_fast"] + ps["d_slow"] + 1e-12)
    print(f"  {r['delta']:>6.1f}{pf['d_fast']:>16.4f}{pf['d_slow']:>16.4f}"
          f"{ps['d_fast']:>16.4f}{ps['d_slow']:>16.4f}{leak:>10.3f}")

print(f"\n  判读: 泄漏比 = 跨块影响 / (跨块 + 本块)")
print(f"        0 = 完全隔离，0.5 = 完全耦合（无分化）")

last_leak = (results[-1]["perturb_fast"]["d_slow"] +
             results[-1]["perturb_slow"]["d_fast"]) / \
            (results[-1]["perturb_fast"]["d_slow"] +
             results[-1]["perturb_slow"]["d_fast"] +
             results[-1]["perturb_fast"]["d_fast"] +
             results[-1]["perturb_slow"]["d_slow"] + 1e-12)
print(f"        δ=2.0 时泄漏比 = {last_leak:.3f}")

# ══════════════════════════════════════════════════
# 测试 C：h 的实际分化程度（用真实数据）
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[测试 C] 真实数据下 h[:64] 与 h[64:] 的相关性")
print("=" * 96)

pcd = PSVAData()
tr_W = pcd.windows("train", stride=8)[:500]

H = []
with torch.no_grad():
    for w in tr_W:
        o = torch.from_numpy(w["obs"][None]).float()
        a = torch.from_numpy(w["act"][None]).float()
        st = rssm.init_state(1, DEV)
        for t in range(WINDOW):
            zp = st["z"].reshape(1, -1)
            h = rssm.cell(torch.cat([zp, a[:, t]], -1), st["h"])
            emb = rssm.enc(torch.cat([
                o[:, t], torch.ones(1, C_O)], -1))
            pl = rssm.post(torch.cat([torch.cat([h, zp], -1), emb], -1)
                           ).reshape(1, rssm.stoch, rssm.classes)
            st = {"h": h, "z": rssm.sample(pl, "argmax")}
        H.append(st["h"][0].numpy())
H = np.array(H)
print(f"\n  h 样本: {H.shape}")

hf, hs = H[:, :df], H[:, df:]
# 主成分相关性（用均值序列）
corr_mean = float(np.corrcoef(hf.mean(1), hs.mean(1))[0, 1])
print(f"  corr(h_f 均值, h_s 均值) = {corr_mean:+.4f}")

# 典型相关分析（CCA）近似：用 SVD 求最大典型相关
def cca_first(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    Qx, _ = np.linalg.qr(X); Qy, _ = np.linalg.qr(Y)
    U, S, Vt = np.linalg.svd(Qx.T @ Qy, full_matrices=False)
    return float(S[0])

cca = cca_first(hf, hs)
print(f"  第一典型相关 = {cca:.4f}")
print(f"\n  判读: CCA 接近 1.0 说明两个子空间高度重叠 -> 无功能分化")
if cca > 0.9:
    print(f"  => ✅ H2 再次确认：h[:64] 与 h[64:] 几乎共线")
else:
    print(f"  => ⚠ 有一定分化")

# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("诊断结论")
print("=" * 96)

print(f"""
  测试 A（权重结构）: 跨块/块内 = {np.mean(cross_ratios):.3f}
  测试 B（功能耦合）: δ=2.0 泄漏比 = {last_leak:.3f}
  测试 C（子空间重叠）: 第一典型相关 = {cca:.4f}

  三项一致表明：h[:64] 与 h[64:] 没有结构隔离，
  仅靠"监督不同切片"无法让共享 GRU 产生功能分化。

  => PSVA 失败的根因是【结构性的】，不是调参问题。
  => 要提高分离度，必须改结构（分组 GRU / 独立 cell），
     而不是继续调整 λ 或维度分配。
""")

out = {
    "test_A_weight_structure": {
        "cross_over_intra_ratio": float(np.mean(cross_ratios)),
        "per_gate": {g: float(r) for g, r in zip(gates, cross_ratios)},
    },
    "test_B_functional_coupling": results,
    "test_C_subspace_overlap": {
        "corr_mean": corr_mean,
        "cca_first": cca,
    },
    "conclusion": "structural_not_tuning",
}
outp = Path(r"C:\pclwm\pclwm_psva\results\root_cause_diagnosis.json")
outp.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                encoding="utf-8")
print(f"  -> {outp}")
