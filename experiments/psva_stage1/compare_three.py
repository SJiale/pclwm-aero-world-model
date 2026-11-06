"""三配置对比：E3 vs P_ctrl vs PSVA（seed=0）

E3 从 PC-RSSM 多seed结果读取（seed0_E3_both）
"""
import json
from pathlib import Path
import numpy as np

PSVA_RES = Path(r"C:\pclwm\pclwm_psva\results\psva_results.json")
MULTI_RES = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
OUT = Path(r"C:\pclwm\pclwm_psva\results")

P = json.loads(PSVA_RES.read_text(encoding="utf-8"))
M = json.loads(MULTI_RES.read_text(encoding="utf-8"))

E3 = M["seed0"]["cfgs"]["E3_both"]
PCTRL = P["P_ctrl_seed0"]
PSVA = P["PSVA_seed0"]

print("=" * 100)
print("三配置对比（seed=0）：E3 / P_ctrl / PSVA")
print("=" * 100)

print("""
  配置说明：
    E3     : PC-RSSM 静态+动态锚定（全 h），动态目标 ΔNdot
    P_ctrl : 全 h + Ndot/N/ΔN 三监督（不切分）  ← PSVA 的直接对照
    PSVA   : h 切分 64/64 + 相同三监督          ← 核心
""")

# ══════════════════════════════════════════════════
# 表 1：核心指标
# ══════════════════════════════════════════════════
print("=" * 100)
print("[表1] 核心指标")
print("=" * 100)


def g(c, *path, default=np.nan):
    cur = c
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


print(f"\n  {'指标':<18}{'E3':>10}{'P_ctrl':>10}{'PSVA':>10}"
      f"{'PSVA vs E3':>13}{'PSVA vs P_ctrl':>16}")
print("  " + "-" * 80)

rows = [
    ("Test-ID", g(E3, "test_id", "mean"), g(PCTRL, "test_id", "mean"),
     g(PSVA, "test_id", "mean")),
    ("Test-OOD", g(E3, "test_ood", "mean"), g(PCTRL, "test_ood", "mean"),
     g(PSVA, "test_ood", "mean")),
    ("连续 10s", g(E3, "continuous", "test_id_10s", "mean_rel"),
     g(PCTRL, "continuous", "test_id_10s", "mean_rel"),
     g(PSVA, "continuous", "test_id_10s", "mean_rel")),
    ("连续 30s", g(E3, "continuous", "test_id_30s", "mean_rel"),
     g(PCTRL, "continuous", "test_id_30s", "mean_rel"),
     g(PSVA, "continuous", "test_id_30s", "mean_rel")),
    ("连续 60s", g(E3, "continuous", "test_id_60s", "mean_rel"),
     g(PCTRL, "continuous", "test_id_60s", "mean_rel"),
     g(PSVA, "continuous", "test_id_60s", "mean_rel")),
]

for name, e3, pc, ps in rows:
    vs_e3 = (ps - e3) / e3 * 100
    vs_pc = (ps - pc) / pc * 100
    print(f"  {name:<18}{e3:>10.4f}{pc:>10.4f}{ps:>10.4f}"
          f"{vs_e3:>+12.2f}%{vs_pc:>+15.2f}%")

# ══════════════════════════════════════════════════
# 表 2：Separation
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表2] Separation（h 前/后 64 维的分工）")
print("=" * 100)

# E3 的 Separation 需要从 e3_reference_split.json 读
ref = Path(r"C:\pclwm\pclwm_psva\results\e3_reference_split.json")
e3_sep = None
if ref.exists():
    R = json.loads(ref.read_text(encoding="utf-8"))
    e3_sep = g(R, "Separation", "sep_total")

print(f"\n  {'配置':<12}{'Sep_fast':>12}{'Sep_slow':>12}{'Sep_total':>12}")
print("  " + "-" * 50)
if e3_sep is not None:
    print(f"  {'E3':<12}{g(R,'Separation','sep_fast'):>+12.4f}"
          f"{g(R,'Separation','sep_slow'):>+12.4f}{e3_sep:>+12.4f}")
print(f"  {'P_ctrl':<12}{g(PCTRL,'probe','Separation','sep_fast'):>+12.4f}"
      f"{g(PCTRL,'probe','Separation','sep_slow'):>+12.4f}"
      f"{g(PCTRL,'probe','Separation','sep_total'):>+12.4f}")
print(f"  {'PSVA':<12}{g(PSVA,'probe','Separation','sep_fast'):>+12.4f}"
      f"{g(PSVA,'probe','Separation','sep_slow'):>+12.4f}"
      f"{g(PSVA,'probe','Separation','sep_total'):>+12.4f}")

print(f"\n  成功门槛: Sep_total > 0.10")
print(f"  实际:     PSVA = {g(PSVA,'probe','Separation','sep_total'):+.4f}"
      f"  -> {'达标' if g(PSVA,'probe','Separation','sep_total') > 0.10 else '未达标（差很远）'}")

# ══════════════════════════════════════════════════
# 表 3：逐轨迹
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表3] 逐轨迹（test_ood，3 条独立工况）")
print("=" * 100)

trajs = list(g(PCTRL, "test_ood", "per_traj", default={}).keys())
print(f"\n  {'轨迹':<18}{'E3':>10}{'P_ctrl':>10}{'PSVA':>10}{'PSVA优势':>12}")
print("  " + "-" * 62)
for rid in trajs:
    e = g(E3, "test_ood", "per_traj", rid, "mean")
    pc = g(PCTRL, "test_ood", "per_traj", rid, "mean")
    ps = g(PSVA, "test_ood", "per_traj", rid, "mean")
    adv = (pc - ps) / pc * 100   # 正 = PSVA 更好
    print(f"  {rid:<18}{e:>10.4f}{pc:>10.4f}{ps:>10.4f}{adv:>+11.2f}%")

print("\n  逐轨迹 test_id:")
print(f"  {'轨迹':<18}{'E3':>10}{'P_ctrl':>10}{'PSVA':>10}{'PSVA优势':>12}")
print("  " + "-" * 62)
for rid in g(PCTRL, "test_id", "per_traj", default={}).keys():
    e = g(E3, "test_id", "per_traj", rid, "mean")
    pc = g(PCTRL, "test_id", "per_traj", rid, "mean")
    ps = g(PSVA, "test_id", "per_traj", rid, "mean")
    adv = (pc - ps) / pc * 100
    print(f"  {rid:<18}{e:>10.4f}{pc:>10.4f}{ps:>10.4f}{adv:>+11.2f}%")

# ══════════════════════════════════════════════════
# 表 4：诚实判断
# ══════════════════════════════════════════════════
print("\n" + "=" * 100)
print("[表4] ★★ 诚实判断")
print("=" * 100)

psva_c60 = g(PSVA, "continuous", "test_id_60s", "mean_rel")
pctrl_c60 = g(PCTRL, "continuous", "test_id_60s", "mean_rel")
e3_c60 = g(E3, "continuous", "test_id_60s", "mean_rel")

psva_sep = g(PSVA, "probe", "Separation", "sep_total")
pctrl_sep = g(PCTRL, "probe", "Separation", "sep_total")

print(f"""
  两个维度：

  (1) 潜状态分工
      P_ctrl = {pctrl_sep:+.4f}
      PSVA   = {psva_sep:+.4f}
      提升   = {psva_sep-pctrl_sep:+.4f}
      {'→ 分工确实变明显了' if psva_sep > pctrl_sep else '→ 分工没变明显'}
      但离成功门槛（0.10）还差 {0.10-psva_sep:.4f}

  (2) 长期预测（60s）
      P_ctrl = {pctrl_c60:.4f}
      PSVA   = {psva_c60:.4f}
      改善   = {(pctrl_c60-psva_c60)/pctrl_c60*100:+.2f}%
      {'→ 有微弱改善' if psva_c60 < pctrl_c60 else '→ 无改善'}

  (3) 跨工况（Test-OOD）
      P_ctrl = {g(PCTRL,'test_ood','mean'):.4f}
      PSVA   = {g(PSVA,'test_ood','mean'):.4f}
      改善   = {(g(PCTRL,'test_ood','mean')-g(PSVA,'test_ood','mean'))/g(PCTRL,'test_ood','mean')*100:+.2f}%
""")

print("  ── 关键结论 ──")
print()
print("  ★ 分工提升（+0.0139）远未达门槛（0.10），差距 7 倍")
print("  ★ 预测改善（60s: 1.08%）在单种子噪声范围内")
print("  ★ Test-OOD 略差（+0.39%）")
print()
print("  => PSVA 相对于 P_ctrl 没有带来明确收益")
print("  => 更关键的是：Separation 本身只到 +0.0026，")
print("     连'分工明显'都谈不上，谈不上'分工带来预测收益'")

# 保存
summary = {
    "configs": ["E3", "P_ctrl", "PSVA"],
    "seed": 0,
    "metrics": {
        n: {"E3": float(a), "P_ctrl": float(b), "PSVA": float(c)}
        for n, a, b, c in rows
    },
    "separation": {
        "E3": float(e3_sep) if e3_sep else None,
        "P_ctrl": float(pctrl_sep),
        "PSVA": float(psva_sep),
        "threshold": 0.10,
    },
    "verdict": "no_clear_benefit",
    "reason": "Separation +0.0026 远低于门槛 0.10；60s 改善 1.08% 在噪声范围；Test-OOD 略差",
}
(OUT / "stage1_three_config.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'stage1_three_config.json'}")
