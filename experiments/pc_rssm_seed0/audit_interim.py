"""实验可信度审查

逐项核查用户列出的 6 个方面：
  1. 数据划分与归一化是否一致、是否泄漏
  2. E0-E3 训练预算/评价条件/检查点选择是否相同
  3. 种子是否真正独立，同种子四组初始化是否可比
  4. 动态监督目标（ΔNdot）是否标签错位/未来泄漏
  5. 评价协议是否统一
  6. 梯度/NaN/反归一化
"""
import json, sys, re
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_pcrssm")))

OUT = Path(r"C:\pclwm\pclwm_pcrssm\audit")
OUT.mkdir(parents=True, exist_ok=True)

print("=" * 96)
print("PC-RSSM 多种子实验 中途有效性审查")
print("=" * 96)

n_pass = n_fail = n_warn = 0
FINDINGS = []


def check(name, cond, detail="", level="fail"):
    global n_pass, n_fail, n_warn
    if cond:
        n_pass += 1
        print(f"  [OK]   {name}" + (f"  {detail}" if detail else ""))
        FINDINGS.append({"item": name, "status": "pass", "detail": detail})
    else:
        if level == "warn":
            n_warn += 1
            print(f"  [WARN] {name}  {detail}")
            FINDINGS.append({"item": name, "status": "warn", "detail": detail})
        else:
            n_fail += 1
            print(f"  [FAIL] {name}  {detail}")
            FINDINGS.append({"item": name, "status": "fail", "detail": detail})


# ══════════════════════════════════════════════════
# 1. 数据划分与归一化
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[1] 数据划分与归一化")
print("=" * 96)

from evaluate import UnifiedData
data = UnifiedData()

# 1.1 划分完整性
all_ids = data.run_ids
split_ids = (data.splits["train"] + data.splits["val"] +
             data.splits["test_id"] + data.splits["test_ood"])
check("划分覆盖全部轨迹", len(split_ids) == len(all_ids),
      f"{len(split_ids)} vs {len(all_ids)}")
check("划分无重复", len(set(split_ids)) == len(split_ids),
      f"unique={len(set(split_ids))}")
check("划分互斥",
      len(set(data.splits["train"]) & set(data.splits["val"])) == 0 and
      len(set(data.splits["train"]) & set(data.splits["test_id"])) == 0 and
      len(set(data.splits["val"]) & set(data.splits["test_id"])) == 0)

print(f"\n  划分: train={len(data.splits['train'])} "
      f"val={len(data.splits['val'])} "
      f"test_id={len(data.splits['test_id'])} "
      f"test_ood={len(data.splits['test_ood'])}")

# 1.2 归一化是否只用 train 拟合
norm_path = Path(r"C:\pclwm\pclwm_data\norm.npz")
if norm_path.exists():
    nm = np.load(norm_path)
    mu, sd = nm["obs_mu"], nm["obs_sd"]
    # 重新用 train 计算
    tr_obs = np.concatenate([data.runs[r]["obs"] for r in data.splits["train"]])
    mu_re, sd_re = tr_obs.mean(0), tr_obs.std(0)
    d_mu = float(np.abs(mu - mu_re).max())
    d_sd = float(np.abs(sd - sd_re).max())
    check("归一化 mu 与 train 一致", d_mu < 1e-3, f"max diff={d_mu:.2e}")
    check("归一化 sd 与 train 一致", d_sd < 1e-3, f"max diff={d_sd:.2e}")
    # 与全体数据对比（若一致说明可能泄漏）
    all_obs = np.concatenate([data.runs[r]["obs"] for r in all_ids])
    mu_all = all_obs.mean(0)
    d_all = float(np.abs(mu - mu_all).max())
    print(f"  （参考：train mu vs 全体 mu 差 {d_all:.4f}，若很小属正常）")
else:
    check("norm.npz 存在", False, str(norm_path))

# 1.3 序列切分是否按轨迹（无窗口泄漏）
tr_W = data.windows("train", stride=4)
te_W = data.windows("test_id", stride=8)
tr_rids = set(w["rid"] for w in tr_W)
te_rids = set(w["rid"] for w in te_W)
check("train/test 窗口无轨迹重叠", len(tr_rids & te_rids) == 0,
      f"train={len(tr_rids)} te={len(te_rids)}")
check("test 窗口只来自 test_id", te_rids == set(data.splits["test_id"]),
      str(sorted(te_rids)))


# ══════════════════════════════════════════════════
# 2. 训练预算 / 评价 / 检查点选择
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[2] 训练预算 / 评价 / 检查点选择")
print("=" * 96)

# 从日志解析
log = Path(r"C:\pclwm\multiseed.log")
if log.exists():
    txt = log.read_text(encoding="utf-8", errors="ignore")

    # 2.1 训练预算（每个配置的 epoch 数）
    cfg_epochs = {}
    cur = None
    for m in re.finditer(r"\[(E\d_\w+)\]|best_val=[\d.]+@(\d+)", txt):
        if m.group(1):
            cur = m.group(1)
        elif cur and m.group(2):
            cfg_epochs.setdefault(cur, []).append(int(m.group(2)))

    print("\n  各配置最佳 epoch（从日志提取）:")
    for k, v in cfg_epochs.items():
        print(f"    {k}: {v}")

    best_eps = [e for v in cfg_epochs.values() for e in v]
    check("所有配置都记录 best_epoch", len(best_eps) > 0, f"{len(best_eps)} 条")
    check("best_epoch <= 60", all(e <= 60 for e in best_eps),
          f"max={max(best_eps) if best_eps else 0}")

    # 2.2 是否出现 NaN
    has_nan = bool(re.search(r"NaN|nan|inf", txt, re.IGNORECASE))
    check("日志中无 NaN/Inf", not has_nan,
          "未发现" if not has_nan else "发现异常")

    # 2.3 早停 vs 跑满
    ran_full = [k for k, v in cfg_epochs.items() if v and v[0] == 60]
    print(f"\n  跑满 60 epoch 的配置: {ran_full if ran_full else '无（都早停）'}")

# 2.4 代码检查：训练预算、早停、检查点选择
src = Path(r"C:\pclwm\pclwm_pcrssm\train_multiseed.py").read_text(encoding="utf-8")
check("MAX_EP=60", "MAX_EP = 60" in src)
check("PATIENCE=15", "PATIENCE = 15" in src)
check("BS=64", "BS = 64" in src)
check("LR 统一", "LR = 4e-4" in src)
check("早停用 val", "batch_iter(va_W" in src or "va_W, 128" in src)
check("检查点按 best_val 保存",
      "if v < best - 1e-7" in src and "torch.save" in src)
# 测试集只在最后用
te_in_train_loop = re.search(r"for ep in range.*?te_W", src, re.S)
check("训练循环内未使用 te_W", te_in_train_loop is None,
      "te_W 只在评价阶段使用" if te_in_train_loop is None else "疑似泄漏")


# ══════════════════════════════════════════════════
# 3. 种子独立性与同种子初始化
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[3] 种子独立性与初始化")
print("=" * 96)

# 3.1 种子设置
check("SEEDS = {0,1,2,3,4}", "SEEDS = [0, 1, 2, 3, 4]" in src)
check("每 seed 设置 torch.manual_seed",
      "torch.manual_seed(seed * 10007 + 1)" in src)
check("每 seed 设置 np.random.seed",
      "np.random.seed(seed * 10007 + 1)" in src)

# 3.2 同种子四组共享主干初始权重
check("同 seed 共享 RSSM 主干初始权重",
      "init_state = copy.deepcopy(ref.rssm.state_dict())" in src and
      "model.rssm.load_state_dict(init_rssm_state)" in src,
      "四组配置从同一初始 w 开始")

# 3.3 数据顺序是否随 seed 变化
check("batch 顺序随 seed 变化",
      "batch_iter(tr_W, BS, True, seed=seed*31+ep)" in src)

# 3.4 验证 log 中是否有 4 组的 best_val 都相同（说明没训练）
if log.exists():
    vals = re.findall(r"best_val=([\d.]+)@", txt)
    uniq = set(vals)
    check("best_val 不全相同（说明确实在训练）", len(uniq) > 2,
          f"{len(uniq)} 个不同值")


# ══════════════════════════════════════════════════
# 4. 动态监督目标（ΔNdot）与因果性
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[4] 动态监督目标与因果性")
print("=" * 96)

# 4.1 物理标签列
d_src = Path(r"C:\pclwm\pclwm_pcrssm\pc_data.py").read_text(encoding="utf-8")
check("PHYS_COLS = [N2dot, N3dot]",
      'PHYS_COLS = ["loop_N2dot", "loop_N3dot"]' in d_src,
      "确认 E3 用 ΔNdot")

# 4.2 时间对齐
import pandas as pd
from pc_data import PCData
pcd = PCData()
w = pcd.windows("train", stride=4)[0]
rid = w["rid"]
o = data.norm_obs(rid)
p = pcd.norm_phys(rid)
pos = int(round(w["t0"] / 0.105))
check("obs 对齐", np.allclose(w["obs"], o[pos - 32:pos], atol=1e-5))
check("phys_now 对齐", np.allclose(w["phys_now"], p[pos - 32:pos], atol=1e-5))
check("fut 对齐", np.allclose(w["fut"], o[pos:pos + 32], atol=1e-5))
check("phys_fut 对齐", np.allclose(w["phys_fut"], p[pos:pos + 32], atol=1e-5))

# 4.3 因果性（复用已验证结果，检查代码）
r_src = Path(r"C:\pclwm\pclwm_pcrssm\pc_rssm.py").read_text(encoding="utf-8")
check("动态监督用先验递推",
      "prior_rollout_from_hist" in r_src)
check("先验递推只接收历史 obs + 未来动作",
      "prior_rollout_from_hist(self, obs, mask, act, af" in r_src)

# 4.4 实际验证因果性
from pc_rssm import PCWorldModel
DEV = "cpu"
C_O, C_A = data.n_obs, data.n_act
m = PCWorldModel(C_O, C_A, deter=32, hidden=32,
                 lam_static=0.01, lam_dynamic=0.003)
m.eval()
torch.manual_seed(0)
B, T, H = 2, 32, 32
obs = torch.randn(B, T, C_O)
mask = torch.ones(B, T, C_O)
act = torch.randn(B, T, C_A)
af = torch.randn(B, H, C_A)
fut = torch.randn(B, H, C_O)
pn = torch.randn(B, T, 2)
pf = torch.randn(B, H, 2)
with torch.no_grad():
    torch.manual_seed(7)
    o1 = m(obs, mask, act, af, fut, obs_true=obs,
           phys_now=pn, phys_future=pf, mode="argmax")
    torch.manual_seed(7)
    o2 = m(obs, mask, act, af, torch.full_like(fut, 999.0),
           obs_true=obs, phys_now=pn, phys_future=pf, mode="argmax")
d1 = float(o1["l_dynamic"]) if o1["l_dynamic"].requires_grad else float(o1["l_dynamic"])
d2 = float(o2["l_dynamic"]) if o2["l_dynamic"].requires_grad else float(o2["l_dynamic"])
check("污染 fut 不改 l_dynamic", abs(d1 - d2) < 1e-6,
      f"{d1:.6f} vs {d2:.6f}")


# ══════════════════════════════════════════════════
# 5. 评价协议统一性
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[5] 评价协议统一性")
print("=" * 96)

# 所有配置用同一 eval_short / eval_continuous
check("评价函数统一", "def eval_short" in src and "def eval_continuous" in src)
check("统一 EVAL_SEED_BASE", "EVAL_SEED_BASE = 777000" in src)
check("统一 N_EVAL_REP=3", "N_EVAL_REP = 3" in src)
check("连续推演用 mean_rel 和 final_rel",
      '"mean_rel"' in src and '"final_rel"' in src)

# 反归一化检查
check("反归一化用各通道自己的 sd",
      "data.obs_sd" in src and "E * np.asarray(data.obs_sd)" in src.replace(" ", ""),
      "非统一用 T4 sd")

# 5.1 实测：反归一化正确性
sd = np.asarray(data.obs_sd).ravel()
print(f"\n  各通道 sd: {dict(zip(data.obs_cols, sd.round(3)))}")
print(f"  （若反归一化错误，会出现某通道误差被放大几百倍）")


# ══════════════════════════════════════════════════
# 6. 检查点有效性
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("[6] 已保存检查点有效性")
print("=" * 96)

ck_dir = Path(r"C:\pclwm\pclwm_pcrssm\checkpoints")
cks = sorted(ck_dir.glob("seed*.pt"))
print(f"\n  多seed检查点: {len(cks)} 个")
for ck in cks:
    try:
        sd_ = torch.load(ck, map_location="cpu", weights_only=True)
        n_key = len(sd_)
        # 检查是否全零/NaN
        has_nan = any(torch.isnan(v).any() for v in sd_.values()
                      if torch.is_tensor(v) and v.is_floating_point())
        all_zero = all((v.abs().sum() == 0) for v in sd_.values()
                       if torch.is_tensor(v) and v.is_floating_point())
        print(f"    {ck.name:<24} keys={n_key:>3}  "
              f"NaN={has_nan}  全零={all_zero}")
        if has_nan or all_zero:
            check(f"{ck.name} 有效", False, f"NaN={has_nan} 零={all_zero}")
        else:
            check(f"{ck.name} 有效", True)
    except Exception as e:
        check(f"{ck.name} 可加载", False, str(e)[:60])


# ══════════════════════════════════════════════════
# 汇总
# ══════════════════════════════════════════════════
print("\n" + "=" * 96)
print("审查汇总")
print("=" * 96)
print(f"  PASS: {n_pass}   WARN: {n_warn}   FAIL: {n_fail}")

if n_fail == 0:
    verdict = "通过"
elif n_fail <= 2:
    verdict = "存疑"
else:
    verdict = "失败"
print(f"\n  可信度判定: 【{verdict}】")

(OUT / "audit_checks.json").write_text(
    json.dumps({"verdict": verdict, "pass": n_pass, "warn": n_warn,
                "fail": n_fail, "findings": FINDINGS},
               ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {OUT/'audit_checks.json'}")
