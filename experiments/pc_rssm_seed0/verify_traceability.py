"""Checkpoint 可追溯性验证

验证每个 checkpoint 的：
  1. 权重可加载
  2. val 指标可从 JSON 查到
  3. best_epoch 可从 hist 验证
  4. 参数量与 JSON 一致
  5. 训练配置可追溯（lam_s, lam_d）
"""
import json
import sys
from pathlib import Path
import torch

CKPT = Path(r"C:\pclwm\pclwm_pcrssm\checkpoints")
RES = Path(r"C:\pclwm\pclwm_pcrssm\results\multiseed_results.json")
LOG = Path(r"C:\pclwm\multiseed.log")

print("=" * 96)
print("Checkpoint 可追溯性验证")
print("=" * 96)

D = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}
print(f"  JSON seeds: {list(D.keys())}")

cks = sorted(CKPT.glob("seed*_*.pt"))
print(f"  检查点数: {len(cks)}")

n_pass = n_fail = 0


def check(name, cond, detail=""):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        print(f"  [OK]   {name}" + (f"  {detail}" if detail else ""))
    else:
        n_fail += 1
        print(f"  [FAIL] {name}  {detail}")


print("\n" + "=" * 96)
print("[1] 逐检查点验证")
print("=" * 96)

rows = []
for ck in cks:
    # 解析文件名: seed0_E0_baseline.pt
    parts = ck.stem.split("_", 2)
    if len(parts) < 3:
        continue
    seed_key = parts[0]            # seed0
    cfg = parts[1] + "_" + parts[2]  # E0_baseline
    seed_num = seed_key.replace("seed", "")

    if seed_key == "seed99":       # 跳过 smoke test
        continue

    # 加载权重
    try:
        sd = torch.load(ck, map_location="cpu", weights_only=True)
        n_param = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
        loadable = True
    except Exception as e:
        loadable = False
        n_param = 0

    # 从 JSON 查
    j = D.get(seed_key, {}).get("cfgs", {}).get(cfg, {})
    j_val = j.get("best_val")
    j_ep = j.get("best_epoch")
    j_params = j.get("params")
    j_lam_s = j.get("lam_s")
    j_lam_d = j.get("lam_d")
    j_hist = j.get("hist", {})
    j_val_curve = j_hist.get("val", [])

    # 验证
    ok_load = loadable
    ok_param = (j_params is not None and n_param == j_params)
    ok_val = (j_val is not None)
    ok_ep = (j_ep is not None)
    # 从 hist 验证 best_epoch
    ok_hist = False
    if j_val_curve and j_ep is not None and 0 < j_ep <= len(j_val_curve):
        v_at_ep = j_val_curve[j_ep - 1]
        v_min = min(j_val_curve)
        ok_hist = abs(v_at_ep - v_min) < 1e-6

    rows.append({
        "ck": ck.name, "seed": seed_key, "cfg": cfg,
        "load": ok_load, "n_param": n_param, "j_params": j_params,
        "param_ok": ok_param, "val_ok": ok_val, "ep_ok": ok_ep,
        "hist_ok": ok_hist, "best_val": j_val, "best_epoch": j_ep,
        "lam_s": j_lam_s, "lam_d": j_lam_d,
    })

print(f"\n  {'检查点':<28}{'可加载':>8}{'参数吻合':>10}{'val':>10}"
      f"{'epoch':>7}{'hist验证':>10}")
print("  " + "-" * 76)
for r in rows:
    print(f"  {r['ck']:<28}{str(r['load']):>8}"
          f"{str(r['param_ok']):>10}"
          f"{r['best_val']:>10.5f}" if r['best_val'] else ""
          f"", end="")
    print(f"{r['best_epoch']:>7}" if r['best_epoch'] else f"{'':>7}", end="")
    print(f"{str(r['hist_ok']):>10}")

# 重新打印（修正格式）
print()
print(f"  {'检查点':<28}{'可加载':>7}{'参数':>10}{'JSON参数':>10}"
      f"{'best_val':>10}{'ep':>4}{'hist验证':>9}")
print("  " + "-" * 82)
for r in rows:
    bv = f"{r['best_val']:.5f}" if r['best_val'] else "n/a"
    be = str(r['best_epoch']) if r['best_epoch'] else "n/a"
    jp = f"{r['j_params']:,}" if r['j_params'] else "n/a"
    print(f"  {r['ck']:<28}{str(r['load']):>7}{r['n_param']:>10,}"
          f"{jp:>10}{bv:>10}{be:>4}{str(r['hist_ok']):>9}")

print("\n" + "=" * 96)
print("[2] 汇总检查")
print("=" * 96)

# 区分：有 JSON 记录的 vs 正在训练的
in_json = [r for r in rows if r["val_ok"]]
not_in_json = [r for r in rows if not r["val_ok"]]

all_load = all(r["load"] for r in rows)
all_param = all(r["param_ok"] for r in in_json) if in_json else True
all_val = all(r["val_ok"] for r in in_json) if in_json else True
all_ep = all(r["ep_ok"] for r in in_json) if in_json else True
all_hist = all(r["hist_ok"] for r in in_json) if in_json else True

check("全部检查点可加载", all_load, f"{len(rows)} 个")
check("参数量与 JSON 一致", all_param, f"{len(in_json)} 个有记录")
check("best_val 可追溯", all_val)
check("best_epoch 可追溯", all_ep)
check("best_epoch 与 hist 曲线一致", all_hist,
      "即 hist.val[best_epoch-1] == min(hist.val)")

print()
if not_in_json:
    print(f"  提示：{len(not_in_json)} 个检查点尚无 JSON 记录（训练进行中）:")
    for r in not_in_json:
        print(f"    {r['ck']}")

# 训练配置追溯
print("\n" + "=" * 96)
print("[3] 训练配置追溯")
print("=" * 96)
print(f"\n  {'seed':<8}{'配置':<14}{'λ_static':>10}{'λ_dynamic':>11}"
      f"{'参数量':>12}")
print("  " + "-" * 58)
for r in rows:
    ls = f"{r['lam_s']:>10}" if r["lam_s"] is not None else f"{'n/a':>10}"
    ld = f"{r['lam_d']:>11}" if r["lam_d"] is not None else f"{'n/a':>11}"
    jp = f"{r['j_params']:>12,}" if r['j_params'] else f"{'n/a':>12}"
    print(f"  {r['seed']:<8}{r['cfg']:<14}{ls}{ld}{jp}")

# 验证 lam 与配置名对应
print("\n  验证 λ 与配置名对应:")
expect = {"E0_baseline": (0.0, 0.0), "E1_static": (0.01, 0.0),
          "E2_dynamic": (0.0, 0.003), "E3_both": (0.01, 0.003)}
lam_ok = True
checked_lam = 0
for r in rows:
    if r["cfg"] in expect and r["lam_s"] is not None:
        e_s, e_d = expect[r["cfg"]]
        checked_lam += 1
        if abs(r["lam_s"] - e_s) > 1e-9 or abs(r["lam_d"] - e_d) > 1e-9:
            lam_ok = False
            print(f"    ✗ {r['seed']}/{r['cfg']}: "
                  f"实际({r['lam_s']},{r['lam_d']}) 期望({e_s},{e_d})")
print(f"    （检查了 {checked_lam} 个有 JSON 记录的配置）")
check("λ 与配置名一致", lam_ok)

print("\n" + "=" * 96)
print(f"结果: {n_pass} 通过, {n_fail} 失败")
print("=" * 96)
if n_fail == 0:
    print("  ✅ Checkpoint 完全可追溯")

