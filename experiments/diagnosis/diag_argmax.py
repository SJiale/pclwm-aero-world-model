"""argmax 推理崩溃的代码级定位

核心问题：同一检查点，argmax 1.30 vs gumbel 0.081（16x 差）

假设 H1：argmax 与 gumbel 的**输出分布**不同 —— 但输出是 decoder，
         z 只是输入，若 z 不同则 decoder 输出不同
假设 H2：argmax 让 z 变成"确定但错误"的类别，而 gumbel 有随机性
         平均掉了错误（类似 dropout 的 ensemble 效果）
假设 H3：训练时用 gumbel（有噪声），推理用 argmax（无噪声）
         -> 训练/推理不一致（exposure bias）

验证：
  A. 检查 z 在 argmax / gumbel / prob 下的分布差异
  B. 检查 decoder 对 z 扰动的敏感度
  C. 检查 prior logits 的熵（是否接近均匀）
  D. 对比 z 的类别与实际"最优"类别
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_data")))
sys.path.insert(0, str(Path(r"C:\pclwm\pclwm_experiments")))
from evaluate import UnifiedData, UnifiedEval, WINDOW, HORIZON
from rssm_impl import load_rssm

OUT = Path(r"C:\pclwm\pclwm_experiments")
DIAG = OUT / "diagnosis"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

data = UnifiedData()
ev = UnifiedEval(data)
C_O, C_A = data.n_obs, data.n_act

model = load_rssm(OUT/"checkpoints"/"RSSM-d128.pt", C_O, C_A,
                  deter=128, hidden=128, device=DEV)
rssm = model.rssm
te_W = data.windows("test_id", stride=8)

print("=" * 88)
print("argmax 崩溃的代码级定位")
print("=" * 88)

# ══════════════════════════════════════════════════════
# A. 训练时与推理时 z 的构造差异
# ══════════════════════════════════════════════════════
print("\n[A] 训练 vs 推理的 z 构造")
print(f"  训练 (observe, model.training=True):  z = Gumbel-softmax(hard=True)")
print(f"  推理 (imagine, mode='gumbel'):        z = Gumbel-softmax(hard=True)")
print(f"  推理 (imagine, mode='argmax'):        z = one_hot(argmax)")
print()
print("  关键差异：")
print("    gumbel 采样的期望 = softmax(logits)")
print("    argmax 恒等 = 取最大概率类")
print("    若 softmax(logits) 接近均匀（熵高），argmax 会**放大微小偏差**")

# ══════════════════════════════════════════════════════
# B. 检查 prior logits 的熵（多步递推中）
# ══════════════════════════════════════════════════════
print("\n[B] 先验 logits 的熵（逐递推步）")


def probe_prior_entropy(ws, n_win=8):
    """沿 32 步递推，记录每步 prior 熵与 max prob"""
    ents, maxps, top2_gap = [], [], []
    with torch.no_grad():
        for i in range(0, min(len(ws), n_win), 4):
            c = ws[i:i+4]
            obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
                .float().to(DEV)
            act = torch.from_numpy(np.stack([w["act"] for w in c])) \
                .float().to(DEV)
            af = torch.from_numpy(np.stack([w["act_fut"] for w in c])) \
                .float().to(DEV)
            B, T, _ = obs.shape
            H = af.shape[1]
            state = rssm.initial(B, DEV)
            # 历史用后验 argmax（与推理一致）
            for t in range(T):
                pz = state["z"].reshape(B, -1)
                h = rssm.cell(torch.cat([pz, act[:, t]], -1), state["h"])
                embed = rssm.enc(obs[:, t])
                pl = rssm.post(torch.cat([h, embed], -1)) \
                    .reshape(B, rssm.stoch, rssm.classes)
                z = rssm.sample(pl, "argmax")
                state = {"h": h, "z": z}
            # 未来用先验
            for k in range(H):
                pz = state["z"].reshape(B, -1)
                h = rssm.cell(torch.cat([pz, af[:, k]], -1), state["h"])
                prior_logits = rssm.prior(h).reshape(B, rssm.stoch,
                                                     rssm.classes)
                probs = F.softmax(prior_logits, -1)
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(-1)
                ents.append(float(ent.mean()))
                maxps.append(float(probs.max(-1).values.mean()))
                # top1 - top2 差距
                top2 = probs.topk(2, -1).values
                top2_gap.append(float((top2[..., 0] - top2[..., 1]).mean()))
                z = rssm.sample(prior_logits, "argmax")
                state = {"h": h, "z": z}
    # 每步平均
    n_steps = len(ents) // max(n_win//4, 1)
    return (np.mean(ents), np.mean(maxps), np.mean(top2_gap),
            np.array(ents))


e_mean, mp_mean, gap_mean, e_arr = probe_prior_entropy(te_W)
print(f"    平均熵 = {e_mean:.4f}  (ln32={np.log(32):.4f}, 均匀=最高熵)")
print(f"    平均最大概率 = {mp_mean:.4f}  (1/32={1/32:.4f})")
print(f"    平均 top1-top2 差 = {gap_mean:.4f}")
print()
print(f"    -> 若熵接近 ln32，说明先验输出**接近均匀分布**")
print(f"       argmax 在这种分布上等于随机选一个类别")
print(f"       gumbel 采样虽然也随机，但有 softmax 概率加权")

# ══════════════════════════════════════════════════════
# C. decoder 对 z 的敏感度
# ══════════════════════════════════════════════════════
print("\n[C] Decoder 对 z 扰动的敏感度")
with torch.no_grad():
    c = te_W[:16]
    obs = torch.from_numpy(np.stack([w["obs"] for w in c])) \
        .float().to(DEV)
    act = torch.from_numpy(np.stack([w["act"] for w in c])) \
        .float().to(DEV)
    af = torch.from_numpy(np.stack([w["act_fut"] for w in c])) \
        .float().to(DEV)
    B, T, _ = obs.shape

    # 编码历史（argmax）
    state = rssm.initial(B, DEV)
    for t in range(T):
        pz = state["z"].reshape(B, -1)
        h = rssm.cell(torch.cat([pz, act[:, t]], -1), state["h"])
        embed = rssm.enc(obs[:, t])
        pl = rssm.post(torch.cat([h, embed], -1)) \
            .reshape(B, rssm.stoch, rssm.classes)
        z = rssm.sample(pl, "argmax")
        state = {"h": h, "z": z}

    # 一步先验，然后比较不同 z 选择下的 decoder 输出
    pz = state["z"].reshape(B, -1)
    h = rssm.cell(torch.cat([pz, af[:, 0]], -1), state["h"])
    prior_logits = rssm.prior(h).reshape(B, rssm.stoch, rssm.classes)
    probs = F.softmax(prior_logits, -1)

    outs = {}
    for mode in ["argmax", "prob", "gumbel"]:
        z = rssm.sample(prior_logits, mode)
        outs[mode] = rssm.dec(torch.cat([h, z.reshape(B, -1)], -1))

    print(f"    先验熵 = {float(-(probs*torch.log(probs.clamp_min(1e-12))).sum(-1).mean()):.4f}")
    print(f"    各模式输出差异（归一化空间）:")
    for a, b in [("argmax", "gumbel"), ("argmax", "prob"),
                 ("prob", "gumbel")]:
        d = (outs[a] - outs[b]).abs().mean()
        rel = d / outs["gumbel"].abs().mean()
        print(f"      |{a} - {b}| = {float(d):.6f}  (相对 {float(rel)*100:.1f}%)")

    # 单步 vs 32 步累积
    print(f"\n    单步输出差异（上）vs 32 步累积（下）:")
    for mode in ["argmax", "prob", "gumbel"]:
        p = rssm.imagine(obs, act, af, mode=mode)
        print(f"      {mode:<8} 32步输出 mean|x| = {float(p.abs().mean()):.4f}")

# ══════════════════════════════════════════════════════
# D. 关键：训练时的 multi 损失用的什么采样方式？
# ══════════════════════════════════════════════════════
print("\n[D] 训练时 multi 损失的采样方式")
print("  代码位置: run_rssm.py -> WorldModel.forward")
print("    pred = self.rssm.imagine(obs, act, act_fut)")
print("    imagine 内部: sm = mode or ('gumbel' if self.training else 'argmax')")
print()
print("  训练时 self.training=True -> 用 gumbel")
print("  推理时 model.eval() -> self.training=False -> 用 **argmax**")
print()
print("  >>> 这就是 bug：")
print("      训练 multi loss 用 gumbel，优化的是 gumbel 分布下的误差")
print("      推理却用 argmax，分布不同 -> exposure bias")
print()
print("  证据：python -c 检查")

# ══════════════════════════════════════════════════════
# E. 定量验证：训练/推理采样一致性
# ══════════════════════════════════════════════════════
print("\n[E] 定量验证：同一输入，训练模式 vs 推理模式的输出差")


def eval_mode_diff(ws, n_win=32):
    """model.train() 用 gumbel，model.eval() 用 argmax，比较输出"""
    c = ws[:n_win]
    b = {k: torch.from_numpy(np.stack([w[k] for w in c]))
         .float().to(DEV) for k in ["obs", "act", "act_fut"]}

    rssm.train()
    with torch.no_grad():
        p_train = model.imagine(b["obs"], b["act"], b["act_fut"])
    rssm.eval()
    with torch.no_grad():
        p_eval = model.imagine(b["obs"], b["act"], b["act_fut"])

    d = (p_train - p_eval).abs()
    return {
        "mean_abs_diff": float(d.mean()),
        "max_abs_diff": float(d.max()),
        "rel": float(d.mean() / p_train.abs().mean()),
        "train_out_scale": float(p_train.abs().mean()),
    }


diff = eval_mode_diff(te_W)
print(f"    train模式(gumbel) vs eval模式(argmax):")
print(f"      平均绝对差 = {diff['mean_abs_diff']:.6f}")
print(f"      最大绝对差 = {diff['max_abs_diff']:.6f}")
print(f"      相对 = {diff['rel']*100:.1f}%")
print(f"      输出量级 = {diff['train_out_scale']:.4f}")

# ══════════════════════════════════════════════════════
# F. 对比 GRU：确定性模型无此问题
# ══════════════════════════════════════════════════════
print("\n[F] 对比：GRU 是确定性模型，train/eval 无采样差异")
print("  -> GRU 的 train 与 eval 输出应该完全一致（除 dropout）")
print("  -> RSSM 因 gumbel/argmax 切换产生 train/eval 不一致")

summary = {
    "prior_entropy_mean": float(e_mean),
    "ln32": float(np.log(32)),
    "prior_maxp_mean": float(mp_mean),
    "top1_top2_gap": float(gap_mean),
    "train_eval_diff": diff,
    "root_cause": "train/inference sampling mismatch (gumbel vs argmax)",
    "code_location": "rssm_impl.py RSSM._sample / imagine mode default",
}
(DIAG / "argmax_root_cause.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n  -> {DIAG/'argmax_root_cause.json'}")
