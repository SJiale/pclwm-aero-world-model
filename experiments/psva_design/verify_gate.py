"""PGTT 门控推导修正

用户指出：外部 α 很小时主要保留旧状态，而非覆盖。
我在 v4 文档中写反了，需要重新推导。

标准 GRU：
  z_t = σ(W_z [h_{t-1}, x_t] + b_z)      # update gate（保留门）
  h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}

  这里 z_t 是"保留旧状态"的系数：
    z_t = 1  -> h_t = h_{t-1}     （完全保留）
    z_t = 0  -> h_t = n_t          （完全覆盖）

外部 α 方案（方案 2：α 作为 update gate 偏置）：
  z_t = σ(W_z [h_{t-1}, x_t] + b_z + logit(α))

  若 logit(α) 很大（α→1），z_t→1，保留旧状态
  若 logit(α) 很小（α→0），z_t→0，完全覆盖

  ★ 所以要让"小 α 对应小更新"，应该是 z_t → 1，即
     b_z = logit(1 - α)

外部 α 方案（方案 3：α 直接乘新信息）：
  h_t = (1 - α) h_{t-1} + α n_t

  这里 α 是"吸收新信息"的系数：
    α 小 -> 保留旧状态多  （τ 大，慢）
    α 大 -> 快速更新      （τ 小，快）

  与 α = 1 - exp(-dt/τ) 一致：τ 大 -> α 小 -> 慢。✓

两方案的语义对照：
  方案 2: z_t（保留门）对应 (1-α_new)，所以 b_z = logit(1-α)
  方案 3: α 直接是吸收率，无需转换
"""
import numpy as np

dt = 0.105
tau_f = 2.0631
tau_s = 9.8392

a_f = 1 - np.exp(-dt / tau_f)
a_s = 1 - np.exp(-dt / tau_s)

print("=" * 78)
print("PGTT 门控语义修正")
print("=" * 78)
print()
print("  物理时间常数:")
print(f"    τ_f = {tau_f:.4f} s  ->  α_f = 1-exp(-dt/τ_f) = {a_f:.6f}")
print(f"    τ_s = {tau_s:.4f} s  ->  α_s = 1-exp(-dt/τ_s) = {a_s:.6f}")
print()
print("  α 的语义：【吸收新信息的比例】")
print(f"    每步保留旧状态 {(1-a_f)*100:.2f}% (快) / {(1-a_s)*100:.2f}% (慢)")
print(f"    每步吸收新信息 {a_f*100:.2f}% (快) / {a_s*100:.2f}% (慢)")
print()
print("=" * 78)
print("方案 2：α 作为 GRU update gate 的偏置")
print("=" * 78)
print()
print("  标准 GRU:")
print("    h_t = (1 - z_t) ⊙ n_t + z_t ⊙ h_{t-1}")
print("    其中 z_t 是【保留门】: z=1 全保留，z=0 全覆盖")
print()
print("  ★ 若要让 α 表示【吸收率】，则保留门应为 (1-α):")
print("    z_t = σ(... + logit(1 - α))")
print()
print("  v4 文档错误：写成了 b_z = logit(α)")
print(f"    logit(α_f={a_f:.4f}) = {np.log(a_f/(1-a_f)):.4f}")
print(f"    -> σ({np.log(a_f/(1-a_f)):.4f}) = {a_f:.4f}  （这是覆盖门，不是保留门）")
print()
print("  正确写法：")
for nm, a in [("快", a_f), ("慢", a_s)]:
    v = 1 - a
    bl = np.log(v / (1 - v))
    print(f"    {nm}分支: b_z = logit(1-α) = logit({v:.6f}) = {bl:+.4f}")
print()
print("  验证:")
for nm, a in [("快", a_f), ("慢", a_s)]:
    v = 1 - a
    bl = np.log(v / (1 - v))
    z0 = 1 / (1 + np.exp(-bl))
    print(f"    {nm}: b_z={bl:+.4f} -> z_0=σ(b_z)={z0:.6f}"
          f"  保留率={z0:.4f} (应={v:.4f})  {'✓' if abs(z0-v)<1e-6 else '✗'}")
print()
print("=" * 78)
print("方案 3：α 直接乘新信息（推荐）")
print("=" * 78)
print()
print("  h_t = (1 - α) h_{t-1} + α n_t")
print("       = (1 - α) h_{t-1} + α · tanh(W_n [r_t ⊙ h_{t-1}, x_t])")
print()
print("  ★ 这里 α 直接是吸收率，与 α = 1-exp(-dt/τ) 语义完全一致：")
for nm, a in [("快", a_f), ("慢", a_s)]:
    print(f"    {nm}: α={a:.6f} -> 保留 {(1-a)*100:.2f}% 旧状态")
print()
print("  优点：")
print("    - 无门控语义歧义")
print("    - α 与物理时间常数一一对应")
print("    - 反推 τ_eff 简单：τ_eff = -dt/ln(1-α)")
print("  缺点：")
print("    - 失去了 update gate 的自适应性")
print("    - 需要靠 reset gate 提供表达力")
print()
print("=" * 78)
print("结论")
print("=" * 78)
print("""
  v4 文档的两处错误：
  
  1. 方案 2 的偏置应为 logit(1-α)，不是 logit(α)
     因为 GRU 的 z_t 是【保留门】，而 α 是【吸收率】
     
  2. 关于"α 很小时主要保留旧状态而非覆盖" —— 这个表述是对的，
     但我在文档里把方案 2 的偏置写反了，导致推导矛盾。
     
  最终建议：
    - 方案 3（α 直接乘新信息）物理语义最清晰，推荐作为主方案
    - 方案 2（α 作为偏置）作为消融，偏置必须用 logit(1-α)
""")
