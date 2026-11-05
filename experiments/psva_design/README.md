# PSVA 归档（物理监督的表征分离）

> 日期：2026-11-05
> **状态：代码与设计完成，未启动训练**
> 授权状态：**未获训练授权**

## 定性

> **PSVA 是「用物理监督诱导潜表征按时间尺度分工」，不构成真正的双时间尺度动力学演化。**

## 设计

```
h = [h_fast(64) ; h_slow(64)]     # 索引切分，不改结构，梯度全通

L_fast    = MSE(H_f(h_f), norm(N2dot, N3dot))          # 加速度
L_slow    = MSE(H_s(h_s), norm(N2, N3))                # 转速
L_dynamic = (1/|K|) Σ_k MSE(H_Δ(h^f_{t+k})-H_Δ(h^f_t), ΔN_{t,k})
            K = {1,4,8,16}, ΔN_{t,k} = N_{t+k} - N_t   # ★ 保留 ΔN，非 ΔNdot
```

**依据 PSG-JEPA**：`Δq_{t+i,k} = q_{t+i+k} - q_{t+i}`（状态差，非导数）

## 参数量

| 模型 | 参数 |
|---|---|
| PC-RSSM E3 | 1,203,209 |
| **PSVA（切分）** | **1,203,083**（−0.01%）|
| P_ctrl（不切分）| 1,203,467（+0.03%）|

## 单元测试：50 项全部通过

| 文件 | 项数 | 覆盖 |
|---|---|---|
| `test_psva.py` | 28 | 结构/前向/梯度/先验递推/因果性/参数量 |
| `test_ctrl.py` | 22 | P_ctrl 对照模式 |

**关键验证**：
- 切分正确性：`concat(split(h)) == h`，diff=0
- **因果性**：污染 `fut` 后 `l_dyn` 完全不变（0.022078 vs 0.022078）
- 梯度隔离：`l_fast` 只作用于 `head_fast`，`head_slow` 梯度为 0
- 不影响已有模型：E3 检查点可正常加载（26 keys）

## 评价方案

### Separation 指标（同一目标在快慢分支间的 R² 差值）

```
Sep_fast = R²(h_f → Ndot) - R²(h_s → Ndot)
Sep_slow = R²(h_s → N)    - R²(h_f → N)
Sep_total = (Sep_fast + Sep_slow) / 2
```

**★ E3 任意 64/64 切分参照（已测）**：

| Probe | R²(h_f) | R²(h_s) | R²(h_all) |
|---|---|---|---|
| Ndot_N2 | 0.8622 | 0.8659 | 0.9234 |
| Ndot_N3 | 0.8837 | 0.8633 | 0.9135 |
| N_N2 | 0.9946 | 0.9963 | 0.9960 |
| N_N3 | 0.9945 | 0.9958 | 0.9968 |

**E3 参照：Sep_fast=+0.0083, Sep_slow=+0.0015, Sep_total=+0.0049**

**成功判据**：PSVA 的 `Sep_total > 0.10`（显著高于 E3 的 0.0049）

### P_ctrl 对照（关键）

**用途**：排除"额外监督目标带来的收益"

| 配置 | 切分 | 监督头输入 |
|---|---|---|
| P1 | ✅ 64/64 | `h_f`(64) / `h_s`(64) / `h_f`(64) |
| **P_ctrl** | ❌ | 全部 **128 维** |

| 若... | 说明 |
|---|---|
| P_ctrl ≈ P1 | **收益来自监督，不来自切分** → PSVA 无价值 |
| P_ctrl < P1 | 切分有额外价值 |
| P_ctrl > P1 | 切分有害 |

## ⚠️ 与 PC-RSSM E3 的重要差异

| 模型 | `PHYS_COLS` | 动态目标 |
|---|---|---|
| **PC-RSSM E3** | `["loop_N2dot", "loop_N3dot"]` | **ΔNdot** |
| **PSVA** | 上述 + `["loop_N2int","loop_N3int"]` | **ΔN** |

**这不是笔误** —— E3 是较早设计（只有 Ndot 标签），PSVA 按 PSG-JEPA 改为 ΔN。

**因此 E3 与 PSVA 不能直接比较动态监督效果，PSVA 的真正对照是 P_ctrl。**

## 完整消融表

| 配置 | 切分 | L_fast | L_slow | L_dyn | 用途 |
|---|---|---|---|---|---|
| E0 | — | — | — | — | 纯基线 |
| E3_both | — | Ndot(全h) | — | ΔNdot(全h) | 主基线 |
| **P_ctrl** | ❌ | Ndot(全h) | N(全h) | ΔN(全h) | **排除监督收益** |
| **P1** | ✅ 64/64 | Ndot(h_f) | N(h_s) | ΔN(h_f) | **核心** |
| P2 | ✅ 64/64 | Ndot(h_f) | Ndot(h_s) | ΔN(h_f) | 不分离对照 |
| P4 | ✅ 64/64 | — | — | ΔN(h_f) | 只动态 |
| P5 | ✅ 32/96 | Ndot(h_f) | N(h_s) | ΔN(h_f) | 维度消融 |
| P_rand | ✅ 随机64 | 同 P1 | 同 P1 | 同 P1 | 索引 vs 随机 |

## PGTT 门控推导修正

**v4 文档错误**：方案 2 写成 `b_z = logit(α)`

**正确**：GRU 的 `z_t` 是**保留门**，α 是**吸收率**，故
```
b_z = logit(1 - α)

快: b_z = logit(0.9504) = +2.9524  →  z_0 = 0.9504 ✓
慢: b_z = logit(0.9894) = +4.5348  →  z_0 = 0.9894 ✓
```

**推荐方案 3**（无门控叠加歧义）：
```
h_t = (1-α) h_{t-1} + α · tanh(W_n[r_t ⊙ h_{t-1}, x_t])
```

## 文件清单

| 文件 | 内容 |
|---|---|
| `psva.py` | PSVA_RSSM + PSVA_WorldModel（含 unsplit）|
| `psva_data.py` | 数据加载 |
| `test_psva.py` | 28 项测试 |
| `test_ctrl.py` | 22 项 P_ctrl 测试 |
| `evaluate_psva.py` | 交叉 Probe + Separation |
| `verify_gate.py` | PGTT 门控验证 |
| `diag_init.py` | 零初始化诊断 |
| `diag_lambda.py` | λ 扫参 |
| `results/e3_reference_split.json` | E3 参照数据 |

## 未启动的训练

**PSVA GPU 训练未获授权，本文档不代表任何训练结果。**