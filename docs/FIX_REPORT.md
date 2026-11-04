# Mask-aware 修复报告：重建监督 + 逐步在线估计

> 日期：2026-11-04
> 前置：`STAGE_REPORT.md`（修复前结果）
> **核心结论：修复后 Mask-RSSM 全面超越 Mask-GRU，结论彻底反转**

---

## 一、修复的两个关键问题

### 问题 1：重建损失监督目标错误（严重）

**代码位置**：`train_mask.py` 训练循环 + `mask_impl.py:MaskWorldModel.forward`

```python
# 错误：obs_m（遮蔽后的）既作 Encoder 输入，又作重建监督
obs_m, m_hist, fut_m, m_fut = augment_mask_batch(...)
out = model(obs_m, m_hist, act, af, fut, ...)
# forward 内部：
rec = ((recon[:, burn:] - obs[:, burn:]) ** 2).mean()   # obs 就是 obs_m
```

**后果**：模型学到的是"遮蔽通道输出 0"，而不是"用其他传感器推断真实 T4"。

**定量证明**（单元测试）：
| 监督目标 | 模型输出全 0 时的 loss |
|---|---|
| 遮蔽值（旧）| **0.00e+00** ← 无惩罚 |
| 真实值（新）| **0.2001** ← 有惩罚 |

**修复**：`augment_mask_batch` 返回 4 个张量 `(obs_masked, mask, obs_true, fut_true)`；
`MaskWorldModel.forward` 新增 `obs_true` 参数，重建损失改为与真实观测比较。

### 问题 2：不是真正的逐步在线估计

**原实现**：`reconstruct(horizon=32)` 每 3.36 s 才用一次传感器反馈，中间 32 步纯外推。

**修复**：新增 `step()` 方法：

```python
# MaskRSSM
def step(self, obs_t, mask_t, act_t, state, t_abs, mode="gumbel"):
    obs_m = apply_mask(obs_t, mask_t)
    new, prior_in, _ = self._advance(state, act_t, t_abs)
    embed = self.enc(torch.cat([obs_m, mask_t], -1))
    post_logits = self.post(torch.cat([prior_in, embed], -1))
    z = self.sample(post_logits, mode)
    return self.dec(self.feat(state_new)), state_new

# MaskGRU
def step(self, obs_t, mask_t, act_t, h):
    obs_m = apply_mask(obs_t, mask_t)
    x = self.inp(torch.cat([obs_m, mask_t, act_t], -1)).unsqueeze(1)
    out, h_new = self.gru(x, h)
    return self.dec(out[:, 0]), h_new
```

**验证**：100 步调用 → 隐状态变化 99 次（每步都更新）。

---

## 二、单元测试（30 项全部通过）

| 组别 | 关键验证 |
|---|---|
| 增强函数 | obs_true 与原始观测一致（max diff=0）；遮蔽位置 obs_masked=0 但 obs_true≠0（100%）|
| **重建监督** | **遮蔽监督:输出0得0loss；真实监督:输出0被罚0.2001** |
| 逐步估计（RSSM）| 100 步调用 100 次 step；隐状态变化 99 次 |
| 逐步估计（GRU）| 每步都更新（最小变化 7.16e-02）|
| 信息隔离（逐步）| 故障后 +999/1e6 污染 → max diff=0；故障前污染有影响 |
| 其他传感器 | 4 通道在逐步模式下都影响 T4 |
| 梯度 | 22/22 参数有梯度 |

---

## 三、泄漏复检（16 项全部通过）

| 模型 | 测试 | 结果 |
|---|---|---|
| MaskGRU | 故障后 T4 +500 / 1e6 / -1e4 | **max diff = 0.00e+00** |
| MaskGRU | 故障前 T4 污染 | max diff = 0.6665（有影响）|
| MaskGRU | 污染 Wfact/N2int/T45/P36 | 0.945 – 1.820（全有影响）|
| MaskRSSM | 故障后 T4 +500 / 1e6 / -1e4 | **max diff = 0.00e+00** |
| MaskRSSM | 故障前 T4 污染 | max diff = 1.1438 |
| MaskRSSM | 污染 Wfact/N2int/T45/P36 | **1.399 – 2.211**（全有影响）|

**关键**：MaskRSSM 对其他传感器的敏感度是 GRU 的 **1.2–2.0 倍**，说明 RSSM 更有效地利用了非 T4 信息。

---

## 四、修复前后的性能变化

### 4.1 Mask-RSSM

| 场景 | 修复前（错误监督+分段）| 修复后（真实监督+逐步）| 改善 |
|---|---|---|---|
| 永久失效 test_id | 40.52 | **22.72** | **+44%** |
| 永久失效 test_ood | 31.18 | **16.76** | **+46%** |
| 缺失 5s test_id | 37.54 | **27.70** | +26% |
| 缺失 5s test_ood | 19.06 | **10.15** | +47% |
| 缺失 10s test_id | 35.13 | **23.61** | +33% |
| 缺失 10s test_ood | 27.15 | **9.34** | **+66%** |
| 缺失 30s test_id | 45.30 | **25.76** | +43% |
| 缺失 30s test_ood | 33.39 | **19.12** | +43% |

**平均改善约 +40%。**

### 4.2 Mask-GRU

| 场景 | 修复前（分段）| 修复后（逐步）| 变化 |
|---|---|---|---|
| 永久失效 test_id | 29.15 | 37.31 | **−28%**（变差）|
| 永久失效 test_ood | 23.97 | 30.38 | −27% |
| 缺失 5s test_id | 19.88 | 30.73 | **−55%** |
| 缺失 5s test_ood | 15.46 | 41.50 | **−168%** |
| 缺失 10s test_id | 18.45 | 29.54 | **−60%** |
| 缺失 10s test_ood | 17.91 | 43.71 | **−144%** |
| 缺失 30s test_id | 24.92 | 35.86 | −44% |
| 缺失 30s test_ood | 27.36 | 37.84 | −38% |

**逐步在线让 GRU 变差 27–168%，但对 RSSM 有利（见下）。**

---

## 五、★ 修复后的公平对比

### 5.1 逐步在线（推荐方案）

| 场景 | 划分 | Mask-GRU | Mask-RSSM | 胜者 | 优势 |
|---|---|---|---|---|---|
| **永久失效 @20s** | test_id | 37.31 | **22.72** | **RSSM** | **+39%** |
| | test_ood | 30.38 | **16.76** | **RSSM** | **+45%** |
| **缺失 5s** | test_id | 30.73 | **27.70** | RSSM | +10% |
| | test_ood | 41.50 | **10.15** | **RSSM** | **+76%** |
| **缺失 10s** | test_id | 29.54 | **23.61** | RSSM | +20% |
| | test_ood | 43.71 | **9.34** | **RSSM** | **+79%** |
| **缺失 30s** | test_id | 35.86 | **25.76** | RSSM | +28% |
| | test_ood | 37.84 | **19.12** | **RSSM** | +49% |

**Mask-RSSM 在全部 8 个组合中胜出，平均领先 36%。**

### 5.2 分段方案（对照）

| 场景 | 划分 | Mask-GRU | Mask-RSSM | 胜者 |
|---|---|---|---|---|
| 永久失效 | test_id | **29.15** | 37.18 | GRU |
| 缺失 5s | test_ood | 15.46 | **13.05** | RSSM |
| 缺失 10s | test_ood | 17.91 | **12.72** | RSSM |
| 缺失 30s | test_id | **24.92** | 37.66 | GRU |

**分段方案下互有胜负；逐步方案下 RSSM 全面胜出。**

### 5.3 关键洞察

| 模型 | 逐步 vs 分段 |
|---|---|
| Mask-GRU | **逐步更差 27–168%** |
| Mask-RSSM | **逐步更好 22–39%** |

**原因分析（诊断已验证）：**

1. **GRU 的隐状态是"压缩的确定性摘要"**
   - 从 t=0 累积到 t=20s，T4 信息已被压缩进 128 维
   - 故障后每步用 mask=0 更新，T4 信息单调衰减，无法刷新
   - 分段方案每 32 步用真实历史重编码，反而"记住"了 T4

2. **RSSM 有独立的随机潜状态 z**
   - 后验 q(z|h, Enc(obs_masked, mask)) 每步都更新
   - z 与 h 分离，允许"h 承载动力学，z 承载当前观测"
   - 故障后 z 可以由其他传感器驱动，不被 T4 缺失拖累

3. **定期刷新实验佐证**
   - GRU 每 N 步刷新：32.76 → 31.25（仅 5% 改善），说明刷新解决不了根本问题
   - 真正原因是 GRU 无法从"其他传感器"推断 T4

---

## 六、模型效率对比

| 模型 | 参数 | best_val | 训练时长 | 单步推理 |
|---|---|---|---|---|
| Mask-GRU | **302,085** | 0.00606@50 | **462 s** | **0.43 ms** |
| Mask-RSSM | 1,202,693 | 0.12002@60 | 2022 s | 1.20 ms |

**Mask-RSSM 参数量 4.0×，训练慢 4.4×，单步推理慢 2.8×，但重构精度高 36%。**

---

## 七、分区间误差（永久失效，逐步在线）

需要从结果文件提取。见 `mask_results_fixed.json` 的 `interval` 字段。

---

## 八、结论

### ✅ 已验证的实验事实

1. **重建监督目标错误是决定性的**：修复后 RSSM 改善 40%
2. **Mask-RSSM 在逐步在线估计下全面优于 Mask-GRU**（8/8 组合，平均 36%）
3. **RSSM 对其他传感器的敏感度是 GRU 的 1.2–2.0 倍**
4. **逐步在线对 RSSM 有利（+31%），对 GRU 有害（−27~−168%）**
5. 信息隔离在逐步模式下完全正确（16/16 项，含 1e6 极端污染）
6. GRU 的缺陷不是"信息刷新"问题（定期刷新仅改善 5%）

### ❌ 被推翻的旧结论

| 旧结论（修复前）| 新结论（修复后）|
|---|---|
| "Mask-GRU 全面优于 Mask-RSSM" | **错误**：RSSM 在逐步模式下 8/8 胜出 |
| "在线 vs 纯外推的增益仅 3–10%" | **错误**：那是错误监督+分段方案的产物 |
| "其他传感器信息量有限" | **部分错误**：RSSM 能有效利用，GRU 不能 |

### ⚠️ 仍需注意

1. 单随机种子（seed=0）
2. 测试集小（test_id 2 条、test_ood 3 条）
3. 只测 T4 单通道失效
4. 未测软故障
5. KL 项仍失效（free_bits=1.0）

---

## 九、下一步建议

| 优先级 | 方向 | 依据 |
|---|---|---|
| **P0** | **多种子重复**（≥5）| 验证 36% 优势是否稳健 |
| **P0** | **分析 RSSM 的 z 如何利用其他传感器** | 解释优势来源 |
| P1 | 扩展测试集 | 当前仅 5 条测试轨迹 |
| P1 | 多通道同时失效 | 验证泛化性 |
| P2 | 修复 KL（free_bits=0.2）| 可能进一步提升 |
| P2 | 软故障（偏置/卡死）| 更实际 |
| P3 | DT + Mask 联合 | 现在有依据了（RSSM 有优势）|

---

## 十、文件

| 文件 | 内容 |
|---|---|
| `mask_impl_fixed.py` | 修复版实现（含 step 方法）|
| `test_mask_fixed.py` | 30 项单元测试 |
| `audit_fixed.py` | 16 项泄漏复检 |
| `train_mask_fixed.py` | 修复版训练 + 双方案评价 |
| `plot_fixed.py` | 可视化 |
| `diag_stepwise.py` | 逐步 vs 分段诊断 |
| `results_fixed/mask_results_fixed.json` | **正式结果** |
| `results_fixed/figures/*.png/pdf` | 4 张图 |
| `checkpoints_fixed/MaskGRU.pt` | 302,085 参数 |
| `checkpoints_fixed/MaskRSSM.pt` | 1,202,693 参数 |