# PC-LWM RSSM 诊断包

> 打包时间：2026-11-03 23:30
> 项目：面向航空发动机数字孪生的机理—数据融合潜在世界模型（PC-LWM）
> 内容：RSSM 基线崩溃的根因诊断 + 修复后的统一性能对比

---

## 一、核心结论

**RSSM 在 A/B/C 数据集上 test_id 误差 1.3027，不是架构缺陷，而是训练/推理采样方式不一致的代码 bug。**

修复后（Gumbel 模式，5 seeds）：

| 模型 | 参数 | test_id | test_ood | 长时 60 s |
|---|---|---|---|---|
| GRU-h128 | 301 K | **0.0766** | **0.0650** | **0.0545** |
| RSSM-d128 (gumbel) | 940 K | 0.0808 ± 0.0016 | 0.0695 ± 0.0006 | 0.0793 |
| RSSM-d128 (argmax) | 940 K | 1.3027 | 0.9832 | 2.2068 |

**差距从 17 倍缩小到 5%。**

---

## 二、根因（代码级）

`rssm_impl.py` 的 `RSSM.imagine`：

```python
sm = mode or ("gumbel" if self.training else "argmax")
```

- 训练（`model.train()`）→ gumbel
- 推理（`model.eval()`）→ argmax

**为什么 argmax 会崩**：先验分布接近均匀。

| 检查项 | 实测 | 均匀基线 |
|---|---|---|
| 先验熵 | 3.333 | ln32 = 3.466 |
| 先验最大概率 | 0.104 | 0.031 |
| **top1 − top2 差** | **0.047** | — |

top1/top2 几乎并列，argmax 等于从近似平局里取一个，并逐步放大偏差。

**train/eval 输出差 159%**（0.678 vs 输出量级 0.425）。

---

## 三、发现的第二个问题（未修）

**`free_bits=1.0` 过高，32 个潜维度的 KL 全部低于阈值（超过比例 0.0%）。**

后果：KL 项被裁剪成常数，梯度为零，先验完全未被约束。
证据：先验熵 2.94 接近均匀 3.47。

---

## 四、包内文件说明

### 报告
| 文件 | 内容 |
|---|---|
| `DIAGNOSIS_REPORT.md` | 完整诊断报告（含 7 项核对、三问回答）|

### 代码（`code/`）
| 文件 | 作用 |
|---|---|
| `rssm_impl.py` | RSSM 实现（可显式指定 mode，无 import 副作用）|
| `run_rssm.py` | RSSM 训练脚本（3 配置）|
| `eval_modes_fixed.py` | 严格模式评价（显式 mode，Gumbel 5 seeds）|
| `diag_rssm_root.py` | 根因诊断（Loss 分解、分布统计、泄漏检查）|
| `diag_argmax.py` | argmax 崩溃的代码级定位 |
| `run_gru_fair.py` | GRU 公平重跑（WINDOW=32）|
| `final_compare.py` | 统一表 + 图生成 |
| `evaluate.py` | 统一数据接口与评价器 |
| `prepare.py` | 数据预处理（26 条 → 统一张量）|

### 配置（`config/`）
| 文件 | 内容 |
|---|---|
| `splits.json` | 数据划分 + 采样参数 + 通道定义 |
| `norm.npz` | 归一化参数（只用 train 拟合）|
| `meta.json` | 26 条轨迹的工况范围 |

**数据划分**：
- train 17 条：A1-A8, B1-B5, C1-C4
- val 4 条：A9, A10, B6, C5
- test_id 2 条：A11, C6
- test_ood 3 条：A12, C7, C8

**训练配置**：
- dt = 0.105 s（step=7），fs = 9.52 Hz
- WINDOW = 32（历史 3.36 s）
- HORIZON = 32（预测 3.36 s）
- 观测 5 通道：Wfact, N2int, S4_Tt(T4), S45_Tt(T45), S36_Pt(P36)
- 动作 3 通道：PLA, Alt, MN

### 诊断数据（`diagnosis/`）
| 文件 | 内容 |
|---|---|
| `rssm_diagnosis.json` | 完整诊断（Loss 分解、分布统计、推理对比、泄漏检查）|
| `argmax_root_cause.json` | argmax 崩溃定量证据 |
| `rssm_mode_eval_fixed.json` | 三模式 × 四划分 × 长时递推 |
| `gru_fair_w32.json` | GRU 公平重跑结果 |
| `final_comparison.json` | 最终统一对比表 |

### 日志（`logs/`）
| 文件 | 内容 |
|---|---|
| `baselines_gru_w64.log` | **旧 GRU（WINDOW=64，已作废，仅留档）** |
| `rssm_train.log` | RSSM 3 配置训练日志 |
| `gru_fair_w32.log` | GRU 公平重跑日志 |
| `diag_root.log` | 根因诊断输出 |
| `eval_modes.log` | 严格模式评价输出 |

### 图（`figures/`）
| 文件 | 内容 |
|---|---|
| `baseline_fixed_compare.png/pdf` | 图 A：修复后基线对比（3 面板）|
| `rollout_curves.png/pdf` | 图 B：长时自由递推误差增长 |

### 检查点（`checkpoints/`）
| 文件 | 内容 |
|---|---|
| `RSSM-d128.pt` | RSSM-d128 权重（3.6 MB，deter=128, stoch=32×32, hidden=128）|

---

## 五、复现方式

```bash
# 环境
python 3.13, torch 2.9.1+cu128, numpy 2.x

# 1. 数据准备
python code/prepare.py

# 2. GRU 公平基线
python code/run_gru_fair.py

# 3. RSSM 训练（约 25 分钟/配置）
python code/run_rssm.py

# 4. 根因诊断
python code/diag_rssm_root.py
python code/diag_argmax.py

# 5. 严格模式评价（Gumbel 5 seeds）
python code/eval_modes_fixed.py

# 6. 生成对比表与图
python code/final_compare.py
```

**注意**：`run_rssm.py` 无 `if __name__ == "__main__"` 保护，直接 import 会触发训练。诊断脚本已改为从 `rssm_impl.py` 导入。

---

## 六、已知问题与待办

### 已修复
- ✅ 训练/推理采样不一致（诊断中用显式 mode 规避）
- ✅ GRU 与 RSSM 的 WINDOW 不公平（已统一为 32）

### 未修（待批准）
1. **`free_bits` 从 1.0 降到 0.2**：让 KL 真正约束先验（预计 25 分钟）
2. **统一推理策略**：
   - 方案 A：训练+推理都用 gumbel（当前最优 0.0808，但有随机性）
   - 方案 B：训练用 soft、推理用 prob（确定性，0.1156）

### 结论限制
- 当前"GRU 略优于 RSSM"的结论建立在 **KL 项失效**的前提下
- 修复 KL 后需重新评估
- **暂不开展双时间尺度与物理约束创新**，等 KL 修复后再定

---

## 七、校验

见 `MANIFEST.txt`（文件清单 + SHA256）。
