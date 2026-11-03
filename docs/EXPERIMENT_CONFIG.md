# 模型与实验配置清单

> 自动生成：2026-11-03 23:32
> 对应检查点：`checkpoints/RSSM-d128.pt`

---

## 1. 数据集

| 项 | 值 |
|---|---|
| 来源 | NASA AGTF30e（T-MATS 1.3.3）|
| 轨迹数 | 26 条（A 组 12 / B 组 6 / C 组 8）|
| 原始采样 | 0.015 s，120 s，8001 点 |
| **训练采样** | **0.105 s（step=7），1143 点** |
| 采样频率 | 9.52 Hz，奈奎斯特 4.76 Hz |
| 归一化 | 只用 train 17 条拟合，persist 到 `norm.npz` |

### 数据划分（按完整轨迹，无窗口泄漏）

| 集合 | 条数 | 轨迹 |
|---|---|---|
| **train** | 17 | A1_step_e1, A2_aprbs_e1, A3_msin_e1, A4_ramp_e1, A5_step_e2, A6_aprbs_e2, A7_msin_e2, A8_ramp_e2, B1_alt_slow, B2_alt_fast, B3_mn_slow, B4_mn_fast, B5_flight_climb, C1_seq_pla_first, C2_seq_env_first, C3_opposite, C4_alternate |
| **val** | 4 | A9_step_e3, A10_aprbs_e3, B6_flight_desc, C5_partial_overlap |
| **test_id** | 2 | A11_msin_e3, C6_mn_step |
| **test_ood** | 3 | A12_ramp_e3, C7_threeway, C8_multiphase |

### 通道

**观测（预测目标，5 通道）**
| 索引 | 名称 | 单位 | train mean | train std |
|---|---|---|---|---|
| 0 | Wfact_lb_s | lb/s | 0.667 | 0.307 |
| 1 | N2int | rpm | 4769.207 | 785.943 |
| 2 | S4_Tt (T4) | °R | 2550.001 | 274.024 |
| 3 | S45_Tt (T45) | °R | 1779.530 | 203.539 |
| 4 | S36_Pt (P36) | psia | 306.090 | 112.945 |

**动作（外部输入，3 通道）**：PLA_deg, Alt_ft, MN

---

## 2. 窗口与评价协议

| 项 | 值 |
|---|---|
| WINDOW | **32**（历史 3.36 s）|
| HORIZON | **32**（预测 3.36 s）|
| act 时序 | `a[t-1] → o[t]`（动作领先观测一步）|
| 窗口构造 | `obs=o[s:s+W]`, `act=a[s-1:s-1+W]`, `fut=o[s+W:s+W+H]`, `act_fut=a[s-1+W:s-1+W+H]` |
| stride | train 4 / val 8 / test 8 |
| 窗口数（W=32）| train 4590 / val 540 / test_id 270 / test_ood 405 |
| 相对化 | `abs_RMSE / mean(obs_sd)`，`mean(obs_sd)=336.95` |
| 聚合顺序 | 通道 RMSE → 时间步平均 → 窗口平均 |
| 长时递推 | 每 32 步重编码一次历史，段内自由递推 |

---

## 3. RSSM-d128 架构

| 组件 | 配置 |
|---|---|
| **Encoder** | Linear(5→128) → SiLU → Linear(128→128) → SiLU |
| **Posterior** q(z\|h,x) | Linear(deter+128→128) → SiLU → Linear(128→32×32) |
| **Prior** p(z\|h) | Linear(deter→128) → SiLU → Linear(128→32×32) |
| **Dynamics cell** | GRUCell(input=32×32+3=1027, hidden=128) |
| **Decoder** | Linear(deter+32×32→128) → SiLU → Linear(128→128) → SiLU → Linear(128→5) |
| deter | **128** |
| stoch | **32** |
| classes | **32** |
| hidden | **128** |
| feat_dim | 128 + 1024 = 1152 |
| **参数量** | **939,909** |
| 采样 | Gumbel-softmax(hard=True), tau=1.0, unimix=0.01 |

### 损失

| 项 | 公式 | 权重 |
|---|---|---|
| Reconstruction | MSE(recon, obs)，burn-in 前 1/4 不计 | 1.0 |
| KL（per-dim free bits）| `clamp(KL_per_dim, min=free_bits)` 后取均值 | **0.1** |
| Multi-step | MSE(imagine(32步先验), fut) | 1.0 |
| **Total** | `rec + 0.1*kl + multi` | — |
| **free_bits** | **1.0**（⚠ 诊断发现过高，导致 KL 全被裁剪）| — |
| dyn_scale / rep_scale | 0.5 / 0.1（定义但当前未分离使用）| — |

### 训练

| 项 | 值 |
|---|---|
| 优化器 | Adam(lr=4e-4, weight_decay=1e-5) |
| 调度 | CosineAnnealingLR(T_max=max_ep) |
| batch size | 64 |
| max_epochs | 60 |
| patience | 15（早停）|
| 梯度裁剪 | 100.0 |
| 实际训练 | 60 epoch，约 25 分钟（RTX 5060 Laptop，~22 s/epoch）|
| 随机种子 | 0（torch + numpy）|

### 训练结果

| epoch | loss | val | rec | kl | multi |
|---|---|---|---|---|---|
| 1 | 1.03443 | 6.44733 | 0.45124 | 1.0000 | 0.48320 |
| 10 | 0.12075 | 1.84729 | 0.00672 | 1.0039 | 0.01364 |
| 20 | 0.11256 | 1.14900 | 0.00385 | 1.0039 | 0.00833 |
| 30 | 0.10939 | 0.83656 | 0.00293 | 1.0032 | 0.00614 |
| 40 | 0.10743 | 0.78626 | 0.00246 | 1.0031 | 0.00467 |
| 50 | 0.10661 | 0.72102 | 0.00219 | 1.0030 | 0.00412 |
| **60** | **0.10637** | **0.72481** | **0.00214** | **1.0029** | **0.00395** |

**最佳检查点**：epoch 60（val=0.72481）

**注意**：日志中的 `rec/kl/multi` 是 batch 平均的训练值，与诊断脚本重算的值（rec=0.056, multi=0.459）不同 —— 因诊断用的是 eval 模式（argmax）且 burn-in 口径不同。以诊断脚本为准。

---

## 4. GRU 基线（公平重跑 WINDOW=32）

| 配置 | hidden | mlp | 参数量 | test_id | test_ood |
|---|---|---|---|---|---|
| GRU96 | 96 | 192 | 170,309 | 0.0829 | 0.0791 |
| **GRU128** | **128** | **256** | **300,805** | **0.0766** | **0.0650** |
| GRU192 | 192 | 384 | 672,389 | 0.0861 | 0.0914 |

架构：
- 输入投影：Linear(5+3→hidden) → LayerNorm → SiLU
- 编码：`nn.GRU(hidden, hidden, batch_first=True)`
- 自回归解码：`nn.GRUCell(hidden, hidden)` + MLP 输出层
- 训练：Adam(lr=1e-3, wd=1e-4), CosineAnnealing, bs=64, max 120 ep, patience 25

---

## 5. ARX 基线

| 项 | 值 |
|---|---|
| 阶数 | na=2, nb=1 |
| 输入 | [PLA, Alt, MN]（各 1 阶滞后）|
| 拟合 | 最小二乘闭式解，**只用 train 17 条** |
| **参数量** | **70**（5 输出 × 14 系数）|
| test_id | 0.1332 |
| test_ood | 0.1007 |

---

## 6. 关键诊断数字

### 训练/推理不一致（根因）

| 检查项 | 值 |
|---|---|
| 先验熵 | 3.333（均匀=3.466）|
| 先验最大概率 | 0.104（均匀=0.031）|
| top1−top2 差 | 0.047 |
| train/eval 输出平均绝对差 | 0.678 |
| train/eval 输出相对差 | **159%** |
| argmax 32步输出量级 | 1.099 |
| gumbel 32步输出量级 | 0.541 |

### 分布健康度

| 项 | train | val | test_id |
|---|---|---|---|
| 后验熵 | 3.014 | 3.257 | 3.296 |
| 先验熵 | 2.940 | 3.260 | 3.316 |
| 后验每维类别数 | 6.3 | 6.4 | 6.2 |
| 每维 KL 均值 | 0.329 | 0.190 | 0.155 |
| **KL > free_bits 的维度占比** | **0.0%** | **0.0%** | **0.0%** |

### 泄漏检查（全部通过）

| 项 | 结果 |
|---|---|
| obs 与 fut 重叠 | 0.000047（仅数值误差，实际不重叠）|
| act 索引 | ✅ 正确（3 个窗口验证）|
| act_fut 索引 | ✅ 正确 |
| 因果 a[t-1]→o[t] | ✅ 正确 |
| `model.training` | ✅ False |
| 初始隐状态 | ✅ 每窗口清零 |
| imagine 未来 stage 是否调 post | ✅ **否**（hook 计数）|

---

## 7. 软件环境

| 项 | 版本 |
|---|---|
| Python | 3.13 |
| PyTorch | 2.9.1+cu128 |
| NumPy | 2.x |
| GPU | NVIDIA RTX 5060 Laptop (8 GB, sm_120) |
| CUDA | 12.8 |
| 训练时 GPU 温度 | 55–60 °C |
| 训练时功耗 | 35–42 W（TGP 95 W）|
