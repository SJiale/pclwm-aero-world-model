# START_HERE — 接手顺序

> 本包是 PC-LWM 项目在 2026-11-02 的工作交接。
> **阅读顺序**：本文件 → `HANDOFF_REPORT.md` §0 → `DATASET_CARD.md` → `HANDOFF_REPORT.md` §6

---

## 1. 三个必须先知道的事实

### ① 实测结果：完整观测下 GRU 优于 RSSM（短时和长时都是）

**短时（H=32, 3.36 s）**

| 模型 | 参数量 | 测试集相对 RMSE |
|---|---|---|
| Persistence | 0 | 0.1701 |
| ARX | 288 | 0.1299 |
| **GRU** | 960,400 | **0.0175** |
| RSSM | 3,243,280 | 0.0328 |

**长时自由递推（384 步 = 42 s，不使用未来真值）**

| 模型 | k=1 | k=384 (42s) | 漂移倍数 |
|---|---|---|---|
| Persistence | 1.0143 | 0.3298 | — |
| **GRU** | 0.0043 | **0.0098** | **2.3×** |
| RSSM | 0.0191 | 0.0648 | 3.4× |

**GRU 在长时稳定性上也更好。** 这不是失败，是正常结果。
实验方案 MD §11 明确写了这种情况该怎么处理：转向 B 组（部分可观测 + 隐藏参数）。

### ② 最大缺口：没有做 B 组实验

当前 30 条轨迹**全部使用标称健康参数**（`s_Eff` 未改动）。
RSSM 的价值只可能在「观测缺失 + 隐藏退化」时体现，而这部分**完全没做**。

### ③ 没有源模型就无法生成新数据

AGTF30e 模型（~200 MB）**未打进本包**。
接手者若无法访问原机 `D:\不太用的项目\AGTF30-e-main\`，需要从
https://github.com/nasa/AGTF30 自行获取。

---

## 2. 接手顺序

### 第 1 步：审查（30 分钟）

```
1. 读 HANDOFF_REPORT.md 全文（尤其 §1 训练结果核查、§5 三模型实测）
2. 读 DATASET_CARD.md（数据来源、字段、单位、已知问题）
3. 打开 results/verify_unified.json，确认数字与报告一致
4. 检查 checkpoints/v2_*.pt 能否加载（见 REPRODUCE.md §3）
```

**验收**：能复述「GRU 0.0175 vs RSSM 0.0328」以及「为什么这不是失败」。

### 第 2 步：跑通评估（1 小时）

```bash
# 见 REPRODUCE.md §3，用已打包的检查点重算指标
python scripts/verify_unified.py
```

**验收**：输出与 `results/verify_unified.json` 一致（v2 部分）。

### 第 3 步：验证关键前提（2–3 小时）

**任务：验证 `MWS.*.s_Eff` 是否真的影响仿真结果。**

这是 B 组实验的**唯一前提**。若无效，整个方向要改。

详细步骤见 `HANDOFF_REPORT.md` §6.4。

**验收**：两次运行（`s_Eff` 差 5%）的 `S4_Tt` 终值差异 > 1%。
若 < 0.1%，记录为「参数未生效」，并尝试 `MWS.HPC.s_Wc` 等备选字段。

### 第 4 步：修 v2 训练（1 小时）

当前 `checkpoints/v2_*.pt` 是**训练中途保存**的，实际 epoch 数未知
（Python print 未 flush，日志丢失）。

重跑时**必须**加 `flush=True` 或用 `python -u`。

### 第 5 步：按 MD 6.1 展开（后续）

---

## 3. 目录结构

```
PC-LWM_handoff_20261102/
├── START_HERE.md              ← 本文件
├── HANDOFF_REPORT.md          ← 完整交接报告
├── EXPERIMENT_INDEX.csv       ← 实验-数据-代码-结果 对应表
├── DATASET_CARD.md            ← 数据集说明
├── REPRODUCE.md               ← 环境与复现命令
├── MANIFEST_SHA256.txt        ← 文件校验值
├── NEXT_STEPS.md              ← 下一步任务清单（MD 逐条对照）
├── requirements.txt
├── data/
│   ├── agtf30_traj/           ← 30 条轨迹原始 CSV（data.csv + action.csv）
│   │   ├── d01_step80_a00000_m00_g010/
│   │   ├── ...
│   │   └── d30_multi_a30000_m05_g010/
│   └── prepared/              ← 训练用张量
│       ├── data.npz           ← 6.1 MB，30 轨迹 × 8001 点 × 16 通道
│       ├── splits.json        ← 15/5/10 划分
│       └── meta.json
├── src/aero_world_model/      ← 模型代码（RSSM、GRU、physics loss）
├── scripts/                   ← 数据处理、训练、评估、画图
├── configs/                   ← 配置（见 REPRODUCE.md）
├── results/                   ← 指标 JSON + loss 曲线数据
├── checkpoints/               ← v2_gru.pt (3.7MB), v2_rssm.pt (12.4MB)
├── figures/                   ← 关键图
└── model_gen/                 ← MATLAB 数据生成脚本（需 AGTF30e 才能跑）
```

---

## 4. 关键文件位置速查

| 要找什么 | 看哪里 |
|---|---|
| 实测精度数字 | `results/verify_unified.json` |
| 数据字段与单位 | `DATASET_CARD.md` §3 |
| 划分详情 | `data/prepared/splits.json` + `DATASET_CARD.md` §4 |
| 模型结构 | `src/aero_world_model/models/rssm.py`、`baselines.py` |
| 训练超参 | `scripts/train_v2.py` 顶部常量 |
| 物理约束（未启用）| `src/aero_world_model/losses/physics.py` |
| 数据生成（MATLAB）| `model_gen/agtf_gen2.m` |
| 三模型失效证据 | `HANDOFF_REPORT.md` §5 |
| 下一步任务 | `NEXT_STEPS.md` |

---

## 5. 重要警告

### ⚠ 不要做的事

1. **不要基于当前结果声称「世界模型优于线性模型/GRU」** —— 实测相反
2. **不要把「仿真数据」说成「真实发动机孪生」** —— MD §0 明确要求克制
3. **不要用 v1 的 `ARX=346.17`** —— 那是 16 通道评估，与 v2 的 11 通道不可比。
   统一后是 423.08（v1）/ 0.1299（v2）
4. **不要假设 `s_Eff` 一定生效** —— 必须先做冒烟测试
5. **不要直接跑训练而不加 flush** —— 本次就是因此丢失了训练日志

### ⚠ 已知的环境陷阱

| 陷阱 | 表现 | 解决 |
|---|---|---|
| Python print 缓冲 | 日志文件 0 字节 | `print(..., flush=True)` 或 `python -u` |
| `S4_FAR` 通道 std 极小 | 相对误差被放大 40× | 评估时剔除 |
| `S0_Tt`/`S0_Pt` 轨迹内 std=0 | 归一化后为常数 | 评估时剔除 |
| T-MATS 版本 | v1.2.0 (31 参数) vs v1.3.3 (32 参数) | 必须用 v1.3.3 |
| MATLAB 中文路径 | `-batch` 模式报「文本字符无效」 | 路径写进 .m 文件，或纯 ASCII 中转 |

---

## 6. 联系与归属

| 项 | 内容 |
|---|---|
| 项目 | PC-LWM：面向航空发动机数字孪生的机理—数据融合潜在世界模型 |
| 数据源 | NASA AGTF30e（公开） |
| 论文目标 | 航空学报（中文） |
| 交接日期 | 2026-11-02 |
| 待用户确认的问题 | 见 `HANDOFF_REPORT.md` §6.5 三个优先任务 |

