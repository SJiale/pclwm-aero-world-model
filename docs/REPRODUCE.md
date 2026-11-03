# REPRODUCE — 环境与复现命令

> **重要**：以下命令中标注 `[已执行]` 的是本次实际运行过的；
> 标注 `[未执行]` 的是给出了命令但**未验证**，接手者需自行确认。

---

## 1. 环境

### 1.1 本次实际使用的环境

| 项 | 值 | 状态 |
|---|---|---|
| OS | Windows | [已执行] |
| Python | 3.13.5 | [已执行] |
| venv 路径 | `D:\涡轴WM\workspace\.venv` | [已执行] |
| torch | 2.9.1+cu128 | [已执行] |
| numpy | 2.5.4 | [已执行] |
| scipy | 1.18.1 | [已执行] |
| pandas | 3.1.0 | [已执行] |
| PyYAML | 6.0.3 | [已执行] |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB) | [已执行] |
| CUDA | 12.8（torch 自带）| [已执行] |
| MATLAB | R2025a (25.1.0.2943329) | [已执行] |
| T-MATS | v1.3.3 | [已执行] |

### 1.2 安装依赖

```bash
# 建议新建 venv，不要用系统 Python（本机系统 Python 有 numpy 版本冲突）
python -m venv .venv
.venv\Scripts\activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install numpy scipy pandas pyyaml matplotlib
```

**说明**：`requirements.txt` 里是精确版本，但 CPU 环境下 torch 版本需相应调整。
**`[未执行]` 在干净环境上按此安装未验证。**

---

## 2. 数据

### 2.1 已包含的数据

| 文件 | 大小 | 内容 |
|---|---|---|
| `data/prepared/data.npz` | 6.1 MB | 30 轨迹 × 8001 点 × 16 通道 + 动作 |
| `data/prepared/splits.json` | <1 KB | train=15 / val=5 / test=10 |
| `data/prepared/meta.json` | 3.9 KB | 每条轨迹的 profile/Alt/MN/增益 |
| `data/agtf30_traj/*/data.csv` | 30 × 3.1 MB | 原始观测（38 列）|
| `data/agtf30_traj/*/action.csv` | 30 × 249 KB | 燃油实际值与指令 |

**直接从 `data/prepared/` 开始，不需要重新生成。**

### 2.2 重新生成数据（需要 AGTF30e 模型）

**[未执行]** 本包不含 AGTF30e 源模型，以下命令未在本包内验证：

```matlab
% MATLAB R2025a，需先确认 AGTF30e 在以下路径
% D:\不太用的项目\AGTF30-e-main\AGTF30-e-main\AGTF30e\

cd('C:\pclwm')
addpath('C:\pclwm')

% 生成 36 个配置（30 成功，6 个 down 失败）
for run_i = 1:36
    agtf_gen2          % 内部用 run_i 索引配置表
end

% 导出为 CSV
agtf_export2
```

**注意**：
- `agtf_gen2.m` 顶部有 `PROFILES`（6 种 PLA 波形）× `CONDS`（6 种工况）表
- `down` 系列（run_i = 31–36）已知会失败
- 每条轨迹约 40 秒仿真

### 2.3 从 CSV 重建训练张量

```bash
python scripts/prepare2.py
```

**[已执行]** 输出 `data/prepared/{data.npz, splits.json, meta.json}`。
划分种子 42，按 profile 分层。

---

## 3. 评估（用已打包的检查点）

### 3.1 快速核验（**唯一在包内验证通过的脚本**）

```bash
python scripts/eval_package.py
```

**[已执行]** 在**解压后的副本**上实际跑过，路径全相对化，无需修改。
输出与 `results/eval_package.json` 一致（RSSM 因随机采样有 ±3% 波动）。

原始版本（需改路径）：

```bash
cd PC-LWM_handoff_20261102
set PYTHONPATH=%CD%\src
python scripts/eval_v2_ckpt.py
```

**[已执行]** 本次实际跑过，输出：

```
GRU  : train 0.0114  val 0.0154  test 0.0175
RSSM : train 0.0304  val 0.0307  test 0.0328
```

**⚠ 脚本内的路径需要改**：`eval_v2_ckpt.py` 硬编码了
`C:\pclwm\agtf30\prepared2` 和 `C:\pclwm\agtf30\train2`。
接手者需改为本包的相对路径：

```python
DATA = Path(__file__).parent.parent / "data" / "prepared"
T2   = Path(__file__).parent.parent / "checkpoints"
# 并把 ckpt 文件名从 gru.pt/rssm.pt 改为 v2_gru.pt/v2_rssm.pt
```

**这一修改未执行**，需要接手者完成。

### 3.2 统一核验（v1 + v2 对比）

```bash
python scripts/verify_unified.py
```

**[已执行]** 但**依赖未打包的 v1 数据与检查点**（见下）。

---

## 4. 训练（重跑 v2）

```bash
set PYTHONPATH=%CD%\src
python -u scripts/train_v2.py
```

**关键**：**必须用 `-u`**（unbuffered）。上次就是因为没加，日志全丢，
导致无法确认训练完成度。

**[未执行]** 在包内未跑过（包内无 GPU 训练验证）。
**[已执行]** 在原始机器上跑过，但**中途被终止**，完成度未知。

**输出**：
- `checkpoints/v2_gru.pt`、`v2_rssm.pt`（覆盖原文件，建议先备份）
- `train2/results2.json`（含 loss 曲线）

**⚠ 注意**：`train_v2.py` 也硬编码了原机路径，需按 §3.1 的方式修改。

---

## 5. 画图

```bash
python scripts/plot_curves.py    # loss 曲线（v1 数据，非 v2）
python scripts/plot_v2.py         # v2 精度对比
```

**[已执行]** 两者都跑过。
**`plot_v2.py` 依赖 `train2/results2.json`**（v2 训练输出），
该文件因训练中断**未生成**，所以 `plot_v2.py` 实际跑不通。
**`figures/` 里的 `fig33_loss_curves.png` 是 v1 的**，不是 v2。

---

## 6. 完整复现链（从零开始）

按顺序执行。**标注 `[未验证]` 的步骤本包内未完整跑通**。

| 步骤 | 命令 | 状态 |
|---|---|---|
| 1. 装环境 | 见 §1.2 | [未验证] 干净环境 |
| 2. 获取 AGTF30e | GitHub nasa/AGTF30 | [未验证] 本包不含 |
| 3. 生成数据 | MATLAB `agtf_gen2` + `agtf_export2` | [未验证] 需 AGTF30e |
| 4. 建张量 | `python scripts/prepare2.py` | [已执行] |
| 5. 训练 | `python -u scripts/train_v2.py` | [部分执行] 原机跑过但中断 |
| 6. 评估 | `python scripts/eval_v2_ckpt.py` | [已执行] 路径需改 |
| 7. 画图 | `python scripts/plot_v2.py` | [未执行] 缺 results2.json |

---

## 7. 已知的复现障碍

| 障碍 | 影响 | 解决 |
|---|---|---|
| 脚本硬编码原机路径 | 直接跑会 FileNotFoundError | 改为相对路径（见 §3.1）|
| 缺 AGTF30e 模型 | 无法生成新数据 | 从 GitHub 获取 |
| 缺 v1 数据/检查点 | v1 结果无法复现 | 从原机取回，或只引用 JSON 指标 |
| `results2.json` 不存在 | `plot_v2.py` 跑不通 | 重跑训练 |
| `v2_*.pt` 训练完成度未知 | 不能作为论文最终结果 | 重跑并记录日志 |

---

## 8. 硬件与耗时记录

| 任务 | 耗时（本次实测）|
|---|---|
| AGTF30e 单条仿真（120 s 仿真时间）| 33–48 秒 |
| `prepare2.py` | < 10 秒 |
| GRU 训练（v2，1 epoch）| ~1.2 秒 |
| RSSM 训练（v2，1 epoch）| ~40 秒（H=32，3915 窗口）|
| `eval_v2_ckpt.py` | ~30 秒 |
| `verify_unified.py` | ~60 秒 |

GPU 占用：RSSM 训练时约 2900–3200 MiB，29% 利用率，57 °C。

