# PC-RSSM 多种子训练：暂停与恢复说明

> 暂停时间：2026-11-05 23:31
> 原因：用户关机
> 状态：**可通过 --resume 无损恢复**

---

## 一、暂停时的状态

### 1.1 完成的种子（3/5）

| seed | E0 | E1 | E2 | E3 | JSON | 检查点 |
|---|---|---|---|---|---|---|
| **seed0** | ✅ | ✅ | ✅ | ✅ | ✅ | 4/4 |
| **seed1** | ✅ | ✅ | ✅ | ✅ | ✅ | 4/4 |
| **seed2** | ✅ | ✅ | ✅ | ✅ | ✅ | 4/4 |
| seed3 | ⚠️ | ⚠️ | ❌ | ❌ | ❌ | 2/4 |
| seed4 | ❌ | ❌ | ❌ | ❌ | ❌ | 0/4 |

**seed3 的 E0/E1 有检查点，但 JSON 无记录** → 明天会重跑覆盖

### 1.2 数据完整性

| 文件 | 状态 |
|---|---|
| `results/multiseed_results.json` | ✅ seed0/1/2 完整（各 4 配置）|
| `checkpoints/seed0_*.pt` | ✅ 4 个 |
| `checkpoints/seed1_*.pt` | ✅ 4 个 |
| `checkpoints/seed2_*.pt` | ✅ 4 个 |
| `checkpoints/seed3_*.pt` | ⚠️ 2 个（E0/E1，将被覆盖）|
| `checkpoints/E*.pt` | ✅ 第一轮实验（已归档，不受影响）|

**seed99 测试数据已清理（4 个检查点）。**

---

## 二、恢复命令

### 2.1 直接恢复（推荐）

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONPATH='C:\pclwm\pclwm_data;C:\pclwm\pclwm_mask;C:\pclwm\pclwm_pcrssm'
$vpy='D:\涡轴WM\workspace\.venv\Scripts\python.exe'
& $vpy -u 'C:\pclwm\pclwm_pcrssm\train_multiseed.py' --seeds 0,1,2,3,4 --resume `
    > 'C:\pclwm\multiseed_resume.log' 2>&1
```

### 2.2 预期行为

```
已加载 3 个 seed 的结果
跳过 seed0（4/4 配置已完成）
跳过 seed1（4/4 配置已完成）
跳过 seed2（4/4 配置已完成）
续跑 seed3（已完成 0/4: []）       ← JSON 中无 seed3
  [E0_baseline] ... 重新训练
  [E1_static]   ... 重新训练
  [E2_dynamic]  ... 训练
  [E3_both]     ... 训练
续跑 seed4（已完成 0/4: []）
  ...
```

**预计耗时**：seed3 (126 分) + seed4 (126 分) = **约 4.2 小时**

### 2.3 验证恢复是否正常

```powershell
# 启动后 30 秒内，应看到：
Get-Content 'C:\pclwm\multiseed_resume.log' -Encoding UTF8 | Select-Object -First 20
# 期望包含 "已加载 3 个 seed 的结果" 和 "跳过 seed0/1/2"
```

---

## 三、本轮已做的代码改进（明天生效）

### 3.1 单配置粒度 resume

**原逻辑**：`if key in ALL: skip` —— 整 seed 跳过  
**新逻辑**：检查 `cfgs` 中每个配置，只跳过已完成的

```python
if key in ALL and _args.resume:
    _have = set(ALL[key].get("cfgs", {}).keys())
    _need = {c[0] for c in CFGS}
    if _need <= _have:
        print(f"跳过 {key}（4/4 配置已完成）")
        continue
    print(f"续跑 {key}（已完成 {len(_have)}/4）")
```

### 3.2 每配置完成后立即落盘

**原逻辑**：JSON 只在 seed 全部结束后写入  
**新逻辑**：每个配置完成后立即写入

```python
# 每配置完成后
ALL[key] = seed_rec
RESFILE.write_text(json.dumps(ALL, ...))
print(f"-> 已增量保存 {key}/{name}")
```

**这防止了本次的问题**：seed3-E0/E1 训练完但 JSON 没记录。

### 3.3 备份

原始脚本已备份为 `train_multiseed.py.bak`。

---

## 四、本轮已有的实验发现（3 个完整种子）

### 4.1 seed0 配对结果

| 配置 | test_id | 30s(mean) | 60s(mean) | N2dot R² | N3dot R² |
|---|---|---|---|---|---|
| E0 | 0.0858 | 0.0894 | 0.0902 | 0.9042 | 0.9162 |
| E1 | **0.0841** | 0.0891 | **0.0883** | 0.9252 | 0.9078 |
| E2 | 0.0851 | 0.0895 | 0.0899 | 0.9097 | 0.9327 |
| **E3** | 0.0847 | 0.0918 | 0.0906 | **0.9365** | **0.9431** |

**关键观察**：
- **物理 Probe**：E3 最好（+3.6% / +2.9%）
- **连续推演**：E3 反而略差（30s −2.7%）
- **短期**：E1/E3 有 1–2.8% 改善

### 4.2 待完成的分析

**seed1/seed2 已完整，可做 3 种子配对分析**，但为保持一致性，建议等 5 种子全部完成后再统一分析。

**若你希望明天开机后先看 3 种子初步结果，可运行**：
```powershell
$env:PYTHONPATH='C:\pclwm\pclwm_data;C:\pclwm\pclwm_mask;C:\pclwm\pclwm_pcrssm'
$vpy='D:\涡轴WM\workspace\.venv\Scripts\python.exe'
& $vpy -u 'C:\pclwm\pclwm_pcrssm\interim_v2.py'
```

---

## 五、最终要回答的问题

> **物理锚定改善的是潜空间可解码性、长期动力学预测能力，还是两者兼有？**

**当前 1 个种子的证据**：
- 可解码性：✅ 改善（+3.2% / +2.7%）
- 长期预测：❌ 无改善（−2.7% / −0.5%）

**需 5 个种子确认是否一致。**

---

## 六、文件清单

| 文件 | 状态 |
|---|---|
| `checkpoints/seed0-2_*.pt` | ✅ 12 个有效 |
| `checkpoints/seed3_E0/E1.pt` | ⚠️ 将被覆盖 |
| `checkpoints/E*.pt` | ✅ 第一轮，已归档 |
| `results/multiseed_results.json` | ✅ seed0/1/2 完整 |
| `train_multiseed.py` | ✅ 已改（单配置 resume + 增量落盘）|
| `train_multiseed.py.bak` | ✅ 原始备份 |
| `PC_RSSM_INTERIM_AUDIT.md` | ✅ 已提交 GitHub |
