# 数据获取

完整数据集发布在 HuggingFace：**[leran0326/pclwm-agtf30e-dynamic](https://huggingface.co/datasets/leran0326/pclwm-agtf30e-dynamic)**

---

## 方式 1：命令行下载（推荐）

```bash
# 安装
pip install huggingface_hub

# 下载全部（142 MB）
huggingface-cli download leran0326/pclwm-agtf30e-dynamic \
    --repo-type dataset \
    --local-dir ./agtf30e_data
```

## 方式 2：Python

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="leran0326/pclwm-agtf30e-dynamic",
    repo_type="dataset",
    local_dir="./agtf30e_data",
)
```

## 方式 3：只下训练张量（最快，6 MB）

```python
from huggingface_hub import hf_hub_download

for f in ["prepared/data.npz", "prepared/splits.json", "prepared/meta.json"]:
    hf_hub_download(
        repo_id="leran0326/pclwm-agtf30e-dynamic",
        repo_type="dataset",
        filename=f,
        local_dir="./agtf30e_data",
    )
```

**如果只做训练/评估，下这三个文件就够了** —— 本仓库 `data/` 里也有相同文件。

---

## 下载后的目录

```
agtf30e_data/
├── README.md              数据集卡片
├── dataset2/              30 条轨迹（主数据集）
│   ├── d01_step80_a00000_m00_g010/
│   │   ├── data.csv       38 列观测（time_s + 7 截面 × W/Tt/Pt/FAR/ht）
│   │   └── action.csv     燃油实际值 + 指令
│   └── ...（共 30 条）
├── dataset1/              10 条历史数据（作对照，可忽略）
└── prepared/
    ├── data.npz           训练张量（30 轨迹 × 8001 × 16 通道）
    ├── splits.json        train=15 / val=5 / test=10
    └── meta.json          每条的 profile/Alt/MN/增益
```

---

## 快速验证

```python
import numpy as np, json

d = np.load("agtf30e_data/prepared/data.npz", allow_pickle=True)
splits = json.load(open("agtf30e_data/prepared/splits.json"))

print("轨迹数:", len(d["run_ids"]))
print("划分:", {k: len(v) for k, v in splits.items()})

rid = splits["train"][0]
obs = d[f"{rid}_obs"]     # (8001, 16)
act = d[f"{rid}_act"]     # (8001, 1)
print(f"{rid}: obs {obs.shape}, act {act.shape}")

# 单位检查：S0_Tt 应为 518.67 °R（= 288.15 K）
cols = [str(c) for c in d["obs_cols"]]
i = cols.index("S0_Tt")
print(f"进气道温度: {obs[0, i]:.2f} °R = {obs[0, i]*5/9:.2f} K")
```

预期输出：
```
轨迹数: 30
划分: {'train': 15, 'val': 5, 'test': 10}
d03_step80_a10000_m03_g010: obs (8001, 16), act (8001, 1)
进气道温度: 518.67 °R = 288.15 K
```

---

## 单位换算

**AGTF30e 使用英制单位。**

| 物理量 | 数据集单位 | 换算 |
|---|---|---|
| 温度 Tt / Ts | °R | `K = R × 5/9` |
| 压力 Pt / Ps | psia | `kPa = psia × 6.89476` |
| 流量 W | lb/s | `kg/s = lb/s × 0.453592` |
| 焓 ht | BTU/lbm | — |
| 油气比 FAR | — | 无量纲 |

---

## 如需重新生成数据

需要本机安装：
- MATLAB R2025a + Simulink
- AGTF30e 模型（https://github.com/nasa/AGTF30）
- T-MATS v1.3.3（AGTF30e 自带）

然后运行本仓库 `model_gen/` 下的脚本：
- `agtf_gen2.m` — 生成 36 个配置（30 成功，6 个 down 系列失败）
- `agtf_export2.m` — 导出为 CSV

详见 [`docs/REPRODUCE.md`](REPRODUCE.md) §2.2。
