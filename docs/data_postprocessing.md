# ECG 训练数据后处理 — 从生成图像到 nnUNet 数据集

本文档说明如何将 `ecg-image-generator` 生成的图像转换为 nnUNet 训练所需的数据格式。

> **前置条件**：已使用 `ecg-image-generator` 生成了带 `--store_config 2` 和 `--mask_unplotted_samples` 的图像和 JSON。

---

## 1. 整体流程

```
ecg-image-generator 输出（clean + aug 两个目录）
│  ├── record-0.png
│  └── record-0.json
│
├─ [步骤1] postprocess.sh (分别对 clean 和 aug 各跑一次)
│     ├── replot_pixels.py     → JSON 中增加 dense_plotted_pixels
│     └── create_train_test.py → 转成 nnUNet 目录格式
│           ├── imagesTr/xxx_0000.png
│           ├── labelsTr/xxx.png
│           └── dataset.json
│
├─ [步骤2] merge_datasets.sh  → 加前缀合并多个数据集，避免文件名冲突
│     ├── imagesTr/clean_xxx_0000.png + aug_xxx_0000.png
│     ├── labelsTr/clean_xxx.png + aug_xxx.png
│     └── dataset.json (自动重新生成)
│
└─ [步骤3] 01_preprocess.sh   → nnUNet 预处理，开始训练
```

---

## 2. 步骤1：像素坐标加密 (replot_pixels.py)

### 为什么需要这步？

`ecg_plot.py` 在记录 `plotted_pixels` 时，**每个采样点只记录一个像素坐标**。但 matplotlib 在两个相邻点之间会画一条直线，这些中间像素没有被记录：

```
plotted_pixels（原始）:  ●           ●           ●
matplotlib 实际画的线:   ●───────────●───────────●
缺失的中间像素:           ○ ○ ○ ○ ○   ○ ○ ○ ○ ○
```

如果直接用 `plotted_pixels` 生成 mask，得到的是一堆**孤立的点**，而非连续的信号线，nnUNet 无法学习。

`replot_pixels.py` 在相邻点之间做**线性插值**，补上中间像素，生成 `dense_plotted_pixels`。

### 像素数对比

| 布局 | plotted_pixels | dense (×3) | dense (×10) |
|------|---------------|------------|-------------|
| 12×1 (500Hz, 10s) | 5,000/导联 | ~15,000 | ~50,000 |
| 3×4 (500Hz, 2.5s) | 1,250/导联 | ~3,750 | ~12,500 |

### 命令

```bash
python -m src.ptb_xl.replot_pixels \
    --dir <生成图像目录> \
    --resample_factor 3 \
    --run_on_subdirs \
    --num_workers 8
```

### 参数说明

| 参数 | 含义 | 建议值 |
|------|------|--------|
| `--dir` | 包含 JSON 文件的目录 | 生成图像的输出目录 |
| `--resample_factor N` | 每两个相邻点之间插入 N-1 个中间点 | 低分辨率(100DPI)用 `3`，高分辨率(200+DPI)用 `5-10` |
| `--run_on_subdirs` | 递归处理子目录 | 数据有子目录结构时加上 |
| `--num_workers N` | 并行进程数 | CPU 核数 |

### 验证

```bash
python -c "
import json
with open('<某个json文件>') as f:
    data = json.load(f)
for lead in data['leads']:
    print(lead['lead_name'], len(lead['plotted_pixels']), '->', len(lead['dense_plotted_pixels']))
"
```

应该看到 `dense_plotted_pixels` 数量 ≈ `plotted_pixels` × resample_factor。

---

## 3. 步骤2：转换为 nnUNet 格式 (create_train_test.py)

### nnUNet 数据集目录结构

```
Dataset500_Signals/
├── dataset.json              # 数据集描述（自动生成）
├── imagesTr/                 # 训练图像
│   ├── 00001_0000.png        # ⚠️ 文件名必须有 _0000 后缀（通道索引）
│   ├── 00002_0000.png
│   └── ...
├── labelsTr/                 # 训练标签（分割 mask）
│   ├── 00001.png             # ⚠️ 同名但没有 _0000 后缀
│   ├── 00002.png
│   └── ...
├── imagesTv/                 # 验证集图像（可选）
└── imagesTs/                 # 测试集图像（可选）
```

### dataset.json 格式

```json
{
    "channel_names": {"0": "Signals"},
    "labels": {
        "background": 0,
        "I": 1, "II": 2, "III": 3,
        "aVR": 4, "aVL": 5, "aVF": 6,
        "V1": 7, "V2": 8, "V3": 9,
        "V4": 10, "V5": 11, "V6": 12
    },
    "numTraining": 样本数量,
    "file_ending": ".png"
}
```

### 标签 mask 说明

标签是**单通道 PNG**，每个像素的值 = 类别 ID：

| 像素值 | 含义 |
|--------|------|
| 0 | 背景 |
| 1 | I 导联 |
| 2 | II 导联 |
| 3 | III 导联 |
| 4 | aVR |
| 5 | aVL |
| 6 | aVF |
| 7 | V1 |
| 8 | V2 |
| 9 | V3 |
| 10 | V4 |
| 11 | V5 |
| 12 | V6 |

mask 的生成逻辑：

```python
mask = np.zeros((height, width), dtype=uint8)   # 全零背景
for lead in json_data["leads"]:
    for [row, col] in lead["dense_plotted_pixels"]:
        mask[row, col] = LEAD_LABEL_MAPPING[lead["lead_name"]]
```

### 命令

```bash
python -m src.ptb_xl.create_train_test \
    -i <生成图像目录> \
    -d <数据集csv（用于划分train/val/test）> \
    -o <输出目录，如 nnUNet_raw/Dataset500_Signals> \
    --mask \
    --mask_multilabel \
    --rgba_to_rgb \
    --gray_to_rgb \
    --rotate_image \
    --plotted_pixels_key dense_plotted_pixels \
    --num_workers 8
```

### 关键参数说明

| 参数 | 含义 | 是否必需 |
|------|------|---------|
| `--mask` | 生成分割 mask | **必需** |
| `--mask_multilabel` | 多类别 mask（每个导联一个类别）| **必需**（否则只有前景/背景二分类）|
| `--plotted_pixels_key` | 使用哪个像素字段 | **必需**，用 `dense_plotted_pixels` |
| `--rgba_to_rgb` | RGBA 转 RGB | 推荐加上 |
| `--gray_to_rgb` | 灰度转 RGB | 推荐加上（黑白图需要）|
| `--rotate_image` | 对旋转增强的图做 mask 同步旋转 | 有旋转增强时加上 |
| `-d <csv>` | 划分 train/val/test 的 CSV | PTB-XL 用 `ptbxl_database.csv`，自定义数据集需自己准备 |

---

## 4. 步骤3：nnUNet 预处理和训练

```bash
# 设置环境变量
export nnUNet_raw='<包含 Dataset500_Signals 的父目录>'
export nnUNet_preprocessed='<预处理输出目录>'
export nnUNet_results='<训练结果目录>'

# 预处理
nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity

# 训练
nnUNetv2_train 500 2d 0 -device cuda

# 找最佳配置
nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling
```

---

## 5. 自定义数据集注意事项

如果你不用 PTB-XL 而是用自己的数据集：

### 划分 train/val/test

`create_train_test.py` 默认依赖 PTB-XL 的 `strat_fold` 列做划分（fold 1-8 训练，9 验证，10 测试）。用自定义数据集时，你可以：

- 准备一个包含 `strat_fold` 列的 CSV，手动标注每条记录的 fold
- 或者跳过 `create_train_test.py`，自己按 nnUNet 的目录结构组织文件和生成 mask

### 手动组织 nnUNet 数据集

如果不用 `create_train_test.py`，你需要自己：

1. 将图像重命名为 `XXXXX_0000.png` 放入 `imagesTr/`
2. 从 JSON 的 `dense_plotted_pixels` 生成 mask，保存为 `XXXXX.png` 放入 `labelsTr/`
3. 编写 `dataset.json`
4. 确保标签值连续（0-12，无间隙）

---

## 6. 完整示例：从头到尾

```bash
conda activate ecgdig

# 1. 生成无增强图像
bash shells/batch_generate_image/gen_12x1_clean.sh

# 2. 生成有增强图像
bash shells/batch_generate_image/gen_12x1_aug.sh

# 3. 后处理无增强数据（像素加密 + 转 nnUNet 格式）
#    修改 postprocess.sh 的 IMAGE_DIR 和 OUTPUT_DIR 指向 clean 数据
bash shells/batch_generate_image/postprocess.sh

# 4. 后处理有增强数据（同上，修改路径指向 aug 数据）
bash shells/batch_generate_image/postprocess.sh

# 5. 合并两个数据集（加前缀避免文件名冲突）
bash shells/batch_generate_image/merge_datasets.sh

# 6. nnUNet 预处理
bash shells/train/01_preprocess.sh

# 7. 训练
bash shells/train/02_train.sh
```

### 合并多数据集说明

无增强和有增强数据的文件名相同（如 `40792771-0_0000.png`），直接合并会冲突。`merge_datasets.sh` 通过给每个数据集的文件加前缀解决：

```
clean 数据集:  40792771-0_0000.png  →  clean_40792771-0_0000.png
aug 数据集:    40792771-0_0000.png  →  aug_40792771-0_0000.png
```

脚本顶部配置要合并的数据集列表，格式为 `"前缀:nnUNet数据集路径"`：

```bash
DATASETS=(
    "clean:/data/.../nnUNet/12x1_clean"
    "aug:/data/.../nnUNet/12x1_aug"
)
OUTPUT_DIR="<合并后的最终数据集路径>"
```

可以添加任意多个数据集。合并后自动生成 `dataset.json`。

---

## 7. 脚本总览

```
shells/batch_generate_image/
├── gen_12x1_clean.sh      # 生成无增强图像
├── gen_12x1_aug.sh        # 生成有增强图像
├── postprocess.sh         # 后处理：像素加密 + 转 nnUNet 格式
└── merge_datasets.sh      # 合并多个数据集（加前缀避免文件名冲突）

shells/train/
├── 01_preprocess.sh       # nnUNet 预处理
├── 02_train.sh            # 训练
├── 03_find_best.sh        # 查找最佳配置
├── 04_predict.sh          # 推理预测
└── run_all.sh             # 一键全流程
```

---

## 8. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| mask 全是 0（空白） | 用了 `plotted_pixels` 而非 `dense_plotted_pixels` | 先跑 `replot_pixels.py`，再用 `--plotted_pixels_key dense_plotted_pixels` |
| mask 是稀疏的点 | `resample_factor` 太小 | 高分辨率图像用 `5-10` |
| 漏了 `--mask` 参数 | `labelsTr/` 为空 | 加上 `--mask` |
| 图像和标签数量不一致 | 文件命名不匹配 | 图像必须是 `XXX_0000.png`，标签是 `XXX.png` |
| nnUNet 报 "Class not in foreground" | 标签值不连续 | 检查 `dataset.json` 的 labels 值为 0-12 无间隙 |
| `dense_plotted_pixels` 全是 0 | 没跑 `replot_pixels.py` | 先跑步骤1 |
