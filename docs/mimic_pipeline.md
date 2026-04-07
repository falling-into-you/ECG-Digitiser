# MIMIC-IV-ECG 训练数据全流程

本文档记录从 MIMIC-IV-ECG 原始数据到 nnUNet 模型训练的完整流程。

---

## 1. 整体流程

```
MIMIC-IV-ECG 原始数据 (WFDB)
│
├─ [步骤1] 01_gen_12x1_clean.sh     生成无增强图像 + 清理空目录
├─ [步骤2] 02_gen_12x1_aug.sh       生成有增强图像 + 清理空目录
│
├─ [步骤3] 03_postprocess.sh        转换为 nnUNet 格式（分别处理 clean 和 aug）
│     ├── replot_pixels（像素坐标加密）
│     ├── create_mimic_dataset（RGBA→RGB、灰度→RGB、生成 mask）
│     └── 校验 imagesTr/labelsTr 配对（自动清理不配对文件）
│
├─ [步骤4] 04_merge_datasets.sh     合并 clean + aug（加前缀 + 自动清理源目录）
│
├─ [步骤5] 05_prepare_nnunet.sh     为 PNG 添加 _0000 后缀
│
├─ [步骤6] shells/train/            nnUNet 预处理 + 训练
│     ├── 01_preprocess.sh
│     ├── 02_train.sh
│     └── run_all.sh（一键）
│
└─ [步骤7] 03_find_best.sh          查找最佳配置（可选，需多折交叉验证）
```

### 端到端测试

用极小数据量验证全流程是否跑通：

```bash
bash shells/test_shells/test_full_pipeline.sh
```

默认 `SAMPLE_COUNT=10`、单卡、1 折。正式跑时修改脚本顶部参数。

---

## 2. 脚本总览

```
shells/
├── batch_generate_image/
│   ├── 01_gen_12x1_clean.sh    # 步骤1: 生成无增强图像 + 清理空目录
│   ├── 02_gen_12x1_aug.sh      # 步骤2: 生成有增强图像 + 清理空目录
│   ├── 03_postprocess.sh       # 步骤3: replot + 转 nnUNet 格式 + 配对校验
│   ├── 04_merge_datasets.sh    # 步骤4: 合并数据集 + 清理源目录
│   ├── 05_prepare_nnunet.sh    # 步骤5: 添加 _0000 后缀
│   ├── clean_empty_dirs.sh     # (工具) 独立清理空目录
│   └── archive/                # 归档的旧脚本
│       ├── generate_masks.sh
│       └── restore_from_nnunet.sh
│
├── train/
│   ├── 01_preprocess.sh        # nnUNet 预处理
│   ├── 02_train.sh             # nnUNet 训练
│   ├── 03_find_best.sh         # 查找最佳配置
│   ├── 04_predict.sh           # 推理预测
│   └── run_all.sh              # 一键: 预处理 + 训练
│
└── test_shells/
    └── test_full_pipeline.sh   # 端到端全流程测试
```

### 脚本约定

- 所有参数在脚本顶部的 `参数配置` 区域修改，不使用命令行参数
- 直接运行：`bash shells/batch_generate_image/01_gen_12x1_clean.sh`

---

## 3. 详细步骤

### 步骤 1: 生成无增强图像

```bash
bash shells/batch_generate_image/01_gen_12x1_clean.sh
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `INPUT_DIR` | MIMIC WFDB 数据路径 | 输入 |
| `OUTPUT_DIR` | `12x1_clean_2w` | 输出 |
| `SAMPLE_COUNT` | `20000` | 生成数量 |
| `NUM_WORKERS` | `64` | 并行进程数 |
| `RESOLUTION` | `200` | DPI |

无旋转、无增强、无 `random_bw`，生成干净的 12×1 布局图像。生成完成后**自动清理空目录**（MIMIC 目录树会产生大量空目录）。

### 步骤 2: 生成有增强图像

```bash
bash shells/batch_generate_image/02_gen_12x1_aug.sh
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OUTPUT_DIR` | `12x1_aug_2w` | 输出 |
| `SAMPLE_COUNT` | `20000` | 生成数量 |

增强参数：`-rot 10`（旋转）、`--wrinkles`（褶皱）、`--augment`（图像增强）、`-noise 40`（噪声）、`--random_bw 0.2`（黑白）。生成完成后自动清理空目录。

### 步骤 3: 后处理（转 nnUNet 格式）

分别对 clean 和 aug 各跑一次（修改脚本顶部 `IMAGE_DIR` 和 `OUTPUT_DIR`）：

```bash
# 第一次：处理 clean 数据
# 修改 03_postprocess.sh 顶部：
#   IMAGE_DIR="<clean 输出目录>"
#   OUTPUT_DIR="<nnUNet_raw>/Dataset500_MIMIC_Clean"
bash shells/batch_generate_image/03_postprocess.sh

# 第二次：处理 aug 数据
# 修改 03_postprocess.sh 顶部：
#   IMAGE_DIR="<aug 输出目录>"
#   OUTPUT_DIR="<nnUNet_raw>/Dataset500_MIMIC_Aug"
bash shells/batch_generate_image/03_postprocess.sh
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `IMAGE_DIR` | | 步骤 1/2 的输出目录 |
| `OUTPUT_DIR` | | nnUNet 数据集目录 |
| `SPLIT_CSV` | `""` | 留空则 `--no_split`，全部作为训练集 |
| `NUM_WORKERS` | `128` | 并行进程数 |
| `RESAMPLE_FACTOR` | `3` | replot_pixels 像素插值倍数 |

执行内容（3 个子步骤）：

1. **像素坐标加密**（`src.mimic.replot_pixels`）— 在相邻像素间线性插值，生成 `dense_plotted_pixels`（全局进度条，跳过空目录）
2. **转换为 nnUNet 格式**（`src.mimic.create_mimic_dataset`）— RGBA→RGB、灰度→RGB、从 JSON 生成多类别分割 mask、生成 `dataset.json`
3. **配对校验** — 自动检查 imagesTr 和 labelsTr 是否一一对应，清理不配对的文件并更新 `dataset.json`

**注意**：此步骤使用 `mv`（移动），源目录的图片会被移走。

### 步骤 4: 合并数据集

```bash
bash shells/batch_generate_image/04_merge_datasets.sh
```

| 配置项 | 说明 |
|--------|------|
| `CLEAN_DIR` | Clean 数据集路径（Dataset500_MIMIC_Clean） |
| `AUG_DIR` | Aug 数据集路径（Dataset500_MIMIC_Aug） |
| `OUTPUT_DIR` | 合并后的 `Dataset500_MIMIC` |
| `NUM_WORKERS` | 并行线程数 |

为避免文件名冲突，自动加前缀：

```
clean 数据: 40792771-0.png → clean_40792771-0.png
aug 数据:   40792771-0.png → aug_40792771-0.png
```

合并后：
- 自动重新生成 `dataset.json`
- **自动删除源目录**（Clean 和 Aug），避免 nnUNet 发现多个 `Dataset500_*` 导致 ID 冲突

### 步骤 5: 添加 _0000 后缀

```bash
bash shells/batch_generate_image/05_prepare_nnunet.sh
```

nnUNet v2 要求图像文件名含 `_0000` 通道后缀：

```
imagesTr/clean_40792771-0.png → imagesTr/clean_40792771-0_0000.png
```

### 步骤 6: 预处理 + 训练

```bash
# 一键运行
bash shells/train/run_all.sh

# 或分步执行
bash shells/train/01_preprocess.sh
bash shells/train/02_train.sh
```

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `nnUNet_raw` | 需修改 | 包含 Dataset500_MIMIC 的父目录 |
| `nnUNet_preprocessed` | 需修改 | 预处理输出目录 |
| `nnUNet_results` | **项目根目录/nnUNet_results** | 训练结果（自动解析） |
| `DATASET_ID` | `500` | 数据集 ID |
| `FOLD` | `0` | 验证集折数（0-9 或 all） |
| `NUM_GPUS` | `2`/`4` | GPU 数量 |
| `CUDA_VISIBLE_DEVICES` | `4,5` | GPU 编号 |

训练结果输出到项目根目录下 `nnUNet_results/`，方便直接查看 `progress.png` 训练曲线。

**PyTorch 2.8 兼容性**：嵌入的 nnUNet fork 已修复 `polylr.py` 的 `LRScheduler` 参数不兼容问题。如果重装了 pip 版 nnU-Net，需要重新执行 `cd nnUNet && pip install .`。

### 步骤 7: 查找最佳配置（可选）

```bash
bash shells/train/03_find_best.sh
```

仅在使用多折交叉验证（训练 fold 0-9）后有意义。如果只训了单折，可跳过。

---

## 4. 数据目录结构

### 步骤 1-2 输出

```
12x1_clean_2w/                         # 无增强图像（空目录已自动清理）
├── p1003/
│   ├── p10030046/
│   │   └── s41234567/
│   │       ├── 41234567-0.png         # ECG 图像
│   │       └── 41234567-0.json        # 像素坐标配置
│   └── ...
└── ...
```

### 步骤 3 输出

```
nnUNet_raw/Dataset500_MIMIC_Clean/     # 步骤 4 后会被自动删除
├── dataset.json
├── imagesTr/
│   ├── 40792771-0_0000.png            # 图像
│   ├── 40792771-0.json                # JSON（保留）
│   └── ...
└── labelsTr/
    ├── 40792771-0.png                 # 分割 mask（RGB 3 通道）
    └── ...
```

### 步骤 4-5 输出（最终训练数据）

```
nnUNet_raw/Dataset500_MIMIC/           # nnUNet_raw 下唯一的 Dataset500_* 目录
├── dataset.json                       # 13 类（背景 + 12 导联）
├── imagesTr/
│   ├── clean_40792771-0_0000.png      # clean 前缀 + _0000 后缀
│   ├── clean_40792771-0.json
│   ├── aug_40000017-0_0000.png        # aug 前缀 + _0000 后缀
│   ├── aug_40000017-0.json
│   └── ...
└── labelsTr/
    ├── clean_40792771-0.png           # mask（无 _0000 后缀）
    ├── aug_40000017-0.png
    └── ...
```

### 步骤 6 输出（项目根目录）

```
ECG-Digitiser-main/nnUNet_results/     # 在 .gitignore 中，不会被 git 跟踪
└── Dataset500_MIMIC/
    └── nnUNetTrainer__nnUNetPlans__2d/
        └── fold_0/
            ├── checkpoint_best.pth    # 验证集最佳权重
            ├── checkpoint_final.pth   # 最终权重
            ├── progress.png           # 训练曲线（每 epoch 更新）
            └── training_log_*.txt     # 文本日志
```

---

## 5. 分割 mask 说明

mask 是 **RGB 3 通道 PNG**（nnUNet 要求 image 和 mask 通道数一致），每个像素的 R 通道值代表类别：

| 像素值 | 类别 | 像素值 | 类别 |
|--------|------|--------|------|
| 0 | 背景 | 7 | V1 |
| 1 | I | 8 | V2 |
| 2 | II | 9 | V3 |
| 3 | III | 10 | V4 |
| 4 | aVR | 11 | V5 |
| 5 | aVL | 12 | V6 |
| 6 | aVF | | |

共 13 类（背景 + 12 导联）。`03_postprocess.sh` 已自动加 `--gray_to_rgb` 确保 mask 为 RGB。

---

## 6. 环境变量

nnUNet 脚本依赖三个环境变量，已在各脚本顶部配置：

```bash
export nnUNet_raw="<包含 Dataset500_MIMIC 的父目录>"
export nnUNet_preprocessed="<预处理输出目录>"
export nnUNet_results="$(cd "$(dirname "$0")/../.." && pwd)/nnUNet_results"  # 自动解析到项目根目录
```

---

## 7. 已知问题与修复

| 问题 | 原因 | 解决 |
|------|------|------|
| `TypeError: LRScheduler.__init__()` | PyTorch 2.8 移除了 `verbose` 参数 | 已修复 `nnUNet/nnunetv2/training/lr_scheduler/polylr.py`，需 `cd nnUNet && pip install .` |
| `RuntimeError: More than one dataset name found for dataset id 500` | `nnUNet_raw` 下有多个 `Dataset500_*` 目录 | `04_merge_datasets.sh` 已自动清理源目录 |
| `FileNotFoundError: labelsTr/xxx.png` | JSON 损坏导致 mask 生成失败，imagesTr/labelsTr 不配对 | `03_postprocess.sh` 已自动校验并清理不配对文件 |
| `AssertionError: Cannot run DDP if batch size < GPUs` | 样本太少，batch size 不够分配到多卡 | 减少 `NUM_GPUS` 或增加数据量 |
| replot_pixels 刷屏 `0it` | MIMIC 目录树有大量空目录 | `src.mimic.replot_pixels` 已跳过空目录，生成脚本已自动清理空目录 |
| `plotted_pixels` KeyError | JSON 中没有该字段 | 确认生成时加了 `--store_config 2` |
| mask 全零 | `plotted_pixels_key` 不匹配 | 检查 JSON 中实际字段名 |
| `assert data.shape[1:] == seg.shape[1:]` | mask 单通道但 image 是 RGB | `03_postprocess.sh` 已加 `--gray_to_rgb` |

---

## 8. 快速参考

### 完整流程命令（逐步执行）

```bash
conda activate ecgdig

# 1. 生成图像
bash shells/batch_generate_image/01_gen_12x1_clean.sh
bash shells/batch_generate_image/02_gen_12x1_aug.sh

# 2. 后处理（修改脚本顶部路径，分别跑 clean 和 aug）
bash shells/batch_generate_image/03_postprocess.sh   # clean
bash shells/batch_generate_image/03_postprocess.sh   # aug

# 3. 合并 + 添加后缀
bash shells/batch_generate_image/04_merge_datasets.sh
bash shells/batch_generate_image/05_prepare_nnunet.sh

# 4. 训练
bash shells/train/run_all.sh

# 5. 查看训练曲线
# nnUNet_results/Dataset500_MIMIC/nnUNetTrainer__nnUNetPlans__2d/fold_0/progress.png
```

### 端到端测试（10 个样本快速验证）

```bash
bash shells/test_shells/test_full_pipeline.sh
```
