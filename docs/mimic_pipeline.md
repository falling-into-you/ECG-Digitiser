# MIMIC-IV-ECG 训练数据全流程

本文档记录从 MIMIC-IV-ECG 原始数据到 nnUNet 模型训练的完整流程。

---

## 1. 整体流程

```
MIMIC-IV-ECG 原始数据 (WFDB)
│
├─ [步骤1] gen_12x1_clean.sh      生成无增强图像（10万张）
├─ [步骤2] gen_12x1_aug.sh        生成有增强图像（10万张）
│
├─ [步骤3] postprocess.sh         转换为 nnUNet 格式（分别处理 clean 和 aug）
│     ├── RGBA → RGB
│     ├── 灰度 → RGB
│     ├── 从 JSON 生成多类别分割 mask
│     └── 生成 dataset.json
│
├─ [步骤4] merge_datasets.sh      合并 clean + aug 数据集（加前缀避免冲突）
│
├─ [步骤5] prepare_nnunet.sh      为 PNG 文件添加 _0000 后缀
│
├─ [步骤6] run_all.sh             nnUNet 预处理 + 训练
│     ├── nnUNetv2_plan_and_preprocess（指纹提取 + 规划 + 预处理）
│     └── nnUNetv2_train（训练）
│
└─ [步骤7] 03_find_best.sh        查找最佳配置（可选，需多折交叉验证）
```

---

## 2. 脚本总览

```
shells/
├── batch_generate_image/
│   ├── gen_12x1_clean.sh       # 步骤1: 生成无增强 ECG 图像
│   ├── gen_12x1_aug.sh         # 步骤2: 生成有增强 ECG 图像
│   ├── postprocess.sh          # 步骤3: 图像 → nnUNet 格式转换
│   ├── merge_datasets.sh       # 步骤4: 合并多个数据集
│   ├── prepare_nnunet.sh       # 步骤5: 添加 _0000 后缀
│   ├── generate_masks.sh       # (备用) 单独生成 mask
│   ├── clean_empty_dirs.sh     # (工具) 清理空目录
│   └── restore_from_nnunet.sh  # (工具) 从 nnUNet 目录还原文件
│
└── train/
    ├── 01_preprocess.sh        # nnUNet 预处理
    ├── 02_train.sh             # nnUNet 训练
    ├── 03_find_best.sh         # 查找最佳配置
    ├── 04_predict.sh           # 推理预测
    └── run_all.sh              # 一键: 预处理 + 训练
```

---

## 3. 详细步骤

### 步骤 1: 生成无增强图像

```bash
bash shells/batch_generate_image/gen_12x1_clean.sh
```

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `INPUT_DIR` | MIMIC WFDB 数据路径 | 输入 |
| `OUTPUT_DIR` | `12x1_clean_10w` | 输出 |
| `SAMPLE_COUNT` | `100000` | 生成 10 万张 |
| `RESOLUTION` | `200` | DPI |

无旋转、无增强、无 `random_bw`，生成干净的 12×1 布局图像。

### 步骤 2: 生成有增强图像

```bash
bash shells/batch_generate_image/gen_12x1_aug.sh
```

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `INPUT_DIR` | MIMIC WFDB 数据路径 | 输入 |
| `OUTPUT_DIR` | `12x1_aug_10w` | 输出 |
| `SAMPLE_COUNT` | `100000` | 生成 10 万张 |

增强参数：`-rot 10`、`--wrinkles`、`--augment`、`-noise 40`。模拟扫描倾斜、纸张褶皱、噪声等。

### 步骤 3: 转换为 nnUNet 格式

分别对 clean 和 aug 各跑一次 `postprocess.sh`（修改顶部 `IMAGE_DIR` 和 `OUTPUT_DIR`）。

```bash
# 第一次：处理 clean 数据
# 修改 postprocess.sh 顶部：
#   IMAGE_DIR="/path/to/12x1_clean_10w"
#   OUTPUT_DIR="/path/to/nnUNet_raw/Dataset500_MIMIC_Clean"
bash shells/batch_generate_image/postprocess.sh

# 第二次：处理 aug 数据
# 修改 postprocess.sh 顶部：
#   IMAGE_DIR="/path/to/12x1_aug_10w"
#   OUTPUT_DIR="/path/to/nnUNet_raw/Dataset500_MIMIC_Aug"
bash shells/batch_generate_image/postprocess.sh
```

| 配置项 | 说明 |
|--------|------|
| `IMAGE_DIR` | 步骤 1/2 的输出目录 |
| `OUTPUT_DIR` | nnUNet 数据集目录 |
| `SPLIT_CSV` | 留空则 `--no_split`，全部作为训练集 |
| `NUM_WORKERS` | 并行进程数，推荐 `128` |
| `RESAMPLE_FACTOR` | 像素插值倍数（已注释掉 replot 步骤） |

执行内容：
- RGBA → RGB 转换
- 灰度 → RGB 转换
- 从 JSON 的 `plotted_pixels` 生成多类别分割 mask
- 生成 `dataset.json`

**注意**：此步骤使用 `mv`（移动），源目录的图片会被移走。

### 步骤 4: 合并数据集

```bash
bash shells/batch_generate_image/merge_datasets.sh
```

| 配置项 | 说明 |
|--------|------|
| `CLEAN_DIR` | Clean 数据集路径 |
| `AUG_DIR` | Aug 数据集路径 |
| `OUTPUT_DIR` | 合并后的 `Dataset500_MIMIC` |
| `NUM_WORKERS` | 并行线程数 |

为避免文件名冲突，自动加前缀：
```
clean 数据: 40792771-0.png → clean_40792771-0.png
aug 数据:   40792771-0.png → aug_40792771-0.png
```

合并后自动重新生成 `dataset.json`。此步骤也使用 `mv`。

### 步骤 5: 添加 _0000 后缀

```bash
bash shells/batch_generate_image/prepare_nnunet.sh
```

| 配置项 | 说明 |
|--------|------|
| `INPUT_DIR` | 步骤 4 的输出目录 |

nnUNet v2 要求图像文件名含 `_0000` 通道后缀：
```
imagesTr/clean_40792771-0.png → imagesTr/clean_40792771-0_0000.png
```

### 步骤 6: 预处理 + 训练

```bash
bash shells/train/run_all.sh
```

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `nnUNet_raw` | `.../nnUNet_raw` | 原始数据父目录 |
| `nnUNet_preprocessed` | `.../nnUNet_preprocessed` | 预处理输出 |
| `nnUNet_results` | `.../nnUNet_results` | 训练结果 |
| `DATASET_ID` | `500` | 数据集 ID |
| `NUM_PROC` | `64` | 预处理并行进程数 |
| `FOLD` | `0` | 验证集折数（0-9） |
| `NUM_GPUS` | `2` | GPU 数量 |
| `CUDA_VISIBLE_DEVICES` | `4,5` | GPU 编号 |

依次执行：
1. **预处理**：`nnUNetv2_plan_and_preprocess`（指纹提取 → 规划 → 预处理）
2. **训练**：`nnUNetv2_train`（自动从 checkpoint 恢复）

也可分步执行：
```bash
bash shells/train/01_preprocess.sh   # 仅预处理
bash shells/train/02_train.sh        # 仅训练
```

### 步骤 7: 查找最佳配置（可选）

```bash
bash shells/train/03_find_best.sh
```

仅在使用多折交叉验证（训练 fold 0-4 或 0-9）后有意义。如果只训了 fold 0，可跳过。

---

## 4. 数据目录结构

### 步骤 1-2 输出

```
12x1_clean_10w/                        # 无增强图像
├── p1003/
│   ├── p10030046/
│   │   ├── p10030046s1000-0.png       # ECG 图像
│   │   └── p10030046s1000-0.json      # 像素坐标配置
│   └── ...
└── ...

12x1_aug_10w/                          # 有增强图像（结构相同）
└── ...
```

### 步骤 3 输出

```
nnUNet_raw/Dataset500_MIMIC_Clean/
├── dataset.json
├── imagesTr/
│   ├── 40792771-0.png                 # 图像（步骤5后加 _0000 后缀）
│   ├── 40792771-0.json                # JSON（保留）
│   └── ...
└── labelsTr/
    ├── 40792771-0.png                 # 分割 mask
    └── ...
```

### 步骤 4-5 输出（最终训练数据）

```
nnUNet_raw/Dataset500_MIMIC/
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

### 步骤 6 输出

```
nnUNet_results/Dataset500_MIMIC/
└── nnUNetTrainer__nnUNetPlans__2d/
    └── fold_0/
        ├── checkpoint_best.pth        # 验证集最佳权重
        ├── checkpoint_final.pth       # 最终权重
        └── progress.png              # 训练曲线
```

---

## 5. 分割 mask 说明

mask 是单通道 PNG，每个像素值代表类别：

| 像素值 | 类别 | 像素值 | 类别 |
|--------|------|--------|------|
| 0 | 背景 | 7 | V1 |
| 1 | I | 8 | V2 |
| 2 | II | 9 | V3 |
| 3 | III | 10 | V4 |
| 4 | aVR | 11 | V5 |
| 5 | aVL | 12 | V6 |
| 6 | aVF | | |

共 13 类（背景 + 12 导联）。

---

## 6. 环境变量

所有 nnUNet 脚本依赖三个环境变量，已在脚本顶部配置：

```bash
export nnUNet_raw="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw"
export nnUNet_preprocessed="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_preprocessed"
export nnUNet_results="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_results"
```

---

## 7. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `plotted_pixels` KeyError | JSON 中没有该字段 | 确认生成时加了 `--store_config 2` |
| mask 全零 | `plotted_pixels_key` 不匹配 | 检查 JSON 中实际字段名，默认用 `plotted_pixels` |
| nnUNet 找不到数据集 | 目录名不以 `Dataset500_` 开头 | 确保 nnUNet_raw 下有 `Dataset500_MIMIC` |
| 图像和标签不配对 | 缺少 `_0000` 后缀 | 跑 `prepare_nnunet.sh` |
| 文件名冲突 | clean 和 aug 有同名文件 | `merge_datasets.sh` 会自动加前缀 |
| 预处理卡住 | 200k 样本 + spawn 模式 | 已修复：fork 模式 + O(n) 文件匹配 |
| `IndexError: boolean index did not match` | image RGB (3通道) 但 mask 单通道 (L) | **mask 必须也是 3 通道 RGB**，见下方说明 |
| `assert data.shape[1:] == seg.shape[1:]` | 同上，nnUNet 要求 image 和 seg 通道数一致 | 同上 |

---

## 8. 关键：mask 必须是 RGB 3通道

nnUNet 使用 `NaturalImage2DIO` 读取 RGB 图片时，返回 shape `(1, 3, H, W)`。预处理阶段有硬断言：

```python
assert data.shape[1:] == seg.shape[1:]
# (3, H, W) != (1, H, W) → 崩溃
```

**image 和 mask 必须都是 RGB 3通道**，否则预处理会在以下位置崩溃：

1. `crop_to_nonzero` — boolean index 维度不匹配
2. `collect_foreground_intensities` — boolean index 维度不匹配
3. `default_preprocessor.py:46` — shape assert 失败
4. resampling — mask 通道数被错误插值

### generate_masks.sh 必须加 --gray_to_rgb

```bash
python3 -m src.mimic.generate_masks \
    -i "$INPUT_DIR" \
    --mask_multilabel \
    --gray_to_rgb \          # ← 必须！让 mask 也变成 3 通道
    --plotted_pixels_key plotted_pixels \
    --num_workers $NUM_WORKERS
```

`create_mimic_dataset.py` 中 `--gray_to_rgb` 会自动将 mask 从 `(H, W)` 扩展为 `(H, W, 3)` 再保存为 RGB PNG。

### 如果 mask 已经是单通道 L 模式

可以用以下命令批量转换（128 进程，约 4 分钟处理 18 万文件）：

```python
from PIL import Image
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

labels_dir = Path('labelsTr路径')

def convert_mask(f):
    img = Image.open(f)
    if img.mode == 'L':
        img = img.convert('RGB')
        img.save(f)

files = list(labels_dir.glob('*.png'))
with ProcessPoolExecutor(max_workers=64) as ex:
    list(ex.map(convert_mask, files))
```

### 验证方法

```bash
python3 -c "
from PIL import Image
img = Image.open('imagesTr/xxx_0000.png')
mask = Image.open('labelsTr/xxx.png')
print(f'image: {img.mode}, mask: {mask.mode}')
# 两个都应该是 RGB
"
```
