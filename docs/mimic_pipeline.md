# MIMIC-IV-ECG 训练数据全流程

本文档记录从 MIMIC-IV-ECG 原始数据到 nnUNet 模型训练的完整流程。

---

## 1. 整体流程

```
MIMIC-IV-ECG 原始数据 (WFDB)
│
├─ [步骤1] 01_gen_optimized_dataset.sh   生成图像（4种增强级别）
│
├─ [步骤2] 02_postprocess.sh             像素插值 + 生成标签
│
├─ [步骤3] 03_merge_datasets.sh          合并数据集（只复制 PNG）
│
├─ [可选]  clean_json.sh                 清理 nnUNet_raw 中的 JSON
│
├─ [步骤4] 04_validate_pairs.sh          验证配对
│
├─ [步骤5] 05_prepare_nnunet.sh          添加 _0000 后缀
│
└─ [步骤6] shells/train/                 nnUNet 预处理 + 训练
      ├── 01_preprocess.sh
      ├── 02_train.sh
      └── run_all.sh（一键）
```

---

## 2. 数据流与 JSON 文件

### JSON 文件生命周期

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 01_gen_optimized_dataset.sh                                             │
│ ecg-image-generator                                                     │
│                                                                         │
│   输入: WFDB 信号 (.dat/.hea)                                           │
│   输出: 图像.png + 坐标.json                                            │
│                                                                         │
│   JSON 内容:                                                            │
│   - plotted_pixels: 原始像素坐标                                        │
│   - leads: 每个导联的元数据                                             │
│   - 图像尺寸、采样频率等                                                │
│                                                                         │
│   JSON 状态: ✅ 生成                                                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 02_postprocess.sh                                                       │
│                                                                         │
│   步骤 1: replot_pixels.py                                              │
│   - 读取 JSON 的 plotted_pixels                                         │
│   - 生成 dense_plotted_pixels（插值加密）                               │
│   - 写回 JSON                                                           │
│   JSON 状态: ✅ 读取 + 写入                                             │
│                                                                         │
│   步骤 2: create_mimic_dataset.py                                       │
│   - 复制 PNG + JSON 到 nnUNet_raw/Dataset*/imagesTr/                    │
│   - 读取 JSON 坐标 → 生成 labelsTr/*.png（灰度分割标签）                │
│   JSON 状态: ✅ 读取（用于生成标签）                                    │
│                                                                         │
│   输出目录结构:                                                         │
│   nnUNet_raw/                                                           │
│   ├── Dataset500_MIMIC_Clean/                                           │
│   │   ├── imagesTr/*.png + *.json                                       │
│   │   └── labelsTr/*.png（灰度）                                        │
│   ├── Dataset501_MIMIC_AugLight/                                        │
│   ├── Dataset502_MIMIC_AugMedium/                                       │
│   └── Dataset503_MIMIC_AugHeavy/                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 03_merge_datasets.sh                                                    │
│ merge_datasets.py                                                       │
│                                                                         │
│   只复制 PNG 文件，不复制 JSON                                          │
│   JSON 状态: ❌ 不需要（标签已在上一步生成）                            │
│                                                                         │
│   输出目录结构:                                                         │
│   nnUNet_merge/                                                         │
│   └── Dataset500_MIMIC/                                                 │
│       ├── imagesTr/*.png（无 JSON）                                     │
│       └── labelsTr/*.png                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ [可选] clean_json.sh                                                    │
│                                                                         │
│   清理 nnUNet_raw/Dataset*/imagesTr/*.json                              │
│   （节省空间，每个 JSON 约 20MB，4万个约 800GB）                         │
│   JSON 状态: ❌ 可删除                                                  │
│                                                                         │
│   注意: 原始数据目录的 JSON 会保留                                      │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 04_validate_pairs.sh → 05_prepare_nnunet.sh → nnUNet 训练               │
│                                                                         │
│   JSON 状态: ❌ 完全不需要（nnUNet 只读 PNG）                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### JSON 用途总结

| 阶段 | JSON 用途 | 之后是否需要 |
|------|----------|-------------|
| 01_gen | 生成像素坐标 | ✅ 需要 |
| 02_postprocess (replot) | 读取+写入加密坐标 | ✅ 需要 |
| 02_postprocess (create_dataset) | 读取坐标生成标签 | ❌ **之后不需要** |
| 03_merge 及之后 | - | ❌ 不需要 |

**结论**: JSON 的唯一作用是生成标签图 (labelsTr/*.png)，生成完成后即可删除。

---

## 3. 脚本总览

```
shells/batch_generate_image/
├── 01_gen_optimized_dataset.sh   # 生成图像（4种增强）
├── 02_postprocess.sh             # 像素插值 + 生成标签
├── 03_merge_datasets.sh          # 合并数据集（只复制 PNG）
├── 04_validate_pairs.sh          # 验证配对
├── 05_prepare_nnunet.sh          # 添加 _0000 后缀
├── clean_json.sh                 # [可选] 清理 JSON
├── clean_empty_dirs.sh           # [工具] 清理空目录
└── fix_dataset_format.sh         # [修复] RGB标签转灰度

shells/train/
├── 01_preprocess.sh              # nnUNet 预处理
├── 02_train.sh                   # nnUNet 训练
├── 03_find_best.sh               # 查找最佳配置
├── 04_predict.sh                 # 推理预测
└── run_all.sh                    # 一键: 预处理 + 训练
```

### 脚本约定

- 所有参数在脚本顶部的 `参数配置` 区域修改，不使用命令行参数
- 直接运行：`bash shells/batch_generate_image/01_gen_optimized_dataset.sh`

---

## 4. 详细步骤

### 步骤 1: 生成图像

```bash
bash shells/batch_generate_image/01_gen_optimized_dataset.sh
```

生成 4 种增强级别的图像：
- `12x1_clean_optimized/` — 无增强
- `12x1_aug_light/` — 轻度增强
- `12x1_aug_medium/` — 中度增强
- `12x1_aug_heavy/` — 重度增强

**输出结构**:
```
12x1_clean_optimized/
└── p1000/
    └── p10001234/
        └── s12345678/
            ├── 12345678-0.png    # ECG 图像 (RGBA)
            └── 12345678-0.json   # 像素坐标
```

### 步骤 2: 后处理

```bash
bash shells/batch_generate_image/02_postprocess.sh
```

执行内容：
1. **像素坐标插值** — 生成 `dense_plotted_pixels`
2. **转换为 nnUNet 格式** — RGBA→RGB，从 JSON 生成灰度标签

**输出结构**:
```
nnUNet_raw/
├── Dataset500_MIMIC_Clean/
│   ├── dataset.json
│   ├── imagesTr/
│   │   ├── 12345678-0.png      # RGB 图像
│   │   └── 12345678-0.json     # JSON（可清理）
│   └── labelsTr/
│       └── 12345678-0.png      # 灰度标签（像素值=类别）
├── Dataset501_MIMIC_AugLight/
├── Dataset502_MIMIC_AugMedium/
└── Dataset503_MIMIC_AugHeavy/
```

### 步骤 3: 合并数据集

```bash
bash shells/batch_generate_image/03_merge_datasets.sh
```

- 合并 4 个数据集，添加前缀避免冲突
- **只复制 PNG，不复制 JSON**

**输出结构**:
```
nnUNet_merge/
└── Dataset500_MIMIC/
    ├── dataset.json
    ├── imagesTr/
    │   ├── clean_12345678-0.png
    │   ├── aug_light_12345678-0.png
    │   └── ...                     # 无 JSON
    └── labelsTr/
        ├── clean_12345678-0.png
        └── ...
```

### [可选] 清理 JSON

```bash
bash shells/batch_generate_image/clean_json.sh
```

清理 `nnUNet_raw/Dataset*/imagesTr/` 中的 JSON 文件，释放约 800GB 空间。

**注意**: 原始数据目录 (`12x1_*/`) 的 JSON 会保留，以便将来重新生成标签。

### 步骤 4: 验证配对

```bash
bash shells/batch_generate_image/04_validate_pairs.sh
```

检查 imagesTr 和 labelsTr 是否一一对应，清理不配对的文件。

### 步骤 5: 添加 _0000 后缀

```bash
bash shells/batch_generate_image/05_prepare_nnunet.sh
```

nnUNet v2 要求图像文件名含 `_0000` 通道后缀：
```
imagesTr/clean_12345678-0.png → imagesTr/clean_12345678-0_0000.png
```

### 步骤 6: 训练

```bash
# 一键运行
bash shells/train/run_all.sh

# 或分步执行
bash shells/train/01_preprocess.sh
bash shells/train/02_train.sh
```

---

## 5. 分割标签说明

标签是 **单通道灰度 PNG**，每个像素值代表类别：

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

**重要**: nnUNet 要求标签是单通道灰度图 (mode='L')，不能是 RGB。

---

## 6. 最终目录结构

```
datasets/ECG-Digital-Dataset/mimic/
├── 12x1_clean_optimized/         # 原始生成数据（保留 JSON）
├── 12x1_aug_light/
├── 12x1_aug_medium/
├── 12x1_aug_heavy/
│
├── nnUNet_raw/                   # 步骤2输出（JSON 可清理）
│   ├── Dataset500_MIMIC_Clean/
│   ├── Dataset501_MIMIC_AugLight/
│   ├── Dataset502_MIMIC_AugMedium/
│   └── Dataset503_MIMIC_AugHeavy/
│
├── nnUNet_merge/                 # 步骤3输出（无 JSON）
│   └── Dataset500_MIMIC/         # ← nnUNet 训练用这个
│
└── nnUNet_preprocessed/          # nnUNet 预处理输出
    └── Dataset500_MIMIC/
```

---

## 7. 已知问题与修复

| 问题 | 原因 | 解决 |
|------|------|------|
| 标签是 RGB 而非灰度 | `create_mimic_dataset.py` 的 bug | 已修复，或运行 `fix_dataset_format.sh` |
| JSON 占用空间太大 | 每个约 20MB | 运行 `clean_json.sh` 清理 |
| `splits_final.json` 样本数错误 | 预处理时数据未复制完成 | 删除后重新训练 |
| DICE 值很低 (~0.4) | 标签格式错误或样本数太少 | 检查标签格式和 splits |

---

## 8. 快速参考

```bash
conda activate ecgdig

# 1. 生成图像
bash shells/batch_generate_image/01_gen_optimized_dataset.sh

# 2. 后处理
bash shells/batch_generate_image/02_postprocess.sh

# 3. 合并
bash shells/batch_generate_image/03_merge_datasets.sh

# 4. [可选] 清理 JSON
bash shells/batch_generate_image/clean_json.sh

# 5. 验证 + 准备
bash shells/batch_generate_image/04_validate_pairs.sh
bash shells/batch_generate_image/05_prepare_nnunet.sh

# 6. 训练
bash shells/train/run_all.sh
```
