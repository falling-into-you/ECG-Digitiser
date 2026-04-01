# ECG 图像生成器 — 数据处理文档

本文档说明如何使用 `ecg-image-generator` 从 WFDB 格式的 ECG 时序信号生成合成 ECG 打印件图像，用于 nnU-Net 分割模型的训练。

> **注意**：本项目的 generator 基于 [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) 做了大量定制修改，与原版不兼容。

---

## 1. 相对原版的主要修改

| 功能 | 原版 ecg-image-kit | 本项目修改版 |
|------|-------------------|-------------|
| 多进程并行 | 不支持 | `--num_workers N`，流式任务队列，支持 256+ 进程 |
| 纯图像模式 | 不支持 | `--image_only` 跳过 WFDB 写入，速度快 30-40% |
| 混合布局生成 | 不支持 | `gen_ecg_images_mixed_layouts.py` 加权随机分配 6 种布局 |
| JSONL 输入 | 不支持 | `gen_ecg_images_from_jsonl.py` 从 JSONL 文件映射生成 |
| 实时日志 | 不支持 | 自动记录所有生成图像路径到 `logs/generated_images_*.txt` |
| 多节律条 | 仅 1 条 | `--full_mode "V1,II,V5"` 支持多条节律条 |
| 生成数量限制 | 不支持 | `--max_num_images N` 限制处理文件数 |

---

## 2. 环境

generator 的所有依赖已包含在主项目的 `ecgdig` 环境中，无需额外安装：

```bash
conda activate ecgdig
cd ecg-image-generator
```

---

## 3. 输入数据格式

输入目录需要包含 WFDB 格式的 ECG 记录：
- `*.hea` — 头文件（采样率、导联名、ADC 增益等）
- `*.dat` — 信号数据文件

支持嵌套子目录结构，generator 会递归扫描。

---

## 4. 四个生成脚本

### 4.1 `gen_ecg_images_from_data_batch.py` — 批量生成（最常用）

从 WFDB 记录批量生成 ECG 图像，固定布局。

```bash
cd ecg-image-generator

# 快速测试（1 张图）
python gen_ecg_images_from_data_batch.py \
    -i <input_dir> \
    -o <output_dir> \
    --config_file config_3x4.yaml \
    --num_columns 4 \
    --full_mode II \
    --max_num_images 1 \
    --image_only

# 大批量生产
python gen_ecg_images_from_data_batch.py \
    -i <input_dir> \
    -o <output_dir> \
    --config_file config_3x4.yaml \
    --num_columns 4 \
    --full_mode II \
    --store_config 2 \
    --mask_unplotted_samples \
    --print_header \
    --random_print_header 0.7 \
    --calibration_pulse 0.6 \
    --fully_random \
    -rot 5 \
    --random_bw 0.1 \
    --num_workers 8 \
    --image_only
```

### 4.2 `gen_ecg_images_mixed_layouts.py` — 混合布局生成

为不同记录**随机分配不同布局**，通过权重控制各布局比例，适合生成多样化训练数据。

```bash
python gen_ecg_images_mixed_layouts.py \
    -i <input_dir> \
    -o <output_dir> \
    -se 42 \
    --num_workers 64 \
    --image_only \
    --layout_weights "6x2_1R:0.30,3x4_1R:0.30,12x1:0.25,6x2:0.05,3x4_3R:0.05,3x4:0.05" \
    --layout_manifest manifest.json
```

6 种预定义布局：

| 布局名 | 配置文件 | 列数 | 节律条 | 每导联时长 |
|--------|---------|------|--------|-----------|
| `3x4_1R` | config_3x4.yaml | 4 | II | 2.5s |
| `3x4_3R` | config_3x4.yaml | 4 | V1,II,V5 | 2.5s |
| `3x4` | config_3x4.yaml | 4 | 无 | 2.5s |
| `6x2_1R` | config_6x2.yaml | 2 | II | 5s |
| `6x2` | config_6x2.yaml | 2 | 无 | 5s |
| `12x1` | config_12x1.yaml | 1 | 无 | 10s |

### 4.3 `gen_ecg_images_from_jsonl.py` — JSONL 输入生成

从 JSONL 文件读取记录映射关系，灵活控制输入输出路径。

```bash
python gen_ecg_images_from_jsonl.py \
    --jsonl_file <input.jsonl> \
    -i <base_input_dir> \
    -o <output_dir> \
    --num_columns 1 \
    --full_mode None \
    --num_workers 64 \
    --image_only
```

### 4.4 `gen_ecg_image_from_data.py` — 单文件生成

底层单文件处理脚本，被上面三个批量脚本调用。通常不直接使用。

---

## 5. 参数详解

### 5.1 布局参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| `--config_file` | YAML 配置文件，定义导联排列顺序 | `config_3x4.yaml` |
| `--num_columns` | 列数 | `4` (3x4), `2` (6x2), `1` (12x1) |
| `--full_mode` | 底部节律条导联，`None` 则不画 | `II`, `"V1,II,V5"`, `None` |

### 5.2 性能参数

| 参数 | 说明 | 建议 |
|------|------|------|
| `--num_workers N` | 并行进程数 | CPU 核数，大数据集建议 8-64 |
| `--image_only` | 只生成 PNG，不写 WFDB 文件 | 训练用图像时加上 |
| `--max_num_images N` | 限制处理文件数 | 测试时用 `1` 或 `10` |

### 5.3 标注参数

| 参数 | 说明 |
|------|------|
| `--store_config 2` | 保存完整 JSON 配置（含网格颜色、增强参数等） |
| `--mask_unplotted_samples` | 未绘制区域标记为 NaN（训练 mask 必需） |
| `--lead_bbox` | 在 JSON 中保存导联**信号波形**的边界框坐标（用于目标检测，不影响图像本身；启用时会禁用裁剪增强） |
| `--lead_name_bbox` | 在 JSON 中保存导联**名称文字**的边界框坐标（用于目标检测，不影响图像本身） |

> `--lead_bbox` 和 `--lead_name_bbox` 只写元数据，**不改变图像**。做分割训练时不需要加。

### 5.4 图像增强参数

增强按 4 个阶段顺序应用：

```
原始 ECG 信号
    ↓
[阶段1] ECG 绘图 — 网格、校准脉冲、黑白、头部信息
    ↓
[阶段2] 手写文字 (--hw_text)
    ↓
[阶段3] 纸张褶皱 (--wrinkles)
    ↓
[阶段4] 图像增强 (--augment) — 旋转 → 噪声 → 裁剪 → 色温
    ↓
最终图像
```

#### 阶段1：ECG 绘图参数

这些参数使用**伯努利分布**，每张图以概率 P 决定是否应用：

| 参数 | 类型 | 默认 | 含义 | 视觉效果 |
|------|------|------|------|---------|
| `--calibration_pulse P` | float | 1.0 | 显示校准脉冲的概率 | 每行开头的 1mV 标准方波 |
| `--random_grid_present P` | float | 1.0 | 显示背景网格的概率 | ECG 纸的红/彩色/灰色栅格线 |
| `--random_bw P` | float | 0.0 | 黑白渲染的概率 | 灰色网格 + 黑色信号（vs 彩色）|
| `--random_print_header P` | float | 0.0 | 打印患者信息的概率 | 图像顶部的患者信息文字 |
| `--print_header` | bool | False | 强制打印患者信息 | 等价于 `--random_print_header 1.0` |

网格颜色控制：

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `--standard_grid_color N` | int | 5 | 网格颜色：1=棕, 2=粉, 3=蓝, 4=绿, **5=红** |
| `--random_grid_color` | bool | False | 忽略上面的固定颜色，每张图随机选颜色 |

导联名显示：

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `--remove_lead_names` | bool | True(显示) | 设为 False 隐藏 "I"、"II"、"V1" 等导联标签 |

#### 阶段2：手写文字 (`--hw_text`)

在图上叠加 TensorFlow 生成的手写医学术语。**需要 TensorFlow**，单张耗时从 ~0.7s 增加到 ~6s。

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `--hw_text` | bool | False | 启用手写文字 |
| `--num_words N` | int | 5 | 手写单词最大数量（实际从 [2, N] 随机）|
| `--x_offset N` | int | 30 | 文字 X 方向最大像素偏移（从 [1, N] 随机）|
| `--y_offset N` | int | 30 | 文字 Y 方向最大像素偏移（从 [1, N] 随机）|
| `--hws F` | float | 0.2 | 手写文字占图像宽度的比例 (0.2 = 20%) |

> 对分割训练影响不大，但极其耗时。建议训练时**不加**。

#### 阶段3：纸张褶皱 (`--wrinkles`)

模拟纸张折叠痕迹和皱纹纹理（从 `CreasesWrinkles/wrinkles-dataset/` 随机选取纹理叠加）。

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `--wrinkles` | bool | False | 启用褶皱和皱纹 |
| `--crease_angle N` | int | 90 | 褶皱线最大角度（0°=水平, 90°=垂直, 从 [0, N] 随机）|
| `--num_creases_vertically N` | int | 10 | 垂直方向最大褶皱数（从 [1, N] 随机）|
| `--num_creases_horizontally N` | int | 10 | 水平方向最大褶皱数（从 [1, N] 随机）|

褶皱效果：用高斯模糊绘制折痕线条，改变局部亮度模拟折叠痕迹。

#### 阶段4：图像增强 (`--augment`)

使用 imgaug 库做几何变换和像素级增强：

| 参数 | 类型 | 默认 | 实际取值方式 | 含义 |
|------|------|------|------------|------|
| `--augment` | bool | False | — | 启用图像增强 |
| `-rot N` | int | 0 | 均匀取 [-N, +N]° | 随机旋转（模拟扫描倾斜）|
| `-noise N` | int | 50 | 随机取 [1, N] | 高斯噪声标准差（模拟扫描噪声）|
| `-c F` | float | 0.01 | 随机取 [0, F] | 四边裁剪比例（0.01 = 每边最多裁 1%）|
| `-t T` | int | 40000 | **代码中被覆盖**：50% 取 [2000,4000]K(冷蓝), 50% 取 [10000,20000]K(暖黄) | 色温（模拟不同光照）|

> **注意**：`-t` 参数的命令行值在代码中不生效，色温始终在冷蓝和暖黄之间随机。
> **注意**：启用 `--lead_bbox` 时，裁剪被强制禁用 (`crop=0`)。

#### 组合开关：`--fully_random`

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `--fully_random` | bool | False | 手写文字 / 褶皱 / 图像增强 各以 **50% 概率**独立随机启用或禁用 |

阶段1 的参数（网格、校准脉冲等）不受 `--fully_random` 影响，始终按各自的概率值生效。

#### 分辨率参数

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `-r N` | int | 200 | 图像分辨率 (DPI) |
| `--random_resolution` | bool | False | 从 [50, N] 随机选择分辨率 |

#### 确定性控制参数

默认情况下所有增强参数在各自范围内随机。以下参数可**固定**某个增强值（用于调试）：

`--deterministic_offset`、`--deterministic_num_words`、`--deterministic_hw_size`、
`--deterministic_angle`、`--deterministic_vertical`、`--deterministic_horizontal`、
`--deterministic_rot`、`--deterministic_noise`、`--deterministic_crop`、`--deterministic_temp`

---

## 6. 配置文件说明

四个 YAML 配置文件控制导联排列顺序：

```yaml
# config_3x4.yaml
paper_len: 10.0             # 纸张总长度 (秒)
abs_lead_step: 10           # 每帧步进 (秒)
format_4_by_3:              # 3x4 布局列分组
  - ["I", "II", "III"]                              # 第1列
  - ["aVR", "aVL", "aVF", "AVR", "AVL", "AVF"]     # 第2列
  - ["V1", "V2", "V3"]                              # 第3列
  - ["V4", "V5", "V6"]                              # 第4列
leadNames_12:               # 从下到上的导联绘制顺序
  ["III", "aVF", "V3", "V6", "II", "aVL", "V2", "V5", "I", "aVR", "V1", "V4"]
tickLength: 8               # 列间分隔线长度
tickSize_step: 0.002        # 分隔线步进
```

各配置的 `leadNames_12` 不同，影响导联从上到下的排列顺序。

---

## 7. 输出结构

```
<output_dir>/
├── [子目录结构保持与输入一致]
│   ├── record-0.png            # 生成的 ECG 图像
│   ├── record-0.json           # 配置信息 (--store_config 时)
│   ├── record.hea              # WFDB 头文件 (非 --image_only 时)
│   └── record.dat              # WFDB 数据文件 (非 --image_only 时)
└── [ecg-image-generator]/logs/
    └── generated_images_YYYYMMDD_HHMMSS.txt   # 生成图像路径清单
```

---

## 8. 典型工作流

### 快速测试

```bash
cd ecg-image-generator
conda activate ecgdig

# 用 1 条记录测试 3x4+1R 布局
python gen_ecg_images_from_data_batch.py \
    -i data/s40689238 \
    -o outputs/test \
    --config_file config_3x4.yaml \
    --num_columns 4 \
    --full_mode II \
    --max_num_images 1 \
    --image_only
```

### 12×1 布局训练数据（带适度增强）

```bash
python gen_ecg_images_from_data_batch.py \
    -i <wfdb_dataset_dir> \
    -o <output_dir> \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --full_mode None \
    --mask_unplotted_samples \
    --store_config 2 \
    --print_header \
    --random_print_header 0.5 \
    --calibration_pulse 0.8 \
    --random_grid_present 0.9 \
    --random_bw 0.2 \
    --standard_grid_color 5 \
    -rot 5 \
    --wrinkles \
    --crease_angle 45 \
    --num_creases_vertically 5 \
    --num_creases_horizontally 5 \
    --augment \
    -noise 40 \
    -c 0.01 \
    --num_workers 8 \
    --image_only
```

### 3×4+1R 布局训练数据（带全部增强）

```bash
python gen_ecg_images_from_data_batch.py \
    -i <wfdb_dataset_dir> \
    -o <output_dir> \
    --config_file config_3x4.yaml \
    --num_columns 4 \
    --full_mode II \
    --store_config 2 \
    --mask_unplotted_samples \
    --print_header \
    --random_print_header 0.7 \
    --calibration_pulse 0.6 \
    --fully_random \
    -rot 5 \
    --random_bw 0.1 \
    --num_workers 8
```

### 混合布局训练数据（提高泛化能力）

```bash
python gen_ecg_images_mixed_layouts.py \
    -i <wfdb_dataset_dir> \
    -o <output_dir> \
    -se 42 \
    --num_workers 64 \
    --image_only \
    --calibration_pulse 1.0 \
    --layout_weights "6x2_1R:0.30,3x4_1R:0.30,12x1:0.25,6x2:0.05,3x4_3R:0.05,3x4:0.05" \
    --layout_manifest layout_manifest.json
```

---

## 9. Shell 脚本参考

`ecg-image-generator/shells/` 下提供了预配置的脚本：

| 脚本 | 布局 | 说明 |
|------|------|------|
| `layout_3x4.sh` | 3×4 无节律条 | 测试用 |
| `layout_3x4_1R.sh` | 3×4 + 1 节律条 (II) | **当前模型训练布局** |
| `layout_3x4_3R.sh` | 3×4 + 3 节律条 (V1,II,V5) | 多节律条布局 |
| `layout_6x2.sh` | 6×2 无节律条 | 测试用 |
| `layout_6x2_1R.sh` | 6×2 + 1 节律条 (II) | 高时间分辨率 |
| `layout_12x1.sh` | 12×1 单列 | 完整时长导联 |
| `mimic_iv.sh` | 12×1 | MIMIC-IV 数据集生成 |
| `mimic_iv_v2.sh` | 混合布局 | MIMIC-IV 混合布局生成 |
| `ptbxl.sh` | 12×1 | PTB-XL 从 JSONL 生成 |

---

## 10. 注意事项

1. **环境**：所有依赖已包含在 `ecgdig` 环境中，直接 `conda activate ecgdig` 即可使用。
2. **磁盘空间**：每张图约 100-200KB，大数据集（如 PTB-XL 21k 记录 × 4 图/记录）需要约 8-15GB。
3. **`--mask_unplotted_samples`**：生成训练 mask 时必须加上，否则未绘制区域不会被正确标记。
4. **手写文字功能 (`--hw_text`)**：依赖 TensorFlow，单张耗时从 ~0.7s 增加到 ~6s，对分割训练影响小，建议不加。
5. **`--full_mode None`**：字符串 `"None"` 表示不画节律条，Python `None` 也可以。
6. **`--lead_bbox`**：启用时会禁用裁剪增强（保护边界框准确性），做分割训练时不要加。
