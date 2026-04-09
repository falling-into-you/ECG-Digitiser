# CODEBUDDY.md

This file provides guidance to CodeBuddy Code when working with code in this repository.

Reply in Chinese.

## Project Overview

ECG Digitiser converts paper ECG printout images into digital WFDB-format signals. Winner of the George B. Moody PhysioNet Challenge 2024. Combines Hough Transform (rotation detection) with nnU-Net deep learning (segmentation). Paper: https://arxiv.org/abs/2410.14185

## Safety Rules

- **禁止执行删除操作**（rm、unlink、os.remove、shutil.rmtree 等）。需要删除文件时，必须把完整的删除命令发给用户，由用户手动执行。

## Script Conventions

- All shell scripts put **every parameter inside the script** as variables at the top (in a `参数配置` section), not as command-line arguments. Users edit the script to change parameters, then run with no arguments: `bash shells/xxx.sh`.
- Do NOT use positional arguments (`$1`, `$2`) or `--flag` parsing in shell scripts.

## Environment Setup

```bash
git lfs install && git lfs pull          # Pull model weights from LFS
conda create -n ecgdig python=3.11 && conda activate ecgdig
pip install -r requirements.txt
cd nnUNet && pip install . && cd ..      # Must use embedded fork (fixes RGB PNG bug)
```

Always use `conda activate ecgdig`, never `conda run`.

## Key Commands

### Run Digitization
```bash
python -m src.run.digitize -d <input_folder> -m <model_path> -o <output_folder> -v
```
Default model path is `models/M3/`. Input folder should contain ECG images (PNG).

### Train Segmentation Model
```bash
export nnUNet_raw='<path_containing_Dataset500_Signals>'
export nnUNet_preprocessed='<path_for_preprocessed>'
export nnUNet_results='<path_for_results>'

nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity
nnUNetv2_train 500 2d 0 -device cuda
nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling
```

### nnUNet Training Parameters

训练参数需要直接修改 nnUNet 源码（本项目使用嵌入式 fork）：

| 参数 | 文件位置 | 变量名 | 默认值 |
|------|----------|--------|--------|
| **epochs** | `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:152` | `self.num_epochs` | 1000 |
| **batch_size** | `nnUNet/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py:63` | `self.UNet_min_batch_size` | 32 |

**注意**：
- `batch_size` 在 plan 生成时确定，修改后需重新运行 `nnUNetv2_plan_and_preprocess`
- 已生成的 `nnUNetPlans.json` 中的 `batch_size` 也可直接修改，但仅影响当前数据集

### Data Pipeline Parameters

数据处理脚本参数（在 shell 脚本顶部配置）：

| 脚本 | 参数 | 说明 |
|------|------|------|
| `02_postprocess.sh` | `RESAMPLE_FACTOR=3` | 像素插值倍数 |
| `03_merge_datasets.sh` | `MOVE_FILES=true` | true=移动文件(省空间), false=复制 |
| `04_validate_pairs.sh` | `--clean -y` | 自动清理不配对文件 |

Python 脚本支持的关键参数：
- `create_mimic_dataset.py -m` — 移动文件而非复制
- `merge_datasets.py -m` — 移动文件而非复制
- `validate_pairs.py --clean -y` — 自动清理且跳过确认

### Lint
```bash
ruff check .
```
Ruff config (pyproject.toml): 90-char line length, ignores E402, F405, F403, E741.

### Test Scripts
Shell-based manual tests live in `shells/test_shells/`:
- `test_digitize.sh` — run digitization on sample data
- `test_generate.sh` / `test_generate_aug.sh` — test ECG image generation
- `plot_output.sh` — plot digitization results

No automated test suite exists. nnUNet has integration tests under `nnUNet/nnunetv2/tests/`.

## Architecture

### Three-Stage Pipeline

1. **Data Preparation** (`src/ptb_xl/`, `src/mimic/`) — Converts WFDB ECG signals → synthetic ECG images using ecg-image-kit. Generates dense pixel masks for segmentation training.

2. **Segmentation** (embedded `nnUNet/` fork, v2.5) — nnU-Net 2D inference produces per-lead segmentation masks (12 ECG leads + background = 13 classes). The embedded fork is required because upstream nnU-Net has a bug with RGB PNG images.

3. **Digitization** (`src/run/digitize.py`) — Main pipeline entry point:
   - `predict_mask_nnunet()` — Runs nnUNet inference via subprocess
   - `get_rotation_angle()` — Detects page rotation using Hough Transform
   - `cut_to_mask()` — Crops image regions per lead using mask bounds
   - `vectorise()` — Extracts 1D signal from binary mask of each lead
   - Outputs WFDB header (`.hea`) + signal (`.dat`) files

### Directory Map

```
config.py                   — Central ECG parameters (frequency, lead mappings, ADC gain, signal lengths)
src/
  run/
    digitize.py             — Full digitization pipeline (main entry point)
  utils/
    helper_code.py          — WFDB I/O utilities (DO NOT EDIT — Challenge code)
    hall_set.py             — Tensor algebra / log-signature utilities
    plot_output.py          — Plot WFDB signals as standard ECG printouts
  ptb_xl/
    prepare_ptbxl_data.py   — Add PTB-XL metadata to WFDB headers
    prepare_image_data.py   — Add image file info to WFDB headers
    replot_pixels.py        — Create dense pixel masks (upsampled)
    create_train_test.py    — Split data into train/val/test in nnUNet format
  mimic/
    create_mimic_dataset.py — Create dataset from MIMIC-IV-ECG (-m 移动文件)
    merge_datasets.py       — Merge multiple datasets (-m 移动文件)
    prepare_nnunet.py       — Rename files to nnUNet naming (_0000 suffix)
    generate_masks.py       — Generate segmentation masks from JSON coordinates
    validate_pairs.py       — Validate imagesTr/labelsTr pairing (--clean -y 自动清理)
ecg-image-generator/        — Synthetic ECG image generation (separate conda env: ecg_gen)
  gen_ecg_images_from_data_batch.py  — Main batch generator (parallel, 64+ workers)
  gen_ecg_images_from_jsonl.py       — Generate from JSONL mapping
  gen_ecg_images_mixed_layouts.py    — Mixed layout generation (6 layouts)
  ecg_plot.py               — ECG plotting engine (matplotlib-based)
nnUNet/                     — Embedded nnU-Net v2.5 fork (DO NOT replace with upstream)
models/
  M1/                       — Single fold (fold_0) trained model
  M3/                       — Full dataset (fold_all) model — preferred for inference
refs/
  config/                   — YAML configs for inference wrapper, lead layouts
  model/                    — Reference implementations (inference_wrapper, lead_identifier, signal_extractor)
shells/
  batch_generate_image/     — Image generation & mask creation scripts
  train/                    — nnUNet training pipeline (01_preprocess → 02_train → 03_find_best → 04_predict)
  test_shells/              — Manual test scripts
docs/                       — Pipeline documentation (data_generation, training, mimic_pipeline)
```

### ECG Image Assumptions (for pretrained models)

Pretrained models expect: 3 rows × 4 leads (2.5s each) + 1 rhythm strip row (10s), 25mm/s horizontal, 10mm/mV vertical. Images generated with random headers, calibration pulse, ±5° rotation, random B&W.

### ECG Constants (`config.py`)

- Sampling frequency: 500 Hz
- 12 standard leads: I, II, III, aVR, aVL, aVF, V1–V6
- Short window: 2.5s per lead, Long window: 10s rhythm strip
- ADC gain: 1000.0, signal units: mV
- `LEAD_LABEL_MAPPING`: lead name → class index (1–12), background = 0
- `Y_SHIFT_RATIO`: per-lead vertical positioning ratios for cropping

### Key Dependencies

- **PyTorch** (≥2.2) + **nnU-Net v2.5** (embedded fork) for segmentation
- **OpenCV** + **scikit-image** for image processing and Hough Transform
- **wfdb** (4.1.2) for medical signal format I/O
- **TensorFlow/Keras** (2.14) — secondary dependency
- All src modules run as `python -m` from repo root (e.g., `python -m src.run.digitize`)

### ecg-image-generator Requires Separate Env

The `ecg-image-generator/` subproject needs its own conda environment (`ecg_gen`) created from `ecg-image-generator/environment_droplet.yml`. Deactivate `ecgdig` before using it, and reactivate after.

### Model Weights

Stored via Git LFS in `models/M1/` and `models/M3/`. Each contains `nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/` with checkpoint files, dataset.json, and plans.json. M3 (fold_all) is the default for inference.

## 数据处理重要说明

### JSON 坐标文件

ecg-image-generator 生成图像时会同时生成 JSON 文件，包含：
- `plotted_pixels`: 原始像素坐标
- `dense_plotted_pixels`: 插值加密后的像素坐标（由 `replot_pixels.py` 生成）
- `leads`: 每个导联的元数据（名称、采样范围、坐标）
- 图像尺寸、采样频率、网格颜色等元信息

**JSON 文件用途**：
1. **生成分割标签** — `create_mimic_dataset.py` 和 `create_train_test.py` 读取 JSON 中的像素坐标生成标签图（labelsTr）
2. **数据增强追踪** — 如果有 `leads_augmented` 字段，可生成增强版本的标签
3. **调试和验证** — 可用于检查像素坐标是否正确

**JSON 文件存放位置**：
- 原始数据目录：`12x1_*/pXXXX/pXXXXXXXX/sXXXXXXXX/*.json` — **必须保留**
- nnUNet 数据集 imagesTr：复制过来的 JSON — **不影响 nnUNet**，但占用空间

**nnUNet 是否读取 JSON**：
- nnUNet **只读取** `dataset.json` 中 `file_ending` 指定的文件格式（本项目是 `.png`）
- imagesTr 中的 `.json` 文件会被 nnUNet **忽略**，不影响训练
- 但会占用存储空间（40000 个 JSON ≈ 几 GB）

**是否需要清理 imagesTr 中的 JSON**：
- 如果磁盘空间充足：**不需要清理**，保留无害
- 如果需要节省空间：可以清理，但**绝对不要清理原始数据目录的 JSON**

### 标签格式要求

nnUNet 对标签图（labelsTr）的格式要求：
- **必须是单通道灰度图** (mode='L')
- 像素值 = 类别索引 (0=背景, 1-12=各导联)
- **不能是 RGB**（即使三通道值相同也不行）

**已知 Bug（已修复）**：
- `create_mimic_dataset.py` 和 `create_train_test.py` 的 `--gray_to_rgb` 参数
- 这个参数本意是把灰度输入图像转 RGB，但被错误地传给了标签生成函数
- 导致标签变成 RGB 三通道（nnUNet 无法正确处理）
- 修复：标签生成时始终传 `rgb=False`
