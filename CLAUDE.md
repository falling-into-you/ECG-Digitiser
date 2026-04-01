# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. Reply in Chinese.

## Project Overview

ECG Digitiser converts paper ECG printout images into digital WFDB-format signals. Winner of the George B. Moody PhysioNet Challenge 2024. Combines Hough Transform (rotation detection) with nnU-Net deep learning (segmentation).

## Key Commands

### Setup
```bash
git lfs install && git lfs pull    # Pull model weights
conda create -n ecgdig python=3.11 && conda activate ecgdig
pip install -r requirements.txt
cd nnUNet && pip install . && cd .. # Must use embedded fork (fixes RGB PNG bug)
```

### Run Digitization
```bash
python -m src.run.digitize -d <input_folder> -m <model_path> -o <output_folder> -v
```
Default model path is `models/M3/`. Input folder should contain ECG images (PNG).

### Train Segmentation Model
```bash
# Set nnUNet env vars first
export nnUNet_raw='<path_containing_Dataset500_Signals>'
export nnUNet_preprocessed='<path_for_preprocessed>'
export nnUNet_results='<path_for_results>'

nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity
nnUNetv2_train 500 2d 0 -device cuda
nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling
```

### Lint
```bash
ruff check .
```
Ruff config: 90-char line length, ignores E402, F405, F403, E741 (in pyproject.toml).

## Architecture

### Three-Stage Pipeline

1. **Data Preparation** (`src/ptb_xl/`) — Converts WFDB ECG signals → synthetic ECG images using ecg-image-kit and PTB-XL dataset. Generates dense pixel masks for segmentation training.

2. **Segmentation** (embedded `nnUNet/` fork, v2.5) — Runs nnU-Net 2D inference to produce per-lead segmentation masks (12 ECG leads + background = 13 classes). The embedded fork is required because upstream nnU-Net has a bug with RGB PNG images.

3. **Digitization** (`src/run/digitize.py`) — Main pipeline entry point:
   - `predict_mask_nnunet()` — Runs nnUNet inference via subprocess
   - `get_rotation_angle()` — Detects page rotation using Hough Transform
   - `cut_to_mask()` — Crops image regions per lead using mask bounds
   - `vectorise()` — Extracts 1D signal from binary mask of each lead
   - Outputs WFDB header + signal files

### Key Source Files

- `config.py` — Central ECG parameters (frequency=500Hz, lead mappings, ADC gain, signal lengths)
- `src/run/digitize.py` — Full digitization pipeline (main entry point)
- `src/utils/helper_code.py` — WFDB I/O utilities (load/save signals, headers, images)
- `src/utils/hall_set.py` — ECG-specific utilities
- `src/ptb_xl/` — Data preparation scripts (prepare_ptbxl_data, prepare_image_data, replot_pixels, create_train_test)

### ECG Image Assumptions

Pretrained models expect: 3 rows × 4 leads (2.5s each) + 1 rhythm strip row (10s), 25mm/s horizontal, 10mm/mV vertical. Images generated with random headers, calibration pulse, ±5° rotation, random B&W.

### Model Weights

Stored in `models/M1/` and `models/M3/` via Git LFS. Contains nnUNet checkpoint files under `nnUNet_results/Dataset500_Signals/`.

## Key Dependencies

- **PyTorch** (≥2.2) + **nnU-Net v2.5** (embedded fork) for segmentation
- **OpenCV** + **scikit-image** for image processing and Hough Transform
- **wfdb** (4.1.2) for medical signal format I/O
- **TensorFlow/Keras** (2.14) — secondary dependency
- All modules run as `python -m` from repo root (e.g., `python -m src.run.digitize`)


## Conda Environment
conda activate ecgdig
NOT conda run

## Safety Rules

- **禁止执行删除操作**（rm、unlink、os.remove、shutil.rmtree 等）。需要删除文件时，必须把完整的删除命令发给用户，由用户手动执行。这包括但不限于：清理坏样本、删除临时文件、清空目录等任何涉及文件删除的场景。

## Script Conventions

- All shell scripts put **every parameter inside the script** as variables at the top (in a `参数配置` section), not as command-line arguments. Users edit the script to change parameters, then run with no arguments: `bash shells/xxx.sh`.
- Do NOT use positional arguments (`$1`, `$2`) or `--flag` parsing in shell scripts.