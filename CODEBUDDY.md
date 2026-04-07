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
    create_mimic_dataset.py — Create dataset from MIMIC-IV-ECG
    prepare_nnunet.py       — Rename files to nnUNet naming (_0000 suffix)
    generate_masks.py       — Generate segmentation masks from JSON coordinates
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
