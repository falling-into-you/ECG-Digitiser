#!/bin/bash
# 端到端全流程测试：图像生成 → 后处理 → 合并 → nnUNet 预处理 → 10 折训练
# 用法: bash shells/test_shells/test_full_pipeline.sh
#
# 每个步骤都有独立的 echo 分隔，方便定位失败位置。
# 默认用极小数据量（SAMPLE_COUNT=10）快速验证流程是否跑通。

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============

# --- 数据源 ---
INPUT_DIR="/mnt/data/jiaruijin/datasets/ECG_R1_Dataset/ecg_timeseries/mimic-iv/files"

# --- 输出根目录（所有中间产物放在这下面）---
BASE_OUTPUT_DIR="/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic/test_pipeline"

# --- 图像生成参数 ---
SAMPLE_COUNT=10          # 测试用极小数量，正式跑改大
SEED=42
GEN_WORKERS=8            # 图像生成并行进程数
RESOLUTION=200

# --- 后处理参数 ---
POST_WORKERS=8           # 后处理并行进程数
RESAMPLE_FACTOR=3        # replot_pixels 插值倍数

# --- nnUNet 路径 ---
NNUNET_RAW="${BASE_OUTPUT_DIR}/nnUNet_raw"
NNUNET_PREPROCESSED="${BASE_OUTPUT_DIR}/nnUNet_preprocessed"
NNUNET_RESULTS="${PROJECT_ROOT}/nnUNet_results"  # 训练输出放在项目根目录，方便查看
DATASET_ID=500
DATASET_NAME="Dataset500_MIMIC"

# --- 训练参数 ---
FOLDS="0"                       # 只跑 1 折验证，全部跑改为 "0 1 2 3 4 5 6 7 8 9"
DEVICE=cuda
NUM_GPUS=1                       # 测试数据少，用单卡；正式跑改为 4
CUDA_DEVICES="4"                  # 正式跑改为 "4,5,6,7"

# =============================================

# 派生路径（不需要修改）
CLEAN_IMG_DIR="${BASE_OUTPUT_DIR}/12x1_clean"
AUG_IMG_DIR="${BASE_OUTPUT_DIR}/12x1_aug"
CLEAN_NNUNET_DIR="${NNUNET_RAW}/${DATASET_NAME}_Clean"
AUG_NNUNET_DIR="${NNUNET_RAW}/${DATASET_NAME}_Aug"
MERGED_DIR="${NNUNET_RAW}/${DATASET_NAME}"

export nnUNet_raw="$NNUNET_RAW"
export nnUNet_preprocessed="$NNUNET_PREPROCESSED"
export nnUNet_results="$NNUNET_RESULTS"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

echo "============================================================"
echo "  ECG 全流程测试"
echo "============================================================"
echo "输入数据:       $INPUT_DIR"
echo "输出根目录:     $BASE_OUTPUT_DIR"
echo "样本数:         $SAMPLE_COUNT (clean) + $SAMPLE_COUNT (aug)"
echo "训练折数:       $FOLDS"
echo "GPU:            $CUDA_DEVICES ($NUM_GPUS 卡)"
echo "============================================================"
echo ""

# # ==============================================================
# # 步骤 1/7: 生成无增强图像
# # ==============================================================
# echo "============================================================"
# echo "  步骤 1/7: 生成无增强图像 (12×1 clean)"
# echo "============================================================"

# mkdir -p "$CLEAN_IMG_DIR"
# cd "$PROJECT_ROOT/ecg-image-generator"

# python gen_ecg_images_from_data_batch.py \
#     -i "$INPUT_DIR" \
#     -o "$CLEAN_IMG_DIR" \
#     -se $SEED \
#     --config_file config_12x1.yaml \
#     --num_columns 1 \
#     --full_mode None \
#     --mask_unplotted_samples \
#     --store_config 2 \
#     --calibration_pulse 1.0 \
#     --random_grid_present 1.0 \
#     --random_bw 0 \
#     --standard_grid_color 5 \
#     -r $RESOLUTION --random_resolution \
#     --max_num_images $SAMPLE_COUNT \
#     --num_workers $GEN_WORKERS \
#     --image_only

# cd "$PROJECT_ROOT"
# echo "--- 清理空目录 (clean) ---"
# find "$CLEAN_IMG_DIR" -type d -empty -delete 2>/dev/null || true
# echo ">>> 步骤 1/7 完成: $CLEAN_IMG_DIR"
# echo ""

# # ==============================================================
# # 步骤 2/7: 生成有增强图像
# # ==============================================================
# echo "============================================================"
# echo "  步骤 2/7: 生成有增强图像 (12×1 aug)"
# echo "============================================================"

# mkdir -p "$AUG_IMG_DIR"
# cd "$PROJECT_ROOT/ecg-image-generator"

# python gen_ecg_images_from_data_batch.py \
#     -i "$INPUT_DIR" \
#     -o "$AUG_IMG_DIR" \
#     -se $SEED \
#     --config_file config_12x1.yaml \
#     --num_columns 1 \
#     --full_mode None \
#     --mask_unplotted_samples \
#     --store_config 2 \
#     --calibration_pulse 0.8 \
#     --random_grid_present 0.9 \
#     --random_bw 0.2 \
#     --standard_grid_color 5 \
#     -r $RESOLUTION --random_resolution \
#     -rot 10 \
#     --wrinkles \
#     --crease_angle 45 \
#     --num_creases_vertically 5 \
#     --num_creases_horizontally 5 \
#     --augment \
#     -noise 40 \
#     -c 0.01 \
#     --max_num_images $SAMPLE_COUNT \
#     --num_workers $GEN_WORKERS \
#     --image_only

# cd "$PROJECT_ROOT"
# echo "--- 清理空目录 (aug) ---"
# find "$AUG_IMG_DIR" -type d -empty -delete 2>/dev/null || true
# echo ">>> 步骤 2/7 完成: $AUG_IMG_DIR"
# echo ""

# # ==============================================================
# # 步骤 3/7: 后处理 — clean 数据
# # ==============================================================
# echo "============================================================"
# echo "  步骤 3/7: 后处理 clean 数据 (replot + 转 nnUNet 格式)"
# echo "============================================================"

# echo "--- 3a: 像素坐标加密 (clean) ---"
# python -m src.mimic.replot_pixels \
#     --dir "$CLEAN_IMG_DIR" \
#     --resample_factor $RESAMPLE_FACTOR \
#     --run_on_subdirs \
#     --num_workers $POST_WORKERS

# echo "--- 3b: 转换为 nnUNet 格式 (clean) ---"
# python -m src.mimic.create_mimic_dataset \
#     -i "$CLEAN_IMG_DIR" \
#     -o "$CLEAN_NNUNET_DIR" \
#     --no_split \
#     -m \
#     --mask \
#     --mask_multilabel \
#     --rgba_to_rgb \
#     --gray_to_rgb \
#     --plotted_pixels_key dense_plotted_pixels \
#     --num_workers $POST_WORKERS

# echo ">>> 步骤 3/7 完成: $CLEAN_NNUNET_DIR"
# echo ""

# # ==============================================================
# # 步骤 4/7: 后处理 — aug 数据
# # ==============================================================
# echo "============================================================"
# echo "  步骤 4/7: 后处理 aug 数据 (replot + 转 nnUNet 格式)"
# echo "============================================================"

# echo "--- 4a: 像素坐标加密 (aug) ---"
# python -m src.mimic.replot_pixels \
#     --dir "$AUG_IMG_DIR" \
#     --resample_factor $RESAMPLE_FACTOR \
#     --run_on_subdirs \
#     --num_workers $POST_WORKERS

# echo "--- 4b: 转换为 nnUNet 格式 (aug) ---"
# python -m src.mimic.create_mimic_dataset \
#     -i "$AUG_IMG_DIR" \
#     -o "$AUG_NNUNET_DIR" \
#     --no_split \
#     -m \
#     --mask \
#     --mask_multilabel \
#     --rgba_to_rgb \
#     --gray_to_rgb \
#     --plotted_pixels_key dense_plotted_pixels \
#     --num_workers $POST_WORKERS

# echo ">>> 步骤 4/7 完成: $AUG_NNUNET_DIR"
# echo ""

# # ==============================================================
# # 步骤 5/7: 合并 clean + aug 数据集
# # ==============================================================
# echo "============================================================"
# echo "  步骤 5/7: 合并数据集 + 添加 _0000 后缀"
# echo "============================================================"

# python3 -c "
# import json, os, shutil
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor

# clean_dir = '$CLEAN_NNUNET_DIR'
# aug_dir = '$AUG_NNUNET_DIR'
# output_dir = Path('$MERGED_DIR')
# num_workers = $POST_WORKERS

# datasets = [
#     ('clean', clean_dir),
#     ('aug', aug_dir),
# ]

# (output_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)
# (output_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)

# def move_file(args):
#     src, dst = args
#     shutil.move(str(src), str(dst))

# total_images = 0
# for prefix, src_dir in datasets:
#     src = Path(src_dir)
#     print(f'合并: {prefix} <- {src_dir}')
#     tasks = []
#     for subdir in ['imagesTr', 'labelsTr']:
#         d = src / subdir
#         if d.is_dir():
#             for f in d.iterdir():
#                 if f.is_file():
#                     tasks.append((f, output_dir / subdir / f'{prefix}_{f.name}'))
#     with ThreadPoolExecutor(max_workers=num_workers) as pool:
#         list(pool.map(move_file, tasks))
#     print(f'  移动 {len(tasks)} 个文件')

# # 生成 dataset.json
# images = sorted([f for f in os.listdir(output_dir / 'imagesTr') if f.endswith('.png')])
# dataset = {
#     'channel_names': {'0': 'Signals'},
#     'labels': {
#         'background': 0,
#         'I': 1, 'II': 2, 'III': 3,
#         'aVR': 4, 'aVL': 5, 'aVF': 6,
#         'V1': 7, 'V2': 8, 'V3': 9,
#         'V4': 10, 'V5': 11, 'V6': 12
#     },
#     'numTraining': len(images),
#     'file_ending': '.png'
# }
# with open(output_dir / 'dataset.json', 'w') as f:
#     json.dump(dataset, f, indent=4)
# print(f'合并完成: {len(images)} 张图像')
# "

# echo "--- 添加 _0000 后缀 ---"
# python3 -m src.mimic.prepare_nnunet \
#     -i "$MERGED_DIR" \
#     --num_workers $POST_WORKERS

# echo "--- 清理源数据集目录（避免 nnUNet 多个 Dataset500_* ID 冲突）---"
# rm -rf "$CLEAN_NNUNET_DIR"
# rm -rf "$AUG_NNUNET_DIR"

# echo ">>> 步骤 5/7 完成: $MERGED_DIR"
# echo ""

# # ==============================================================
# # 步骤 6/7: nnUNet 预处理
# # ==============================================================

# # 清理中间数据集目录，避免 nnUNet 发现多个 Dataset500_* 导致 ID 冲突
# if [ -d "$CLEAN_NNUNET_DIR" ]; then
#     echo "清理中间目录: $CLEAN_NNUNET_DIR"
#     rm -rf "$CLEAN_NNUNET_DIR"
# fi
# if [ -d "$AUG_NNUNET_DIR" ]; then
#     echo "清理中间目录: $AUG_NNUNET_DIR"
#     rm -rf "$AUG_NNUNET_DIR"
# fi

# # 校验 imagesTr 和 labelsTr 一一对应，清理不配对的文件
# echo "--- 校验 imagesTr/labelsTr 配对 ---"
# python3 -c "
# import os
# from pathlib import Path

# merged = Path('$MERGED_DIR')
# images_dir = merged / 'imagesTr'
# labels_dir = merged / 'labelsTr'

# # 提取 case name: imagesTr 里 xxx_0000.png -> xxx, labelsTr 里 xxx.png -> xxx
# image_cases = {f.replace('_0000.png', '') for f in os.listdir(images_dir) if f.endswith('_0000.png')}
# label_cases = {f.replace('.png', '') for f in os.listdir(labels_dir) if f.endswith('.png')}

# missing_labels = image_cases - label_cases
# missing_images = label_cases - image_cases

# removed = 0
# for case in missing_labels:
#     # 删除 imagesTr 中没有对应 mask 的文件
#     for f in images_dir.iterdir():
#         if f.stem.replace('_0000', '') == case or (f.suffix == '.json' and f.stem == case):
#             f.unlink()
#             removed += 1
# for case in missing_images:
#     (labels_dir / f'{case}.png').unlink()
#     removed += 1

# if removed > 0:
#     # 更新 dataset.json 中的 numTraining
#     import json
#     ds_path = merged / 'dataset.json'
#     with open(ds_path) as f:
#         ds = json.load(f)
#     ds['numTraining'] = len([f for f in os.listdir(images_dir) if f.endswith('_0000.png')])
#     with open(ds_path, 'w') as f:
#         json.dump(ds, f, indent=4)

# print(f'缺 mask: {len(missing_labels)} 个, 缺图片: {len(missing_images)} 个, 共清理 {removed} 个文件')
# "

# echo "============================================================"
# echo "  步骤 6/7: nnUNet 预处理"
# echo "  nnUNet_raw:          $NNUNET_RAW"
# echo "  nnUNet_preprocessed: $NNUNET_PREPROCESSED"
# echo "  nnUNet_results:      $NNUNET_RESULTS"
# echo "  Dataset ID:          $DATASET_ID"
# echo "============================================================"

# nnUNetv2_plan_and_preprocess \
#     -d $DATASET_ID \
#     --clean \
#     -c 2d \
#     -np $POST_WORKERS

# echo ">>> 步骤 6/7 完成"
# echo ""

# ==============================================================
# 步骤 7/7: 10 折交叉验证训练
# ==============================================================
echo "============================================================"
echo "  步骤 7/7: 10 折交叉验证训练"
echo "  Folds: $FOLDS"
echo "  Device: $DEVICE | GPUs: $CUDA_DEVICES ($NUM_GPUS 卡)"
echo "============================================================"

for FOLD in $FOLDS; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  训练 fold $FOLD / 9"
    echo "------------------------------------------------------------"

    nnUNetv2_train \
        $DATASET_ID \
        2d \
        $FOLD \
        -device $DEVICE \
        -num_gpus $NUM_GPUS \
        --c
done

echo ""
echo "============================================================"
echo "  全流程完成!"
echo "============================================================"
echo "数据集:    $MERGED_DIR"
echo "预处理:    $NNUNET_PREPROCESSED"
echo "模型权重:  $NNUNET_RESULTS"
echo ""
echo "下一步（可选）:"
echo "  # 查找最佳配置"
echo "  nnUNetv2_find_best_configuration $DATASET_ID -c 2d --disable_ensembling"
echo "============================================================"
