#!/bin/bash
# 将 nnUNet 输出目录中的文件还原到原始嵌套目录结构
# 用法: bash shells/batch_generate_image/restore_from_nnunet.sh

set -e

# ============ 参数配置（按需修改）============
NNUNET_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC_Aug"
ORIGINAL_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/12x1_aug_10w"
NUM_WORKERS=64
# ============================================

python3 -c "
import os, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

nnunet_dir = Path('$NNUNET_DIR/imagesTr')
original_dir = Path('$ORIGINAL_DIR')
num_workers = $NUM_WORKERS

# 步骤1: 建立 study_id -> 目录 映射
print('步骤 1/2: 扫描原始目录结构...')
study_map = {}
for d in original_dir.rglob('s*'):
    if d.is_dir():
        study_map[d.name[1:]] = d  # 去掉 's' 前缀
print(f'找到 {len(study_map)} 个 study 目录')

# 步骤2: 按目标目录分组，批量移动
print('步骤 2/2: 还原文件...')
files = [f for f in nnunet_dir.iterdir() if f.is_file()]
print(f'共 {len(files)} 个文件待还原')

def move_file(f):
    study_id = f.name.split('-')[0]
    target = study_map.get(study_id)
    if target:
        shutil.move(str(f), str(target / f.name))
        return True
    return False

moved = 0
not_found = 0
with ThreadPoolExecutor(max_workers=num_workers) as pool:
    futures = {pool.submit(move_file, f): f for f in files}
    for future in tqdm(as_completed(futures), total=len(futures), mininterval=0.5):
        if future.result():
            moved += 1
        else:
            not_found += 1

print()
print(f'还原完成! 总文件: {len(files)}, 已还原: {moved}, 未找到: {not_found}')
"
