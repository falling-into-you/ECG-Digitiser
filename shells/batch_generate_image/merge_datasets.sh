#!/bin/bash
# 合并多个 nnUNet 数据集（解决文件名冲突），用 mv 移动
# 用法: bash shells/batch_generate_image/merge_datasets.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置（按需修改）============
CLEAN_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC_Clean"
AUG_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC_Aug"
OUTPUT_DIR="/data/jinjiarui/datasets/ECG-Digital-Dataset/mimic/nnUNet_raw/Dataset500_MIMIC"
NUM_WORKERS=64
# =============================================

python3 -c "
import json, os, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

clean_dir = '$CLEAN_DIR'
aug_dir = '$AUG_DIR'
output_dir = Path('$OUTPUT_DIR')
num_workers = $NUM_WORKERS

datasets = [
    ('clean', clean_dir),
    ('aug', aug_dir),
]

(output_dir / 'imagesTr').mkdir(parents=True, exist_ok=True)
(output_dir / 'labelsTr').mkdir(parents=True, exist_ok=True)

def move_file(args):
    src, dst = args
    shutil.move(str(src), str(dst))
    return True

total = 0
for prefix, src_dir in datasets:
    src = Path(src_dir)
    print(f'=============================')
    print(f'合并数据集: {prefix}')
    print(f'来源: {src_dir}')
    print(f'=============================')

    tasks = []

    # imagesTr
    img_dir = src / 'imagesTr'
    if img_dir.is_dir():
        for f in img_dir.iterdir():
            if f.is_file():
                tasks.append((f, output_dir / 'imagesTr' / f'{prefix}_{f.name}'))
        img_count = len([t for t in tasks])
        print(f'  imagesTr: {img_count} 个文件')

    # labelsTr
    lbl_dir = src / 'labelsTr'
    lbl_count = 0
    if lbl_dir.is_dir():
        for f in lbl_dir.iterdir():
            if f.is_file():
                tasks.append((f, output_dir / 'labelsTr' / f'{prefix}_{f.name}'))
                lbl_count += 1
        print(f'  labelsTr: {lbl_count} 个文件')

    # 多线程移动
    from tqdm import tqdm
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(move_file, t) for t in tasks]
        for f in tqdm(futures, total=len(futures), desc=f'Moving {prefix}', mininterval=1.0):
            f.result()

    total += len(tasks)
    print()

# dataset.json
images = sorted([f for f in os.listdir(output_dir / 'imagesTr') if f.endswith('.png')])
dataset = {
    'channel_names': {'0': 'Signals'},
    'labels': {
        'background': 0,
        'I': 1, 'II': 2, 'III': 3,
        'aVR': 4, 'aVL': 5, 'aVF': 6,
        'V1': 7, 'V2': 8, 'V3': 9,
        'V4': 10, 'V5': 11, 'V6': 12
    },
    'numTraining': len(images),
    'file_ending': '.png'
}
with open(output_dir / 'dataset.json', 'w') as f:
    json.dump(dataset, f, indent=4)

print(f'=============================')
print(f'合并完成!')
print(f'输出: {output_dir}')
print(f'总样本数: {len(images)}')
print(f'imagesTr/: {len(os.listdir(output_dir / \"imagesTr\"))} 个文件')
print(f'labelsTr/: {len(os.listdir(output_dir / \"labelsTr\"))} 个文件')
print(f'=============================')
"
