#!/usr/bin/env python

"""
合并多个 nnUNet 数据集到新目录。
为每个源数据集的文件添加前缀，避免文件名冲突。

用法: python -m src.mimic.merge_datasets \
        --sources "clean:/path/to/Clean" "aug:/path/to/Aug" \
        -o /path/to/output/Dataset500_MIMIC \
        --num_workers 64 \
        -m  # 可选：移动文件而非复制
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge multiple nnUNet datasets into one"
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        help="Source datasets as 'prefix:dir' pairs, e.g. clean:/path/to/Clean aug:/path/to/Aug",
    )
    parser.add_argument(
        "-o", "--output_dir",
        type=str,
        required=True,
        help="Output dataset directory",
    )
    parser.add_argument(
        "-m", "--move",
        action="store_true",
        help="Move files instead of copying (saves disk space)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="Thread count for parallel transfer",
    )
    return parser


def parse_sources(sources):
    """解析 prefix:dir 对"""
    result = []
    for s in sources:
        if ":" not in s:
            raise ValueError(f"格式错误: '{s}'，应为 'prefix:dir'")
        prefix, dir_path = s.split(":", 1)
        result.append((prefix, dir_path))
    return result


def transfer_file(args):
    src, dst, move = args
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(str(src), str(dst))


def run(args):
    datasets = parse_sources(args.sources)
    output_dir = Path(args.output_dir)
    move = args.move
    action = "移动" if move else "复制"

    # 清空目标目录（如果存在）
    for subdir in ["imagesTr", "labelsTr"]:
        target = output_dir / subdir
        if target.is_dir():
            shutil.rmtree(target)
    
    (output_dir / "imagesTr").mkdir(parents=True, exist_ok=True)
    (output_dir / "labelsTr").mkdir(parents=True, exist_ok=True)

    total = 0
    for prefix, src_dir in datasets:
        src = Path(src_dir)
        print("=" * 60)
        print(f"合并数据集: {prefix}")
        print(f"来源: {src_dir}")

        if not src.is_dir():
            print("  目录不存在，跳过")
            print()
            continue

        tasks = []

        # imagesTr (只复制 PNG，跳过 JSON)
        img_dir = src / "imagesTr"
        if img_dir.is_dir():
            img_tasks = []
            for f in img_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".png":
                    img_tasks.append(
                        (f, output_dir / "imagesTr" / f"{prefix}_{f.name}", move)
                    )
            tasks.extend(img_tasks)
            print(f"  imagesTr: {len(img_tasks)} 个 PNG 文件")

        # labelsTr (只复制 PNG)
        lbl_dir = src / "labelsTr"
        if lbl_dir.is_dir():
            lbl_tasks = []
            for f in lbl_dir.iterdir():
                if f.is_file() and f.suffix.lower() == ".png":
                    lbl_tasks.append(
                        (f, output_dir / "labelsTr" / f"{prefix}_{f.name}", move)
                    )
            tasks.extend(lbl_tasks)
            print(f"  labelsTr: {len(lbl_tasks)} 个 PNG 文件")

        # 多线程传输
        with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(transfer_file, t) for t in tasks]
            for fut in tqdm(
                futures, total=len(futures),
                desc=f"{action} {prefix}", mininterval=1.0,
            ):
                fut.result()

        total += len(tasks)
        print()

    # dataset.json (numTraining 只统计 PNG)
    all_img = os.listdir(output_dir / "imagesTr")
    all_lbl = os.listdir(output_dir / "labelsTr")
    img_png_count = sum(1 for f in all_img if f.endswith(".png"))
    lbl_png_count = sum(1 for f in all_lbl if f.endswith(".png"))
    dataset = {
        "channel_names": {"0": "Signals"},
        "labels": {
            "background": 0,
            "I": 1, "II": 2, "III": 3,
            "aVR": 4, "aVL": 5, "aVF": 6,
            "V1": 7, "V2": 8, "V3": 9,
            "V4": 10, "V5": 11, "V6": 12,
        },
        "numTraining": img_png_count,
        "file_ending": ".png",
    }
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)

    print("=" * 60)
    print("合并完成!")
    print(f"输出: {output_dir}")
    print(f"imagesTr/: {len(all_img)} 个文件 ({img_png_count} PNG)")
    print(f"labelsTr/: {len(all_lbl)} 个文件 ({lbl_png_count} PNG)")
    print(f"numTraining: {img_png_count}")
    if move:
        print("源文件已移动。")
    else:
        print("源文件已保留，未删除。")
    print("=" * 60)


if __name__ == "__main__":
    run(get_parser().parse_args())
