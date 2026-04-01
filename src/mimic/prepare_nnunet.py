#!/usr/bin/env python

"""
为 imagesTr 中的 PNG 文件添加 _0000 后缀，适配 nnUNet v2 命名规范。
labelsTr 中的 mask 文件不需要修改。

nnUNet 要求:
  imagesTr/casename_0000.png  (通道0)
  labelsTr/casename.png       (不带_0000)

用法: python -m src.mimic.prepare_nnunet -i <nnUNet_dataset_dir>
"""

import argparse
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def get_parser():
    parser = argparse.ArgumentParser(description="Add _0000 suffix for nnUNet v2")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="nnUNet dataset folder (e.g. Dataset500_MIMIC)")
    parser.add_argument("--num_workers", type=int, default=64, help="Process count")
    parser.add_argument("--dry_run", action="store_true", help="Only show what would be renamed")
    return parser


def rename_file(src, dst):
    os.rename(src, dst)
    return True


def run(args):
    base = Path(args.input_dir)
    images_dir = base / "imagesTr"

    if not images_dir.is_dir():
        print(f"错误: {images_dir} 不存在")
        return

    # 找出需要重命名的 png（不带 _0000 后缀的）
    tasks = []
    for f in images_dir.iterdir():
        if f.is_file() and f.name.endswith(".png") and not f.name.endswith("_0000.png"):
            new_name = f.name.replace(".png", "_0000.png")
            tasks.append((str(f), str(images_dir / new_name)))

    print(f"需要重命名: {len(tasks)} 个文件")

    if args.dry_run:
        for src, dst in tasks[:5]:
            print(f"  {Path(src).name} -> {Path(dst).name}")
        if len(tasks) > 5:
            print(f"  ... 共 {len(tasks)} 个")
        return

    from tqdm import tqdm
    with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
        futures = [pool.submit(rename_file, s, d) for s, d in tasks]
        for f in tqdm(as_completed(futures), total=len(futures), mininterval=1.0):
            f.result()

    # 验证
    pngs = list(images_dir.glob("*_0000.png"))
    print(f"完成! imagesTr 中 _0000.png 文件数: {len(pngs)}")


if __name__ == "__main__":
    run(get_parser().parse_args())
