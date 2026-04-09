#!/usr/bin/env python
"""
清理 nnUNet 数据集目录中的 JSON 文件。

JSON 文件在生成标签后已无用，但占用大量空间（每个约 20MB）。
此脚本用于清理 imagesTr 目录中的 JSON 文件。

用法:
    python -m src.mimic.clean_json -i /path/to/dataset       # 交互确认
    python -m src.mimic.clean_json -i /path/to/dataset -y    # 直接执行
    python -m src.mimic.clean_json -i /path/to/dataset --dry-run  # 只统计不删除
"""

import argparse
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="清理 nnUNet 数据集中 imagesTr 的 JSON 文件"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="数据集目录（包含 imagesTr 子目录）",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="跳过确认，直接删除",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只统计不删除",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=64,
        help="并行删除的线程数 (默认: 64)",
    )
    return parser


def get_file_size_str(size_bytes):
    """将字节数转换为人类可读的字符串"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def delete_file(path):
    """删除单个文件，返回 (成功, 文件大小或错误信息)"""
    try:
        size = path.stat().st_size
        path.unlink()
        return True, size
    except Exception as e:
        return False, str(e)


def run(args):
    input_dir = Path(args.input_dir)
    images_dir = input_dir / "imagesTr"

    if not images_dir.is_dir():
        print(f"错误: imagesTr 目录不存在: {images_dir}")
        sys.exit(1)

    # 查找所有 JSON 文件
    print(f"扫描目录: {images_dir}")
    json_files = list(images_dir.glob("*.json"))

    if not json_files:
        print("没有找到 JSON 文件，无需清理。")
        return

    # 统计大小
    total_size = 0
    print(f"统计文件大小...")
    for f in tqdm(json_files, desc="统计中", mininterval=0.5):
        try:
            total_size += f.stat().st_size
        except:
            pass

    print()
    print("=" * 50)
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print(f"总大小: {get_file_size_str(total_size)}")
    print("=" * 50)

    if args.dry_run:
        print()
        print("(--dry-run 模式，不执行删除)")
        return

    # 确认
    if not args.yes:
        print()
        confirm = input("确认删除这些文件? [y/N]: ").strip().lower()
        if confirm != 'y':
            print("已取消。")
            return

    # 并行删除
    print()
    print(f"删除中 (使用 {args.num_workers} 个线程)...")
    
    deleted_count = 0
    deleted_size = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
        futures = {pool.submit(delete_file, f): f for f in json_files}
        for future in tqdm(as_completed(futures), total=len(futures), desc="删除中", mininterval=0.5):
            success, result = future.result()
            if success:
                deleted_count += 1
                deleted_size += result
            else:
                errors.append((futures[future], result))

    print()
    print("=" * 50)
    print(f"删除完成!")
    print(f"已删除: {deleted_count} 个文件")
    print(f"释放空间: {get_file_size_str(deleted_size)}")
    if errors:
        print(f"错误: {len(errors)} 个文件删除失败")
        for f, e in errors[:5]:
            print(f"  {f.name}: {e}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors) - 5} 个错误")
    print("=" * 50)


if __name__ == "__main__":
    run(get_parser().parse_args())
