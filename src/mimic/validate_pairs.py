#!/usr/bin/env python

"""
验证 nnUNet 数据集 imagesTr / labelsTr 的配对完整性。
默认只检查不修改。加 --clean 可删除不配对的文件并修复 dataset.json。

检查项:
  1. 目录存在性
  2. 文件统计 (按扩展名)
  3. imagesTr ↔ labelsTr 配对
  4. dataset.json 一致性
  5. 汇总

用法:
  检查:  python -m src.mimic.validate_pairs -i <dir>
  清理:  python -m src.mimic.validate_pairs -i <dir> --clean
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser(
        description="Validate nnUNet dataset imagesTr/labelsTr pairing"
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str,
        required=True,
        help="nnUNet dataset directory (e.g. Dataset500_MIMIC)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete unpaired files and fix dataset.json (requires confirmation)",
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Skip confirmation prompt (use with --clean)",
    )
    return parser


def run(args):
    dataset_dir = Path(args.input_dir)
    if not dataset_dir.is_dir():
        print(f"错误: 目录不存在: {dataset_dir}")
        sys.exit(1)

    print("=" * 60)
    print("nnUNet 数据集配对验证")
    print(f"目录: {dataset_dir}")
    print("=" * 60)

    images_dir = dataset_dir / "imagesTr"
    labels_dir = dataset_dir / "labelsTr"
    errors = []

    # ---------- 1. 目录存在性 ----------
    print()
    print("【1】目录检查")
    for name, d in [("imagesTr", images_dir), ("labelsTr", labels_dir)]:
        if d.is_dir():
            print(f"  {name}: 存在")
        else:
            print(f"  {name}: 不存在 !!!")
            errors.append(f"{name} 目录不存在")

    if not images_dir.is_dir() and not labels_dir.is_dir():
        print("\nimagesTr 和 labelsTr 均不存在，无法继续。")
        sys.exit(1)

    # ---------- 2. 文件统计 ----------
    print()
    print("【2】文件统计")
    for name, d in [("imagesTr", images_dir), ("labelsTr", labels_dir)]:
        if not d.is_dir():
            continue
        files = [f.name for f in d.iterdir() if f.is_file()]
        ext_counts = Counter(Path(f).suffix for f in files)
        total = len(files)
        print(f"  {name}: {total} 个文件")
        for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
            marker = " (非预期)" if ext not in (".png",) else ""
            print(f"    {ext}: {cnt}{marker}")
        if total == 0:
            errors.append(f"{name} 目录为空")

    # ---------- 3. 配对检查 ----------
    print()
    print("【3】配对检查 (imagesTr <-> labelsTr)")

    # case_id -> [文件路径列表] (一个 case 可能有 .png 和 .json)
    img_by_case = {}
    if images_dir.is_dir():
        for f in images_dir.iterdir():
            if f.is_file():
                name = f.stem
                case_id = name[:-5] if name.endswith("_0000") else name
                img_by_case.setdefault(case_id, []).append(f)

    lbl_by_case = {}
    if labels_dir.is_dir():
        for f in labels_dir.iterdir():
            if f.is_file():
                lbl_by_case.setdefault(f.stem, []).append(f)

    # 兼容旧逻辑: 取第一个文件名用于显示
    img_files = {cid: fs[0].name for cid, fs in img_by_case.items()}
    lbl_files = {cid: fs[0].name for cid, fs in lbl_by_case.items()}

    img_ids = set(img_files.keys())
    lbl_ids = set(lbl_files.keys())
    paired = img_ids & lbl_ids
    only_img = img_ids - lbl_ids
    only_lbl = lbl_ids - img_ids

    print(f"  imagesTr case 数: {len(img_ids)}")
    print(f"  labelsTr case 数: {len(lbl_ids)}")
    print(f"  已配对:           {len(paired)}")
    print(f"  有 image 无 label: {len(only_img)}")
    print(f"  有 label 无 image: {len(only_lbl)}")

    SHOW_LIMIT = 20

    if only_img:
        errors.append(f"{len(only_img)} 个 image 缺少对应 label")
        print()
        print(f"  --- 有 image 无 label (前 {min(SHOW_LIMIT, len(only_img))} 个) ---")
        for cid in sorted(only_img)[:SHOW_LIMIT]:
            print(f"    imagesTr/{img_files[cid]}")
        if len(only_img) > SHOW_LIMIT:
            print(f"    ... 共 {len(only_img)} 个")

    if only_lbl:
        errors.append(f"{len(only_lbl)} 个 label 缺少对应 image")
        print()
        print(f"  --- 有 label 无 image (前 {min(SHOW_LIMIT, len(only_lbl))} 个) ---")
        for cid in sorted(only_lbl)[:SHOW_LIMIT]:
            print(f"    labelsTr/{lbl_files[cid]}")
        if len(only_lbl) > SHOW_LIMIT:
            print(f"    ... 共 {len(only_lbl)} 个")

    # ---------- 4. dataset.json 一致性 ----------
    print()
    print("【4】dataset.json 检查")
    ds_json = dataset_dir / "dataset.json"
    if ds_json.is_file():
        with open(ds_json) as f:
            ds = json.load(f)
        num_training = ds.get("numTraining", "(缺失)")
        file_ending = ds.get("file_ending", "(缺失)")
        print(f"  numTraining: {num_training}")
        print(f"  file_ending: {file_ending}")
        if isinstance(num_training, int) and num_training != len(img_ids):
            print(
                f"  numTraining ({num_training}) != imagesTr case 数 ({len(img_ids)})"
            )
            errors.append("dataset.json numTraining 与实际不一致")
        else:
            print("  numTraining 与 imagesTr 一致: OK")
    else:
        print("  dataset.json 不存在 !!!")
        errors.append("dataset.json 不存在")

    # ---------- 5. 汇总 ----------
    print()
    print("=" * 60)
    if not errors:
        print("验证通过  所有检查项均正常")
    else:
        print(f"发现 {len(errors)} 个问题:")
        for i, e in enumerate(errors, 1):
            print(f"  {i}. {e}")
    print("=" * 60)

    # ---------- --clean: 删除不配对文件 ----------
    if getattr(args, "clean", False) and (only_img or only_lbl):
        to_delete = []
        for cid in only_img:
            to_delete.extend(img_by_case.get(cid, []))
        for cid in only_lbl:
            to_delete.extend(lbl_by_case.get(cid, []))

        print()
        print(f"--clean: 将删除 {len(to_delete)} 个不配对文件:")
        for f in sorted(to_delete)[:30]:
            print(f"  {f}")
        if len(to_delete) > 30:
            print(f"  ... 共 {len(to_delete)} 个")

        confirm = "y" if args.yes else input("\n确认删除? (y/N): ").strip().lower()
        if confirm == "y":
            for f in to_delete:
                f.unlink()
            print(f"已删除 {len(to_delete)} 个文件")

            # 修复 dataset.json
            ds_json = dataset_dir / "dataset.json"
            if ds_json.is_file():
                img_png_count = sum(
                    1 for f in images_dir.iterdir()
                    if f.is_file() and f.suffix == ".png"
                )
                with open(ds_json) as f:
                    ds = json.load(f)
                ds["numTraining"] = img_png_count
                with open(ds_json, "w") as f:
                    json.dump(ds, f, indent=4)
                print(f"已更新 dataset.json numTraining = {img_png_count}")

            print("清理完成")
        else:
            print("已取消")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    run(get_parser().parse_args())
