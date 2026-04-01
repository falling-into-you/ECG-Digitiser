#!/usr/bin/env python

"""
从 nnUNet 格式的 imagesTr 目录中读取 JSON，生成对应的 segmentation mask 到 labelsTr。

注意：此脚本应在 postprocess.sh 完成 RGBA→RGB 图片格式转换之后单独运行。
因为 create_mimic_dataset.py 中的 mask 生成步骤对 MIMIC 数据存在兼容性问题
（full_mode_lead 为 "None" 导致 max() 空序列错误），所以将 mask 生成独立为
此脚本，支持多进程并行，速度更快。

用法: python -m src.mimic.generate_masks -i <nnUNet_dataset_dir>
"""

import json
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import LEAD_LABEL_MAPPING


def get_parser():
    parser = argparse.ArgumentParser(description="Generate masks from JSON config files")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="nnUNet dataset folder")
    parser.add_argument("--mask_multilabel", action="store_true", help="Multi-label mask (per-lead classes)")
    parser.add_argument("--gray_to_rgb", action="store_true", help="Save mask as RGB")
    parser.add_argument("--plotted_pixels_key", type=str, default="plotted_pixels", help="JSON key for pixel coordinates")
    parser.add_argument("--num_workers", type=int, default=64, help="Process count")
    return parser


def create_mask(json_path, mask_path, rgb=False, multilabel=False, plotted_pixels_key="plotted_pixels"):
    try:
        with open(json_path) as f:
            data_dict = json.load(f)

        mask_values = LEAD_LABEL_MAPPING
        full_mode_lead = data_dict.get("full_mode_lead")

        keys = ["leads"]
        if "leads_augmented" in data_dict:
            keys.append("leads_augmented")

        for idx, key in enumerate(keys):
            leads = data_dict[key]

            # Filter full_mode_lead if present
            if full_mode_lead and full_mode_lead != "None":
                matching = [
                    lead["end_sample"] - lead["start_sample"]
                    for lead in leads
                    if lead["lead_name"] == full_mode_lead
                ]
                if matching:
                    full_len = max(matching)
                    leads = [
                        lead for lead in leads
                        if lead["lead_name"] != full_mode_lead
                        or lead["end_sample"] - lead["start_sample"] == full_len
                    ]

            # Collect pixel coordinates
            plotted_pixels = {}
            for lead in leads:
                pixels = lead.get(plotted_pixels_key, [])
                name = lead["lead_name"]
                for item in pixels:
                    coord = tuple(np.array(item).astype("int"))
                    plotted_pixels[coord] = name

            # Filter out-of-bounds
            h, w = data_dict["height"], data_dict["width"]
            plotted_pixels = {k: v for k, v in plotted_pixels.items() if k[0] < h and k[1] < w}

            if not plotted_pixels:
                mask = np.zeros((h, w), dtype=np.uint8)
            else:
                if multilabel:
                    values = np.array([mask_values[v] for v in plotted_pixels.values()])
                else:
                    values = np.ones(len(plotted_pixels), dtype=np.uint8)

                coords = np.array(list(plotted_pixels.keys()))
                mask = np.zeros((h, w), dtype=np.uint8)
                mask[coords[:, 0], coords[:, 1]] = values

            if rgb:
                mask = np.stack([mask] * 3, axis=-1)

            out_path = mask_path if idx == 0 else mask_path.replace(".png", "_augmented.png")
            Image.fromarray(mask).save(out_path)

        return True

    except Exception as e:
        return f"ERROR {json_path}: {e}"


def _process_task(args):
    return create_mask(*args)


def run(args):
    for split in ["imagesTr", "imagesTv", "imagesTs"]:
        images_dir = os.path.join(args.input_dir, split)
        if not os.path.isdir(images_dir):
            continue

        labels_dir = images_dir.replace("imagesT", "labelsT")
        os.makedirs(labels_dir, exist_ok=True)

        json_files = sorted([f for f in os.listdir(images_dir) if f.endswith(".json")])
        if not json_files:
            continue

        print(f"Generating masks for {split}: {len(json_files)} files, {args.num_workers} workers")

        tasks = []
        for jf in json_files:
            json_path = os.path.join(images_dir, jf)
            mask_path = os.path.join(labels_dir, jf.replace(".json", ".png"))
            tasks.append((json_path, mask_path, args.gray_to_rgb, args.mask_multilabel, args.plotted_pixels_key))

        errors = 0
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = {pool.submit(_process_task, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), mininterval=1.0):
                result = future.result()
                if result is not True:
                    errors += 1
                    if errors <= 10:
                        print(result)

        if errors > 10:
            print(f"... 共 {errors} 个 mask 创建失败")
        else:
            print(f"{split}: 完成, {errors} 个失败")

    # Generate dataset.json
    imagesTr_path = os.path.join(args.input_dir, "imagesTr")
    num_training = len([f for f in os.listdir(imagesTr_path) if f.endswith(".png")]) if os.path.isdir(imagesTr_path) else 0

    if args.mask_multilabel:
        labels_dict = {"background": 0}
        labels_dict.update(LEAD_LABEL_MAPPING)
    else:
        labels_dict = {"background": 0, "signal": 1}

    dataset_json = {
        "channel_names": {"0": "Signals"},
        "labels": labels_dict,
        "numTraining": num_training,
        "file_ending": ".png",
    }
    json_path = os.path.join(args.input_dir, "dataset.json")
    with open(json_path, "w") as f:
        json.dump(dataset_json, f, indent=4)
    print(f"dataset.json saved: numTraining={num_training}")


if __name__ == "__main__":
    run(get_parser().parse_args())
