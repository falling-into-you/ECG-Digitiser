#!/usr/bin/env python3
"""
分析 JSON 标注内容
"""

import os
import json
from pathlib import Path
from collections import defaultdict


def main():
    data_dir = Path("results/test_data/test120_chestpain")
    
    print("=" * 70)
    print("JSON 标注分析")
    print("=" * 70)
    
    # 收集所有 JSON
    json_files = sorted([f for f in data_dir.iterdir() 
                        if f.suffix == '.json' and f.stem.isdigit()],
                       key=lambda x: int(x.stem))
    
    print(f"共找到 {len(json_files)} 个 JSON 文件")
    print()
    
    # 统计标注信息
    label_counts = defaultdict(int)
    shape_type_counts = defaultdict(int)
    shapes_per_image = []
    images_with_labels = defaultdict(list)
    no_shapes = []
    
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        shapes = data.get("shapes", [])
        shapes_per_image.append(len(shapes))
        
        if len(shapes) == 0:
            no_shapes.append(json_file.stem)
        
        for shape in shapes:
            label = shape.get("label", "unknown")
            shape_type = shape.get("shape_type", "unknown")
            label_counts[label] += 1
            shape_type_counts[shape_type] += 1
            images_with_labels[label].append(json_file.stem)
    
    # 输出统计
    print("=" * 70)
    print("1. 标注标签分布")
    print("=" * 70)
    for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
        img_count = len(set(images_with_labels[label]))
        print(f"  {label:20s}: {count:4d} 个标注, 出现在 {img_count:3d} 张图像中")
    
    print()
    print("=" * 70)
    print("2. 标注形状类型")
    print("=" * 70)
    for shape_type, count in sorted(shape_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {shape_type:20s}: {count}")
    
    print()
    print("=" * 70)
    print("3. 每张图像的标注数量")
    print("=" * 70)
    import numpy as np
    shapes_arr = np.array(shapes_per_image)
    print(f"  min: {shapes_arr.min()}")
    print(f"  max: {shapes_arr.max()}")
    print(f"  mean: {shapes_arr.mean():.1f}")
    print(f"  median: {np.median(shapes_arr):.0f}")
    
    # 分布
    print()
    print("  标注数量分布:")
    for n in range(0, max(shapes_per_image) + 1):
        count = sum(1 for x in shapes_per_image if x == n)
        if count > 0:
            bar = "█" * count
            print(f"    {n:2d} 个标注: {count:3d} 张图像 {bar}")
    
    print()
    print("=" * 70)
    print("4. 无标注的图像")
    print("=" * 70)
    if no_shapes:
        print(f"  共 {len(no_shapes)} 张: {', '.join(no_shapes[:20])}" + ("..." if len(no_shapes) > 20 else ""))
    else:
        print("  所有图像都有标注")
    
    # 检查缺少 JSON 的 PNG
    print()
    print("=" * 70)
    print("5. 缺少 JSON 标注的图像")
    print("=" * 70)
    png_files = [f.stem for f in data_dir.iterdir() 
                 if f.suffix.lower() == '.png' and f.stem.isdigit()]
    json_stems = [f.stem for f in json_files]
    missing_json = set(png_files) - set(json_stems)
    if missing_json:
        print(f"  共 {len(missing_json)} 张: {', '.join(sorted(missing_json, key=int))}")
    else:
        print("  所有 PNG 都有对应的 JSON")
    
    # 原始文件名分类统计
    print()
    print("=" * 70)
    print("6. 原始数据来源分布 (基于原始文件名)")
    print("=" * 70)
    source_counts = defaultdict(int)
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        orig_name = data.get("original_filename", "")
        if orig_name.startswith("NSTEMI"):
            source_counts["NSTEMI"] += 1
        elif orig_name.startswith("STEMI"):
            source_counts["STEMI"] += 1
        elif "不稳定性心绞痛" in orig_name or "涓嶇ǔ瀹氭€у績缁炵棝" in orig_name:
            source_counts["不稳定性心绞痛"] += 1
        elif "主动脉夹层" in orig_name or "涓诲姩鑴夊す灞" in orig_name:
            source_counts["主动脉夹层"] += 1
        elif "肺栓塞" in orig_name or "鑲烘爴濉" in orig_name:
            source_counts["肺栓塞"] += 1
        elif "其他类型" in orig_name or "鍏朵粬绫诲瀷" in orig_name:
            source_counts["其他类型"] += 1
        else:
            source_counts["未知"] += 1
    
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {source:20s}: {count:3d} ({count/len(json_files)*100:.1f}%)")


if __name__ == "__main__":
    main()
