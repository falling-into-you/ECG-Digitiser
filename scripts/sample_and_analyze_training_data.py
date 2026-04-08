#!/usr/bin/env python3
"""
随机复制训练数据样本并分析像素分布
对比 clean vs aug vs 真实世界数据
"""

import os
import random
import shutil
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import subprocess


def find_all_png_files(base_dir):
    """递归查找所有 PNG 文件"""
    result = subprocess.run(
        ["find", str(base_dir), "-name", "*.png"],
        capture_output=True, text=True
    )
    return [Path(p) for p in result.stdout.strip().split("\n") if p]


def copy_random_samples(src_dir, dst_dir, n_samples=200):
    """随机复制 n 个样本到目标目录"""
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有 PNG
    all_pngs = find_all_png_files(src_dir)
    print(f"  找到 {len(all_pngs)} 个 PNG 文件")
    
    # 随机选择
    random.seed(42)  # 固定种子保证可复现
    selected = random.sample(all_pngs, min(n_samples, len(all_pngs)))
    
    # 复制文件
    for i, src_path in enumerate(selected):
        dst_path = dst_dir / f"{i+1}.png"
        shutil.copy(src_path, dst_path)
        
        # 同时复制对应的 JSON（如果有）
        json_path = src_path.with_suffix(".json")
        if json_path.exists():
            shutil.copy(json_path, dst_dir / f"{i+1}.json")
    
    print(f"  复制了 {len(selected)} 个样本到 {dst_dir}")
    return dst_dir


def analyze_images(img_dir, name):
    """分析目录中所有图像的像素统计"""
    img_dir = Path(img_dir)
    png_files = sorted(img_dir.glob("*.png"), key=lambda x: int(x.stem) if x.stem.isdigit() else 0)
    
    stats = {
        "name": name,
        "count": len(png_files),
        "means": [],
        "stds": [],
        "mins": [],
        "maxs": [],
        "widths": [],
        "heights": [],
    }
    
    for png_path in png_files:
        img = Image.open(png_path).convert("RGB")
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
        
        stats["means"].append(float(gray.mean()))
        stats["stds"].append(float(gray.std()))
        stats["mins"].append(float(gray.min()))
        stats["maxs"].append(float(gray.max()))
        stats["widths"].append(img.size[0])
        stats["heights"].append(img.size[1])
    
    return stats


def print_stats_comparison(stats_list):
    """打印多个数据集的对比"""
    print("\n" + "=" * 80)
    print("数据集像素分布对比")
    print("=" * 80)
    
    # 表头
    print(f"\n{'指标':<15}", end="")
    for s in stats_list:
        print(f"{s['name']:<20}", end="")
    print()
    print("-" * 80)
    
    # Mean
    print(f"{'Mean 均值':<15}", end="")
    for s in stats_list:
        print(f"{np.mean(s['means']):<20.1f}", end="")
    print()
    
    print(f"{'Mean 范围':<15}", end="")
    for s in stats_list:
        print(f"[{min(s['means']):.0f}, {max(s['means']):.0f}]{'':>10}", end="")
    print()
    
    print(f"{'Mean 标准差':<15}", end="")
    for s in stats_list:
        print(f"{np.std(s['means']):<20.1f}", end="")
    print()
    
    print("-" * 80)
    
    # Std
    print(f"{'Std 均值':<15}", end="")
    for s in stats_list:
        print(f"{np.mean(s['stds']):<20.1f}", end="")
    print()
    
    print(f"{'Std 范围':<15}", end="")
    for s in stats_list:
        print(f"[{min(s['stds']):.0f}, {max(s['stds']):.0f}]{'':>10}", end="")
    print()
    
    print("-" * 80)
    
    # 尺寸
    print(f"{'宽度范围':<15}", end="")
    for s in stats_list:
        print(f"[{min(s['widths'])}, {max(s['widths'])}]{'':>8}", end="")
    print()
    
    print(f"{'高度范围':<15}", end="")
    for s in stats_list:
        print(f"[{min(s['heights'])}, {max(s['heights'])}]{'':>8}", end="")
    print()
    
    # Mean 分布直方图
    print("\n" + "=" * 80)
    print("Mean 分布直方图")
    print("=" * 80)
    
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 256)]
    bin_labels = ["0-50", "50-100", "100-150", "150-200", "200-255"]
    
    print(f"\n{'区间':<12}", end="")
    for s in stats_list:
        print(f"{s['name']:<20}", end="")
    print()
    print("-" * 80)
    
    for (low, high), label in zip(bins, bin_labels):
        print(f"{label:<12}", end="")
        for s in stats_list:
            count = sum(1 for m in s["means"] if low <= m < high)
            pct = count / len(s["means"]) * 100
            print(f"{count:>3} ({pct:>5.1f}%){'':>8}", end="")
        print()


def main():
    base_dir = Path("/mnt/data/jiaruijin/datasets/ECG-Digital-Dataset/mimic")
    output_dir = Path("results/test_data/training_samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("训练数据采样与分析")
    print("=" * 80)
    
    # 1. 复制 clean 样本
    print("\n1. 采样 Clean 数据集...")
    clean_dir = copy_random_samples(
        base_dir / "12x1_clean_2w",
        output_dir / "clean_200",
        n_samples=200
    )
    
    # 2. 复制 aug 样本
    print("\n2. 采样 Aug 数据集...")
    aug_dir = copy_random_samples(
        base_dir / "12x1_aug_2w",
        output_dir / "aug_200",
        n_samples=200
    )
    
    # 3. 分析三个数据集
    print("\n3. 分析像素分布...")
    
    clean_stats = analyze_images(clean_dir, "Clean (训练)")
    aug_stats = analyze_images(aug_dir, "Aug (增强)")
    
    # 真实世界数据
    real_dir = Path("results/test_data/test120_chestpain")
    real_stats = analyze_images(real_dir, "Real World")
    
    # 4. 对比输出
    print_stats_comparison([clean_stats, aug_stats, real_stats])
    
    # 5. 详细分析
    print("\n" + "=" * 80)
    print("关键发现")
    print("=" * 80)
    
    clean_mean = np.mean(clean_stats["means"])
    aug_mean = np.mean(aug_stats["means"])
    real_mean = np.mean(real_stats["means"])
    
    print(f"""
  1. 像素亮度 (Mean):
     - Clean:      {clean_mean:.1f}
     - Aug:        {aug_mean:.1f}
     - Real World: {real_mean:.1f}
     - Gap (Clean vs Real): {abs(clean_mean - real_mean):.1f}
     - Gap (Aug vs Real):   {abs(aug_mean - real_mean):.1f}
""")
    
    clean_std = np.mean(clean_stats["stds"])
    aug_std = np.mean(aug_stats["stds"])
    real_std = np.mean(real_stats["stds"])
    
    print(f"""  2. 对比度 (Std):
     - Clean:      {clean_std:.1f}
     - Aug:        {aug_std:.1f}
     - Real World: {real_std:.1f}
""")
    
    # 检查增强是否覆盖了真实数据的分布
    aug_mean_range = (min(aug_stats["means"]), max(aug_stats["means"]))
    real_mean_range = (min(real_stats["means"]), max(real_stats["means"]))
    
    coverage = "是" if aug_mean_range[0] <= real_mean_range[0] and aug_mean_range[1] >= real_mean_range[1] else "否"
    
    print(f"""  3. 增强数据是否覆盖真实数据分布:
     - Aug Mean 范围:  [{aug_mean_range[0]:.0f}, {aug_mean_range[1]:.0f}]
     - Real Mean 范围: [{real_mean_range[0]:.0f}, {real_mean_range[1]:.0f}]
     - 完全覆盖: {coverage}
""")
    
    # 保存结果
    results = {
        "clean": {
            "mean_avg": clean_mean,
            "mean_range": [min(clean_stats["means"]), max(clean_stats["means"])],
            "std_avg": clean_std,
            "std_range": [min(clean_stats["stds"]), max(clean_stats["stds"])],
        },
        "aug": {
            "mean_avg": aug_mean,
            "mean_range": [min(aug_stats["means"]), max(aug_stats["means"])],
            "std_avg": aug_std,
            "std_range": [min(aug_stats["stds"]), max(aug_stats["stds"])],
        },
        "real_world": {
            "mean_avg": real_mean,
            "mean_range": [min(real_stats["means"]), max(real_stats["means"])],
            "std_avg": real_std,
            "std_range": [min(real_stats["stds"]), max(real_stats["stds"])],
        }
    }
    
    with open(output_dir / "_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}/_comparison_results.json")
    print(f"Clean 样本: {clean_dir}")
    print(f"Aug 样本: {aug_dir}")


if __name__ == "__main__":
    main()
