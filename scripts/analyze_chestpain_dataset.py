#!/usr/bin/env python3
"""
分析 test120_chestpain 数据集的图像统计信息
为后续分割模型提供参考
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict


def analyze_image(img_path):
    """分析单张图像"""
    img = Image.open(img_path)
    img_array = np.array(img)
    
    stats = {
        "path": str(img_path),
        "mode": img.mode,
        "size": img.size,  # (width, height)
        "shape": img_array.shape,
    }
    
    # 转为 RGB 进行分析
    if img.mode != "RGB":
        img_rgb = img.convert("RGB")
        img_array = np.array(img_rgb)
    
    # 灰度统计
    gray = np.mean(img_array, axis=2)
    stats["gray_mean"] = float(gray.mean())
    stats["gray_std"] = float(gray.std())
    stats["gray_min"] = float(gray.min())
    stats["gray_max"] = float(gray.max())
    stats["gray_median"] = float(np.median(gray))
    
    # RGB 通道统计
    stats["r_mean"] = float(img_array[:,:,0].mean())
    stats["g_mean"] = float(img_array[:,:,1].mean())
    stats["b_mean"] = float(img_array[:,:,2].mean())
    
    # 判断是否为白底（mean > 200）还是深底（mean < 100）
    if gray.mean() > 200:
        stats["background_type"] = "white"
    elif gray.mean() < 100:
        stats["background_type"] = "dark"
    else:
        stats["background_type"] = "mixed"
    
    # 像素直方图（简化版，10 个 bin）
    hist, _ = np.histogram(gray.flatten(), bins=10, range=(0, 255))
    stats["gray_histogram"] = hist.tolist()
    
    return stats


def main():
    data_dir = Path("results/test_data/test120_chestpain")
    output_file = data_dir / "_dataset_analysis.json"
    
    # 训练数据参考值
    train_mean = 87.2
    train_std = 62.5
    
    print("=" * 70)
    print("ECG 真实图像数据集分析")
    print("=" * 70)
    print(f"数据目录: {data_dir}")
    print(f"训练数据参考: mean={train_mean}, std={train_std}")
    print()
    
    # 收集所有图像
    png_files = sorted([f for f in data_dir.iterdir() 
                       if f.suffix.lower() == '.png' and f.stem.isdigit()],
                      key=lambda x: int(x.stem))
    
    print(f"共找到 {len(png_files)} 张图像")
    print()
    
    # 分析每张图像
    all_stats = []
    all_means = []
    all_stds = []
    all_widths = []
    all_heights = []
    background_counts = defaultdict(int)
    mode_counts = defaultdict(int)
    
    for img_path in png_files:
        stats = analyze_image(img_path)
        all_stats.append(stats)
        all_means.append(stats["gray_mean"])
        all_stds.append(stats["gray_std"])
        all_widths.append(stats["size"][0])
        all_heights.append(stats["size"][1])
        background_counts[stats["background_type"]] += 1
        mode_counts[stats["mode"]] += 1
    
    # 汇总统计
    print("=" * 70)
    print("1. 图像尺寸统计")
    print("=" * 70)
    print(f"  宽度: min={min(all_widths)}, max={max(all_widths)}, mean={np.mean(all_widths):.0f}")
    print(f"  高度: min={min(all_heights)}, max={max(all_heights)}, mean={np.mean(all_heights):.0f}")
    print(f"  总像素范围: {min(all_widths)*min(all_heights):,} ~ {max(all_widths)*max(all_heights):,}")
    
    print()
    print("=" * 70)
    print("2. 图像模式分布")
    print("=" * 70)
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count} ({count/len(png_files)*100:.1f}%)")
    
    print()
    print("=" * 70)
    print("3. 像素灰度统计")
    print("=" * 70)
    print(f"  Mean 分布:")
    print(f"    min: {min(all_means):.1f}")
    print(f"    max: {max(all_means):.1f}")
    print(f"    平均: {np.mean(all_means):.1f}")
    print(f"    中位数: {np.median(all_means):.1f}")
    print(f"    标准差: {np.std(all_means):.1f}")
    print()
    print(f"  Std 分布:")
    print(f"    min: {min(all_stds):.1f}")
    print(f"    max: {max(all_stds):.1f}")
    print(f"    平均: {np.mean(all_stds):.1f}")
    
    print()
    print("=" * 70)
    print("4. 背景类型分布")
    print("=" * 70)
    for bg_type, count in sorted(background_counts.items()):
        print(f"  {bg_type}: {count} ({count/len(png_files)*100:.1f}%)")
    
    print()
    print("=" * 70)
    print("5. 与训练数据对比")
    print("=" * 70)
    print(f"  训练数据: mean={train_mean}, std={train_std}")
    print(f"  本数据集: mean={np.mean(all_means):.1f}, std={np.mean(all_stds):.1f}")
    print(f"  Mean 差距: {abs(np.mean(all_means) - train_mean):.1f}")
    print(f"  Std 比值: {np.mean(all_stds) / train_std:.2f}x")
    
    # 分析哪些图像需要预处理
    need_preprocess = []
    for stats in all_stats:
        mean_diff = abs(stats["gray_mean"] - train_mean)
        if mean_diff > 50:  # 差距超过 50 需要预处理
            need_preprocess.append({
                "file": Path(stats["path"]).name,
                "mean": stats["gray_mean"],
                "diff": mean_diff,
                "background": stats["background_type"]
            })
    
    print()
    print("=" * 70)
    print(f"6. 需要预处理的图像 (mean 差距 > 50): {len(need_preprocess)} 张")
    print("=" * 70)
    if need_preprocess:
        # 按背景类型分组
        white_bg = [x for x in need_preprocess if x["background"] == "white"]
        dark_bg = [x for x in need_preprocess if x["background"] == "dark"]
        mixed_bg = [x for x in need_preprocess if x["background"] == "mixed"]
        print(f"  白底: {len(white_bg)} 张")
        print(f"  深底: {len(dark_bg)} 张")
        print(f"  混合: {len(mixed_bg)} 张")
    
    # Mean 分布直方图
    print()
    print("=" * 70)
    print("7. Mean 值分布直方图")
    print("=" * 70)
    bins = [0, 50, 100, 150, 200, 256]
    bin_labels = ["0-50 (深)", "50-100 (偏深)", "100-150 (中)", "150-200 (偏亮)", "200-255 (白)"]
    hist, _ = np.histogram(all_means, bins=bins)
    for i, (label, count) in enumerate(zip(bin_labels, hist)):
        bar = "█" * (count // 2) if count > 0 else ""
        print(f"  {label:20s}: {count:3d} {bar}")
    
    # 保存详细结果
    summary = {
        "total_images": len(png_files),
        "size_stats": {
            "width": {"min": min(all_widths), "max": max(all_widths), "mean": float(np.mean(all_widths))},
            "height": {"min": min(all_heights), "max": max(all_heights), "mean": float(np.mean(all_heights))}
        },
        "gray_stats": {
            "mean": {"min": min(all_means), "max": max(all_means), "avg": float(np.mean(all_means)), "std": float(np.std(all_means))},
            "std": {"min": min(all_stds), "max": max(all_stds), "avg": float(np.mean(all_stds))}
        },
        "background_distribution": dict(background_counts),
        "mode_distribution": dict(mode_counts),
        "train_reference": {"mean": train_mean, "std": train_std},
        "preprocess_needed": len(need_preprocess),
        "individual_stats": all_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print()
    print(f"详细分析结果已保存到: {output_file}")
    
    # 建议
    print()
    print("=" * 70)
    print("8. 预处理建议")
    print("=" * 70)
    if len(white_bg) > 0:
        print(f"  - {len(white_bg)} 张白底图像需要反转 (255-pixel) 后再调整分布")
    if len(mixed_bg) > 0:
        print(f"  - {len(mixed_bg)} 张混合背景图像需要单独检查")
    print(f"  - 推荐使用 invert_adjust 方法使 mean 接近 {train_mean}")


if __name__ == "__main__":
    main()
