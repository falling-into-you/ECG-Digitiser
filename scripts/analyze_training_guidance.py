#!/usr/bin/env python3
"""
分析真实世界 ECG 数据特点，为训练集构造提供指导
"""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict


def analyze_for_training_guidance():
    data_dir = Path("results/test_data/test120_chestpain")
    
    # 加载之前的分析结果
    with open(data_dir / "_dataset_analysis.json") as f:
        analysis = json.load(f)
    
    # 训练数据参考
    train_mean = 87.2
    train_std = 62.5
    
    print("=" * 70)
    print("真实世界 ECG 数据分析 → 训练集构造指导")
    print("=" * 70)
    
    stats = analysis["individual_stats"]
    
    # ========== 1. 像素分布差异分析 ==========
    print("\n" + "=" * 70)
    print("1. 像素分布差异 (Domain Gap)")
    print("=" * 70)
    
    means = [s["gray_mean"] for s in stats]
    stds = [s["gray_std"] for s in stats]
    
    print(f"\n  训练数据:  mean={train_mean:.1f}, std={train_std:.1f}")
    print(f"  真实数据:  mean={np.mean(means):.1f}±{np.std(means):.1f}, std={np.mean(stds):.1f}±{np.std(stds):.1f}")
    print(f"\n  差距分析:")
    print(f"    - Mean 差距: {abs(np.mean(means) - train_mean):.1f} (训练偏暗, 真实偏亮)")
    print(f"    - Std 比值: {np.mean(stds)/train_std:.2f}x (真实数据对比度更低)")
    
    # 按 mean 分段统计
    bins = [(0, 100, "深色背景"), (100, 150, "中灰背景"), (150, 200, "浅色背景"), (200, 256, "白色背景")]
    print(f"\n  Mean 分布:")
    for low, high, label in bins:
        count = sum(1 for m in means if low <= m < high)
        pct = count / len(means) * 100
        bar = "█" * int(pct / 2)
        print(f"    {label:10s} ({low:3d}-{high:3d}): {count:3d} ({pct:5.1f}%) {bar}")
    
    # ========== 2. 图像尺寸分析 ==========
    print("\n" + "=" * 70)
    print("2. 图像尺寸分布")
    print("=" * 70)
    
    widths = [s["size"][0] for s in stats]
    heights = [s["size"][1] for s in stats]
    aspects = [w/h for w, h in zip(widths, heights)]
    
    print(f"\n  宽度: {min(widths)} ~ {max(widths)}, 平均 {np.mean(widths):.0f}")
    print(f"  高度: {min(heights)} ~ {max(heights)}, 平均 {np.mean(heights):.0f}")
    print(f"  宽高比: {min(aspects):.2f} ~ {max(aspects):.2f}, 平均 {np.mean(aspects):.2f}")
    
    # 尺寸分布
    size_bins = [(0, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]
    print(f"\n  宽度分布:")
    for low, high in size_bins:
        count = sum(1 for w in widths if low <= w < high)
        if count > 0:
            print(f"    {low:4d}-{high:4d}: {count:3d} ({count/len(widths)*100:.1f}%)")
    
    # ========== 3. 颜色通道分析 ==========
    print("\n" + "=" * 70)
    print("3. 颜色特征")
    print("=" * 70)
    
    r_means = [s["r_mean"] for s in stats]
    g_means = [s["g_mean"] for s in stats]
    b_means = [s["b_mean"] for s in stats]
    
    print(f"\n  RGB 通道均值:")
    print(f"    R: {np.mean(r_means):.1f} ± {np.std(r_means):.1f}")
    print(f"    G: {np.mean(g_means):.1f} ± {np.std(g_means):.1f}")
    print(f"    B: {np.mean(b_means):.1f} ± {np.std(b_means):.1f}")
    
    # 检查是否有明显色偏
    color_bias = []
    for s in stats:
        r, g, b = s["r_mean"], s["g_mean"], s["b_mean"]
        if abs(r - g) > 20 or abs(g - b) > 20 or abs(r - b) > 20:
            color_bias.append(Path(s["path"]).name)
    print(f"\n  有明显色偏的图像: {len(color_bias)} 张 ({len(color_bias)/len(stats)*100:.1f}%)")
    
    # ========== 4. 训练集构造建议 ==========
    print("\n" + "=" * 70)
    print("4. 训练集构造建议")
    print("=" * 70)
    
    print("""
  【问题 1】像素分布不匹配
  ─────────────────────────────────────────────────────────
  现状: 训练数据 mean=87 (偏暗), 真实数据 mean=178 (偏亮)
  
  建议:
    A) 数据增强 - 在训练时添加:
       - 亮度变换: 随机调整 mean 到 80-220 范围
       - 对比度变换: 随机调整 std 到 20-80 范围
       - 反转增强: 以 50% 概率反转图像 (255-pixel)
       
    B) 归一化策略 - 修改预处理:
       - 当前: ZScore(mean=87.2, std=62.5)
       - 建议: 改用 [0,1] 归一化 或 per-image 标准化
       
    C) 混合训练数据:
       - 添加白底 ECG 图像到训练集
       - 或合成不同亮度的变体
""")
    
    print("""
  【问题 2】图像尺寸差异大
  ─────────────────────────────────────────────────────────
  现状: 真实数据尺寸 326-2822 像素，变化范围大
  
  建议:
    A) 多尺度训练:
       - 使用多尺度输入 (如 512, 768, 1024)
       - 或随机 resize 增强
       
    B) 推理时策略:
       - 对大图使用滑动窗口
       - 保持宽高比 resize
""")
    
    print("""
  【问题 3】真实场景干扰
  ─────────────────────────────────────────────────────────
  现状: 真实照片包含背景、阴影、倾斜、手写标注等
  
  建议:
    A) 增强模拟真实场景:
       - 添加随机背景
       - 添加阴影/光照变化
       - 随机旋转 (±15°)
       - 添加噪声/模糊
       - 添加手写文字覆盖
       
    B) 数据采集:
       - 收集更多真实拍摄的 ECG 图像
       - 包含不同设备、光照条件
""")

    # ========== 5. 具体增强参数建议 ==========
    print("\n" + "=" * 70)
    print("5. 推荐的数据增强参数")
    print("=" * 70)
    
    print(f"""
  基于真实数据分布，建议的增强参数:
  
  亮度调整:
    - target_mean: [{min(means):.0f}, {max(means):.0f}] → 建议 [80, 240]
    
  对比度调整:
    - target_std: [{min(stds):.0f}, {max(stds):.0f}] → 建议 [15, 80]
    
  尺寸变换:
    - resize_range: [0.5, 2.0]
    - 最小边: {min(min(widths), min(heights))} → 建议最小 320
    
  几何变换:
    - rotation: ±10°
    - perspective: 轻微透视变换
    
  颜色增强:
    - 色偏图像比例: {len(color_bias)/len(stats)*100:.1f}%
    - 建议: 添加轻微色调偏移
""")

    # ========== 6. 关键数值总结 ==========
    print("\n" + "=" * 70)
    print("6. 关键数值总结 (用于配置增强)")
    print("=" * 70)
    
    summary = {
        "pixel_distribution": {
            "real_world_mean": float(np.mean(means)),
            "real_world_std": float(np.mean(stds)),
            "train_mean": train_mean,
            "train_std": train_std,
            "mean_range": [float(min(means)), float(max(means))],
            "std_range": [float(min(stds)), float(max(stds))],
        },
        "size": {
            "width_range": [min(widths), max(widths)],
            "height_range": [min(heights), max(heights)],
            "aspect_ratio_range": [round(min(aspects), 2), round(max(aspects), 2)],
        },
        "augmentation_recommendations": {
            "brightness_range": [80, 240],
            "contrast_range": [15, 80],
            "rotation_range": [-10, 10],
            "scale_range": [0.5, 2.0],
            "invert_probability": 0.5,
        }
    }
    
    print(f"""
  像素分布:
    - 真实数据 mean: {summary['pixel_distribution']['real_world_mean']:.1f}
    - 真实数据 std:  {summary['pixel_distribution']['real_world_std']:.1f}
    - Mean 范围: {summary['pixel_distribution']['mean_range']}
    - Std 范围:  {summary['pixel_distribution']['std_range']}
    
  尺寸:
    - 宽度范围: {summary['size']['width_range']}
    - 高度范围: {summary['size']['height_range']}
    - 宽高比范围: {summary['size']['aspect_ratio_range']}
    
  推荐增强参数:
    - 亮度范围: {summary['augmentation_recommendations']['brightness_range']}
    - 对比度范围: {summary['augmentation_recommendations']['contrast_range']}
    - 旋转范围: {summary['augmentation_recommendations']['rotation_range']}°
    - 缩放范围: {summary['augmentation_recommendations']['scale_range']}
    - 反转概率: {summary['augmentation_recommendations']['invert_probability']}
""")
    
    # 保存建议
    output_file = data_dir / "_training_guidance.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n数值总结已保存到: {output_file}")


if __name__ == "__main__":
    analyze_for_training_guidance()
