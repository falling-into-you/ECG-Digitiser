#!/usr/bin/env python3
"""
对比训练数据和测试数据的分布情况
分析像素统计、尺寸、颜色分布等特征
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict
import json


def find_all_pngs(base_dir):
    """递归查找所有 PNG 文件"""
    base_dir = Path(base_dir)
    return list(base_dir.rglob("*.png"))


def analyze_single_image(img_path):
    """分析单张图像的各项指标"""
    img = Image.open(img_path)
    
    # 基本信息
    width, height = img.size
    mode = img.mode
    
    # 转灰度分析
    if mode == 'RGB' or mode == 'RGBA':
        arr = np.array(img.convert('RGB'))
        gray = np.mean(arr, axis=2)
    elif mode == 'L':
        gray = np.array(img)
        arr = np.stack([gray]*3, axis=2)
    else:
        img = img.convert('RGB')
        arr = np.array(img)
        gray = np.mean(arr, axis=2)
    
    # RGB 通道分析
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    
    return {
        'path': str(img_path),
        'width': width,
        'height': height,
        'aspect_ratio': width / height if height > 0 else 0,
        'pixels': width * height,
        # 灰度统计
        'gray_mean': float(gray.mean()),
        'gray_std': float(gray.std()),
        'gray_min': float(gray.min()),
        'gray_max': float(gray.max()),
        'gray_median': float(np.median(gray)),
        # RGB 通道均值
        'r_mean': float(r.mean()),
        'g_mean': float(g.mean()),
        'b_mean': float(b.mean()),
        # 亮度分布（百分位数）
        'gray_p5': float(np.percentile(gray, 5)),
        'gray_p25': float(np.percentile(gray, 25)),
        'gray_p75': float(np.percentile(gray, 75)),
        'gray_p95': float(np.percentile(gray, 95)),
        # 边缘像素（通常是背景）
        'border_mean': float(np.concatenate([gray[0,:], gray[-1,:], gray[:,0], gray[:,-1]]).mean()),
    }


def analyze_dataset(data_dir, name):
    """分析整个数据集"""
    pngs = find_all_pngs(data_dir)
    
    if not pngs:
        return None
    
    print(f"  分析 {name}: {len(pngs)} 张图像...")
    
    all_stats = []
    for png in pngs:
        try:
            stats = analyze_single_image(png)
            all_stats.append(stats)
        except Exception as e:
            print(f"    跳过 {png}: {e}")
    
    return {
        'name': name,
        'count': len(all_stats),
        'images': all_stats
    }


def compute_summary(dataset):
    """计算数据集的汇总统计"""
    if not dataset or not dataset['images']:
        return None
    
    images = dataset['images']
    
    def stats_for_field(field):
        values = [img[field] for img in images]
        return {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'p5': float(np.percentile(values, 5)),
            'p95': float(np.percentile(values, 95)),
        }
    
    return {
        'name': dataset['name'],
        'count': dataset['count'],
        'width': stats_for_field('width'),
        'height': stats_for_field('height'),
        'aspect_ratio': stats_for_field('aspect_ratio'),
        'gray_mean': stats_for_field('gray_mean'),
        'gray_std': stats_for_field('gray_std'),
        'r_mean': stats_for_field('r_mean'),
        'g_mean': stats_for_field('g_mean'),
        'b_mean': stats_for_field('b_mean'),
        'border_mean': stats_for_field('border_mean'),
    }


def histogram_distribution(values, bins=None):
    """计算值的直方图分布"""
    if bins is None:
        bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 256)]
    
    result = []
    for low, high in bins:
        count = sum(1 for v in values if low <= v < high)
        pct = count / len(values) * 100 if values else 0
        result.append({'range': f'{low}-{high}', 'count': count, 'pct': pct})
    return result


def print_comparison(train_summary, test_summary):
    """打印对比结果"""
    print("\n" + "=" * 90)
    print("训练数据 vs 测试数据 分布对比")
    print("=" * 90)
    
    def print_field(field_name, display_name, fmt=".1f"):
        train_s = train_summary[field_name]
        test_s = test_summary[field_name]
        
        print(f"\n{display_name}:")
        print(f"  {'指标':<12} {'训练数据':<25} {'测试数据':<25} {'差异':<15}")
        print(f"  {'-'*75}")
        
        for metric in ['mean', 'std', 'min', 'max', 'median']:
            train_val = train_s[metric]
            test_val = test_s[metric]
            diff = train_val - test_val
            print(f"  {metric:<12} {train_val:<25{fmt}} {test_val:<25{fmt}} {diff:+{fmt}}")
    
    # 基本信息
    print(f"\n数据集规模:")
    print(f"  训练数据: {train_summary['count']} 张")
    print(f"  测试数据: {test_summary['count']} 张")
    
    # 尺寸对比
    print_field('width', '图像宽度', '.0f')
    print_field('height', '图像高度', '.0f')
    print_field('aspect_ratio', '宽高比', '.2f')
    
    # 像素分布对比
    print_field('gray_mean', '灰度均值 (亮度)')
    print_field('gray_std', '灰度标准差 (对比度)')
    print_field('border_mean', '边缘像素均值 (背景亮度)')
    
    # RGB 通道
    print_field('r_mean', 'R通道均值')
    print_field('g_mean', 'G通道均值')
    print_field('b_mean', 'B通道均值')


def print_histogram_comparison(train_dataset, test_dataset):
    """打印直方图对比"""
    print("\n" + "=" * 90)
    print("灰度均值分布直方图")
    print("=" * 90)
    
    train_means = [img['gray_mean'] for img in train_dataset['images']]
    test_means = [img['gray_mean'] for img in test_dataset['images']]
    
    bins = [(0, 50), (50, 100), (100, 150), (150, 200), (200, 256)]
    
    print(f"\n{'区间':<12} {'训练数据':<30} {'测试数据':<30}")
    print("-" * 75)
    
    for low, high in bins:
        train_count = sum(1 for m in train_means if low <= m < high)
        test_count = sum(1 for m in test_means if low <= m < high)
        train_pct = train_count / len(train_means) * 100
        test_pct = test_count / len(test_means) * 100
        
        train_bar = '█' * int(train_pct / 5) + '░' * (20 - int(train_pct / 5))
        test_bar = '█' * int(test_pct / 5) + '░' * (20 - int(test_pct / 5))
        
        print(f"{low:>3}-{high:<3}     {train_count:>3} ({train_pct:>5.1f}%) {train_bar}  {test_count:>3} ({test_pct:>5.1f}%) {test_bar}")


def print_coverage_analysis(train_dataset, test_dataset):
    """分析训练数据是否覆盖测试数据的分布"""
    print("\n" + "=" * 90)
    print("分布覆盖分析 (训练数据是否覆盖测试数据的范围)")
    print("=" * 90)
    
    fields = [
        ('gray_mean', '灰度均值'),
        ('gray_std', '灰度标准差'),
        ('width', '图像宽度'),
        ('height', '图像高度'),
        ('border_mean', '背景亮度'),
    ]
    
    print(f"\n{'指标':<15} {'训练范围':<25} {'测试范围':<25} {'覆盖情况':<20}")
    print("-" * 90)
    
    for field, display in fields:
        train_vals = [img[field] for img in train_dataset['images']]
        test_vals = [img[field] for img in test_dataset['images']]
        
        train_min, train_max = min(train_vals), max(train_vals)
        test_min, test_max = min(test_vals), max(test_vals)
        
        # 检查覆盖
        covers_min = train_min <= test_min
        covers_max = train_max >= test_max
        
        if covers_min and covers_max:
            status = "✓ 完全覆盖"
        elif covers_min or covers_max:
            status = "△ 部分覆盖"
        else:
            status = "✗ 未覆盖"
        
        print(f"{display:<15} [{train_min:>8.1f}, {train_max:>8.1f}]   [{test_min:>8.1f}, {test_max:>8.1f}]   {status}")


def print_recommendations(train_summary, test_summary, train_dataset, test_dataset):
    """打印建议"""
    print("\n" + "=" * 90)
    print("分析结论与建议")
    print("=" * 90)
    
    issues = []
    
    # 1. 亮度差异
    train_mean = train_summary['gray_mean']['mean']
    test_mean = test_summary['gray_mean']['mean']
    diff = abs(train_mean - test_mean)
    
    if diff > 30:
        issues.append(f"亮度差异较大 ({diff:.1f}): 训练={train_mean:.1f}, 测试={test_mean:.1f}")
    
    # 2. 对比度差异
    train_std = train_summary['gray_std']['mean']
    test_std = test_summary['gray_std']['mean']
    std_diff = abs(train_std - test_std)
    
    if std_diff > 15:
        issues.append(f"对比度差异较大 ({std_diff:.1f}): 训练={train_std:.1f}, 测试={test_std:.1f}")
    
    # 3. 尺寸差异
    train_w = train_summary['width']['mean']
    test_w = test_summary['width']['mean']
    train_h = train_summary['height']['mean']
    test_h = test_summary['height']['mean']
    
    if abs(train_w - test_w) / test_w > 0.3 or abs(train_h - test_h) / test_h > 0.3:
        issues.append(f"尺寸差异较大: 训练={train_w:.0f}x{train_h:.0f}, 测试={test_w:.0f}x{test_h:.0f}")
    
    # 4. 分布范围覆盖
    train_means = [img['gray_mean'] for img in train_dataset['images']]
    test_means = [img['gray_mean'] for img in test_dataset['images']]
    
    if min(train_means) > min(test_means):
        issues.append(f"训练数据缺少低亮度样本: 训练最小={min(train_means):.1f}, 测试最小={min(test_means):.1f}")
    
    if max(train_means) < max(test_means):
        issues.append(f"训练数据缺少高亮度样本: 训练最大={max(train_means):.1f}, 测试最大={max(test_means):.1f}")
    
    if issues:
        print("\n发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n建议:")
        print("  - 调整数据增强参数，使训练数据分布更接近测试数据")
        print("  - 增加训练样本数量，确保覆盖测试数据的分布范围")
        print("  - 考虑使用 domain adaptation 技术")
    else:
        print("\n✓ 训练数据和测试数据分布基本一致")


def main():
    train_dir = Path("results/train_data")
    test_dir = Path("results/test_data/test120_chestpain")
    
    print("=" * 90)
    print("训练数据 vs 测试数据 分布分析")
    print("=" * 90)
    
    # 分析训练数据（包含子目录）
    print("\n1. 分析训练数据...")
    train_dataset = analyze_dataset(train_dir, "训练数据")
    
    # 分析测试数据
    print("\n2. 分析测试数据...")
    test_dataset = analyze_dataset(test_dir, "测试数据")
    
    if not train_dataset or not test_dataset:
        print("错误: 数据集为空")
        return
    
    # 计算汇总
    train_summary = compute_summary(train_dataset)
    test_summary = compute_summary(test_dataset)
    
    # 打印对比
    print_comparison(train_summary, test_summary)
    print_histogram_comparison(train_dataset, test_dataset)
    print_coverage_analysis(train_dataset, test_dataset)
    print_recommendations(train_summary, test_summary, train_dataset, test_dataset)
    
    # 按增强级别分析训练数据
    print("\n" + "=" * 90)
    print("训练数据各增强级别明细")
    print("=" * 90)
    
    aug_dirs = sorted(train_dir.iterdir())
    for aug_dir in aug_dirs:
        if aug_dir.is_dir():
            aug_dataset = analyze_dataset(aug_dir, aug_dir.name)
            if aug_dataset and aug_dataset['images']:
                means = [img['gray_mean'] for img in aug_dataset['images']]
                stds = [img['gray_std'] for img in aug_dataset['images']]
                print(f"\n  {aug_dir.name}:")
                print(f"    数量: {len(means)}")
                print(f"    灰度均值: {np.mean(means):.1f} (范围: [{min(means):.1f}, {max(means):.1f}])")
                print(f"    灰度标准差: {np.mean(stds):.1f} (范围: [{min(stds):.1f}, {max(stds):.1f}])")
    
    # 保存详细结果
    output_file = Path("results/train_test_distribution_analysis.json")
    results = {
        'train': {
            'summary': train_summary,
            'images': train_dataset['images']
        },
        'test': {
            'summary': test_summary,
            'images': test_dataset['images']
        }
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
