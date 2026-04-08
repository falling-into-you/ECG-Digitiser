#!/usr/bin/env python3
"""
预处理 Real World ECG 图像，使其像素分布接近训练数据

问题：
- 训练数据: mean=87.2, std=62.5 (深色背景，浅色线条)
- Real World: mean=217-230 (白色背景，深色线条)

解决方案：反转图像颜色
"""

import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def invert_image(img_array):
    """反转图像颜色: 255 - pixel"""
    return 255 - img_array


def adjust_contrast(img_array, target_mean=87.2, target_std=62.5):
    """
    调整图像对比度使其接近目标分布
    使用线性变换: new = (old - old_mean) * (target_std / old_std) + target_mean
    """
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    old_mean = gray.mean()
    old_std = gray.std()
    
    if old_std < 1e-6:
        return img_array
    
    # 对每个通道应用变换
    result = img_array.astype(np.float32)
    for c in range(result.shape[2]):
        result[:, :, c] = (result[:, :, c] - old_mean) * (target_std / old_std) + target_mean
    
    return np.clip(result, 0, 255).astype(np.uint8)


def preprocess_image(input_path, output_path, method="invert"):
    """预处理单张图像"""
    img = Image.open(input_path).convert("RGB")
    img_array = np.array(img)
    
    if method == "invert":
        # 方法1: 简单反转
        processed = invert_image(img_array)
    elif method == "adjust":
        # 方法2: 调整到目标分布
        processed = adjust_contrast(img_array)
    elif method == "invert_adjust":
        # 方法3: 先反转再调整
        inverted = invert_image(img_array)
        processed = adjust_contrast(inverted)
    else:
        processed = img_array
    
    # 保存
    Image.fromarray(processed).save(output_path)
    return processed


def analyze_stats(img_array, name=""):
    """输出图像统计"""
    gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
    print(f"  {name}: mean={gray.mean():.1f}, std={gray.std():.1f}, min={gray.min():.0f}, max={gray.max():.0f}")


def main():
    parser = argparse.ArgumentParser(description="预处理 Real World ECG 图像")
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="输入目录")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--method", type=str, default="invert", 
                       choices=["invert", "adjust", "invert_adjust"],
                       help="预处理方法: invert(反转), adjust(调整分布), invert_adjust(先反转再调整)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"预处理方法: {args.method}")
    print(f"训练数据参考: mean=87.2, std=62.5")
    print()
    
    for f in sorted(input_dir.iterdir()):
        if f.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            output_path = output_dir / f"{f.stem}.png"  # 统一输出为 PNG
            
            # 原始统计
            original = np.array(Image.open(f).convert("RGB"))
            print(f"{f.name}:")
            analyze_stats(original, "原始")
            
            # 预处理
            processed = preprocess_image(f, output_path, args.method)
            analyze_stats(processed, "处理后")
            print()
    
    print(f"预处理完成，输出目录: {output_dir}")


if __name__ == "__main__":
    main()
