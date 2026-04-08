#!/usr/bin/env python3
"""
分析 nnUNet 分割模型的输出
直接输出数值统计，不显示图片
"""

import os
import numpy as np
from PIL import Image


def main():
    print("=" * 60)
    print("1. Real World 分割掩码分析")
    print("=" * 60)
    
    mask_dir = "results/test_data/real_world_pred"
    for f in sorted(os.listdir(mask_dir)):
        if f.endswith(".png") and not f.startswith("_") and "_color" not in f and "_overlay" not in f:
            mask = np.array(Image.open(os.path.join(mask_dir, f)))
            unique, counts = np.unique(mask, return_counts=True)
            fg_ratio = 1 - counts[0] / mask.size if 0 in unique else 1.0
            print(f"{f}: unique_labels={list(unique)}, 前景比例={fg_ratio:.4%}")
    
    print()
    print("=" * 60)
    print("2. Real World 输入图像像素统计")
    print("=" * 60)
    
    train_mean, train_std = 87.2, 62.5
    input_dir = "results/test_data/real_world"
    for f in sorted(os.listdir(input_dir)):
        if f.lower().endswith((".png", ".jpg", ".jpeg")):
            img = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"))
            gray = np.mean(img, axis=2)
            print(f"{f}: mean={gray.mean():.1f}, std={gray.std():.1f}, min={gray.min():.0f}, max={gray.max():.0f}")
            print(f"    vs训练数据: mean差={abs(gray.mean() - train_mean):.1f}, std比={gray.std() / train_std:.2f}x")
    
    print()
    print("=" * 60)
    print("3. 12x1 标准数据输入图像像素统计（对比）")
    print("=" * 60)
    
    input_dir = "results/test_data/12x1"
    if os.path.exists(input_dir):
        for f in sorted(os.listdir(input_dir)):
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                img = np.array(Image.open(os.path.join(input_dir, f)).convert("RGB"))
                gray = np.mean(img, axis=2)
                print(f"{f}: mean={gray.mean():.1f}, std={gray.std():.1f}, min={gray.min():.0f}, max={gray.max():.0f}")
                print(f"    vs训练数据: mean差={abs(gray.mean() - train_mean):.1f}, std比={gray.std() / train_std:.2f}x")
    else:
        print("目录不存在")
    
    print()
    print("=" * 60)
    print("训练数据参考: mean=87.2, std=62.5")
    print("=" * 60)


if __name__ == "__main__":
    main()
