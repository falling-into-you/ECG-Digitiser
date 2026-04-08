#!/usr/bin/env python3
"""
可视化分割结果对比
"""

import os
import numpy as np
from PIL import Image
import colorsys


def get_distinct_colors(n):
    """生成 n 个视觉可区分的颜色"""
    colors = [(0, 0, 0)]  # 背景黑色
    for i in range(1, n):
        hue = (i - 1) / (n - 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
        colors.append(tuple(int(c * 255) for c in rgb))
    return colors


def create_colored_mask(mask, num_classes=13):
    """将灰度掩码转换为彩色"""
    colors = get_distinct_colors(num_classes)
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in enumerate(colors):
        colored[mask == label] = color
    return colored


def create_overlay(image, mask, alpha=0.5):
    """创建叠加图"""
    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=-1)
    colored_mask = create_colored_mask(mask)
    fg_mask = (mask > 0).astype(np.float32)[:, :, None]
    overlay = image * (1 - alpha * fg_mask) + colored_mask * alpha * fg_mask
    return overlay.astype(np.uint8)


def main():
    # 定义目录
    orig_input = "results/test_data/real_world"
    orig_pred = "results/test_data/real_world_pred"
    adj_input = "results/test_data/real_world_adjusted"
    adj_pred = "results/test_data/real_world_adjusted_pred"
    output_dir = "results/test_data/comparison"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理每个图像
    for i in range(1, 5):
        orig_img_path = os.path.join(orig_input, f"{i}.jpg")
        adj_img_path = os.path.join(adj_input, f"{i}.png")
        orig_mask_path = os.path.join(orig_pred, f"{i}.png")
        adj_mask_path = os.path.join(adj_pred, f"{i}.png")
        
        if not os.path.exists(orig_img_path):
            continue
            
        # 加载数据
        orig_img = np.array(Image.open(orig_img_path).convert("RGB"))
        adj_img = np.array(Image.open(adj_img_path).convert("RGB"))
        orig_mask = np.array(Image.open(orig_mask_path))
        adj_mask = np.array(Image.open(adj_mask_path))
        
        # 创建叠加图
        orig_overlay = create_overlay(orig_img, orig_mask)
        adj_overlay = create_overlay(adj_img, adj_mask)
        
        # 保存
        Image.fromarray(orig_overlay).save(os.path.join(output_dir, f"{i}_orig_overlay.png"))
        Image.fromarray(adj_overlay).save(os.path.join(output_dir, f"{i}_adj_overlay.png"))
        Image.fromarray(create_colored_mask(orig_mask)).save(os.path.join(output_dir, f"{i}_orig_mask_color.png"))
        Image.fromarray(create_colored_mask(adj_mask)).save(os.path.join(output_dir, f"{i}_adj_mask_color.png"))
        
        print(f"图像 {i}:")
        print(f"  原始掩码唯一值: {np.unique(orig_mask)}")
        print(f"  调整后掩码唯一值: {np.unique(adj_mask)}")
    
    print(f"\n对比图已保存到: {output_dir}")


if __name__ == "__main__":
    main()
