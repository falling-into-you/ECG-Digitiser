#!/usr/bin/env python3
"""
将 test120_chestpain 目录中的所有图像和对应的 JSON 重命名为连续编号 (1, 2, 3, ...)
"""

import os
import json
import shutil
from pathlib import Path


def main():
    input_dir = Path("results/test_data/test120_chestpain")
    
    # 获取所有 PNG 文件（排除临时文件和 .DS_Store）
    png_files = sorted([f for f in input_dir.iterdir() 
                       if f.suffix.lower() == '.png' and not f.name.startswith('.')])
    
    print(f"找到 {len(png_files)} 个 PNG 文件")
    
    # 创建映射：原始名称 -> 新编号
    rename_map = {}
    
    for idx, png_file in enumerate(png_files, start=1):
        stem = png_file.stem  # 不带扩展名的文件名
        rename_map[stem] = idx
    
    # 先备份原始文件名映射
    mapping_file = input_dir / "_rename_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump({str(k): v for k, v in rename_map.items()}, f, ensure_ascii=False, indent=2)
    print(f"已保存映射到: {mapping_file}")
    
    # 为避免重名冲突，先重命名到临时名称，再重命名到最终名称
    temp_suffix = "_temp_rename_"
    
    # 第一步：重命名到临时名称
    for stem, new_idx in rename_map.items():
        old_png = input_dir / f"{stem}.png"
        old_json = input_dir / f"{stem}.json"
        
        if old_png.exists():
            temp_png = input_dir / f"{temp_suffix}{new_idx}.png"
            shutil.move(str(old_png), str(temp_png))
        
        if old_json.exists():
            temp_json = input_dir / f"{temp_suffix}{new_idx}.json"
            shutil.move(str(old_json), str(temp_json))
    
    # 第二步：从临时名称重命名到最终名称
    for stem, new_idx in rename_map.items():
        temp_png = input_dir / f"{temp_suffix}{new_idx}.png"
        temp_json = input_dir / f"{temp_suffix}{new_idx}.json"
        
        final_png = input_dir / f"{new_idx}.png"
        final_json = input_dir / f"{new_idx}.json"
        
        if temp_png.exists():
            shutil.move(str(temp_png), str(final_png))
            print(f"  {stem}.png -> {new_idx}.png")
        
        if temp_json.exists():
            # 读取 JSON 并更新内容
            with open(temp_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 更新 JSON 中的图像引用（如果有的话）
            if 'image_path' in data:
                data['image_path'] = f"{new_idx}.png"
            if 'image' in data:
                data['image'] = f"{new_idx}.png"
            if 'filename' in data:
                data['filename'] = f"{new_idx}.png"
            # 保留原始文件名作为参考
            data['original_filename'] = f"{stem}.png"
            
            with open(final_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"  {stem}.json -> {new_idx}.json")
    
    # 删除 .DS_Store 如果存在
    ds_store = input_dir / ".DS_Store"
    if ds_store.exists():
        ds_store.unlink()
        print("已删除 .DS_Store")
    
    print(f"\n完成！共重命名 {len(rename_map)} 对文件")
    print(f"文件现在编号为: 1 到 {len(rename_map)}")


if __name__ == "__main__":
    main()
