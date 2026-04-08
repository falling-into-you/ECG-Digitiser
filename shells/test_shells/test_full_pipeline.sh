#!/bin/bash
# 测试完整的 ECG 图像生成 + 后处理流程
# 用法: bash shells/test_shells/test_full_pipeline.sh

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$PROJECT_ROOT"

# ============ 参数配置 ============
INPUT_DIR="$PROJECT_ROOT/results/test_data/timeseries/40792771"
OUTPUT_DIR="$PROJECT_ROOT/results/test_data/pipeline_test"
RESAMPLE_FACTOR=20  # 像素插值倍数
# ==================================

# 清理旧结果
rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "ECG 图像生成 + 后处理完整流程测试"
echo "============================================================"
echo "输入: $INPUT_DIR"
echo "输出: $OUTPUT_DIR"
echo ""

# ============================================================
# 步骤 1: 生成 ECG 图像
# ============================================================
echo "============================="
echo "步骤 1/4: 生成 ECG 图像"
echo "============================="

cd "$PROJECT_ROOT/ecg-image-generator"

python gen_ecg_images_from_data_batch.py \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR/generated" \
    --config_file config_12x1.yaml \
    --num_columns 1 \
    --mask_unplotted_samples \
    --store_config 2 \
    --calibration_pulse 1.0 \
    --random_grid_present 1.0 \
    --random_bw 0 \
    --standard_grid_color 5 \
    --max_num_images 1 \
    -r 200 \
    -se 42

cd "$PROJECT_ROOT"

# 检查生成结果
echo ""
echo "生成的文件:"
find "$OUTPUT_DIR/generated" -type f \( -name "*.png" -o -name "*.json" \) | head -10

# ============================================================
# 步骤 2: 像素坐标加密
# ============================================================
echo ""
echo "============================="
echo "步骤 2/4: 像素坐标加密 (resample_factor=$RESAMPLE_FACTOR)"
echo "============================="

python -m src.mimic.replot_pixels \
    --dir "$OUTPUT_DIR/generated" \
    --resample_factor $RESAMPLE_FACTOR \
    --num_workers 1

# ============================================================
# 步骤 3: 分析加密效果
# ============================================================
echo ""
echo "============================="
echo "步骤 3/4: 分析像素坐标密度"
echo "============================="

python3 << 'EOF'
import json
import numpy as np
import os
import glob

output_dir = os.environ.get('OUTPUT_DIR', 'results/test_data/pipeline_test')
json_files = glob.glob(f"{output_dir}/generated/**/*.json", recursive=True)

if not json_files:
    print("未找到 JSON 文件")
    exit(1)

json_path = json_files[0]
print(f"分析文件: {json_path}")

with open(json_path, 'r') as f:
    data = json.load(f)

print(f"\n图像尺寸: {data['width']} x {data['height']}")
print(f"导联数量: {len(data['leads'])}")

# 分析第一个导联
lead = data['leads'][0]
name = lead['lead_name']
orig = np.array(lead['plotted_pixels'])
dense = np.array(lead.get('dense_plotted_pixels', []))

print(f"\n导联 {name}:")
print(f"  原始点数: {len(orig)}")
print(f"  加密后点数: {len(dense)}")

if len(dense) > 0:
    # 分析相邻点距离
    orig_diffs = np.diff(orig, axis=0)
    orig_dist = np.sqrt(orig_diffs[:, 0]**2 + orig_diffs[:, 1]**2)
    
    dense_diffs = np.diff(dense, axis=0)
    dense_dist = np.sqrt(dense_diffs[:, 0]**2 + dense_diffs[:, 1]**2)
    
    print(f"\n原始坐标相邻点距离:")
    print(f"  平均: {np.mean(orig_dist):.4f} px")
    print(f"  最大: {np.max(orig_dist):.4f} px")
    print(f"  > 1px 的数量: {np.sum(orig_dist > 1)}")
    
    print(f"\n加密后坐标相邻点距离:")
    print(f"  平均: {np.mean(dense_dist):.4f} px")
    print(f"  最大: {np.max(dense_dist):.4f} px")
    print(f"  > 1px 的数量: {np.sum(dense_dist > 1)}")
    
    if np.max(dense_dist) < 1:
        print("\n✓ 加密后所有间隙 < 1px，mask 应该连续")
    else:
        print(f"\n⚠ 仍有 {np.sum(dense_dist > 1)} 个间隙 > 1px")
EOF

# ============================================================
# 步骤 4: 生成 mask 对比
# ============================================================
echo ""
echo "============================="
echo "步骤 4/4: 生成 mask 对比"
echo "============================="

# 准备目录
mkdir -p "$OUTPUT_DIR/mask_comparison/imagesTr"
mkdir -p "$OUTPUT_DIR/mask_comparison/labelsTr_original"
mkdir -p "$OUTPUT_DIR/mask_comparison/labelsTr_dense"

# 复制文件
cp "$OUTPUT_DIR/generated/"*.png "$OUTPUT_DIR/mask_comparison/imagesTr/" 2>/dev/null || true
cp "$OUTPUT_DIR/generated/"*.json "$OUTPUT_DIR/mask_comparison/imagesTr/" 2>/dev/null || true
# 处理子目录中的文件
find "$OUTPUT_DIR/generated" -mindepth 2 -name "*.png" -exec cp {} "$OUTPUT_DIR/mask_comparison/imagesTr/" \; 2>/dev/null || true
find "$OUTPUT_DIR/generated" -mindepth 2 -name "*.json" -exec cp {} "$OUTPUT_DIR/mask_comparison/imagesTr/" \; 2>/dev/null || true

# 用原始坐标生成 mask
echo "生成 mask (原始坐标 plotted_pixels)..."
python -m src.mimic.generate_masks \
    -i "$OUTPUT_DIR/mask_comparison" \
    --plotted_pixels_key plotted_pixels \
    --num_workers 1

mv "$OUTPUT_DIR/mask_comparison/labelsTr/"*.png "$OUTPUT_DIR/mask_comparison/labelsTr_original/"

# 用加密坐标生成 mask
echo "生成 mask (加密坐标 dense_plotted_pixels)..."
python -m src.mimic.generate_masks \
    -i "$OUTPUT_DIR/mask_comparison" \
    --plotted_pixels_key dense_plotted_pixels \
    --num_workers 1

mv "$OUTPUT_DIR/mask_comparison/labelsTr/"*.png "$OUTPUT_DIR/mask_comparison/labelsTr_dense/"

# ============================================================
# 生成可视化对比图
# ============================================================
echo ""
echo "生成可视化对比图..."

python3 << 'EOF'
import os
import numpy as np
from PIL import Image
import glob

output_dir = os.environ.get('OUTPUT_DIR', 'results/test_data/pipeline_test')

# 找到图像和 mask
img_files = glob.glob(f"{output_dir}/mask_comparison/imagesTr/*.png")
img_files = [f for f in img_files if 'coordinate' not in f]

if not img_files:
    print("未找到图像文件")
    exit(1)

img_path = img_files[0]
basename = os.path.basename(img_path)

mask_orig_path = f"{output_dir}/mask_comparison/labelsTr_original/{basename}"
mask_dense_path = f"{output_dir}/mask_comparison/labelsTr_dense/{basename}"

print(f"图像: {img_path}")
print(f"原始 mask: {mask_orig_path}")
print(f"加密 mask: {mask_dense_path}")

# 加载图像
img = np.array(Image.open(img_path))
mask_orig = np.array(Image.open(mask_orig_path)) if os.path.exists(mask_orig_path) else None
mask_dense = np.array(Image.open(mask_dense_path)) if os.path.exists(mask_dense_path) else None

# 统计 mask
if mask_orig is not None:
    print(f"\n原始 mask 非零像素: {np.sum(mask_orig > 0)}")
if mask_dense is not None:
    print(f"加密 mask 非零像素: {np.sum(mask_dense > 0)}")
    if mask_orig is not None:
        ratio = np.sum(mask_dense > 0) / max(np.sum(mask_orig > 0), 1)
        print(f"密度提升: {ratio:.2f}x")

# 保存对比图 (裁剪一个小区域便于查看细节)
# 找一个有信号的区域
if mask_dense is not None and mask_dense.ndim == 2:
    rows, cols = np.where(mask_dense > 0)
    if len(rows) > 0:
        # 取中间区域
        cy, cx = int(np.median(rows)), int(np.median(cols))
        size = 200
        y1, y2 = max(0, cy - size), min(mask_dense.shape[0], cy + size)
        x1, x2 = max(0, cx - size), min(mask_dense.shape[1], cx + size)
        
        # 裁剪
        img_crop = img[y1:y2, x1:x2]
        mask_orig_crop = mask_orig[y1:y2, x1:x2] if mask_orig is not None else np.zeros_like(img_crop[:,:,0])
        mask_dense_crop = mask_dense[y1:y2, x1:x2]
        
        # 保存裁剪区域
        Image.fromarray(img_crop).save(f"{output_dir}/crop_image.png")
        Image.fromarray((mask_orig_crop * 255).astype(np.uint8)).save(f"{output_dir}/crop_mask_original.png")
        Image.fromarray((mask_dense_crop * 255).astype(np.uint8)).save(f"{output_dir}/crop_mask_dense.png")
        
        print(f"\n裁剪区域已保存:")
        print(f"  {output_dir}/crop_image.png")
        print(f"  {output_dir}/crop_mask_original.png")
        print(f"  {output_dir}/crop_mask_dense.png")

# 保存完整对比
if mask_orig is not None and mask_dense is not None:
    # 创建差异图
    diff = (mask_dense > 0).astype(np.uint8) - (mask_orig > 0).astype(np.uint8)
    diff_img = np.zeros((*diff.shape, 3), dtype=np.uint8)
    diff_img[diff > 0] = [0, 255, 0]   # 绿色: 加密后新增
    diff_img[diff < 0] = [255, 0, 0]   # 红色: 加密后丢失 (应该没有)
    diff_img[mask_orig > 0] = [255, 255, 255]  # 白色: 原始就有
    
    Image.fromarray(diff_img).save(f"{output_dir}/mask_diff.png")
    print(f"  {output_dir}/mask_diff.png (白=原始, 绿=新增)")
EOF

# ============================================================
# 总结
# ============================================================
echo ""
echo "============================================================"
echo "测试完成!"
echo "============================================================"
echo ""
echo "输出文件:"
ls -lh "$OUTPUT_DIR"/*.png 2>/dev/null || echo "  (无)"
echo ""
echo "目录结构:"
find "$OUTPUT_DIR" -type d | head -20
echo ""
echo "查看结果:"
echo "  原始图像: $OUTPUT_DIR/mask_comparison/imagesTr/"
echo "  原始坐标 mask: $OUTPUT_DIR/mask_comparison/labelsTr_original/"
echo "  加密坐标 mask: $OUTPUT_DIR/mask_comparison/labelsTr_dense/"
echo "  裁剪对比: $OUTPUT_DIR/crop_*.png"
echo "  差异图: $OUTPUT_DIR/mask_diff.png"
echo "============================================================"
