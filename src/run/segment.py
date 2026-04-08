"""
ECG 图像分割推理脚本
使用训练好的 nnUNet 模型对 ECG 图像进行语义分割

用法:
    python -m src.run.segment -i <输入目录> -o <输出目录> -m <模型目录>

示例:
    python -m src.run.segment -i results/test_data -o results/predictions -m results/nnUNet_1
"""

import argparse
import os
import sys
import shutil
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# 导联标签映射
LEAD_LABELS = {
    0: "background",
    1: "I",
    2: "II", 
    3: "III",
    4: "aVR",
    5: "aVL",
    6: "aVF",
    7: "V1",
    8: "V2",
    9: "V3",
    10: "V4",
    11: "V5",
    12: "V6",
}

# 颜色映射 (RGB)
LEAD_COLORS = {
    0: (0, 0, 0),        # background - black
    1: (31, 119, 180),   # I - blue
    2: (255, 127, 14),   # II - orange
    3: (44, 160, 44),    # III - green
    4: (214, 39, 40),    # aVR - red
    5: (148, 103, 189),  # aVL - purple
    6: (140, 86, 75),    # aVF - brown
    7: (227, 119, 194),  # V1 - pink
    8: (127, 127, 127),  # V2 - gray
    9: (188, 189, 34),   # V3 - olive
    10: (23, 190, 207),  # V4 - cyan
    11: (255, 255, 0),   # V5 - yellow
    12: (0, 255, 255),   # V6 - cyan-light
}


def get_parser():
    parser = argparse.ArgumentParser(
        description="ECG 图像分割推理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m src.run.segment -i results/test_data -o results/predictions
    python -m src.run.segment -i results/test_data -o results/predictions --overlay
        """
    )
    parser.add_argument(
        "-i", "--input_dir",
        type=str, required=True,
        help="输入图像目录 (包含 PNG 格式的 ECG 图像)"
    )
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, required=True,
        help="输出目录 (保存分割结果)"
    )
    parser.add_argument(
        "-m", "--model_dir",
        type=str, default="results/nnUNet_1",
        help="nnUNet 模型目录 (默认: results/nnUNet_1)"
    )
    parser.add_argument(
        "-d", "--dataset",
        type=str, default="Dataset500_MIMIC",
        help="数据集名称 (默认: Dataset500_MIMIC)"
    )
    parser.add_argument(
        "-f", "--fold",
        type=int, default=0,
        help="使用的 fold (默认: 0)"
    )
    parser.add_argument(
        "-c", "--checkpoint",
        type=str, default="checkpoint_best.pth",
        help="checkpoint 文件名 (默认: checkpoint_best.pth)"
    )
    parser.add_argument(
        "--device",
        type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="推理设备 (默认: cuda)"
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="生成叠加可视化 (原图 + 分割结果)"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="不生成彩色可视化"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细输出"
    )
    return parser


def prepare_input_files(input_dir, temp_dir):
    """准备输入文件，重命名为 nnUNet 格式 (xxx_0000.png)"""
    os.makedirs(temp_dir, exist_ok=True)
    
    file_mapping = {}  # 原始文件名 -> 临时文件名
    
    for f in os.listdir(input_dir):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            src_path = os.path.join(input_dir, f)
            basename = os.path.splitext(f)[0]
            dst_name = f"{basename}_0000.png"
            dst_path = os.path.join(temp_dir, dst_name)
            
            # 如果是 jpg，转换为 png
            if f.lower().endswith(('.jpg', '.jpeg')):
                img = Image.open(src_path).convert('RGB')
                img.save(dst_path)
            else:
                shutil.copy(src_path, dst_path)
            
            file_mapping[basename] = dst_name
    
    return file_mapping


def run_nnunet_predict(temp_input_dir, output_dir, model_dir, dataset, fold, checkpoint, device, verbose=False):
    """运行 nnUNet 推理"""
    import subprocess
    
    # 设置环境变量
    env = os.environ.copy()
    env["nnUNet_results"] = model_dir
    
    cmd = [
        "nnUNetv2_predict",
        "-d", dataset,
        "-i", temp_input_dir,
        "-o", output_dir,
        "-f", str(fold),
        "-tr", "nnUNetTrainer",
        "-c", "2d",
        "-p", "nnUNetPlans",
        "-device", device,
        "-chk", checkpoint,
    ]
    
    if verbose:
        cmd.append("--verbose")
        print(f"运行命令: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, env=env, capture_output=not verbose)
    
    if result.returncode != 0:
        print(f"nnUNet 推理失败!")
        if result.stderr:
            print(result.stderr.decode())
        sys.exit(1)


def create_color_mask(mask):
    """将分割掩码转换为彩色图像"""
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    
    for label, color in LEAD_COLORS.items():
        color_img[mask == label] = color
    
    return color_img


def create_overlay(original_img, mask, alpha=0.5):
    """创建原图与分割结果的叠加可视化"""
    # 确保原图是 RGB
    if len(original_img.shape) == 2:
        original_img = np.stack([original_img] * 3, axis=-1)
    elif original_img.shape[2] == 4:
        original_img = original_img[:, :, :3]
    
    color_mask = create_color_mask(mask)
    
    # 只在非背景区域叠加
    overlay = original_img.copy().astype(np.float32)
    fg_mask = mask > 0
    overlay[fg_mask] = (
        overlay[fg_mask] * (1 - alpha) + 
        color_mask[fg_mask].astype(np.float32) * alpha
    )
    
    return overlay.astype(np.uint8)


def create_legend():
    """创建导联颜色图例"""
    fig, ax = plt.subplots(figsize=(2, 4))
    
    for i, (label, name) in enumerate(LEAD_LABELS.items()):
        if label == 0:
            continue
        color = np.array(LEAD_COLORS[label]) / 255.0
        ax.add_patch(plt.Rectangle((0, 12 - i), 1, 0.8, color=color))
        ax.text(1.2, 12 - i + 0.4, name, va='center', fontsize=10)
    
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 13)
    ax.axis('off')
    ax.set_title('导联图例', fontsize=12)
    
    return fig


def process_outputs(output_dir, input_dir, file_mapping, overlay=False, no_color=False, verbose=False):
    """处理 nnUNet 输出，生成可视化"""
    results = []
    
    for basename, temp_name in tqdm(file_mapping.items(), desc="生成可视化"):
        mask_path = os.path.join(output_dir, f"{basename}.png")
        
        if not os.path.exists(mask_path):
            print(f"警告: 未找到 {basename} 的分割结果")
            continue
        
        # 读取分割掩码
        mask = np.array(Image.open(mask_path))
        unique_labels = np.unique(mask)
        detected_leads = [LEAD_LABELS[l] for l in unique_labels if l > 0]
        
        if verbose:
            print(f"{basename}: 检测到 {len(detected_leads)} 个导联 - {detected_leads}")
        
        results.append({
            "name": basename,
            "shape": mask.shape,
            "detected_leads": detected_leads,
        })
        
        # 生成彩色可视化
        if not no_color:
            color_mask = create_color_mask(mask)
            color_path = os.path.join(output_dir, f"{basename}_color.png")
            Image.fromarray(color_mask).save(color_path)
        
        # 生成叠加可视化
        if overlay:
            # 查找原始图像
            original_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(input_dir, f"{basename}{ext}")
                if os.path.exists(candidate):
                    original_path = candidate
                    break
            
            if original_path:
                original_img = np.array(Image.open(original_path).convert('RGB'))
                overlay_img = create_overlay(original_img, mask)
                overlay_path = os.path.join(output_dir, f"{basename}_overlay.png")
                Image.fromarray(overlay_img).save(overlay_path)
    
    return results


def print_summary(results, output_dir):
    """打印分割结果摘要"""
    print("\n" + "=" * 50)
    print("分割结果摘要")
    print("=" * 50)
    
    total_leads = 0
    for r in results:
        n_leads = len(r["detected_leads"])
        total_leads += n_leads
        print(f"  {r['name']}: {n_leads} 个导联")
    
    print("-" * 50)
    print(f"共处理 {len(results)} 张图像")
    print(f"平均每张检测 {total_leads / len(results):.1f} 个导联" if results else "")
    print(f"输出目录: {output_dir}")
    print("=" * 50)


def main():
    args = get_parser().parse_args()
    
    # 验证输入目录
    if not os.path.exists(args.input_dir):
        print(f"错误: 输入目录不存在: {args.input_dir}")
        sys.exit(1)
    
    # 验证模型目录
    model_path = os.path.join(
        args.model_dir, args.dataset, 
        "nnUNetTrainer__nnUNetPlans__2d", 
        f"fold_{args.fold}", args.checkpoint
    )
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 创建临时输入目录
    temp_input_dir = os.path.join(args.output_dir, "_temp_input")
    if os.path.exists(temp_input_dir):
        shutil.rmtree(temp_input_dir)
    
    print("=" * 50)
    print("ECG 图像分割推理")
    print("=" * 50)
    print(f"输入目录: {args.input_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型: {args.model_dir}/{args.dataset}")
    print(f"Fold: {args.fold}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"设备: {args.device}")
    print("=" * 50)
    
    # 1. 准备输入文件
    print("\n[1/3] 准备输入文件...")
    file_mapping = prepare_input_files(args.input_dir, temp_input_dir)
    print(f"  找到 {len(file_mapping)} 张图像")
    
    if not file_mapping:
        print("错误: 未找到任何图像文件")
        sys.exit(1)
    
    # 2. 运行 nnUNet 推理
    print("\n[2/3] 运行 nnUNet 推理...")
    run_nnunet_predict(
        temp_input_dir, args.output_dir,
        args.model_dir, args.dataset,
        args.fold, args.checkpoint,
        args.device, args.verbose
    )
    
    # 3. 生成可视化
    print("\n[3/3] 生成可视化...")
    results = process_outputs(
        args.output_dir, args.input_dir, file_mapping,
        overlay=args.overlay,
        no_color=args.no_color,
        verbose=args.verbose
    )
    
    # 清理临时文件
    shutil.rmtree(temp_input_dir, ignore_errors=True)
    
    # 保存图例
    legend_fig = create_legend()
    legend_fig.savefig(
        os.path.join(args.output_dir, "_legend.png"),
        bbox_inches='tight', dpi=150
    )
    plt.close(legend_fig)
    
    # 打印摘要
    print_summary(results, args.output_dir)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
