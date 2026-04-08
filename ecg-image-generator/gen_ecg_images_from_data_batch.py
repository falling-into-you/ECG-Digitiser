import os, sys, argparse
import random
import csv
from helper_functions import find_records, create_output_dirs_for_files
from gen_ecg_image_from_data import run_single_file
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
import threading
from datetime import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

def process_single_record(file_info):
    """处理单个文件的包装函数，用于并行处理"""
    full_header_file, full_recording_file, args_dict, input_directory, original_output_dir = file_info
    
    # 从字典重建 args 对象
    args = argparse.Namespace(**args_dict)
    
    filename = full_recording_file
    header = full_header_file
    args.input_file = os.path.join(input_directory, filename)
    args.header_file = os.path.join(input_directory, header)
    args.start_index = -1
    
    # 根据 flat_output 决定输出目录
    if args_dict.get('flat_output', False):
        # 扁平化输出：直接放在输出目录下
        args.output_directory = original_output_dir
    else:
        # 保留原始目录结构
        folder_struct_list = full_header_file.split('/')[:-1]
        args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
    
    args.encoding = os.path.split(os.path.splitext(filename)[0])[1]
    
    # 确保输出目录存在
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory, exist_ok=True)
    
    try:
        num_images = run_single_file(args)
        
        # 收集生成的图像路径
        # 根据输出规则，图像文件名格式为: basename-0.png, basename-1.png, ...
        base_name = os.path.splitext(os.path.basename(full_recording_file))[0]
        generated_paths = []
        for i in range(num_images):
            image_path = os.path.join(args.output_directory, f"{base_name}-{i}.png")
            if os.path.exists(image_path):
                generated_paths.append(image_path)
        
        return (num_images, generated_paths)
    except Exception as e:
        # 返回错误信息而不是直接打印，避免干扰进度条
        return (0, [], f"错误: {filename} - {str(e)}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, required=True)
    parser.add_argument('-o', '--output_directory', type=str, required=True)
    parser.add_argument('-se', '--seed', type=int, required=False, default = -1)
    parser.add_argument('--num_leads',type=str,default='twelve')
    parser.add_argument('--max_num_images',type=int,default = -1)
    parser.add_argument('--config_file', type=str, default='config.yaml')
    
    parser.add_argument('-r','--resolution',type=int,required=False,default = 200)
    parser.add_argument('--pad_inches',type=int,required=False,default=0)
    parser.add_argument('-ph','--print_header', action="store_true",default=False)
    parser.add_argument('--num_columns',type=int,default = -1)
    parser.add_argument('--full_mode', type=str,default='II')
    parser.add_argument('--mask_unplotted_samples', action="store_true", default=False)
    parser.add_argument('--add_qr_code', action="store_true", default=False)

    parser.add_argument('-l', '--link', type=str, required=False,default='')
    parser.add_argument('-n','--num_words',type=int,required=False,default=5)
    parser.add_argument('--x_offset',dest='x_offset',type=int,default = 30)
    parser.add_argument('--y_offset',dest='y_offset',type=int,default = 30)
    parser.add_argument('--hws',dest='handwriting_size_factor',type=float,default = 0.2)
    
    parser.add_argument('-ca','--crease_angle',type=int,default=90)
    parser.add_argument('-nv','--num_creases_vertically',type=int,default=10)
    parser.add_argument('-nh','--num_creases_horizontally',type=int,default=10)

    parser.add_argument('-rot','--rotate',type=int,default=0)
    parser.add_argument('-noise','--noise',type=int,default=50)
    parser.add_argument('-c','--crop',type=float,default=0.01)
    parser.add_argument('-t','--temperature',type=int,default=40000)

    parser.add_argument('--random_resolution',action="store_true",default=False)
    parser.add_argument('--random_padding',action="store_true",default=False)
    parser.add_argument('--random_grid_color',action="store_true",default=False)
    parser.add_argument('--standard_grid_color', type=int, default=5)
    parser.add_argument('--calibration_pulse',type=float,default=1)
    parser.add_argument('--random_grid_present',type=float,default=1)
    parser.add_argument('--random_print_header',type=float,default=0)
    parser.add_argument('--random_bw',type=float,default=0)
    parser.add_argument('--remove_lead_names',action="store_false",default=True)
    parser.add_argument('--lead_name_bbox',action="store_true",default=False)
    parser.add_argument('--store_config', type=int, nargs='?', const=1, default=0)

    parser.add_argument('--deterministic_offset',action="store_true",default=False)
    parser.add_argument('--deterministic_num_words',action="store_true",default=False)
    parser.add_argument('--deterministic_hw_size',action="store_true",default=False)

    parser.add_argument('--deterministic_angle',action="store_true",default=False)
    parser.add_argument('--deterministic_vertical',action="store_true",default=False)
    parser.add_argument('--deterministic_horizontal',action="store_true",default=False)

    parser.add_argument('--deterministic_rot',action="store_true",default=False)
    parser.add_argument('--deterministic_noise',action="store_true",default=False)
    parser.add_argument('--deterministic_crop',action="store_true",default=False)
    parser.add_argument('--deterministic_temp',action="store_true",default=False)

    parser.add_argument('--fully_random',action='store_true',default=False)
    parser.add_argument('--hw_text',action='store_true',default=False)
    parser.add_argument('--wrinkles',action='store_true',default=False)
    parser.add_argument('--augment',action='store_true',default=False)
    parser.add_argument('--lead_bbox',action='store_true',default=False)
    
    parser.add_argument('--num_workers',type=int,default=1, help='并行处理的进程数，默认为1（不并行）')
    parser.add_argument('--image_only',action='store_true',default=False, help='只生成PNG图像，不生成原始数据文件(.dat和.hea)')
    parser.add_argument('--random_sample', action='store_true', default=False,
                        help='当使用max_num_images限制数量时，随机抽样而不是取前N个')
    parser.add_argument('--flat_output', action='store_true', default=False,
                        help='扁平化输出，所有文件直接放在输出目录下，不保留原始目录结构')

    return parser

def run(args):
        random.seed(args.seed)
        print('args.input_directory: ', args.input_directory)
        if os.path.isabs(args.input_directory) == False:
            args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
        if os.path.isabs(args.output_directory) == False:
            original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
        else:
            original_output_dir = args.output_directory
        
        if os.path.exists(args.input_directory) == False or os.path.isdir(args.input_directory) == False:
            raise Exception("The input directory does not exist, Please re-check the input arguments!")

        if os.path.exists(original_output_dir) == False:
            os.makedirs(original_output_dir)

        # 扫描文件但不预创建目录（延迟到抽样后）
        full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir, create_dirs=False)
        
        # 限制处理的文件数量
        if args.max_num_images != -1 and args.max_num_images < len(full_header_files):
            if args.random_sample:
                # 随机抽样
                indices = list(range(len(full_header_files)))
                random.shuffle(indices)
                selected_indices = sorted(indices[:args.max_num_images])  # 排序保持目录结构一致性
                full_header_files = [full_header_files[i] for i in selected_indices]
                full_recording_files = [full_recording_files[i] for i in selected_indices]
                print(f'随机抽样 {args.max_num_images} 个文件（从 {len(indices)} 个文件中）')
            else:
                # 取前N个
                full_header_files = full_header_files[:args.max_num_images]
                full_recording_files = full_recording_files[:args.max_num_images]
        
        # 抽样完成后，只为选中的文件创建目录结构（除非使用扁平化输出）
        if not args.flat_output:
            create_output_dirs_for_files(full_recording_files, original_output_dir)
        
        total_files = len(full_header_files)
        print(f'开始处理 {total_files} 个文件...')
        print(f'使用 {args.num_workers} 个并行进程')
        
        # 将 args 转换为字典，方便传递给子进程
        args_dict = vars(args).copy()
        
        i = 0
        
        # 准备输出路径记录文件（实时写入）
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_list_file = os.path.join(log_dir, f"generated_images_{timestamp}.txt")
        
        # 创建文件并写入头部信息
        with open(output_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# ECG图像生成路径记录\n")
            f.write(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 输入目录: {args.input_directory}\n")
            f.write(f"# 输出目录: {original_output_dir}\n")
            f.write(f"# 总文件数: {total_files}\n")
            f.write(f"#{'='*60}\n")
        
        # 使用锁保护文件写入
        file_lock = threading.Lock()
        log_progress = not sys.stdout.isatty()
        progress_interval_s = float(os.environ.get("PROGRESS_LOG_INTERVAL", "10"))
        last_progress_ts = time.time()
        def _maybe_log_progress(done_files, total, images, errors):
            nonlocal last_progress_ts
            if not log_progress:
                return
            now = time.time()
            if now - last_progress_ts >= progress_interval_s:
                last_progress_ts = now
                print(f"progress: {done_files}/{total}, images={images}, errors={errors}", flush=True)
        
        if args.num_workers > 1:
            # 使用多进程并行处理（流式提交，避免内存爆炸）
            print('正在启动并行处理...')
            print(f'图像路径将记录到: {output_list_file}')
            
            # 使用生成器创建任务（不预先创建所有任务）
            def task_generator():
                for header, recording in zip(full_header_files, full_recording_files):
                    yield (header, recording, args_dict, args.input_directory, original_output_dir)
            
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                # 使用滑动窗口方式提交任务，保持有限的内存占用
                # 同时保持足够的任务队列让所有进程都忙碌
                max_queue_size = args.num_workers * 4  # 队列大小为进程数的4倍
                futures = {}
                task_iter = task_generator()
                error_count = 0
                
                # 使用 tqdm 显示进度
                with tqdm(total=total_files, 
                         desc='生成ECG图像', 
                         unit='个文件',
                         position=0,
                         leave=True,
                         disable=False,
                         dynamic_ncols=True) as pbar:
                    
                    # 初始填充任务队列
                    for _ in range(min(max_queue_size, total_files)):
                        try:
                            task = next(task_iter)
                            future = executor.submit(process_single_record, task)
                            futures[future] = task
                        except StopIteration:
                            break
                    
                    # 处理任务：每完成一个就提交一个新的
                    while futures:
                        # 等待至少一个任务完成
                        for done in as_completed(futures):
                            try:
                                result = done.result()
                                # 检查返回值格式
                                if isinstance(result, tuple):
                                    if len(result) == 3:
                                        # 错误情况: (count, paths, error_msg)
                                        count, paths, error_msg = result
                                        i += count
                                        error_count += 1
                                        tqdm.write(error_msg)
                                        # 实时写入路径
                                        if paths:
                                            with file_lock:
                                                with open(output_list_file, 'a', encoding='utf-8') as f:
                                                    for path in paths:
                                                        f.write(path + '\n')
                                    elif len(result) == 2:
                                        # 成功情况: (count, paths)
                                        count, paths = result
                                        i += count
                                        # 实时写入路径
                                        if paths:
                                            with file_lock:
                                                with open(output_list_file, 'a', encoding='utf-8') as f:
                                                    for path in paths:
                                                        f.write(path + '\n')
                                else:
                                    # 向后兼容：只返回数量
                                    i += result
                                
                                pbar.update(1)
                                # 实时显示统计信息
                                if error_count > 0:
                                    pbar.set_postfix({'已生成': i, '错误': error_count}, refresh=True)
                                else:
                                    pbar.set_postfix({'已生成': i}, refresh=True)
                                _maybe_log_progress(pbar.n, total_files, i, error_count)
                            except Exception as e:
                                error_count += 1
                                tqdm.write(f"处理任务时出错: {str(e)}")
                                pbar.update(1)
                                _maybe_log_progress(pbar.n, total_files, i, error_count)
                            finally:
                                # 移除已完成的任务
                                del futures[done]
                                
                                # 提交新任务以保持队列满载
                                try:
                                    task = next(task_iter)
                                    future = executor.submit(process_single_record, task)
                                    futures[future] = task
                                except StopIteration:
                                    pass  # 没有更多任务了
                            
                            # 只处理一个完成的任务后就跳出，继续循环
                            break
                
                # 写入完成信息
                with open(output_list_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n#{'='*60}\n")
                    f.write(f"# 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# 处理文件数: {total_files}\n")
                    f.write(f"# 生成图像数: {i}\n")
                    if error_count > 0:
                        f.write(f"# 错误数量: {error_count}\n")
                    f.write(f"# 成功率: {(total_files-error_count)/total_files*100:.2f}%\n")
                
                print(f'\n✓ 并行处理完成！')
                print(f'  - 处理文件数: {total_files}')
                print(f'  - 生成图像数: {i}')
                if error_count > 0:
                    print(f'  - 错误数量: {error_count}')
                print(f'  - 成功率: {(total_files-error_count)/total_files*100:.2f}%')
                print(f'✓ 图像路径已实时记录到: {output_list_file}')
        else:
            # 单进程处理（原有逻辑）
            print(f'图像路径将记录到: {output_list_file}')
            
            for done_files, (full_header_file, full_recording_file) in enumerate(
                tqdm(
                    zip(full_header_files, full_recording_files),
                    total=total_files,
                    desc='生成ECG图像',
                    unit='个文件',
                ),
                start=1,
            ):
                filename = full_recording_file
                header = full_header_file
                args.input_file = os.path.join(args.input_directory, filename)
                args.header_file = os.path.join(args.input_directory, header)
                args.start_index = -1
                
                # 根据 flat_output 决定输出目录
                if args.flat_output:
                    args.output_directory = original_output_dir
                else:
                    folder_struct_list = full_header_file.split('/')[:-1]
                    args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
                args.encoding = os.path.split(os.path.splitext(filename)[0])[1]
                
                num_images = run_single_file(args)
                i += num_images
                
                # 实时写入生成的图像路径
                base_name = os.path.splitext(os.path.basename(full_recording_file))[0]
                with open(output_list_file, 'a', encoding='utf-8') as f:
                    for j in range(num_images):
                        image_path = os.path.join(args.output_directory, f"{base_name}-{j}.png")
                        if os.path.exists(image_path):
                            f.write(image_path + '\n')
                _maybe_log_progress(done_files, total_files, i, 0)
            
            # 写入完成信息
            with open(output_list_file, 'a', encoding='utf-8') as f:
                f.write(f"\n#{'='*60}\n")
                f.write(f"# 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 处理文件数: {total_files}\n")
                f.write(f"# 生成图像数: {i}\n")
            
            print(f'\n✓ 处理完成！共生成 {i} 张图像')
            print(f'✓ 图像路径已实时记录到: {output_list_file}')

if __name__=='__main__':
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run(get_parser().parse_args(sys.argv[1:]))
