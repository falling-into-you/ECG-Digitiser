import os, sys, argparse
import random
import csv
from helper_functions import find_records 
from gen_ecg_image_from_data import run_single_file
import warnings
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import multiprocessing
import threading
from datetime import datetime
import json 
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
warnings.filterwarnings("ignore")

def process_single_record(file_info):
    """(多进程) 处理单个文件的包装函数，包含详细错误日志"""
    
    try:
        full_header_file, full_recording_file, args_dict, input_directory, original_output_dir, target_image_paths = file_info
    except ValueError:
        full_header_file, full_recording_file, args_dict, input_directory, original_output_dir = file_info
        target_image_paths = []

    # 从字典重建 args 对象
    args = argparse.Namespace(**args_dict)
    
    filename = full_recording_file
    header = full_header_file
    
    args.input_file = os.path.join(input_directory, filename)
    args.header_file = os.path.join(input_directory, header)
    args.start_index = -1
    
    if target_image_paths and target_image_paths[0]:
        target_rel_dir = os.path.dirname(target_image_paths[0])
        args.output_directory = os.path.join(original_output_dir, target_rel_dir)
    else:
        folder_struct_list = full_header_file.split('/')[:-1]
        args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
    
    args.encoding = os.path.split(os.path.splitext(filename)[0])[1]
    
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory, exist_ok=True)
    
    try:
        if not os.path.exists(args.input_file):
            raise FileNotFoundError(f"输入文件未找到: {args.input_file}")
        if not os.path.exists(args.header_file):
            raise FileNotFoundError(f"头文件未找到: {args.header_file}")

        # --- 新增修复: 转换 'None' 字符串为 None 对象 ---
        if hasattr(args, 'full_mode') and args.full_mode == 'None':
            args.full_mode = None
        # --- 修复结束 ---

        num_images = run_single_file(args)
        
        if num_images is None:
            raise ValueError("run_single_file 返回了 None，表示内部处理失败。")
        if not isinstance(num_images, int):
             raise TypeError(f"run_single_file 返回了 {type(num_images)} 类型，而不是 int。")
        
        base_name = args.encoding
        generated_paths = []
        
        for i in range(num_images):
            image_path = os.path.join(args.output_directory, f"{base_name}-{i}.png")
            if os.path.exists(image_path):
                generated_paths.append(image_path)
        
        return (num_images, generated_paths)
    except Exception as e:
        input_path_display = args.input_file if 'input_file' in args else os.path.join(input_directory, filename)
        error_type = type(e).__name__
        error_repr = repr(e) 
        
        return (0, [], f"错误: {input_path_display} - Type: {error_type}, Repr: {error_repr}")

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_directory', type=str, required=True, help="存放ECG原始数据(.dat, .hea)的基目录")
    parser.add_argument('-o', '--output_directory', type=str, required=True, help="存放生成图像的基目录")
    parser.add_argument('--jsonl_file', type=str, default=None, help='指向包含ECG路径和目标图像路径的jsonl文件') 
    parser.add_argument('-se', '--seed', type=int, required=False, default = -1)
    parser.add_argument('--num_leads',type=str,default='twelve')
    parser.add_argument('--max_num_images',type=int,default = -1, help="处理的最大ECG文件数")
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
    parser.add_argument('--image_only',action='store_true',default=False, help='(此参数可能在 find_records 中用于跳过已生成的文件)')

    return parser

def run(args):
        random.seed(args.seed)
        
        if not os.path.isabs(args.input_directory):
            args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
        print(f'输入基目录: {args.input_directory}')
            
        if not os.path.isabs(args.output_directory):
            original_output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
        else:
            original_output_dir = args.output_directory
        print(f'输出基目录: {original_output_dir}')

        if not os.path.exists(args.input_directory) or not os.path.isdir(args.input_directory):
            raise Exception("输入目录 (-i) 不存在, 请检查!")

        if not os.path.exists(original_output_dir):
            os.makedirs(original_output_dir)

        full_header_files = []
        full_recording_files = []
        target_output_paths_list = [] 

        if args.jsonl_file:
            print(f'正在从 {args.jsonl_file} 加载文件列表...')
            if not os.path.exists(args.jsonl_file):
                raise Exception(f"JSONL 文件 {args.jsonl_file} 不存在!")
                
            with open(args.jsonl_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        ecg_base_path = item['ecg'] 
                        images_list = item['images']
                        
                        full_header_files.append(ecg_base_path + '.hea')
                        full_recording_files.append(ecg_base_path + '.dat') 
                        target_output_paths_list.append(images_list)
                    except json.JSONDecodeError:
                        print(f"警告: 跳过无效的JSON行: {line.strip()}")
                    except KeyError:
                        print(f"警告: 跳过缺少 'ecg' 或 'images' 字段的行: {line.strip()}")
            print(f'从 JSONL 文件中加载了 {len(full_header_files)} 个记录。')
        
        else:
            print(f'未提供 --jsonl_file，正在扫描输入目录: {args.input_directory} ...')
            full_header_files, full_recording_files = find_records(args.input_directory, original_output_dir)
            target_output_paths_list = [[] for _ in full_header_files]
            print(f'扫描发现 {len(full_header_files)} 个记录。')

        
        if args.max_num_images != -1 and args.max_num_images < len(full_header_files):
            print(f'限制处理的文件数量为: {args.max_num_images}')
            full_header_files = full_header_files[:args.max_num_images]
            full_recording_files = full_recording_files[:args.max_num_images]
            target_output_paths_list = target_output_paths_list[:args.max_num_images]
        
        total_files = len(full_header_files)
        if total_files == 0:
            print('没有找到需要处理的文件。退出。')
            return

        print(f'开始处理 {total_files} 个文件...')
        print(f'使用 {args.num_workers} 个并行进程')
        
        args_dict = vars(args).copy()
        
        i = 0
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_list_file = os.path.join(log_dir, f"generated_images_{timestamp}.txt")
        
        with open(output_list_file, 'w', encoding='utf-8') as f:
            f.write(f"# ECG图像生成路径记录\n")
            f.write(f"# 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# 输入基目录: {args.input_directory}\n")
            f.write(f"# 输出基目录: {original_output_dir}\n")
            if args.jsonl_file:
                f.write(f"# JSONL文件: {args.jsonl_file}\n")
            f.write(f"# 总文件数: {total_files}\n")
            f.write(f"#{'='*60}\n")
        
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
            # --- 多进程逻辑 ---
            print('正在启动并行处理...')
            print(f'图像路径将记录到: {output_list_file}')
            
            def task_generator():
                for header, recording, targets in zip(full_header_files, full_recording_files, target_output_paths_list):
                    yield (header, recording, args_dict, args.input_directory, original_output_dir, targets)
            
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                max_queue_size = args.num_workers * 4
                futures = {}
                task_iter = task_generator()
                error_count = 0
                
                with tqdm(total=total_files, 
                         desc='生成ECG图像', 
                         unit='个文件',
                         position=0,
                         leave=True,
                         disable=False,
                         dynamic_ncols=True) as pbar:
                    
                    for _ in range(min(max_queue_size, total_files)):
                        try:
                            task = next(task_iter)
                            future = executor.submit(process_single_record, task)
                            futures[future] = task
                        except StopIteration:
                            break
                    
                    while futures:
                        for done in as_completed(futures):
                            try:
                                result = done.result()
                                if isinstance(result, tuple):
                                    if len(result) == 3:
                                        count, paths, error_msg = result
                                        i += count
                                        error_count += 1
                                        tqdm.write(error_msg) 
                                        if paths:
                                            with file_lock:
                                                with open(output_list_file, 'a', encoding='utf-8') as f:
                                                    for path in paths:
                                                        f.write(path + '\n')
                                    elif len(result) == 2:
                                        count, paths = result
                                        i += count
                                        if paths:
                                            with file_lock:
                                                with open(output_list_file, 'a', encoding='utf-8') as f:
                                                    for path in paths:
                                                        f.write(path + '\n')
                                else:
                                    i += result
                                
                                pbar.update(1)
                                if error_count > 0:
                                    pbar.set_postfix({'已生成': i, '错误': error_count}, refresh=True)
                                else:
                                    pbar.set_postfix({'已生成': i}, refresh=True)
                                _maybe_log_progress(pbar.n, total_files, i, error_count)
                            except Exception as e:
                                error_count += 1
                                task_info = futures[done] 
                                tqdm.write(f"处理任务 {task_info[0]} 时出错: {str(e)}")
                                pbar.update(1)
                                _maybe_log_progress(pbar.n, total_files, i, error_count)
                            finally:
                                del futures[done]
                                try:
                                    task = next(task_iter)
                                    future = executor.submit(process_single_record, task)
                                    futures[future] = task
                                except StopIteration:
                                    pass
                            
                            break
                
                with open(output_list_file, 'a', encoding='utf-8') as f:
                    f.write(f"\n#{'='*60}\n")
                    f.write(f"# 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"# 处理文件数: {total_files}\n")
                    f.write(f"# 生成图像数: {i}\n")
                    if error_count > 0:
                        f.write(f"# 错误数量: {error_count}\n")
                    if total_files > 0:
                        f.write(f"# 成功率: {(total_files-error_count)/total_files*100:.2f}%\n")
                
                print(f'\n✓ 并行处理完成！')
                print(f'  - 处理文件数: {total_files}')
                print(f'  - 生成图像数: {i}')
                if error_count > 0:
                    print(f'  - 错误数量: {error_count}')
                if total_files > 0:
                    print(f'  - 成功率: {(total_files-error_count)/total_files*100:.2f}%')
                print(f'✓ 图像路径已实时记录到: {output_list_file}')
        
        else:
            # --- 单进程处理 ---
            print(f'图像路径将记录到: {output_list_file}')
            error_count = 0
            
            for done_files, (full_header_file, full_recording_file, target_image_paths) in enumerate(
                tqdm(
                    zip(full_header_files, full_recording_files, target_output_paths_list),
                    total=total_files,
                    desc='生成ECG图像',
                    unit='个文件',
                ),
                start=1,
            ):
                
                current_input_file = os.path.join(args.input_directory, full_recording_file)
                current_header_file = os.path.join(args.input_directory, full_header_file)
                
                try:
                    args.input_file = current_input_file
                    args.header_file = current_header_file
                    args.start_index = -1
                    
                    if target_image_paths and target_image_paths[0]:
                        target_rel_dir = os.path.dirname(target_image_paths[0])
                        args.output_directory = os.path.join(original_output_dir, target_rel_dir)
                    else:
                        folder_struct_list = full_header_file.split('/')[:-1]
                        args.output_directory = os.path.join(original_output_dir, '/'.join(folder_struct_list))
                    
                    args.encoding = os.path.split(os.path.splitext(full_recording_file)[0])[1]
                    
                    if not os.path.exists(args.output_directory):
                        os.makedirs(args.output_directory, exist_ok=True)
                    
                    if not os.path.exists(args.input_file):
                        raise FileNotFoundError(f"输入文件未找到: {args.input_file}")
                    if not os.path.exists(args.header_file):
                        raise FileNotFoundError(f"头文件未找到: {args.header_file}")

                    # --- 新增修复: 转换 'None' 字符串为 None 对象 ---
                    if hasattr(args, 'full_mode') and args.full_mode == 'None':
                        args.full_mode = None
                    # --- 修复结束 ---

                    num_images = run_single_file(args)

                    if num_images is None:
                        raise ValueError("run_single_file 返回了 None")
                    if not isinstance(num_images, int):
                         raise TypeError(f"run_single_file 返回了 {type(num_images)} 类型")

                    i += num_images
                    
                    base_name = args.encoding
                    with open(output_list_file, 'a', encoding='utf-8') as f:
                        for j in range(num_images):
                            image_path = os.path.join(args.output_directory, f"{base_name}-{j}.png")
                            if os.path.exists(image_path):
                                f.write(image_path + '\n')
                    _maybe_log_progress(done_files, total_files, i, error_count)
                
                except Exception as e:
                    error_count += 1
                    error_type = type(e).__name__
                    error_repr = repr(e)
                    tqdm.write(f"错误: {current_input_file} - Type: {error_type}, Repr: {error_repr}")
                    _maybe_log_progress(done_files, total_files, i, error_count)
            
            with open(output_list_file, 'a', encoding='utf-8') as f:
                f.write(f"\n#{'='*60}\n")
                f.write(f"# 完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 处理文件数: {total_files}\n")
                f.write(f"# 生成图像数: {i}\n")
                if error_count > 0:
                    f.write(f"# 错误数量: {error_count}\n")
                if total_files > 0:
                    f.write(f"# 成功率: {(total_files-error_count)/total_files*100:.2f}%\n")
            
            print(f'\n✓ 处理完成！')
            print(f'  - 处理文件数: {total_files}')
            print(f'  - 生成图像数: {i}')
            if error_count > 0:
                print(f'  - 错误数量: {error_count}')
            if total_files > 0:
                print(f'  - 成功率: {(total_files-error_count)/total_files*100:.2f}%')
            print(f'✓ 图像路径已实时记录到: {output_list_file}')

if __name__=='__main__':
    # 设置多进程启动方法 (必须在 run() 之前)
    if os.name != 'nt': # Windows 默认是 'spawn'
        multiprocessing.set_start_method('spawn', force=True) 
    
    run(get_parser().parse_args(sys.argv[1:]))
