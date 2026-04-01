import os
import sys
import argparse
import random
import json
import warnings
import threading
import time
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

from helper_functions import find_records
from gen_ecg_image_from_data import run_single_file

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")


def _default_layout_catalog():
    return {
        "6x2_1R": {"config_file": "config_6x2.yaml", "num_columns": 2, "full_mode": "II"},
        "3x4_1R": {"config_file": "config_3x4.yaml", "num_columns": 4, "full_mode": "II"},
        "12x1": {"config_file": "config_12x1.yaml", "num_columns": 1, "full_mode": None},
        "6x2": {"config_file": "config_6x2.yaml", "num_columns": 2, "full_mode": None},
        "3x4_3R": {"config_file": "config_3x4.yaml", "num_columns": 4, "full_mode": "V1,II,V5"},
        "3x4": {"config_file": "config_3x4.yaml", "num_columns": 4, "full_mode": None},
    }


def _parse_layout_weights(layout_weights):
    items = []
    for part in layout_weights.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"layout_weights 格式错误: {part}")
        key, value = part.split(":", 1)
        key = key.strip()
        value = value.strip()
        items.append((key, float(value)))
    if not items:
        raise ValueError("layout_weights 不能为空")
    weights_sum = sum(w for _, w in items)
    if weights_sum <= 0:
        raise ValueError("layout_weights 权重和必须 > 0")
    items = [(k, w / weights_sum) for k, w in items]
    return items


def _allocate_counts(total, weights_items):
    raw = [(k, total * w) for k, w in weights_items]
    base = {k: int(v) for k, v in raw}
    used = sum(base.values())
    remain = total - used
    if remain > 0:
        frac = sorted(((k, v - int(v)) for k, v in raw), key=lambda x: x[1], reverse=True)
        for i in range(remain):
            base[frac[i % len(frac)][0]] += 1
    return base


def _assign_layouts(records, counts, seed):
    rng = random.Random(seed)
    layout_list = []
    for k, c in counts.items():
        layout_list.extend([k] * c)
    rng.shuffle(layout_list)
    recs = list(records)
    rng.shuffle(recs)
    return list(zip(recs, layout_list))


def _process_single_record(task):
    header_rel, recording_rel, base_args_dict, input_directory, output_directory, layout_key, layout_cfg = task
    args = argparse.Namespace(**base_args_dict)

    args.input_file = os.path.join(input_directory, recording_rel)
    args.header_file = os.path.join(input_directory, header_rel)
    args.start_index = -1

    folder_struct_list = header_rel.split("/")[:-1]
    args.output_directory = os.path.join(output_directory, "/".join(folder_struct_list))
    args.encoding = os.path.split(os.path.splitext(recording_rel)[0])[1]

    args.config_file = layout_cfg["config_file"]
    args.num_columns = layout_cfg["num_columns"]
    args.full_mode = layout_cfg["full_mode"]

    if hasattr(args, "full_mode") and args.full_mode == "None":
        args.full_mode = None

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory, exist_ok=True)

    try:
        num_images = run_single_file(args)
        base_name = os.path.splitext(os.path.basename(recording_rel))[0]
        generated_paths = []
        for i in range(num_images):
            image_path = os.path.join(args.output_directory, f"{base_name}-{i}.png")
            if os.path.exists(image_path):
                generated_paths.append(image_path)

        return {
            "recording": recording_rel,
            "header": header_rel,
            "layout": layout_key,
            "num_images": int(num_images),
            "images": generated_paths,
            "error": None,
        }
    except Exception as e:
        return {
            "recording": recording_rel,
            "header": header_rel,
            "layout": layout_key,
            "num_images": 0,
            "images": [],
            "error": f"{type(e).__name__}: {repr(e)}",
        }


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_directory", type=str, required=True)
    parser.add_argument("-o", "--output_directory", type=str, required=True)
    parser.add_argument("-se", "--seed", type=int, required=False, default=-1)
    parser.add_argument("--num_leads", type=str, default="twelve")
    parser.add_argument("--max_num_images", type=int, default=-1)
    parser.add_argument("--config_file", type=str, default="config.yaml")

    parser.add_argument("-r", "--resolution", type=int, required=False, default=200)
    parser.add_argument("--pad_inches", type=int, required=False, default=0)
    parser.add_argument("-ph", "--print_header", action="store_true", default=False)
    parser.add_argument("--num_columns", type=int, default=-1)
    parser.add_argument("--full_mode", type=str, default="II")
    parser.add_argument("--mask_unplotted_samples", action="store_true", default=False)
    parser.add_argument("--add_qr_code", action="store_true", default=False)

    parser.add_argument("-l", "--link", type=str, required=False, default="")
    parser.add_argument("-n", "--num_words", type=int, required=False, default=5)
    parser.add_argument("--x_offset", dest="x_offset", type=int, default=30)
    parser.add_argument("--y_offset", dest="y_offset", type=int, default=30)
    parser.add_argument("--hws", dest="handwriting_size_factor", type=float, default=0.2)

    parser.add_argument("-ca", "--crease_angle", type=int, default=90)
    parser.add_argument("-nv", "--num_creases_vertically", type=int, default=10)
    parser.add_argument("-nh", "--num_creases_horizontally", type=int, default=10)

    parser.add_argument("-rot", "--rotate", type=int, default=0)
    parser.add_argument("-noise", "--noise", type=int, default=50)
    parser.add_argument("-c", "--crop", type=float, default=0.01)
    parser.add_argument("-t", "--temperature", type=int, default=40000)

    parser.add_argument("--random_resolution", action="store_true", default=False)
    parser.add_argument("--random_padding", action="store_true", default=False)
    parser.add_argument("--random_grid_color", action="store_true", default=False)
    parser.add_argument("--standard_grid_color", type=int, default=5)
    parser.add_argument("--calibration_pulse", type=float, default=1)
    parser.add_argument("--random_grid_present", type=float, default=1)
    parser.add_argument("--random_print_header", type=float, default=0)
    parser.add_argument("--random_bw", type=float, default=0)
    parser.add_argument("--remove_lead_names", action="store_false", default=True)
    parser.add_argument("--lead_name_bbox", action="store_true", default=False)
    parser.add_argument("--store_config", type=int, nargs="?", const=1, default=0)

    parser.add_argument("--deterministic_offset", action="store_true", default=False)
    parser.add_argument("--deterministic_num_words", action="store_true", default=False)
    parser.add_argument("--deterministic_hw_size", action="store_true", default=False)

    parser.add_argument("--deterministic_angle", action="store_true", default=False)
    parser.add_argument("--deterministic_vertical", action="store_true", default=False)
    parser.add_argument("--deterministic_horizontal", action="store_true", default=False)

    parser.add_argument("--deterministic_rot", action="store_true", default=False)
    parser.add_argument("--deterministic_noise", action="store_true", default=False)
    parser.add_argument("--deterministic_crop", action="store_true", default=False)
    parser.add_argument("--deterministic_temp", action="store_true", default=False)

    parser.add_argument("--fully_random", action="store_true", default=False)
    parser.add_argument("--hw_text", action="store_true", default=False)
    parser.add_argument("--wrinkles", action="store_true", default=False)
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--lead_bbox", action="store_true", default=False)

    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--image_only", action="store_true", default=False)

    parser.add_argument(
        "--layout_weights",
        type=str,
        required=True,
        help="逗号分隔: layout:weight，例如 6x2_1R:0.3,3x4_1R:0.3,...",
    )
    parser.add_argument("--layout_manifest", type=str, default="")

    return parser


def run(args):
    random.seed(args.seed)

    if not os.path.isabs(args.input_directory):
        args.input_directory = os.path.normpath(os.path.join(os.getcwd(), args.input_directory))
    if not os.path.isabs(args.output_directory):
        output_dir = os.path.normpath(os.path.join(os.getcwd(), args.output_directory))
    else:
        output_dir = args.output_directory

    if not os.path.exists(args.input_directory) or not os.path.isdir(args.input_directory):
        raise Exception("输入目录 (-i) 不存在, 请检查!")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    repo_root = os.path.dirname(os.path.abspath(__file__))
    layout_catalog = _default_layout_catalog()
    weights_items = _parse_layout_weights(args.layout_weights)

    print(f"input_directory: {args.input_directory}", flush=True)
    print(f"output_directory: {output_dir}", flush=True)
    print(f"seed: {args.seed}", flush=True)
    print(f"num_workers: {args.num_workers}", flush=True)
    print(f"layout_weights: {args.layout_weights}", flush=True)

    for key, _ in weights_items:
        if key not in layout_catalog:
            raise ValueError(f"未知布局类型: {key}")

    for layout_key in layout_catalog:
        cfg_path = os.path.join(repo_root, layout_catalog[layout_key]["config_file"])
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"未找到布局配置文件: {cfg_path}")

    print("扫描输入目录中(.dat/.hea)记录中...", flush=True)
    headers, recordings = find_records(args.input_directory, output_dir)
    print(f"扫描完成，记录数: {len(headers)}", flush=True)
    if args.max_num_images != -1 and args.max_num_images < len(headers):
        headers = headers[: args.max_num_images]
        recordings = recordings[: args.max_num_images]

    total_files = len(headers)
    if total_files == 0:
        print("没有找到需要处理的文件。退出。")
        return

    counts = _allocate_counts(total_files, weights_items)
    assigned = _assign_layouts(list(zip(headers, recordings)), counts, args.seed)

    print(f"布局分配(按record数): {counts}", flush=True)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.layout_manifest:
        manifest_path = args.layout_manifest
        if not os.path.isabs(manifest_path):
            manifest_path = os.path.normpath(os.path.join(os.getcwd(), manifest_path))
    else:
        manifest_path = os.path.join(script_dir, "logs", f"layout_manifest_{timestamp}.json")

    manifest_dir = os.path.dirname(manifest_path)
    if manifest_dir and not os.path.exists(manifest_dir):
        os.makedirs(manifest_dir, exist_ok=True)

    log_dir = manifest_dir if manifest_dir else os.path.join(script_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    output_list_file = os.path.join(log_dir, f"generated_images_{timestamp}.txt")
    error_log_file = os.path.join(log_dir, f"error_log_{timestamp}.txt")

    print(f"generated_images_list: {output_list_file}", flush=True)
    print(f"error_log_file: {error_log_file}", flush=True)
    print(f"layout_manifest: {manifest_path}", flush=True)

    meta = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_directory": args.input_directory,
        "output_directory": output_dir,
        "seed": args.seed,
        "total_records": total_files,
        "layout_weights": {k: w for k, w in weights_items},
        "layout_counts": counts,
    }

    with open(output_list_file, "w", encoding="utf-8") as f:
        f.write("# ECG图像生成路径记录\n")
        f.write(f"# 开始时间: {meta['created_at']}\n")
        f.write(f"# 输入基目录: {meta['input_directory']}\n")
        f.write(f"# 输出基目录: {meta['output_directory']}\n")
        f.write(f"# 总文件数: {meta['total_records']}\n")
        f.write(f"#{'='*60}\n")

    with open(error_log_file, "w", encoding="utf-8") as f:
        f.write("# ECG图像生成错误记录\n")
        f.write(f"# 开始时间: {meta['created_at']}\n")
        f.write(f"#{'='*60}\n")

    file_lock = threading.Lock()
    layout_by_record = {}
    errors_by_record = {}
    error_count = 0
    total_images = 0
    log_progress = not sys.stdout.isatty()
    progress_interval_s = float(os.environ.get("PROGRESS_LOG_INTERVAL", "2.0"))
    last_progress_ts = 0.0
    def _maybe_log_progress(done_files, total, images, errors):
        nonlocal last_progress_ts
        if not log_progress:
            return
        now = time.time()
        if now - last_progress_ts >= progress_interval_s:
            last_progress_ts = now
            print(f"progress: {done_files}/{total} ({(done_files/total)*100:.1f}%), images={images}, errors={errors}", flush=True)

    base_args_dict = vars(args).copy()

    def make_tasks():
        for ((header_rel, recording_rel), layout_key) in assigned:
            yield (
                header_rel,
                recording_rel,
                base_args_dict,
                args.input_directory,
                output_dir,
                layout_key,
                layout_catalog[layout_key],
            )

    if args.num_workers > 1:
        print(f"启动并行处理: workers={args.num_workers}", flush=True)
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            max_queue_size = args.num_workers * 4
            futures = {}
            task_iter = iter(make_tasks())

            with tqdm(
                total=total_files,
                desc="生成ECG图像(混合布局)",
                unit="个文件",
                position=0,
                leave=True,
                disable=False,
                dynamic_ncols=True,
            ) as pbar:
                for _ in range(min(max_queue_size, total_files)):
                    try:
                        task = next(task_iter)
                    except StopIteration:
                        break
                    future = executor.submit(_process_single_record, task)
                    futures[future] = task
                print(f"初始任务已提交: {len(futures)}", flush=True)

                while futures:
                    for done in as_completed(futures):
                        result = done.result()
                        rec = result["recording"]
                        layout_by_record[rec] = result["layout"]
                        if result["error"]:
                            errors_by_record[rec] = result["error"]
                            error_count += 1
                            tqdm.write(f"错误: {rec} - {result['error']}")
                            with file_lock:
                                with open(error_log_file, "a", encoding="utf-8") as f:
                                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {rec} | {result['error']}\n")
                        total_images += int(result["num_images"])

                        if result["images"]:
                            with file_lock:
                                with open(output_list_file, "a", encoding="utf-8") as f:
                                    for path in result["images"]:
                                        f.write(path + "\n")

                        pbar.update(1)
                        pbar.set_postfix({"已生成": total_images, "错误": error_count}, refresh=True)
                        _maybe_log_progress(pbar.n, total_files, total_images, error_count)

                        del futures[done]
                        try:
                            task = next(task_iter)
                            future = executor.submit(_process_single_record, task)
                            futures[future] = task
                        except StopIteration:
                            pass
                        break
    else:
        print("使用单进程处理", flush=True)
        with tqdm(
            total=total_files,
            desc="生成ECG图像(混合布局)",
            unit="个文件",
            disable=False,
            dynamic_ncols=True,
        ) as pbar:
            for task in make_tasks():
                result = _process_single_record(task)
                rec = result["recording"]
                layout_by_record[rec] = result["layout"]
                if result["error"]:
                    errors_by_record[rec] = result["error"]
                    error_count += 1
                    tqdm.write(f"错误: {rec} - {result['error']}")
                    with open(error_log_file, "a", encoding="utf-8") as f:
                        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | {rec} | {result['error']}\n")
                total_images += int(result["num_images"])

                if result["images"]:
                    with open(output_list_file, "a", encoding="utf-8") as f:
                        for path in result["images"]:
                            f.write(path + "\n")

                pbar.update(1)
                pbar.set_postfix({"已生成": total_images, "错误": error_count}, refresh=True)
                _maybe_log_progress(pbar.n, total_files, total_images, error_count)

    meta["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta["total_images"] = total_images
    meta["error_records"] = error_count

    with open(output_list_file, "a", encoding="utf-8") as f:
        f.write(f"\n#{'='*60}\n")
        f.write(f"# 完成时间: {meta['finished_at']}\n")
        f.write(f"# 处理文件数: {meta['total_records']}\n")
        f.write(f"# 生成图像数: {meta['total_images']}\n")
        if error_count:
            f.write(f"# 错误数量: {error_count}\n")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {"meta": meta, "layout_by_record": layout_by_record, "errors_by_record": errors_by_record},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"✓ 混合布局生成完成: records={total_files}, images={total_images}, errors={error_count}")
    print(f"✓ 图像路径记录: {output_list_file}")
    print(f"✓ 错误日志记录: {error_log_file}")
    print(f"✓ 布局清单JSON: {manifest_path}")


if __name__ == "__main__":
    path = os.path.join(os.getcwd(), sys.argv[0])
    parentPath = os.path.dirname(path)
    os.chdir(parentPath)
    run(get_parser().parse_args(sys.argv[1:]))
