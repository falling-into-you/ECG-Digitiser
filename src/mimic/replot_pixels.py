import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def resample_single_file(file_path, resample_factor):
    """处理单个 JSON 文件，返回 None 或 (error, file_path)"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        leads = data["leads"]
        for i in range(len(leads)):
            pixels = np.array(leads[i]["plotted_pixels"])
            new_pixels = np.zeros(((len(pixels) - 1) * resample_factor, 2))
            for j in range(len(pixels) - 1):
                new_pixels[
                    j * resample_factor : (j + 1) * resample_factor, 0
                ] = np.linspace(
                    pixels[j, 0], pixels[j + 1, 0], resample_factor
                )
                new_pixels[
                    j * resample_factor : (j + 1) * resample_factor, 1
                ] = np.linspace(
                    pixels[j, 1], pixels[j + 1, 1], resample_factor
                )
            data["leads"][i]["dense_plotted_pixels"] = new_pixels.tolist()

        if "leads_augmented" in data.keys():
            leads = data["leads_augmented"]
            for i in range(len(leads)):
                pixels = np.array(leads[i]["plotted_pixels"])
                new_pixels = np.zeros(
                    ((len(pixels) - 1) * resample_factor, 2)
                )
                for j in range(len(pixels) - 1):
                    new_pixels[
                        j * resample_factor : (j + 1) * resample_factor, 0
                    ] = np.linspace(
                        pixels[j, 0], pixels[j + 1, 0], resample_factor
                    )
                    new_pixels[
                        j * resample_factor : (j + 1) * resample_factor, 1
                    ] = np.linspace(
                        pixels[j, 1], pixels[j + 1, 1], resample_factor
                    )
                data["leads_augmented"][i]["dense_plotted_pixels"] = (
                    new_pixels.tolist()
                )

        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
        return None
    except Exception as e:
        return (str(e), file_path)


def collect_json_files(dir):
    """递归收集目录下所有 JSON 文件路径"""
    json_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Resample plotted pixels in a directory."
    )

    parser.add_argument(
        "--resample_factor",
        type=int,
        default=20,
        help="Multiplicative factor for resampling the plotted pixels.",
    )
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Directory containing plotted pixels to be resampled.",
    )
    parser.add_argument(
        "--run_on_subdirs",
        action="store_true",
        help="Whether to run on folder itself or in parallel on subdirs?.",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Whether to plot the resampled pixels."
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers.")

    args = parser.parse_args()

    dir = args.dir

    # 收集所有 JSON 文件
    print(f"扫描 JSON 文件: {dir} ...")
    json_files = collect_json_files(dir)
    print(f"共找到 {len(json_files)} 个 JSON 文件")

    if not json_files:
        print("没有找到 JSON 文件，退出。")
        exit(0)

    # 并行处理，全局进度条
    resample_fn = partial(resample_single_file, resample_factor=args.resample_factor)
    error_list = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            for result in tqdm(
                executor.map(resample_fn, json_files),
                total=len(json_files),
                desc="replot_pixels",
            ):
                if result is not None:
                    error_list.append(result)
    else:
        for fp in tqdm(json_files, desc="replot_pixels"):
            result = resample_fn(fp)
            if result is not None:
                error_list.append(result)

    if error_list:
        print(f"\n{len(error_list)} 个文件处理失败:")
        for err, fp in error_list[:20]:
            print(f"  {fp}: {err}")
        if len(error_list) > 20:
            print(f"  ... 共 {len(error_list)} 个错误")
    else:
        print("全部成功，无错误。")

    print(f"All files saved. ({len(json_files) - len(error_list)}/{len(json_files)})")

    # Plot
    if args.plot:
        for file in os.listdir(dir):
            if file.endswith(".json"):
                file_path = os.path.join(dir, file)

                with open(file_path, "r") as f:
                    data = json.load(f)

                leads = data["leads"]
                for i in range(len(leads)):
                    pixels = np.array(leads[i]["plotted_pixels"])
                    plt.scatter(pixels[:, 1], -pixels[:, 0], s=1)
                plt.figure()
                for i in range(len(leads)):
                    dense_pixels = np.array(leads[i]["dense_plotted_pixels"])
                    plt.scatter(dense_pixels[:, 1], -dense_pixels[:, 0], s=1)
                plt.show()
                break

    print("Done with replot_pixels.py")
