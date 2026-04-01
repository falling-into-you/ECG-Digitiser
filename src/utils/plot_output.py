#!/usr/bin/env python
"""将数字化输出的 WFDB 信号绘制为 ECG 打印纸风格图。

风格完全对齐 ecg-image-generator/ecg_plot.py:
  - 3行×4列 短导联 (每列 2.5s) + 底部 1 行节律条 (10s)
  - 标准 ECG 网格: 大格 0.2s / 0.5mV, 小格 0.04s / 0.1mV
  - 红色网格 + 黑色信号线
  - 导联名标注 + 底部 25mm/s / 10mm/mV 标注

用法:
    python shells/plot_output.py -i test/output/M3/40792771-0
    python shells/plot_output.py -i test/output/M3/  # 目录下所有记录
    python shells/plot_output.py -i test/output/M3/40792771-0 -o my_plot.png
"""

import argparse
import os
import glob
import numpy as np
import wfdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


# ── ECG 纸标准参数 (同 ecg_plot.py) ──────────────────────────
PAPER_WIDTH = 11           # inches
PAPER_HEIGHT = 8.5         # inches
RESOLUTION = 200           # dpi

Y_GRID_SIZE = 0.5          # mV per major grid
X_GRID_SIZE = 0.2          # s  per major grid
Y_GRID_INCH = 5 / 25.4     # 5mm per major grid
X_GRID_INCH = 5 / 25.4

LINE_WIDTH = 0.75
GRID_LINE_WIDTH = 0.5
LEAD_FONTSIZE = 11
LEAD_NAME_OFFSET = 0.5
DC_OFFSET_LEN = 0.2        # seconds

# 红色网格配色 (colour5 in ecg_plot.py)
COLOR_MAJOR = (1, 0, 0)
COLOR_MINOR = (0.996, 0.8745, 0.8588)
COLOR_LINE = (0, 0, 0)

# 标准 12 导联 3×4 排布 (同 config_3x4.yaml)
FORMAT_3X4 = [
    ["I", "aVR", "V1", "V4"],     # row 0
    ["II", "aVL", "V2", "V5"],    # row 1
    ["III", "aVF", "V3", "V6"],   # row 2
]
FULL_MODE_LEAD = "II"   # 底部节律条导联
COLUMNS = 4
LEAD_SEC = 2.5           # 每列 10s/4=2.5s


def plot_ecg_record(record_path, output_path=None):
    """按 ECG 打印纸风格绘制单条 WFDB 记录。"""
    record = wfdb.rdrecord(record_path)
    sig = record.p_signal          # (n_samples, n_leads)
    lead_names = record.sig_name
    fs = record.fs
    n_samples = sig.shape[0]
    total_sec = n_samples / fs

    # 构建 lead_name -> column_index 的信号字典
    lead_data = {name: sig[:, i] for i, name in enumerate(lead_names)}

    # ── 计算坐标系 ─────────────────────────────────────
    rows = 3  # 3×4 rows
    full_rows = 1  # 节律条
    total_rows = rows + full_rows
    row_height = (PAPER_HEIGHT * Y_GRID_SIZE / Y_GRID_INCH) / (total_rows + 2)
    x_max = PAPER_WIDTH * X_GRID_SIZE / X_GRID_INCH
    y_max = PAPER_HEIGHT * Y_GRID_SIZE / Y_GRID_INCH

    secs = LEAD_SEC
    x_gap = np.floor(((x_max - COLUMNS * secs) / 2) / 0.2) * 0.2
    step = 1.0 / fs

    # ── 创建画布 ──────────────────────────────────────
    fig, ax = plt.subplots(
        figsize=(PAPER_WIDTH, PAPER_HEIGHT), dpi=RESOLUTION
    )
    fig.subplots_adjust(
        hspace=0, wspace=0, left=0, right=1, bottom=0, top=1
    )
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    # ── 画网格 ────────────────────────────────────────
    ax.set_xticks(np.arange(0, x_max, X_GRID_SIZE))
    ax.set_yticks(np.arange(0, y_max, Y_GRID_SIZE))
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.grid(
        which='major', linestyle='-',
        linewidth=GRID_LINE_WIDTH, color=COLOR_MAJOR
    )
    ax.grid(
        which='minor', linestyle='-',
        linewidth=GRID_LINE_WIDTH, color=COLOR_MINOR
    )

    dc_offset = DC_OFFSET_LEN  # seconds offset for calibration pulse

    # ── 绘制 3×4 短导联 ─────────────────────────────────
    for row_idx, row_leads in enumerate(FORMAT_3X4):
        # y_offset: 从上往下, row0 最高
        y_offset = row_height * (total_rows - row_idx) + row_height / 2

        for col_idx, lead_name in enumerate(row_leads):
            x_offset = col_idx * secs

            # 取对应时间段的信号
            col_start_sec = col_idx * secs
            s_start = int(col_start_sec * fs)
            s_end = int((col_start_sec + secs) * fs)

            if lead_name in lead_data:
                data = lead_data[lead_name]
                seg = data[s_start:min(s_end, len(data))]
            else:
                # 导联缺失, 画零线
                seg = np.zeros(int(secs * fs))

            t = np.arange(len(seg)) * step
            ax.plot(
                t + x_offset + dc_offset + x_gap,
                seg + y_offset,
                linewidth=LINE_WIDTH, color=COLOR_LINE
            )

            # 导联名标注
            ax.text(
                x_offset + x_gap + dc_offset,
                y_offset - LEAD_NAME_OFFSET - 0.2,
                lead_name, fontsize=LEAD_FONTSIZE
            )

            # 每行第一列画校准脉冲
            if col_idx == 0:
                x_dc = np.arange(
                    0, fs * DC_OFFSET_LEN * step + 4 * step, step
                )
                dc_pulse = np.ones(len(x_dc))
                dc_pulse = np.concatenate(
                    ((0, 0), dc_pulse[2:-2], (0, 0))
                )
                ax.plot(
                    x_dc + x_gap,
                    dc_pulse + y_offset,
                    linewidth=LINE_WIDTH * 1.5, color=COLOR_LINE
                )

        # 列间分隔竖线
        for col_idx in range(COLUMNS - 1):
            sep_x_val = (col_idx + 1) * secs + dc_offset + x_gap
            sep_y = np.linspace(
                y_offset - 0.8, y_offset + 0.8, 20
            )
            ax.plot(
                [sep_x_val] * len(sep_y), sep_y,
                linewidth=LINE_WIDTH * 3, color=COLOR_LINE
            )

    # ── 绘制底部节律条 (full_mode lead, 10s) ────────────
    y_center = row_height * 0.5
    full_lead = FULL_MODE_LEAD

    if full_lead in lead_data:
        full_data = lead_data[full_lead][:int(10 * fs)]
    else:
        # 用第一个可用导联
        first_available = lead_names[0] if lead_names else None
        if first_available:
            full_data = lead_data[first_available][:int(10 * fs)]
            full_lead = first_available
        else:
            full_data = np.zeros(int(10 * fs))

    t_full = np.arange(len(full_data)) * step
    ax.plot(
        t_full + x_gap + dc_offset,
        full_data + y_center,
        linewidth=LINE_WIDTH, color=COLOR_LINE
    )
    ax.text(
        x_gap + dc_offset,
        y_center - LEAD_NAME_OFFSET,
        full_lead, fontsize=LEAD_FONTSIZE
    )

    # 节律条校准脉冲
    x_dc = np.arange(0, fs * DC_OFFSET_LEN * step + 4 * step, step)
    dc_pulse = np.ones(len(x_dc))
    dc_pulse = np.concatenate(((0, 0), dc_pulse[2:-2], (0, 0)))
    ax.plot(
        x_dc + x_gap,
        dc_pulse + y_center,
        linewidth=LINE_WIDTH * 1.5, color=COLOR_LINE
    )

    # ── 底部标注 ──────────────────────────────────────
    ax.text(2, 0.5, '25mm/s', fontsize=LEAD_FONTSIZE)
    ax.text(4, 0.5, '10mm/mV', fontsize=LEAD_FONTSIZE)

    # ── 保存 ─────────────────────────────────────────
    if output_path is None:
        output_path = record_path + "_plot.png"
    plt.savefig(output_path, dpi=RESOLUTION)
    plt.close(fig)
    plt.clf()
    plt.cla()
    print(f"已保存: {output_path}")


def find_records(path):
    """从路径找到所有 WFDB 记录 (不含扩展名)。"""
    if os.path.isfile(path + ".hea") or os.path.isfile(path):
        return [path.replace(".hea", "").replace(".dat", "")]
    hea_files = sorted(glob.glob(os.path.join(path, "*.hea")))
    return [f.replace(".hea", "") for f in hea_files]


def main():
    parser = argparse.ArgumentParser(
        description="绘制 WFDB 数字化 ECG 信号 (ECG 打印纸风格)"
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="WFDB 记录路径 (不含扩展名) 或包含记录的目录"
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="输出图片路径, 默认为 <input>_plot.png"
    )
    args = parser.parse_args()

    records = find_records(args.input)
    if not records:
        print(f"错误: 在 {args.input} 中未找到 WFDB 记录 (.hea 文件)")
        return

    print(f"找到 {len(records)} 条记录")
    for rec in records:
        out = args.output if (args.output and len(records) == 1) else None
        plot_ecg_record(rec, out)


if __name__ == "__main__":
    main()
