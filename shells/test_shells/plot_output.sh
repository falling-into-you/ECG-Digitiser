#!/bin/bash
# 绘制数字化输出的 ECG 信号 (ECG 打印纸风格)
# 用法:
#   bash shells/plot_output.sh                          # 默认绘制 test/output/M3/
#   bash shells/plot_output.sh test/output/M1/          # 指定输入目录
#   bash shells/plot_output.sh test/output/M3/record -o out.png  # 指定输出路径

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INPUT="${1:-test/output/M3}"
shift 2>/dev/null || true

python -m src.utils.plot_output -i "$INPUT" "$@"
