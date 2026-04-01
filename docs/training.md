# nnUNet 训练文档

本文档说明如何使用 nnUNet 训练 ECG 分割模型。

> **前置条件**：已通过 `shells/batch_generate_image/postprocess.sh` 生成 nnUNet 格式的数据集。

---

## 1. 环境变量

所有 nnUNet 命令依赖三个环境变量，每个脚本顶部都需要配置：

| 变量 | 含义 | 示例 |
|------|------|------|
| `nnUNet_raw` | 包含 `DatasetXXX_Signals` 的**父目录** | `/data/nnUNet_raw` |
| `nnUNet_preprocessed` | 预处理输出目录（自动创建） | `/data/nnUNet_preprocessed` |
| `nnUNet_results` | 训练结果输出目录（自动创建） | `/data/nnUNet_results` |

目录关系：

```
/data/nnUNet_raw/                      ← nnUNet_raw 指向这里
└── Dataset500_Signals/                ← postprocess.sh 生成的数据集
    ├── dataset.json
    ├── imagesTr/
    └── labelsTr/

/data/nnUNet_preprocessed/             ← nnUNet_preprocessed 指向这里
└── Dataset500_Signals/                ← 01_preprocess.sh 自动生成

/data/nnUNet_results/                  ← nnUNet_results 指向这里
└── Dataset500_Signals/                ← 02_train.sh 自动生成
    └── nnUNetTrainer__nnUNetPlans__2d/
        ├── fold_0/
        │   ├── checkpoint_best.pth
        │   └── checkpoint_final.pth
        └── fold_all/
            ├── checkpoint_best.pth
            └── checkpoint_final.pth
```

---

## 2. 训练脚本

```
shells/train/
├── 01_preprocess.sh    # 预处理
├── 02_train.sh         # 训练
├── 03_find_best.sh     # 查找最佳配置
├── 04_predict.sh       # 推理预测
└── run_all.sh          # 一键全流程
```

---

## 3. 分步说明

### 3.1 预处理 (01_preprocess.sh)

```bash
bash shells/train/01_preprocess.sh
```

执行 `nnUNetv2_plan_and_preprocess`，包含：

1. **verify_dataset_integrity** — 检查 imagesTr/labelsTr 文件配对、标签值连续性、dataset.json 合法性
2. **plan** — 根据数据自动规划网络架构（patch size、batch size、网络深度等）
3. **preprocess** — 将图像和标签转为 nnUNet 内部格式（.npz/.npy）

| 参数 | 含义 |
|------|------|
| `-d 500` | 数据集 ID，对应 `Dataset500_Signals` |
| `--clean` | 清除已有预处理结果，重新开始 |
| `-c 2d` | 只做 2D 配置（ECG 图像是 2D 的）|
| `--verify_dataset_integrity` | 预处理前先校验数据集 |

### 3.2 训练 (02_train.sh)

```bash
bash shells/train/02_train.sh
```

#### 命令行参数（脚本顶部配置）

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `FOLD` | `0` | `0-9` 为 10 折中某一折（90% 训练 + 10% 验证）；`all` 全量训练无验证 |
| `DEVICE` | `cuda` | 训练设备：`cuda` / `cpu` / `mps` |
| `NUM_GPUS` | `1` | 多 GPU 训练时的 GPU 数量 |
| `PRETRAINED` | 空 | 预训练权重路径（迁移学习），留空从头训练 |
| `CONTINUE` | `true` | 断点续训——中断后重新执行从最近 checkpoint 恢复 |
| `SAVE_SOFTMAX` | `false` | 保存 softmax 预测（模型集成时需要） |
| `USE_COMPRESSED` | `false` | 读压缩数据（省存储但耗 CPU） |

#### 模型超参（需改源码）

以下参数位于 `nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py` 的 `__init__` 中。修改后需重新安装：`cd nnUNet && pip install . && cd ..`

| 超参 | 默认值 | 说明 |
|------|--------|------|
| `initial_lr` | `1e-2` | 初始学习率 |
| `weight_decay` | `3e-5` | 权重衰减（L2 正则化） |
| `num_epochs` | `500` | 总训练轮数 |
| `num_iterations_per_epoch` | `250` | 每轮训练迭代次数（每轮看到 250 × batch_size 个样本） |
| `num_val_iterations_per_epoch` | `50` | 每轮验证迭代次数 |
| `oversample_foreground_percent` | `0.33` | 前景过采样比例（33% 的 batch 保证包含前景像素，应对背景占 97% 的不平衡） |
| `enable_deep_supervision` | `True` | 深度监督（中间层也计算 loss，帮助梯度传播） |
| `save_every` | `50` | 每 N 轮保存一次中间 checkpoint |

固定设置（不建议改）：
- **优化器**：SGD（momentum=0.99, nesterov=True）
- **学习率调度**：PolyLR（多项式衰减，从 initial_lr 衰减到 0）
- **数据划分**：10 折，seed=42

#### 多 GPU 训练

```bash
# 不同 fold 分配到不同 GPU
CUDA_VISIBLE_DEVICES=0 bash shells/train/02_train.sh 0
CUDA_VISIBLE_DEVICES=1 bash shells/train/02_train.sh 1
CUDA_VISIBLE_DEVICES=2 bash shells/train/02_train.sh 2
# ...
```

#### 训练时长参考

nnUNet 默认训练 1000 个 epoch。实际时间取决于数据集大小和 GPU：

| 数据量 | GPU | 大约时长 |
|--------|-----|---------|
| 10k 样本 | V100 | ~12 小时 |
| 50k 样本 | V100 | ~48 小时 |
| 100k 样本 | V100 | ~4-5 天 |

### 3.3 查找最佳配置 (03_find_best.sh)

```bash
bash shells/train/03_find_best.sh
```

执行 `nnUNetv2_find_best_configuration`，自动评估所有已训练的配置和 fold，找出最佳组合。

| 参数 | 含义 |
|------|------|
| `-c 2d` | 只评估 2D 配置 |
| `--disable_ensembling` | 禁用模型集成（单模型即可）|

> 只有完成了**五折交叉验证**（fold 0-4 全训完）才有意义。如果只训了 `fold all`，可以跳过这步。

### 3.4 推理预测 (04_predict.sh)

```bash
bash shells/train/04_predict.sh
```

在脚本顶部配置：

| 配置项 | 默认值 | 含义 |
|--------|--------|------|
| `FOLD` | `all` | 使用哪个 fold 的模型 |
| `INPUT_DIR` | — | 待预测图像目录（PNG 格式，文件名需含 `_0000` 后缀）|
| `OUTPUT_DIR` | — | 预测结果输出目录（输出 mask PNG）|
| `DEVICE` | `cuda` | `cuda` 或 `cpu` |

输入图像命名要求：`xxx_0000.png`（跟 nnUNet 训练格式一致）。

---

## 4. 一键全流程 (run_all.sh)

```bash
bash shells/train/run_all.sh
```

依次执行：预处理 → 全量训练 (fold=all) → 查找最佳配置。

适合**最终部署**场景——用全部数据训练一个最强模型。

---

## 5. 训练产出

训练完成后，模型权重在：

```
$nnUNet_results/Dataset500_Signals/nnUNetTrainer__nnUNetPlans__2d/
├── fold_0/                          # 交叉验证 fold
│   ├── checkpoint_best.pth          # 验证集最佳 checkpoint
│   ├── checkpoint_final.pth         # 最终 checkpoint
│   └── progress.png                 # 训练曲线
└── fold_all/                        # 全量训练
    ├── checkpoint_best.pth
    └── checkpoint_final.pth
```

要用训练好的模型做数字化，将整个 `nnUNet_results` 目录复制到 `models/M_new/` 下：

```bash
# 部署新模型
mkdir -p models/M_new
cp -r $nnUNet_results models/M_new/nnUNet_results

# 用新模型做数字化
python -m src.run.digitize -d test/input -m models/M_new -o test/output/M_new -v
```

---

## 6. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 预处理报 "dataset integrity check failed" | imagesTr 和 labelsTr 文件不配对，或标签值不连续 | 检查文件命名（图像 `_0000.png`，标签无后缀）和 dataset.json |
| 训练 OOM (Out of Memory) | GPU 显存不足 | nnUNet 会自动适配，但极大图像可能溢出；可降低输入分辨率 |
| 训练中断后如何恢复 | — | 直接重新执行同样的 `02_train.sh` 命令，`--c` 参数自动从 checkpoint 恢复 |
| CPU 训练太慢 | 正常，nnUNet 设计用 GPU | CPU 仅用于测试验证，正式训练务必用 GPU |
| fold_all 没有验证指标 | 全量训练无验证集 | 正常行为；如需评估请用五折交叉验证 |
| 推理报 SIGBUS | `/dev/shm` 太小 | 本项目已修复（export 在主进程执行），确保用的是本地 nnUNet fork |
