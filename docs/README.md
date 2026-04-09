# 文档总览

本目录包含 ECG Digitiser 项目的详细技术文档。

---

## 文档阅读顺序

根据工作流程，建议按以下顺序阅读：

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. mimic_pipeline.md (必读)                                             │
│    MIMIC-IV-ECG 完整训练流程                                            │
│    - 从原始数据到模型训练的端到端流程                                   │
│    - 每个步骤的输入/输出/数据格式                                       │
│    - JSON 文件生命周期说明                                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. data_generation.md                                                   │
│    ECG 图像生成器详解                                                   │
│    - ecg-image-generator 的使用方法                                     │
│    - 各种参数和布局选项                                                 │
│    - 与原版 ecg-image-kit 的差异                                        │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. data_postprocessing.md                                               │
│    数据后处理详解                                                       │
│    - 像素坐标插值 (replot_pixels) 原理                                  │
│    - 标签生成流程                                                       │
│    - nnUNet 数据格式要求                                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. training.md                                                          │
│    nnUNet 训练详解                                                      │
│    - 环境变量配置                                                       │
│    - 预处理、训练、推理命令                                             │
│    - 多卡训练和断点续训                                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 文档列表

### 核心流程文档

| 文档 | 内容 | 适用场景 |
|------|------|---------|
| **mimic_pipeline.md** | MIMIC-IV-ECG 完整训练流程 | 快速上手、了解全貌 |
| **data_generation.md** | ECG 图像生成器详解 | 需要定制图像生成参数 |
| **data_postprocessing.md** | 数据后处理详解 | 理解像素插值和标签生成 |
| **training.md** | nnUNet 训练详解 | 配置训练参数、多卡训练 |

### 参考文档

| 文档 | 内容 | 适用场景 |
|------|------|---------|
| **dataset_construction_guide.md** | 数据集构建指南 | 设计增强策略、分析像素分布 |
| **post_processing_reference.md** | 后处理参考 | 改进 digitize.py 的信号提取 |

---

## 各文档详细说明

### 1. mimic_pipeline.md — MIMIC-IV-ECG 完整训练流程

**内容**：
- 6 步完整流程图
- 每个脚本的输入/输出
- JSON 文件在各阶段的用途
- 目录结构说明
- 已知问题与修复

**核心流程**：
```
01_gen → 02_postprocess → 03_merge → [clean_json] → 04_validate → 05_prepare → train
```

**重点章节**：
- "数据流与 JSON 文件" — 解释 JSON 什么时候需要、什么时候可删除
- "分割标签说明" — 标签格式要求（单通道灰度）

---

### 2. data_generation.md — ECG 图像生成器详解

**内容**：
- ecg-image-generator 的 4 个生成脚本
- 所有参数说明（布局、增强、输出格式）
- 与原版 ecg-image-kit 的差异
- JSON 输出格式

**关键参数**：
| 参数 | 说明 |
|------|------|
| `--store_config 2` | 保存像素坐标到 JSON（必需） |
| `--num_workers N` | 并行进程数 |
| `-rot N` | 随机旋转角度 |
| `--random_bw P` | 黑白图像概率 |

---

### 3. data_postprocessing.md — 数据后处理详解

**内容**：
- 像素坐标插值原理（为什么需要 dense_plotted_pixels）
- create_mimic_dataset.py 的处理流程
- nnUNet 数据格式要求
- 常见问题排查

**关键概念**：
- `plotted_pixels` — 原始采样点坐标（稀疏）
- `dense_plotted_pixels` — 插值后的坐标（稠密）
- 标签必须是单通道灰度图，像素值 = 类别索引

---

### 4. training.md — nnUNet 训练详解

**内容**：
- 三个环境变量的配置
- 预处理、训练、推理命令
- 多卡训练配置
- 断点续训
- 训练参数修改方法

**关键环境变量**：
```bash
export nnUNet_raw="..."
export nnUNet_preprocessed="..."
export nnUNet_results="..."
```

---

### 5. dataset_construction_guide.md — 数据集构建指南

**内容**：
- 真实世界 ECG 图像的像素分布分析
- 四种增强级别的设计依据
- 数据集构成比例建议

**背景**：解释为什么需要 clean/aug_light/aug_medium/aug_heavy 四种数据。

---

### 6. post_processing_reference.md — 后处理参考

**内容**：
- Open-ECG-Digitizer 的后处理方法
- 像素尺寸校准
- 信号线追踪
- 与当前 digitize.py 的对比

**用途**：参考其他项目的实现，用于未来改进 `src/run/digitize.py`。

---

## 快速定位

| 我想... | 看哪个文档 |
|---------|-----------|
| 跑通完整训练流程 | mimic_pipeline.md |
| 定制图像生成参数 | data_generation.md |
| 理解标签生成原理 | data_postprocessing.md |
| 配置多卡训练 | training.md |
| 设计数据增强策略 | dataset_construction_guide.md |
| 改进数字化算法 | post_processing_reference.md |
| 清理 JSON 节省空间 | mimic_pipeline.md 第 2 章 |
| 修复标签 RGB 问题 | mimic_pipeline.md 第 7 章 |
