# ECG 后处理参考文档

> 参考项目：Open-ECG-Digitizer
> 参考文件位于本仓库 `refs/` 目录
> 本文档记录其后处理方法，供未来改造 ECG-Digitiser 的 `digitize.py` 时参考

---

## 1. 当前问题

ECG-Digitiser 的后处理存在以下硬编码限制：

| 问题 | 位置 | 说明 |
|------|------|------|
| `Y_SHIFT_RATIO` 硬编码 | `config.py` | 基线位置按 Letter 纸 (21.59cm) 3×4 布局写死 |
| `sec_per_pixel` 启发式估算 | `digitize.py:440-447` | 取中位数 mask 宽度估算，无网格校准 |
| 5秒阈值二分判断 | `digitize.py:248` | 导联被分为 2.5s 或 10s，边界情况易误判 |
| 无信号线追踪 | `cut_to_mask()` | 直接裁剪 mask 边界框，不做质心追踪 |
| 无断线修复 | 全局 | 分割缺口直接传递到最终信号 |

---

## 2. Open-ECG-Digitizer 后处理总览

完整管线共 6 个阶段：

```
ECG图像
  │
  ├─ 1. 透视校正（Hough 变换检测网格线 → 旋转/透视变换）
  │
  ├─ 2. 像素尺寸校准（自相关检测网格间距 → mm/pixel）
  │
  ├─ 3. UNet 分割（4类：信号/网格/文字/背景）
  │
  ├─ 4. 信号线提取（质心追踪 → 线段匹配合并 → 端点外推 → 间隙插值）
  │
  ├─ 5. 导联识别（布局模板点云匹配 → 标准12导联排列）
  │
  └─ 6. 归一化输出（基线去除 → 缩放到 µV → 重采样到5000点）
```

**对比我们当前管线**：我们只有阶段 1（Hough 旋转）、3（nnUNet 分割）和简化版的 4+6。缺少阶段 2（像素校准）和 5（布局识别），阶段 4 的信号提取也很粗糙。

---

## 3. 像素尺寸校准（PixelSizeFinder）

> 参考文件：`refs/model/pixel_size_finder.py`

### 目的

从 ECG 网格线自动推算 mm/pixel，**替代硬编码的 `sec_per_pixel`**。

### 算法：自相关网格检测

```
输入：网格概率图 grid_prob (H×W)

1. 沿 x/y 轴分别求和，得到 1D 信号
2. 减去均值，计算自相关：auto = correlate(signal, signal, mode='full')
3. 迭代缩放搜索（10轮 zoom）：
   初始搜索范围：[image_size/120, image_size/15] 像素
   每轮：
     a. 在 1000 个候选间距中搜索
     b. 对每个候选间距，生成理想网格自相关模板：
        - 大格线 (5mm) 权重 = 1.0
        - 小格线 (1mm) 权重 = 0.5（lower_grid_line_factor）
        - 距离衰减加权
     c. 评分 = Σ(实际自相关 × 理想模板)
     d. 最佳间距 = argmax(评分)
     e. 缩小搜索范围 ×10 倍
4. mm_per_pixel = 5mm / 最佳像素间距
```

### 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mm_between_grid_lines` | 5 | 标准 ECG 网格大格间距 |
| `samples` | 1000 | 每轮搜索候选数 |
| `min_number_of_grid_lines` | 15-30 | 预期最少网格线数 |
| `max_number_of_grid_lines` | 70-120 | 预期最多网格线数 |
| `max_zoom` | 10 | zoom 迭代次数 |
| `lower_grid_line_factor` | 0.3-0.5 | 小格线权重 |

### 我们如何使用

当前 `digitize.py` 用 mask 宽度中位数估算 `sec_per_pixel`，可替换为：
- 如果我们的 nnUNet 输出增加网格类，直接用此方法
- 如果不增加网格类，可对原图做自相关（效果会差些）
- 12×1 布局下尤其重要，因为所有导联等长，无法用宽度差异区分长短

---

## 4. 信号线提取（SignalExtractor）

> 参考文件：`refs/model/signal_extractor.py`（baseline）、`refs/model/signal_extractor_improved.py`（improved）

### 4.1 质心追踪（核心算法）

替代我们的 `cut_to_mask()` 裁剪方式，逐列计算信号 y 坐标：

```
输入：信号概率图 fmap (H×W)，二值 mask (H×W)

对每一列 x：
  概率归一化：prob[:, x] /= sum(prob[:, x])
  y 坐标 = Σ(row_index × prob[row, x]) / Σ(prob[row, x])

  即加权质心：line[x] = Σ(i × P(i,x)) / Σ(P(i,x))

如果某列概率和 < 阈值 → line[x] = NaN（缺失）
```

**优势**：不是简单取 mask 边界框，而是精确追踪信号曲线的 y 位置。

### 4.2 连通域分析 + 线段匹配

```
1. 对 fmap 做连通域标记（4-连通，label_thresh=0.1）
2. 过滤：region.sum() >= threshold_sum (10.0)
3. 对每个连通域提取质心线
4. 验证：>= 95% 的线点落在 mask 内
5. 用匈牙利算法匹配线段端点，合并属于同一导联的碎片

代价矩阵：
  cost = (|Δx| + |Δy|) × (1 + 30 × |高度差|)
  - Δx：两段端点的水平距离（考虑环绕）
  - Δy：两段端点的垂直距离
  - 高度差惩罚：不同粗细/高度的线段不太可能属于同一导联
```

### 4.3 端点外推（improved 版，PCC +42%）

最有价值的改进，**填补分割在导联两端的缺口**：

```
对每条合并后的线：
  valid = 非 NaN 的位置
  first = min(valid), last = max(valid)

  左端外推（如果 first > 0 且 first ≤ 30px）：
    取前 15 个有效点
    线性回归求斜率：slope = Σ((x-μ)(y-μ)) / Σ((x-μ)²)
    向左延伸：y[px] = y[first] + slope × (px - first)

  右端外推（同理，如果距右边界 ≤ 30px）
```

**关键参数**：
- `slope_window = 15`：用于估算斜率的点数
- `max_extrapolation_gap = 30`：最大外推距离（像素）

### 4.4 覆盖率加权合并（improved 版，PCC +1%）

当多条线段在某些列重叠时，用覆盖率加权平均取代简单取第一个值：

```
对每个合并组内的多条线：
  coverage[i] = 该线有效点数
  weight[i] = coverage[i] / Σ(coverage)

  对每列：
    如果多条线都有值：merged[col] = Σ(line[i][col] × weight[i])
    如果只有一条线有值：直接取该值
```

### 4.5 内部间隙插值

```
对每条合并后的线，在 [first, last] 范围内：
  如果发现 NaN 间隙 ≤ 20px：
    线性插值填补
  如果间隙 > 20px：
    保留 NaN（间隙过大不可靠）
```

### 4.6 Viterbi 动态规划路径提取（可选）

全局最优路径提取，适用于导联重叠严重的情况：

```
1. 逐列检测节点：prob > 0.1 的连续区域 → 计算加权中心
2. DP 前向传播：
   cost(j) = min_k[ cost(k) + α × dist(k,j)/prob(k) + (1-α) × |角度变化| ]
   - α=0.5：距离与平滑度等权
   - 1/prob：偏好高概率节点
3. 回溯最优路径
4. 填补 ≤5 列的间隙
```

### 数据流对比

```
当前 ECG-Digitiser：
  mask → cut_to_mask(边界框裁剪) → 直接用 y_min 作为信号位置 → resample

Open-ECG-Digitizer：
  fmap → 连通域标记 → 质心追踪 → 线段匹配(匈牙利) → 端点外推 → 加权合并 → 间隙插值
```

---

## 5. 导联识别与布局匹配（LeadIdentifier）

> 参考文件：`refs/model/lead_identifier.py`、`refs/model/lead_identifier_improved.py`

### 目的

自动识别 ECG 图的导联布局并重排为标准 12 导联顺序。**替代我们硬编码的 `LEAD_LABEL_MAPPING`**。

### 5.1 布局模板系统

> 参考文件：`refs/config/lead_layouts_all.yml`

预定义的布局模板包括：

| 布局 | 结构 | 导联顺序 |
|------|------|----------|
| standard_3x4 | 3行×4列 | I,aVR,V1,V4 / II,aVL,V2,V5 / III,aVF,V3,V6 |
| cabrera_12x1 | 12行×1列 | aVL,I,-aVR,II,aVF,III,V1-V6 |
| 6x2 | 6行×2列 | I,V1 / II,V2 / III,V3 / aVR,V4 / aVL,V5 / aVF,V6 |
| precordial_6x1 | 6行×1列 | V1-V6 |

每个布局可附加 rhythm strip 配置（0/1/2/3 条）。

**对 12×1 布局的意义**：我们只需定义一个 12×1 模板，匹配逻辑自动适配。

### 5.2 导联位置检测

利用 UNet 输出的文字区域概率图（13类：12导联+背景），计算每个导联标签的质心：

```
对每个导联 lead_idx (0-11)：
  channel = prob[0, lead_idx, :, :]
  如果 sum(channel) == 0：跳过

  x_com = Σ(x × channel) / Σ(channel)  # x 质心
  y_com = Σ(y × channel) / Σ(channel)  # y 质心

  输出：[(lead_name, x_com, y_com), ...]
```

### 5.3 点云匹配算法

将检测到的导联位置与模板网格对齐：

```
对每个布局模板：
  1. 模板网格归一化到 [0,1]：
     norm_x(j) = j / (cols-1)
     norm_y(i) = i / (rows-1)

  2. 找交集导联（检测到的 ∩ 模板中的）

  3. 刚性点云对齐（缩放+平移）：
     Pc = P - mean(P)，Gc = G - mean(G)
     scale = Σ(Pc·Gc) / Σ(Pc²)
     translation = mean(G) - scale × mean(P)

  4. 残差：
     matched_residual = ||P_aligned - G||₂
     unmatched_penalty = 0.5
     cost = mean(residuals) × scaling_factor

  5. scaling_factor = max(n_grid, n_detected) / min(n_grid, n_detected)
                    × (1 + |rows_diff| × 3)

选择 cost 最小的布局
```

### 5.4 翻转检测

自动检测 ECG 图是否上下颠倒：

```
对每个布局，同时测试正常和翻转两种情况
翻转时：y 坐标取反，线序反转，幅值取反
选择残差更小的方向
```

### 5.5 Rhythm Strip 匹配

```
rhythm 导联通过余弦相似度与标准导联匹配：
  1. 计算相关矩阵 (n_rhythm × 12)
  2. 对常见 rhythm 导联加权（lead II 最常见）
  3. 匈牙利算法最优匹配
```

### 5.6 Cabrera 肢导联一致性检查

```
用 Cabrera 关系验证肢导联：
  aVL = I - 0.5×II
  -aVR = 0.5×I + 0.5×II
  aVF = -0.5×I + II
  III = -I + II

余弦相似度 > 0.992 → 强 Cabrera 信号（限制为 Cabrera 布局）
余弦相似度 < 0.95  → 排除 Cabrera 布局
```

---

## 6. 归一化与基线去除

> 参考文件：`refs/model/lead_identifier_improved.py`

### 当前方式（ECG-Digitiser）

```python
# 硬编码基线位置
signal_shifted = (1 - Y_SHIFT_RATIO[lead]) * image_height - signal_y
predicted_signal = (signal_shifted - non_zero_mean) * mV_per_pixel
```

### 改进方式（Open-ECG-Digitizer improved）

```python
# 用 nanmedian 自适应去基线
baseline = lines.nanmedian(dim=1, keepdim=True).values
lines = lines - baseline

# 动态缩放
lines = lines * (mv_per_mm / avg_pixel_per_mm) * 1000  # → µV
```

**为什么用 nanmedian**：
- ECG 信号大部分时间在基线附近，QRS 波是短暂的大幅偏移
- `nanmean` 被 QRS 峰拉偏 → 基线估计不准
- `nanmedian` 对离群值鲁棒 → 基线估计更准确
- 这个改动是所有改进的基础

### 重采样

```
1. 裁剪有效区域：找到 ≥3 个导联同时有值的列范围
2. 线性插值到 5000 点（500Hz × 10s）
```

---

## 7. 实验记录与经验总结

> 参考文件：`refs/git_history.txt`（188 条迭代记录）

### 有效改进（按贡献度排序）

| 改进 | PCC 提升 | 说明 |
|------|----------|------|
| nanmedian 基线去除 | 基础 | 所有后续改进的前提 |
| 端点外推 | +42% | 最大单项改进，填补分割边缘缺口 |
| 离群值拒绝 | +14% | 去除异常质心点 |
| 覆盖率加权合并 | +1% | 重叠区域用线段长度加权 |
| 确定性种子 | +0.5% | 消除随机性 |

### 失败尝试（需避免）

| 尝试 | 结果 | 原因 |
|------|------|------|
| 去畸变 (dewarping) | PCC 降至 0.59 | 过度矫正，扭曲信号 |
| 三次样条插值 | RMSE 爆炸到 124,884 | 边界震荡 |
| 分水岭分割 | PCC 0.54 | 过度分割 |
| 形态学操作 | 更差 | 断开细线 |
| 二次外推 | 更差 | 边界过拟合 |
| 高斯模糊预处理 | 无改善 | 模糊了有用细节 |

### 最终效果对比

| 指标 | Baseline | Improved |
|------|----------|----------|
| PCC (相关系数) | 0.5452 | **0.9024** |
| RMSE | 148.00 µV | **61.26 µV** |
| SNR | 0.63 dB | **9.94 dB** |

---

## 8. 适配 12×1 布局的路线图

### 阶段 1：训练新分割模型

- 生成 12×1 布局的合成训练数据
- 训练 nnUNet（12 导联类 + 背景 = 13 类）
- **无需改后处理**，只改数据生成

### 阶段 2：替换信号提取

优先级从高到低：

1. **质心追踪**替代 `cut_to_mask()`
   - 实现 `SignalExtractor._extract_line_from_region()`
   - 需要连通域标记 + 质心计算

2. **端点外推**（最高 ROI 改进）
   - 实现 `SignalExtractorImproved._extrapolate_endpoints()`
   - 参数：`slope_window=15`, `max_extrapolation_gap=30`

3. **内部间隙插值**
   - 线性插值 ≤20px 的 NaN 间隙

### 阶段 3：替换幅值换算

1. **nanmedian 基线去除**替代 `Y_SHIFT_RATIO`
   - 一行改动，效果显著

2. **像素校准**替代启发式 `sec_per_pixel`
   - 需要网格概率图（可能需要改分割模型为 4 类）
   - 或对原图做自相关（降级方案）

### 阶段 4：（可选）布局自适应

- 引入布局模板系统
- 点云匹配自动识别布局
- 使项目支持多种布局

---

## 9. 参考文件索引

```
refs/
├── model/
│   ├── signal_extractor.py             # §4 baseline 信号提取
│   ├── signal_extractor_improved.py    # §4 improved 信号提取
│   ├── lead_identifier.py             # §5 布局识别 (nanmean)
│   ├── lead_identifier_improved.py    # §6 布局识别 (nanmedian)
│   ├── pixel_size_finder.py           # §3 像素尺寸校准
│   └── inference_wrapper.py           # §2 完整管线串联
├── config/
│   ├── lead_layouts_all.yml           # §5.1 布局模板定义
│   ├── lead_name_unet.yml            # UNet 导联名检测模型配置
│   ├── inference_wrapper.yml          # baseline 推理配置
│   └── inference_wrapper_viterbi.yml  # improved 推理配置
├── evaluation/
│   ├── evaluate_vs_gt.py             # 与 ground truth 对比评估
│   ├── comparison.py                  # baseline vs improved 对比
│   └── ablation_study.py             # 消融实验
└── git_history.txt                    # 188 条迭代记录
```
