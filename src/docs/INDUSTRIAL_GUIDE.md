# 工业设备声音异常检测指南

## 针对工业场景的特殊优化

### 工业设备声音的特点

1. **正常和异常差距小** - 异常通常是微妙的偏离
2. **持续背景音** - 环境噪声、设备基础噪声
3. **周期性运转** - 电机、泵、风扇等有周期性模式
4. **异常类型多样** - 轴承磨损、松动、不平衡、堵塞等

### 为什么监督学习效果不好？

监督学习（GMM分类）要求：
- ❌ 两类样本有明显差异
- ❌ 需要大量标注的异常样本
- ❌ 假设异常类型固定且已知

工业场景的现实：
- ✓ 正常和异常差距很小
- ✓ 异常样本难以收集
- ✓ 异常类型未知或多变

---

## 推荐方案：异常检测 + 工业特征

### 核心思路

1. **只学习"正常"的模式** - 不需要异常样本
2. **背景噪声抑制** - 减少持续噪声的干扰
3. **关注"变化"而非绝对值** - 异常表现为与正常模式的偏离
4. **提取周期性特征** - 捕捉设备运转规律

### 技术实现

#### 1. 背景噪声抑制
```python
# 方法：频谱减法
# - 从音频开头估计背景噪声特征
# - 从整个信号中减去噪声成分
# - 保持信号相位，只减弱噪声幅度
```

#### 2. 差分特征提取
```python
# 关注时间上的变化，而不是绝对值
# - MFCC的一阶、二阶差分
# - 频谱质心的变化
# - RMS能量的突变
# - 过零率的变化
```

#### 3. 周期性特征
```python
# 工业设备有周期性运转模式
# - 自相关函数检测周期性
# - Tempogram节奏特征
# - 异常会破坏周期性
```

---

## 使用方法

### 第1步：准备数据

**重要**：确保训练集只包含正常样本！

```bash
# 数据结构
data/
├── normal/              # 只放正常样本（100个）
│   ├── normal_001.wav
│   ├── normal_002.wav
│   └── ...
└── anomaly/             # 异常样本（用于测试）
    ├── anomaly_001.wav
    ├── anomaly_002.wav
    └── ...
```

### 第2步：训练模型（启用降噪）

```bash
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --denoise \
    --encoding_dim 32 \
    --epochs 150 \
    --threshold_percentile 95
```

**参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--normal_train_dir` | 必需 | 正常样本目录（训练用） |
| `--mixed_test_dir` | 可选 | 测试集目录（包含正常和异常） |
| `--denoise` | True | 启用背景噪声抑制 |
| `--no_denoise` | - | 禁用降噪（如果背景音不强） |
| `--encoding_dim` | 32 | 编码维度（32-64推荐） |
| `--epochs` | 150 | 训练轮数 |
| `--threshold_percentile` | 95 | 异常阈值（95表示95%正常） |

### 第3步：评估结果

训练完成后会输出：

```
评估结果:
  准确率: 0.XXXX
  精确率: 0.XXXX
  召回率: 0.XXXX
  F1分数: 0.XXXX
  AUC: 0.XXXX

重构误差分析:
  正常样本 - 均值: 0.00XX, 标准差: 0.00XX
  异常样本 - 均值: 0.0XXX, 标准差: 0.00XX
```

**期望结果：**
- AUC > 0.7 表示模型有区分能力
- 异常样本的重构误差明显大于正常样本

---

## 与之前方法的对比

| 方法 | 训练数据 | 适用场景 | 优势 | 劣势 |
|------|----------|----------|------|------|
| **监督学习 (GMM)** | 需要正常+异常 | 异常类型固定、差异明显 | 简单快速 | 需要异常样本、差异小时失效 |
| **标准异常检测** | 只需正常样本 | 异常类型未知 | 不需要异常样本 | 对背景噪声敏感 |
| **工业异常检测** | 只需正常样本 | **工业设备、有背景音** | **降噪+差分特征** | 计算稍复杂 |

---

## 参数调优建议

### 如果F1分数 < 0.6

#### 尝试1：调整阈值
```bash
# 降低阈值（更敏感，可能误报多）
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --threshold_percentile 90

# 提高阈值（更保守，可能漏报多）
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --threshold_percentile 98
```

#### 尝试2：增加模型容量
```bash
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --encoding_dim 64 \
    --epochs 200
```

#### 尝试3：禁用降噪（如果背景音不强）
```bash
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --no_denoise
```

### 如果F1分数 0.6-0.7

模型效果一般，可以：
1. 增加正常样本数量（至少150个）
2. 确保正常样本覆盖多种正常状态
3. 收集更明显的异常样本

### 如果F1分数 > 0.7

✓ 模型效果良好，可以部署使用

---

## 实时检测

训练完成后，使用实时检测：

```bash
# 麦克风实时检测
python realtime_detection.py \
    --model models/saved_models/industrial_anomaly_model.pth \
    --preprocessor models/saved_models/industrial_preprocessor.pkl \
    --consecutive_threshold 3

# 文件模拟流式检测
python realtime_detection.py \
    --model models/saved_models/industrial_anomaly_model.pth \
    --preprocessor models/saved_models/industrial_preprocessor.pkl \
    --mode file \
    --test_file data/test_audio.wav

# 可视化版本
python realtime_detection_visual.py \
    --model models/saved_models/industrial_anomaly_model.pth \
    --preprocessor models/saved_models/industrial_preprocessor.pkl
```

---

## 故障排查

### 问题1：训练后F1分数仍然很低 (~0.5)

**可能原因：**
1. 训练集中混入了异常样本
2. 正常样本变化太大（不同设备、不同工况）
3. 异常类型与正常太相似

**解决方法：**
```bash
# 1. 仔细检查训练集，确保全是正常样本
# 2. 只用同一设备、同一工况的正常样本训练
# 3. 调整阈值
```

### 问题2：重构误差正常和异常差距很小

查看输出：
```
正常样本 - 均值: 0.0050
异常样本 - 均值: 0.0055  # 差距太小！
```

**解决方法：**
- 增加训练轮数 `--epochs 200`
- 增大编码维度 `--encoding_dim 64`
- 尝试不降噪 `--no_denoise`

### 问题3：误报率高（很多正常被误判为异常）

**解决方法：**
- 提高阈值 `--threshold_percentile 98`
- 增加更多正常样本（覆盖更多正常模式）
- 增加连续异常帧数 `--consecutive_threshold 5`

---

## 下一步优化

如果效果仍不理想，可以尝试：

1. **数据增强** - 对正常样本添加轻微扰动
2. **集成学习** - 训练多个模型投票
3. **半监督学习** - 利用少量异常样本微调
4. **深度学习** - 使用CNN或RNN提取更高层特征

---

## 快速开始命令

```bash
# 一键训练（推荐配置）
python train_industrial.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --denoise \
    --encoding_dim 48 \
    --epochs 150 \
    --threshold_percentile 95
```

期待结果：
- AUC > 0.65（相比之前的0.52是明显提升）
- F1分数 > 0.6（相比之前的0.5是明显提升）
