# 训练指南

本项目支持两种训练模式，根据您的数据情况选择：

## 模式1：监督学习（有标签的正常和异常样本）

### 数据准备

准备两个文件夹：
```
data/
├── normal/          # 正常样本（.wav文件）
│   ├── normal_001.wav
│   ├── normal_002.wav
│   └── ...
└── anomaly/         # 异常样本（.wav文件）
    ├── anomaly_001.wav
    ├── anomaly_002.wav
    └── ...
```

### 训练命令

```bash
python train_supervised.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --output_dir models/saved_models \
    --n_components 5
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--normal_dir` | 正常样本目录 | 必需 |
| `--anomaly_dir` | 异常样本目录 | 必需 |
| `--output_dir` | 模型保存目录 | models/saved_models |
| `--sr` | 采样率 | 22050 |
| `--n_components` | GMM组件数量 | 5 |
| `--test_size` | 测试集比例 | 0.2 |
| `--val_size` | 验证集比例 | 0.2 |

### 输出

训练完成后会生成：
- `supervised_gmm_model.pkl` - 训练好的模型
- `supervised_gmm_model_info.json` - 训练信息和评估指标

---

## 模式2：异常检测（只有正常样本）

### 数据准备

#### 训练数据（纯净正常样本）
```
data/
└── normal_train/    # 全部是正常声音
    ├── normal_001.wav
    ├── normal_002.wav
    └── ...
```

#### 测试数据（可选，大部分正常+少量异常）
```
data/
└── mixed_test/
    ├── normal/      # 正常样本
    │   ├── test_normal_001.wav
    │   └── ...
    └── anomaly/     # 异常样本
        ├── test_anomaly_001.wav
        └── ...
```

### 训练命令

```bash
# 只有训练数据
python train_anomaly_detection.py \
    --normal_train_dir data/normal_train \
    --output_dir models/saved_models

# 同时提供测试数据
python train_anomaly_detection.py \
    --normal_train_dir data/normal_train \
    --mixed_test_dir data/mixed_test \
    --output_dir models/saved_models
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--normal_train_dir` | 纯净正常样本目录 | 必需 |
| `--mixed_test_dir` | 混合测试样本目录 | 可选 |
| `--output_dir` | 模型保存目录 | models/saved_models |
| `--sr` | 采样率 | 22050 |
| `--encoding_dim` | 编码器维度 | 64 |
| `--epochs` | 训练轮数 | 100 |
| `--batch_size` | 批次大小 | 32 |
| `--threshold_percentile` | 阈值百分位 | 95 |
| `--val_ratio` | 验证集比例 | 0.2 |

### 工作原理

1. 模型只在正常样本上训练
2. 学习正常声音的特征模式
3. 计算重构误差阈值（基于正常样本的第95百分位）
4. 测试时，重构误差超过阈值的样本判定为异常

### 输出

训练完成后会生成：
- `anomaly_detection_model.pth` - 训练好的模型
- `anomaly_detection_model_info.json` - 训练信息和评估指标

---

## 使用训练好的模型

### 实时检测

```bash
# 监督学习模型
python realtime_detection.py \
    --model models/saved_models/supervised_gmm_model.pkl

# 异常检测模型
python realtime_detection.py \
    --model models/saved_models/anomaly_detection_model.pth
```

### 批量预测

```python
# 监督学习模型
from models.gmm_model import GMMModel

model = GMMModel.load('models/saved_models/supervised_gmm_model.pkl')
# 使用模型进行预测...

# 异常检测模型
from models.autoencoder import AudioAutoencoder

model = AudioAutoencoder.load('models/saved_models/anomaly_detection_model.pth')
# 使用模型进行异常检测...
```

---

## 选择哪种模式？

| 情况 | 推荐模式 | 原因 |
|------|----------|------|
| 有大量标注好的正常和异常样本 | 监督学习 | 利用标签信息，效果更好 |
| 只有正常样本，异常样本难获取 | 异常检测 | 无需异常样本标注 |
| 异常模式未知或多变 | 异常检测 | 更灵活，能检测未见过的异常 |
| 异常模式明确且固定 | 监督学习 | 更精确 |

---

## 常见问题

### Q: 训练数据需要多少？

- **监督学习**: 建议每个类别至少50个样本
- **异常检测**: 建议至少100个正常样本

### Q: 音频文件格式要求？

- 格式: WAV（推荐）
- 采样率: 22050Hz（默认，可调整）
- 声道: 单声道（自动转换）

### Q: 训练时间？

- **监督学习**: 通常几分钟
- **异常检测**: 根据epochs，通常10-30分钟

### Q: 如何提高检测准确率？

1. 增加训练样本数量
2. 调整模型参数（GMM的n_components，自动编码器的encoding_dim）
3. 调整阈值（supervised的threshold，anomaly的threshold_percentile）
4. 确保训练数据质量（清晰、无噪声）

---

## 完整示例

### 监督学习完整流程

```bash
# 1. 准备数据
mkdir -p data/normal data/anomaly

# 2. 训练模型
python train_supervised.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --n_components 8

# 3. 实时检测
python realtime_detection.py \
    --model models/saved_models/supervised_gmm_model.pkl
```

### 异常检测完整流程

```bash
# 1. 准备数据
mkdir -p data/normal_train data/mixed_test/{normal,anomaly}

# 2. 训练模型
python train_anomaly_detection.py \
    --normal_train_dir data/normal_train \
    --mixed_test_dir data/mixed_test \
    --epochs 150 \
    --encoding_dim 128

# 3. 实时检测
python realtime_detection.py \
    --model models/saved_models/anomaly_detection_model.pth
```
