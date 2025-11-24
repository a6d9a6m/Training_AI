# 声音异常检测系统

一个基于机器学习的声音异常检测系统，支持监督学习和无监督异常检测两种模式。

## 快速开始

### 安装依赖

```bash
pip install numpy scipy librosa scikit-learn torch matplotlib seaborn sounddevice
```

### 训练模型

**方式1：监督学习**（有标签的正常和异常样本）

```bash
python train_supervised.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly
```

**方式2：异常检测**（只有正常样本）

```bash
python train_anomaly_detection.py \
    --normal_train_dir data/normal_train
```

### 实时检测

```bash
python realtime_detection.py --model models/saved_models/supervised_gmm_model.pkl
```

## 两种训练模式

### 1. 监督学习模式

**适用场景**: 有大量标注好的正常和异常样本

**数据组织**:
```
data/
├── normal/      # 正常样本
└── anomaly/     # 异常样本
```

**模型**: GMM（高斯混合模型）- 为每个类别训练独立的GMM

**使用**:
```bash
python train_supervised.py --normal_dir data/normal --anomaly_dir data/anomaly
```

### 2. 异常检测模式

**适用场景**: 只有正常样本，异常样本难获取或未知

**数据组织**:
```
data/
├── normal_train/    # 纯净正常样本（训练用）
└── mixed_test/      # 混合样本（测试用，可选）
    ├── normal/
    └── anomaly/
```

**模型**: 自动编码器 - 只在正常样本上训练，学习正常模式

**原理**: 重构误差大的样本判定为异常

**使用**:
```bash
python train_anomaly_detection.py --normal_train_dir data/normal_train --mixed_test_dir data/mixed_test
```

## 实时检测

系统支持实时音频流异常检测：

```bash
# 命令行版本
python realtime_detection.py --model models/saved_models/xxx_model.pkl

# 可视化版本（实时显示波形和异常状态）
python realtime_detection_visual.py --model models/saved_models/xxx_model.pkl

# 从文件模拟流（测试用）
python realtime_detection.py --mode file --audio_file test.wav --model models/saved_models/xxx_model.pkl
```

**实时检测特性**:
- ✅ 麦克风实时输入
- ✅ 边播放边检测，延迟约100-200ms
- ✅ 连续异常判定，减少误报
- ✅ 可视化界面（波形、异常分数曲线、状态指示器）
- ✅ 自定义警告回调

## 项目结构

```
Project/
├── train_supervised.py           # 监督学习训练脚本
├── train_anomaly_detection.py    # 异常检测训练脚本
├── realtime_detection.py          # 实时检测（命令行版）
├── realtime_detection_visual.py   # 实时检测（可视化版）
├── features/                      # 特征提取
│   └── extract_features.py
├── models/                        # 模型实现
│   ├── gmm_model.py              # GMM模型
│   ├── autoencoder.py            # 自动编码器
│   └── threshold_detector.py     # 阈值检测
├── utils/                         # 工具类
│   ├── simple_data_loader.py     # 简化的数据加载器
│   ├── evaluator.py              # 模型评估
│   ├── realtime_classifier.py    # 实时分类器
│   ├── audio_stream_handler.py   # 音频流处理
│   └── feature_extractor_wrapper.py  # 特征提取包装
└── TRAINING_GUIDE.md             # 详细训练指南
```

## 功能特点

### 核心功能
- **两种训练模式**: 监督学习 + 异常检测
- **音频特征提取**: MFCC、梅尔频谱、色度、频谱对比度、过零率、RMS能量等
- **多种模型**: GMM、自动编码器
- **完整评估**: 准确率、精确率、召回率、F1分数、混淆矩阵、ROC曲线

### 实时检测
- **麦克风输入**: 实时捕获并检测音频流
- **文件模拟**: 从文件模拟实时流（方便测试）
- **智能警告**: 连续异常帧判定，避免误报
- **可视化**: 实时波形、异常分数曲线、状态指示

### 其他特性
- **特征缓存**: 自动缓存提取的特征，提高效率
- **配置灵活**: 通过命令行参数灵活配置
- **数值稳定**: 优化的统计特征计算

## 详细文档

- **[训练指南](TRAINING_GUIDE.md)**: 详细的训练说明、参数解释、常见问题
- **[实时检测指南](REALTIME_DETECTION_GUIDE.md)**: 实时检测的详细使用说明
- **[架构文档](CLAUDE.md)**: 系统架构说明（供AI助手参考）

## 使用示例

### 完整流程示例

```bash
# 1. 准备数据（监督学习）
mkdir -p data/normal data/anomaly
# 将音频文件放入对应目录...

# 2. 训练模型
python train_supervised.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --n_components 8 \
    --output_dir models/saved_models

# 3. 实时检测
python realtime_detection.py \
    --model models/saved_models/supervised_gmm_model.pkl

# 4. 可视化版本
python realtime_detection_visual.py \
    --model models/saved_models/supervised_gmm_model.pkl
```

### Python API 使用

```python
# 加载模型
from models.gmm_model import GMMModel
from models.autoencoder import AudioAutoencoder

# GMM模型
gmm_model = GMMModel.load('models/saved_models/supervised_gmm_model.pkl')

# 自动编码器
ae_model = AudioAutoencoder.load('models/saved_models/anomaly_detection_model.pth')

# 提取特征并预测
from features.extract_features import extract_all_features
features, _ = extract_all_features(audio_data, sr=22050)

# GMM预测
prediction = gmm_model.predict(features.reshape(1, -1))

# 自动编码器异常检测
error = ae_model.calculate_reconstruction_error(features.reshape(1, -1))
is_anomaly = error > ae_model.threshold
```

## 选择哪种模式？

| 场景 | 推荐模式 | 原因 |
|------|----------|------|
| 有大量标注数据 | 监督学习 | 利用标签，效果更好 |
| 只有正常样本 | 异常检测 | 无需异常样本 |
| 异常模式未知 | 异常检测 | 更灵活 |
| 异常模式明确 | 监督学习 | 更精确 |

## 性能

- **训练时间**:
  - 监督学习: 通常几分钟
  - 异常检测: 10-30分钟（取决于epochs）
- **实时检测延迟**: 100-200ms
- **CPU占用**: 单核15-25%
- **内存占用**: 约100-200MB

## 常见问题

**Q: 训练数据需要多少？**
- 监督学习: 建议每类至少50个样本
- 异常检测: 建议至少100个正常样本

**Q: 支持的音频格式？**
- WAV格式（推荐）
- 采样率: 22050Hz（默认，可调整）
- 声道: 单声道（自动转换）

**Q: 如何提高准确率？**
1. 增加训练样本数量
2. 调整模型参数（n_components, encoding_dim）
3. 调整阈值
4. 确保训练数据质量

## 许可证

MIT License
