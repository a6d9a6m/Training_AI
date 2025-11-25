# 声音异常检测系统

这是一个基于高斯混合模型(GMM)的声音异常检测系统，可以对音频数据进行特征提取、模型训练和分类，用于区分正常声音和异常声音。系统支持批量处理和预留了实时分类接口。

## 项目结构

```
Project/
├── config/
│   └── config.yaml          # 配置文件
├── dev_data/                # 训练和测试数据
│   ├── normal/              # 正常声音数据（传统结构）
│   ├── anomaly/             # 异常声音数据（传统结构）
│   └── fan/                 # 设备类型目录示例
│       ├── train/           # 训练数据
│       │   ├── normal/      # 正常样本
│       │   └── anomaly/     # 异常样本
│       ├── source_test/     # 源域测试数据
│       │   ├── normal/      # 正常样本
│       │   └── anomaly/     # 异常样本
│       └── target_test/     # 目标域测试数据（可选）
├── features/                # 特征提取模块
│   ├── __init__.py
│   └── extract_features.py  # 音频特征提取功能
├── models/                  # 模型相关模块
│   ├── __init__.py
│   ├── gmm_model.py         # GMM模型实现
│   └── threshold_detector.py # 阈值检测方法
├── utils/                   # 工具类
│   ├── __init__.py
│   ├── config_manager.py    # 配置管理
│   ├── data_loader.py       # 数据加载和预处理
│   ├── evaluator.py         # 模型评估
│   └── realtime_classifier.py # 实时分类接口
├── main.py                  # 主程序入口
└── README.md                # 项目说明文档
```

## 功能特点

- **音频数据处理**：支持音频文件加载、重采样、分段等预处理功能
- **特征提取**：使用librosa库提取多种音频特征，包括MFCC、梅尔频谱、色度等
- **GMM模型**：实现基于高斯混合模型的声音分类
- **阈值优化**：提供基于F1分数、精确率-召回率曲线、ROC曲线等多种阈值优化方法
- **模型评估**：支持准确率、精确率、召回率、F1分数等多种评估指标计算和可视化
- **配置管理**：通过YAML文件灵活配置系统参数
- **实时音频流异常检测**：完整实现的实时音频流处理功能，支持麦克风输入和文件模拟，实时检测并警告异常
- **特征缓存机制**：自动缓存提取的特征，提高多次运行的效率，支持基于配置哈希的自动失效机制
- **数值稳定性改进**：优化了偏度和峰度计算，解决了因数据值接近导致的精度损失问题

## 安装依赖

确保您已安装Python 3.7或更高版本，然后使用pip安装所需依赖：

```bash
# 基础依赖
pip install numpy scipy librosa scikit-learn pyyaml matplotlib seaborn torch

# 实时音频输入依赖（必需）
pip install sounddevice
```

## 使用方法

### 1. 准备数据

系统支持两种数据组织方式：

#### 方式一：设备类型组织（推荐）

为每个设备类型创建单独的文件夹，如`dev_data/fan/`，内部结构如下：
- `train/normal/` - 训练用的正常声音数据
- `train/anomaly/` - 训练用的异常声音数据
- `source_test/normal/` - 测试用的正常声音数据
- `source_test/anomaly/` - 测试用的异常声音数据

目前系统已配置支持的设备类型有：`fan`, `pump`, `slider`, `valve`, `gearbox`, `bearing`, `others`

#### 方式二：传统结构（向后兼容）

将正常和异常的音频文件分别放入`dev_data/normal/`和`dev_data/anomaly/`目录。

### 2. 配置参数

根据需要修改`config/config.yaml`文件中的参数：

- **paths**：数据和模型保存路径
  - `base_data_dir`：基础数据目录（用于设备类型加载）
  - `normal_data_dir`：传统结构的正常数据目录
  - `anomaly_data_dir`：传统结构的异常数据目录
  - 其他路径参数...
  
- **data**：音频处理和数据加载参数
  - `sample_rate`：采样率
  - `duration`：音频时长（null表示加载整个文件）
  - `normalize`：是否归一化音频
  - `test_size`、`val_size`：数据集分割比例
  - `use_device_type`：是否使用设备类型加载数据（true/false）
  - `device_type`：设备类型，默认为'fan'
  - `supported_devices`：支持的设备类型列表
  
- **features**：特征提取参数（MFCC数量、窗口大小等）
  - `use_cache`：是否使用特征缓存机制，默认为true
  - `features_save_path`：特征缓存保存路径，默认为'./features/saved_features'
- **model**：GMM模型参数（组件数量等）
- **training**：训练参数
- **evaluation**：评估参数
- **real_time**：实时分类参数

### 3. 运行主程序

主程序`main.py`提供了训练、评估和推理的功能。根据实际需求修改并运行该程序。

### 4. 实时音频异常检测

系统已完整实现实时音频流异常检测功能，可以从麦克风实时捕获音频并检测异常。

#### 快速开始

```bash
# 1. 训练模型（如果还没有）
python main.py --mode train --config config/config.yaml

# 2. 从麦克风进行实时检测
python realtime_detection.py --model models/saved_models/gmm_model.pkl --threshold 0.5

# 3. 从音频文件模拟流（用于测试）
python realtime_detection.py \
    --mode file \
    --audio_file path/to/test_audio.wav \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.5
```

#### 可视化界面版本

```bash
# 启动带图形界面的实时检测（实时显示波形和异常状态）
python realtime_detection_visual.py \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.5
```

#### 主要功能

- ✅ **实时麦克风输入**: 从麦克风实时捕获音频并检测异常
- ✅ **文件模拟流**: 从音频文件模拟实时流，方便测试
- ✅ **实时警告**: 检测到异常时立即输出警告信息
- ✅ **连续异常判定**: 连续N帧异常才触发警告，避免误报
- ✅ **统计信息**: 实时显示检测统计数据
- ✅ **可视化界面**: 实时波形和异常分数曲线显示（visual版本）

详细使用说明请参考 [实时检测使用指南](REALTIME_DETECTION_GUIDE.md)。

## 核心模块说明

### 数据加载和预处理 (`utils/data_loader.py`)

- `load_audio`：加载音频文件
- `preprocess_audio`：预处理音频信号（归一化、降噪等）
- `load_dataset`：加载并分割数据集（训练集、验证集、测试集）
  - 支持两种数据加载方式：
    - 设备类型方式：通过`device_type`和`base_data_dir`参数加载特定设备的数据
    - 传统方式：通过`normal_dir`和`anomaly_dir`参数加载数据
- `load_audio_batch`：批量加载音频文件

### 特征提取 (`features/extract_features.py`)

- `extract_mfcc`：提取梅尔频率倒谱系数
- `extract_melspectrogram`：提取梅尔频谱
- `extract_chroma`：提取色度特征
- `extract_spectral_contrast`：提取频谱对比度
- `extract_zero_crossing_rate`：提取过零率
- `extract_rms_energy`：提取RMS能量
- `extract_all_features`：提取所有特征并组合
- `extract_features_from_files`：从音频文件批量提取特征

### GMM模型 (`models/gmm_model.py`)

- `GMMModel`：高斯混合模型类，支持训练、预测、概率计算等
- `train_gmm_model`：训练模型的便捷函数
- `find_optimal_components`：通过交叉验证确定最佳组件数量

### 阈值确定 (`models/threshold_detector.py`)

- `ThresholdDetector`：阈值检测器类，支持多种阈值优化方法
- `find_optimal_threshold`：寻找最佳阈值的便捷函数

### 模型评估 (`utils/evaluator.py`)

- `ModelEvaluator`：模型评估器类，支持多种评估指标和可视化
- `evaluate_model`：评估模型的便捷函数

### 配置管理 (`utils/config_manager.py`)

- `ConfigManager`：配置管理器类，支持YAML配置加载和参数访问
- `load_config`：加载配置的便捷函数
- `create_default_config`：创建默认配置的便捷函数

### 实时音频流处理 (`utils/`)

- `realtime_classifier.py`：实时分类器核心逻辑
  - `RealTimeClassifier`：实时分类器类
  - `AudioBuffer`：音频缓冲区类
  - `FrameProcessor`：帧处理器类
- `audio_stream_handler.py`：音频流处理器实现
  - `SoundDeviceStreamHandler`：基于sounddevice的麦克风输入处理器
  - `FileStreamSimulator`：文件流模拟器（用于测试）
- `feature_extractor_wrapper.py`：特征提取器包装类

完整的实时检测系统已实现，支持从麦克风实时捕获音频并检测异常。详见 [实时检测使用指南](REALTIME_DETECTION_GUIDE.md)。

## 示例代码

### 训练模型

```python
from utils.data_loader import load_dataset
from features.extract_features import extract_all_features
from models.gmm_model import train_gmm_model, find_optimal_components
from models.threshold_detector import find_optimal_threshold
from utils.config_manager import load_config

# 加载配置
config = load_config()

# 加载数据集（使用设备类型方式）
train_data, val_data, test_data = load_dataset(
    device_type=config.get('data.device_type', 'fan'),
    base_data_dir=config.get('paths.base_data_dir', '../dev_data'),
    sr=config.get('data.sample_rate')
)

# 或者使用传统方式（向后兼容）
# train_data, val_data, test_data = load_dataset(
#     normal_dir=config.get('paths.normal_data_dir'),
#     anomaly_dir=config.get('paths.anomaly_data_dir'),
#     sr=config.get('data.sample_rate')
# )

# 提取特征
train_features = [extract_all_features(audio, config.get('data.sample_rate')) 
                  for audio, _ in train_data]
train_labels = [label for _, label in train_data]

# 寻找最佳组件数量
optimal_components = find_optimal_components(
    train_features, train_labels,
    min_components=1, max_components=10,
    cv_folds=5
)

# 训练模型
model = train_gmm_model(
    train_features, train_labels,
    n_components=optimal_components
)

# 确定最佳阈值
val_features = [extract_all_features(audio, config.get('data.sample_rate')) 
                for audio, _ in val_data]
val_labels = [label for _, label in val_data]

optimal_threshold = find_optimal_threshold(
    model, val_features, val_labels,
    method='f1_score'
)

print(f"最佳组件数量: {optimal_components}")
print(f"最佳阈值: {optimal_threshold}")
```

### 使用实时分类接口

```python
from utils.realtime_classifier import create_realtime_classifier, AudioStreamHandler
from utils.config_manager import load_config

# 加载配置和模型
config = load_config()
# 注意：这里需要加载已训练好的模型和特征提取器
# model = load_model()
# feature_extractor = FeatureExtractor(config)

# 创建实时分类器
classifier = create_realtime_classifier(
    model=model,
    feature_extractor=feature_extractor,
    config=config
)

# 创建音频流处理器
stream_handler = AudioStreamHandler(
    classifier=classifier,
    sample_rate=config.get('data.sample_rate')
)

# 启动分类器和音频流
classifier.start()
stream_handler.start_stream()

try:
    # 运行一段时间
    import time
    time.sleep(60)  # 运行1分钟
finally:
    # 停止
    stream_handler.stop_stream()
    classifier.stop()
```

## 注意事项

1. 请确保音频文件格式兼容，推荐使用WAV格式
2. 对于实时分类功能，需要额外安装音频输入库并实现具体的音频采集功能
3. 参数调优对于模型性能至关重要，特别是GMM组件数量和检测阈值
4. 建议使用验证集来优化参数，测试集仅用于最终评估
5. 系统已优化偏度和峰度计算的数值稳定性，通过添加微小扰动和警告抑制机制解决精度损失问题

## 许可证

本项目采用MIT许可证。详情请参阅LICENSE文件。