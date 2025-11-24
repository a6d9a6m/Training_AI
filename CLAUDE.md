# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 项目概述

这是一个**声音异常检测系统**，使用机器学习模型来区分正常和异常音频。系统支持两种模型类型：
- **GMM (高斯混合模型)**: 有监督分类方法
- **Autoencoder (自动编码器)**: 基于重构误差的无监督异常检测

系统包含高级的领域偏移/适应功能，允许在一个领域（如某个声学环境）上训练的模型在不同的目标领域上测试。

## 关键命令

### 运行系统

```bash
# 训练模型（标准模式）
python main.py --mode train --config config/config.yaml

# 领域偏移训练（源域 → 目标域）
python main.py --mode domain_shift --config config/config.yaml --adaptation_method mean_shift

# 评估已训练的模型
python main.py --mode evaluate --model_path models/saved_models/gmm_model.pkl --threshold 0.5

# 预测单个音频文件
python main.py --mode predict --audio_file path/to/audio.wav --model_path models/saved_models/gmm_model.pkl

# 实时音频流异常检测（麦克风输入）
python realtime_detection.py --model models/saved_models/gmm_model.pkl --threshold 0.5

# 从文件模拟实时流（用于测试）
python realtime_detection.py --mode file --audio_file path/to/test.wav --model models/saved_models/gmm_model.pkl

# 可视化版本（实时显示波形和异常状态）
python realtime_detection_visual.py --model models/saved_models/gmm_model.pkl --threshold 0.5
```

### 测试

```bash
# 运行领域适应性能测试
python tests/test_domain_adaptation_performance.py
```

### 依赖项

```bash
pip install numpy scipy librosa scikit-learn pyyaml matplotlib seaborn torch
# 实时音频检测必需:
pip install sounddevice
```

## 架构

### 数据加载系统

系统支持**两种数据组织模式**：

1. **设备类型模式**（推荐）: `config.yaml` → `data.use_device_type: true`
   - 结构: `dev_data/{device_type}/train/{normal|anomaly}/`, `dev_data/{device_type}/source_test/`, `dev_data/{device_type}/target_test/`
   - 通过分离源域和目标域测试集来启用领域偏移训练
   - 支持的设备: fan, pump, slider, valve, gearbox, bearing, others

2. **传统模式**: `config.yaml` → `data.use_device_type: false`
   - 结构: `dev_data/normal/`, `dev_data/anomaly/`
   - 简单的二分类，不考虑领域问题

`utils/data_loader.py` 中的 `load_dataset()` 函数处理两种模式，并返回不同的元组格式：
- 标准模式: `(train_data, val_data, test_data)`
- 领域偏移模式: `(source_train_data, source_test_data, target_train_data, target_test_data)`

### 模型架构

#### 双模型系统

代码库支持通过 `config.yaml` → `model.type` 切换模型：

1. **GMM 模型** (`models/gmm_model.py`):
   - 为每个类别（正常/异常）训练独立的 GMM
   - 通过似然概率比较进行分类
   - 需要有标签的训练数据

2. **自动编码器** (`models/autoencoder.py`):
   - 基于 PyTorch 的深度自动编码器
   - 无监督：仅在正常样本上训练
   - 通过重构误差阈值检测异常
   - 架构：4层编码器/解码器，带有 dropout

两个模型共享相同的预测接口，但内部实现不同：
- GMM: 使用 `predict()` 和基于阈值的评分
- 自动编码器: 使用 `calculate_reconstruction_error()` 和学习得到的阈值

### 特征提取管道

`features/extract_features.py` 提取丰富的音频特征：
- **MFCC**: 梅尔频率倒谱系数及其一阶、二阶差分
- **梅尔频谱图**: 时频表示
- **色度特征**: 音高类特征
- **频谱对比度**: 频率内容变化
- **过零率**: 时域特征
- **RMS 能量**: 信号能量
- **高级统计特征**: 均值、标准差、最小值、最大值、中位数、偏度、峰度、百分位数

**特征缓存**: 系统自动缓存提取的特征（见 `main.py` 中的 `extract_dataset_features()`）。缓存键包含配置哈希值，因此配置更改时会重新生成特征。

### 领域适应系统

`utils/domain_adaptation.py` 提供全面的领域偏移处理：

**核心概念**: 在源域（如某个录音环境）上训练的模型在目标域（不同的声学条件）上可能表现不佳。领域适应对齐域之间的特征分布。

**关键适应方法**:
- `mean_shift`: 将源域均值对齐到目标域
- `standardization`: 在组合域上拟合 StandardScaler
- `minmax`: MinMaxScaler 归一化
- `coral_adaptation`: CORAL（相关对齐）- 对齐协方差
- `joint_distribution_adaptation`: 同时对齐均值和协方差
- `pca`: 降维获取领域不变特征
- `ensemble`: 通过加权平均组合多种方法
- `auto`: 通过交叉验证自动选择最佳方法

**MMD (最大均值差异)**: 用于量化适应前后的领域差距（见 `calculate_mmd()`）。

**阈值校准**: 在源域训练后，决策阈值在目标域上重新校准，使用：
- `f1_optimization`: 网格搜索寻找最佳 F1 分数
- `dynamic`: 基于 Z 分数的统计调整
- `percentile`: 基于目标域分数分布
- `isotonic`: 等渗回归校准

**特征选择**: `select_domain_invariant_features()` 和 `ensemble_feature_selection()` 识别跨域泛化的特征。

### 评估系统

`utils/evaluator.py` 提供全面的模型评估：
- 分类指标：准确率、精确率、召回率、F1
- 基于阈值的方法：ROC 曲线、PR 曲线
- 可视化：混淆矩阵、分数分布

### 配置管理

`utils/config_manager.py` 以点符号访问方式加载 YAML 配置：
```python
config.get('data.sample_rate')  # 返回 22050
config.get('paths.model_dir')  # 返回 './models/saved_models'
```

### 实时音频流检测系统

**新增功能**：完整实现的实时音频流异常检测系统，支持边播放边检测。

#### 核心组件

1. **FeatureExtractorWrapper** (`utils/feature_extractor_wrapper.py`):
   - 包装项目的特征提取函数，适配实时分类器接口
   - 使用与训练时相同的特征配置确保一致性

2. **RealTimeClassifier** (`utils/realtime_classifier.py`):
   - 核心实时分类器，维护音频缓冲区和帧处理器
   - 后台线程持续处理音频帧并进行分类
   - 通过队列提供异步结果访问

3. **AudioStreamHandler** (`utils/audio_stream_handler.py`):
   - **SoundDeviceStreamHandler**: 基于 sounddevice 的麦克风实时输入
   - **FileStreamSimulator**: 从文件模拟实时流（用于测试）
   - 监控线程检测连续异常帧并触发警告

#### 实时检测流程

```
麦克风/文件 → sounddevice回调 → 音频块
                  ↓
        RealTimeClassifier.process_audio_data()
                  ↓
         AudioBuffer（缓冲区管理）
                  ↓
         FrameProcessor（分帧处理）
                  ↓
    FeatureExtractorWrapper（特征提取）
                  ↓
         Model（GMM/Autoencoder）
                  ↓
      ThresholdDetector（阈值判断）
                  ↓
         Result Queue（结果队列）
                  ↓
    Monitor Thread（监控异常并触发警告）
```

#### 关键设计决策

- **缓冲区大小**: 默认2秒（sample_rate * 2），平衡延迟和稳定性
- **帧重叠**: 50% 重叠（hop_length = frame_size // 2），提高检测连续性
- **连续异常判定**: 默认连续3帧异常才触发警告，减少误报
- **线程安全**: 使用 threading.Lock 保护共享数据结构
- **异步处理**: 音频采集和分类在不同线程，避免阻塞

#### 使用脚本

- `realtime_detection.py`: 命令行版本，输出文本警告
- `realtime_detection_visual.py`: 可视化版本，实时显示波形和异常状态

## 重要模式

### 主流程协调

`main.py` 协调整个工作流程。关键函数：

1. **prepare_data()**: 加载音频文件，处理标准和领域偏移两种模式
2. **extract_dataset_features()**: 提取并缓存特征
3. **train_model()**: 根据配置训练 GMM 或自动编码器
4. **determine_threshold()**: 设置决策边界（模型特定逻辑）
5. **evaluate_final_model()**: 计算指标并生成图表
6. **domain_shift_mode()**: 领域适应训练的完整管道

### 特征处理流程

```
音频文件 → load_dataset() → 原始音频 + 标签
           ↓
原始音频 → extract_all_features() → 特征字典
           ↓
特征字典 → extract_dataset_features() → 缓存的特征数组
           ↓
特征数组 → 领域适应（可选） → 适应后的特征
           ↓
适应后的特征 → 模型训练 → 训练好的模型
```

### 阈值确定

阈值行为因模型类型而异：
- **GMM**: `models/threshold_detector.py` 中的 `find_optimal_threshold()` 搜索最佳分离点
- **自动编码器**: 基于百分位数（默认：正常样本重构误差的第 95 百分位）

处理阈值逻辑时，首先使用 `hasattr(model, 'calculate_reconstruction_error')` 检查模型类型。

### 数值稳定性修复

最近的提交解决了偏度/峰度计算中的精度损失问题。添加统计特征时：
- 为接近零方差的特征添加小扰动
- 使用 `warnings.filterwarnings` 抑制数值警告
- 参见 `features/extract_features.py` 中的 `extract_mfcc()` 示例

## 代码组织注意事项

- **模型保存**: GMM 和自动编码器都实现了 `save()`/`load()` 方法，但使用不同的后端（pickle vs PyTorch）
- **标签约定**: 0 = 正常，1 = 异常（整个代码库保持一致）
- **缩放器管理**: 两种模型类型都在内部维护自己的 `StandardScaler`
- **设备处理**: 自动编码器自动检测 CUDA 可用性

## 配置建议

- 切换模型：在 `config.yaml` 中将 `model.type` 改为 'gmm' 或 'autoencoder'
- 启用领域偏移：设置 `data.enable_domain_shift: true` 和 `data.use_device_type: true`
- 禁用特征缓存：设置 `features.use_cache: false`
- 自动编码器配置：调整 `model.autoencoder.hidden_dims`、`model.epochs`、`model.threshold_percentile`
- 领域适应配置：配置 `domain_shift.adaptation_method` 及相关参数

## 常见问题

- 如果特征有 NaN 值，领域适应函数使用 `handle_nan()` 进行填充
- 领域偏移模式需要特定的目录结构，包含 source_test/ 和 target_test/
- 自动编码器仅在正常样本上训练 - 过滤在 `train_model()` 中进行
- 特征缓存失效基于配置哈希自动进行，但如果文件结构改变可能需要手动删除
