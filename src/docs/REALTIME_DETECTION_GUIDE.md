# 实时音频异常检测使用指南

本指南说明如何使用实时音频流异常检测功能。

## 功能特性

- ✅ **实时麦克风输入**: 从麦克风实时捕获音频并检测异常
- ✅ **文件模拟流**: 从音频文件模拟实时流，方便测试
- ✅ **实时警告**: 检测到异常时立即输出警告信息
- ✅ **连续异常判定**: 连续N帧异常才触发警告，避免误报
- ✅ **统计信息**: 实时显示检测统计数据
- ✅ **自定义回调**: 支持自定义警告处理逻辑

## 安装依赖

首先安装实时音频处理所需的额外依赖：

```bash
pip install sounddevice
```

## 快速开始

### 1. 列出可用的音频设备

```bash
python realtime_detection.py --list_devices
```

这会显示所有可用的音频输入设备及其ID。

### 2. 从麦克风进行实时检测

```bash
# 使用默认麦克风
python realtime_detection.py --model models/saved_models/gmm_model.pkl --threshold 0.5

# 指定特定的音频设备
python realtime_detection.py --model models/saved_models/gmm_model.pkl --threshold 0.5 --device 1

# 运行60秒后自动停止
python realtime_detection.py --model models/saved_models/gmm_model.pkl --threshold 0.5 --duration 60
```

### 3. 从音频文件模拟流（推荐用于测试）

```bash
# 从文件模拟实时流
python realtime_detection.py \
    --mode file \
    --audio_file path/to/test_audio.wav \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.5

# 加速播放（2倍速）
python realtime_detection.py \
    --mode file \
    --audio_file path/to/test_audio.wav \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.5 \
    --playback_speed 2.0
```

## 命令行参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config` | string | config/config.yaml | 配置文件路径 |
| `--model` | string | **必需** | 训练好的模型路径 |
| `--threshold` | float | 模型默认 | 分类阈值 |
| `--mode` | string | microphone | 输入模式：microphone 或 file |
| `--audio_file` | string | - | 音频文件路径（file模式必需） |
| `--device` | int | 默认设备 | 音频输入设备ID |
| `--list_devices` | flag | - | 列出所有可用的音频设备 |
| `--duration` | int | 0 | 运行时长（秒），0表示手动停止 |
| `--consecutive_threshold` | int | 3 | 触发警告所需的连续异常帧数 |
| `--chunk_size` | int | 2048 | 音频块大小 |
| `--playback_speed` | float | 1.0 | 文件播放速度（仅file模式） |

## 使用示例

### 示例1: 工厂设备监控

```bash
# 使用训练好的风扇异常检测模型监控实时音频
python realtime_detection.py \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.3 \
    --consecutive_threshold 5 \
    --device 0
```

### 示例2: 测试模型性能

```bash
# 使用测试音频文件快速验证模型
python realtime_detection.py \
    --mode file \
    --audio_file dev_data/fan/source_test/anomaly/test_001.wav \
    --model models/saved_models/autoencoder_model.pth \
    --playback_speed 1.5
```

### 示例3: 长时间监控

```bash
# 监控8小时（28800秒）
python realtime_detection.py \
    --model models/saved_models/gmm_model.pkl \
    --threshold 0.5 \
    --duration 28800 \
    --consecutive_threshold 10
```

## 输出说明

### 正常运行输出

```
==============================================================
实时音频异常检测系统
==============================================================
已加载配置文件: config/config.yaml
正在加载模型: models/saved_models/gmm_model.pkl
已加载GMM模型
使用模型保存的阈值: 0.5
初始化特征提取器...
初始化实时分类器...
初始化音频流处理器 (模式: microphone)...
音频流已启动 - 采样率: 22050 Hz, 通道数: 1
使用设备: 麦克风 (Realtek High Definition Audio)

==============================================================
开始实时检测...
按 Ctrl+C 停止检测
==============================================================

正在监听音频流...
```

### 检测到异常时的警告输出

```
==================================================
⚠️  检测到声音异常！
时间: 14:23:45
异常分数: 0.7234
置信度: 0.7234
连续异常帧数: 5
==================================================
```

### 停止时的统计输出

```
音频流已停止

统计信息:
总处理帧数: 1250
异常帧数: 48
异常率: 3.84%

==============================================================
检测会话统计:
总处理帧数: 1250
异常帧数: 48
异常率: 3.84%
==============================================================

实时检测已结束
```

## 自定义警告处理

您可以在 `realtime_detection.py` 中修改 `custom_alert_callback` 函数来自定义警告处理逻辑：

```python
def custom_alert_callback(result):
    """
    自定义警告回调函数
    """
    # 示例1: 发送邮件通知
    # send_email_alert(result)

    # 示例2: 发送到监控系统
    # monitoring_system.send_alert(result)

    # 示例3: 播放警告声音
    # play_alert_sound()

    # 示例4: 记录到数据库
    # db.log_anomaly(result)

    # 当前实现: 保存到日志文件
    log_file = "anomaly_log.txt"
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',
                                  time.localtime(result['timestamp']))
        f.write(f"{timestamp} - 异常分数: {result['score']:.4f}, "
                f"置信度: {result['confidence']:.4f}\n")
```

## 调优建议

### 1. 调整异常阈值

- **阈值过低**: 会产生较多误报（正常声音被判定为异常）
- **阈值过高**: 可能漏检异常（真正的异常未被检测到）
- 建议使用验证集调优阈值，使F1分数最大化

### 2. 调整连续异常帧数阈值

- `--consecutive_threshold` 参数控制需要连续多少帧异常才触发警告
- **值较小**: 响应更快，但可能产生更多误报
- **值较大**: 更稳定，但响应较慢
- 推荐值: 3-10，根据具体应用场景调整

### 3. 调整音频块大小

- `--chunk_size` 参数控制每次处理的样本数
- **较小值**: 响应更快，计算负载更高
- **较大值**: 响应较慢，计算负载较低
- 推荐值: 1024-4096

### 4. 选择合适的模型

- **GMM模型**: 训练快，推理快，适合实时应用
- **自动编码器**: 可能更准确，但计算量较大
- 实时应用优先考虑GMM模型

## 故障排除

### 问题1: 找不到音频设备

**错误信息**: `PortAudioError: No input device found`

**解决方案**:
1. 检查麦克风是否正确连接
2. 运行 `--list_devices` 查看可用设备
3. 使用 `--device` 参数指定正确的设备ID

### 问题2: 音频质量差/有噪音

**解决方案**:
1. 检查麦克风质量
2. 确保采样率与训练时一致
3. 考虑添加降噪预处理

### 问题3: CPU占用率过高

**解决方案**:
1. 增大 `--chunk_size` 参数
2. 使用较简单的模型（GMM而非自动编码器）
3. 减少特征提取的复杂度

### 问题4: 检测延迟大

**解决方案**:
1. 减小 `--chunk_size` 参数
2. 减小 `--consecutive_threshold` 参数
3. 优化特征提取代码

## 高级用法

### 集成到自己的应用

您可以将实时检测功能集成到自己的Python应用中：

```python
from utils.config_manager import load_config
from utils.feature_extractor_wrapper import FeatureExtractorWrapper
from utils.realtime_classifier import RealTimeClassifier
from utils.audio_stream_handler import SoundDeviceStreamHandler
from models.gmm_model import GMMModel

# 加载配置和模型
config = load_config('config/config.yaml')
model = GMMModel.load('models/saved_models/gmm_model.pkl')

# 创建特征提取器和分类器
feature_extractor = FeatureExtractorWrapper(config)
classifier = RealTimeClassifier(
    model=model,
    feature_extractor=feature_extractor,
    sample_rate=22050,
    threshold=0.5
)

# 自定义警告回调
def my_alert_handler(result):
    print(f"警告！异常分数: {result['score']}")
    # 你的处理逻辑...

# 创建流处理器
stream_handler = SoundDeviceStreamHandler(
    classifier=classifier,
    alert_callback=my_alert_handler,
    consecutive_anomaly_threshold=3
)

# 启动检测
stream_handler.start_stream()

# ... 你的应用逻辑 ...

# 停止检测
stream_handler.stop_stream()
```

## 性能指标

在标准配置下（采样率22050Hz，块大小2048，GMM模型）：

- **延迟**: 约100-200ms
- **CPU占用**: 单核15-25%
- **内存占用**: 约100-200MB

## 相关文件

- `realtime_detection.py`: 主程序
- `utils/realtime_classifier.py`: 实时分类器核心逻辑
- `utils/audio_stream_handler.py`: 音频流处理器
- `utils/feature_extractor_wrapper.py`: 特征提取器包装
- `anomaly_log.txt`: 异常检测日志（自动生成）

## 联系与支持

如遇到问题或有改进建议，请参考项目的README.md文件。
