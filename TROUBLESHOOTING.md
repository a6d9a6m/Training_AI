# 常见问题修复

## 已修复的问题

### 1. ❌ 错误：`ValueError: 不支持的优化方法: f1_score`

**原因**: `find_optimal_threshold`函数的方法参数错误

**已修复**: 将`method='f1_score'`改为`method='f1'`

**支持的方法**:
- `'f1'` - 基于F1分数优化（推荐）
- `'precision_recall'` - 基于精确率-召回率曲线
- `'roc'` - 基于ROC曲线

---

### 2. ❌ 错误：`ModuleNotFoundError: No module named 'pywt'`

**原因**: PyWavelets库未安装（小波特征提取的可选依赖）

**已修复**: 将pywt改为可选依赖，未安装时跳过小波特征

**如果需要小波特征**（可选，不影响训练）:
```bash
pip install PyWavelets
```

---

## 确认修复

运行以下命令确认修复成功：

```bash
# 查看帮助信息
python train_supervised.py --help

# 查看异常检测训练脚本
python train_anomaly_detection.py --help
```

---

## 快速开始训练

### 方式1：监督学习训练

```bash
# 1. 准备数据结构
mkdir -p data/normal data/anomaly

# 2. 将WAV文件放入对应目录
# data/normal/xxx.wav - 正常样本
# data/anomaly/xxx.wav - 异常样本

# 3. 开始训练
python train_supervised.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --n_components 5
```

### 方式2：异常检测训练

```bash
# 1. 准备数据
mkdir -p data/normal_train

# 2. 将正常样本WAV文件放入目录
# data/normal_train/xxx.wav

# 3. 开始训练
python train_anomaly_detection.py \
    --normal_train_dir data/normal_train \
    --epochs 100
```

---

## 训练输出

训练成功后会在`models/saved_models/`目录下生成：

**监督学习**:
- `supervised_gmm_model.pkl` - 模型文件
- `supervised_gmm_model_info.json` - 训练信息

**异常检测**:
- `anomaly_detection_model.pth` - 模型文件
- `anomaly_detection_model_info.json` - 训练信息

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

### 可视化版本

```bash
python realtime_detection_visual.py \
    --model models/saved_models/supervised_gmm_model.pkl
```

---

## 其他常见问题

### Q: 训练时出现内存不足？

**解决方案**:
1. 减少训练样本数量
2. 对于异常检测，减少`--batch_size`参数
3. 对于监督学习，减少`--n_components`参数

### Q: 训练速度太慢？

**解决方案**:
1. 异常检测：减少`--epochs`参数（如50）
2. 检查音频文件是否过大
3. 使用GPU（自动编码器模型）

### Q: 模型准确率不高？

**解决方案**:
1. 增加训练样本数量（建议每类至少50个）
2. 确保训练数据质量（清晰、无噪声）
3. 调整模型参数（n_components或encoding_dim）
4. 确保正常和异常样本有明显差异

### Q: 实时检测误报太多？

**解决方案**:
```bash
python realtime_detection.py \
    --model models/saved_models/xxx_model.pkl \
    --consecutive_threshold 5  # 增加连续异常帧数阈值
```

---

## 需要帮助？

查看详细文档：
- **训练指南**: `TRAINING_GUIDE.md`
- **实时检测指南**: `REALTIME_DETECTION_GUIDE.md`
- **项目说明**: `README.md`
