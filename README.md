# 声音异常检测系统

一个基于机器学习的声音异常检测系统，使用增强版GMM模型和集成学习方法，特别针对工业设备异常声音检测场景优化。

## 核心组件

当前系统主要使用两个核心训练脚本：

1. **train_gmm_with_score_export.py** - 增强版GMM异常检测模型，支持多种特征提取和分数导出
2. **train_ensemble_from_scores.py** - 集成模型训练器，融合多种异常检测算法

## 快速开始

### 安装依赖

```bash
pip install numpy scipy librosa scikit-learn matplotlib seaborn pandas sounddevice
```

### 数据准备

系统需要两类数据：
- 正常样本：用于训练模型
- 异常样本：用于测试和校准

数据目录结构：
```
data/
├── normal/          # 正常样本（训练用）
└── anomaly/         # 异常样本（测试用）
```

### 训练流程

#### 第一步：训练基础GMM模型并导出分数

```powershell
# PowerShell命令
python src/train/train_gmm_with_score_export.py `
    --normal_train_dir data/normal `
    --anomaly_test_dir data/anomaly `
    --use_deep_features `
    --use_ensemble `
    --use_augmentation `
    --output_dir models/saved_models_optimized
```

此命令将：
- 提取深度特征和增强特征
- 应用数据增强提高模型鲁棒性
- 训练GMM模型并导出多种异常检测分数
- 保存模型和分数到指定目录

#### 第二步：训练集成模型

```powershell
# PowerShell命令
python src/train/train_ensemble_from_scores.py `
    --scores_csv models/saved_models_optimized/sample_scores.csv `
    --output_dir models/saved_models_optimized
```

此命令将：
- 读取第一步导出的分数
- 训练多种集成模型（随机森林、逻辑回归、SVM）
- 选择最佳模型并保存

### 实时检测

```bash
python src/realtime/realtime_detection.py --model models/saved_models_optimized/ensemble_model.pkl
```

## 模型特性

### 增强版GMM模型 (train_gmm_with_score_export.py)

- **多种特征提取**：
  - 增强特征集：MFCC、色度、频谱特征、节奏特征等
  - 深度特征：基于梅尔频谱的深度特征提取
  
- **数据增强**：
  - 时间拉伸（6种速率）
  - 音调偏移（6种步长）
  - 噪声添加（3种强度）
  - 音量变化（4种增益）

- **多种分离方法**：
  - Fisher线性判别分析（LDA）
  - 核方法（RBF、多项式）
  - 对比度增强
  - 监督特征变换

- **多算法分数导出**：
  - GMM分数
  - IsolationForest分数
  - OneClassSVM分数
  - 集成分数

### 集成模型 (train_ensemble_from_scores.py)

- **多种集成算法**：
  - 随机森林
  - 逻辑回归
  - SVM（RBF核）
  - 简单平均（基线）

- **自动模型选择**：
  - 基于F1分数选择最佳模型
  - 保存模型参数和评估指标

 **快速链接：**
- 后端 API: http://localhost:5001/api/status
- 前端界面: 直接打开 `web_interface/frontend/index.html`


## 性能优化

系统提供了多种优化方案来解决常见问题，特别是假阳性（正常样本被误判为异常）问题。详细优化指南请参考：`src/docs/MODEL_OPTIMIZATION_GUIDE.md`

### 主要优化方向

1. **调整模型参数**：
   - 降低IsolationForest的敏感度
   - 调整OneClassSVM参数
   - 增加GMM组件数

2. **增强训练数据多样性**：
   - 增加数据增强强度
   - 使用更多训练数据

3. **调整阈值策略**：
   - 使用更保守的阈值
   - 集成模型使用软投票

4. **特征工程优化**：
   - 增加特征鲁棒性
   - 减少过拟合的特征选择

## 项目结构

```
Project/
├── src/
│   ├── train/
│   │   ├── train_gmm_with_score_export.py    # 增强版GMM训练脚本
│   │   └── train_ensemble_from_scores.py     # 集成模型训练脚本
│   ├── predict/
│   │   └── predict_with_ensemble.py          # 预测脚本
│   ├── realtime/
│   │   ├── realtime_detection.py             # 实时检测
│   │   └── realtime_detection_visual.py      # 可视化实时检测
│   ├── features/
│   │   ├── extract_features.py               # 特征提取
│   │   └── deep_features.py                 # 深度特征
│   ├── models/
│   │   ├── gmm_model.py                      # GMM模型
│   │   └── threshold_detector.py             # 阈值检测
│   └── docs/
│       ├── MODEL_OPTIMIZATION_GUIDE.md       # 模型优化指南
│       └── TRAINING_GUIDE.md                 # 训练指南
├── data/                                     # 数据目录
├── models/                                   # 模型保存目录
└── dev_data/                                 # 开发数据
```

## 使用示例

### 完整训练流程

```powershell
# 第一步：训练基础GMM模型
python src/train/train_gmm_with_score_export.py `
    --normal_train_dir data/normal `
    --anomaly_test_dir data/anomaly `
    --use_deep_features `
    --use_ensemble `
    --use_augmentation `
    --separation_method lda `
    --auto_tune `
    --output_dir models/saved_models_optimized

# 第二步：训练集成模型
python src/train/train_ensemble_from_scores.py `
    --scores_csv models/saved_models_optimized/sample_scores.csv `
    --output_dir models/saved_models_optimized

# 第三步：测试模型
python src/predict/predict_with_ensemble.py `
    --base_model models/saved_models_optimized/gmm_with_scores.pkl `
    --ensemble_model models/saved_models_optimized/ensemble_model.pkl `
    --audio_dir data/test
```

### 高级训练选项

```powershell
# 使用更多优化选项
python src/train/train_gmm_with_score_export.py `
    --normal_train_dir data/normal `
    --anomaly_test_dir data/anomaly `
    --use_deep_features `
    --use_ensemble `
    --use_augmentation `
    --separation_method all `
    --kernel_type rbf `
    --contrast_alpha 2.0 `
    --auto_tune `
    --k_features 60 `
    --output_dir models/saved_models_optimized
```

## 常见问题与解决方案

### Q: 模型误报率高（正常样本被误判为异常）？

A: 参考 `src/docs/MODEL_OPTIMIZATION_GUIDE.md` 中的优化方案：

1. 调整IsolationForest的contamination参数从0.1降至0.05
2. 调整OneClassSVM的nu参数从0.1降至0.05
3. 增加GMM组件数到12个
4. 使用更保守的阈值（95%或98%百分位数）

### Q: 训练数据需要多少？

A: 建议至少：
- 正常样本：100个以上
- 异常样本：20个以上（用于测试和校准）

### Q: 如何提高模型性能？

A: 尝试以下方法：
1. 启用数据增强：`--use_augmentation`
2. 使用深度特征：`--use_deep_features`
3. 尝试不同分离方法：`--separation_method all`
4. 自动调优：`--auto_tune`

## 性能指标

训练完成后，系统会输出以下关键指标：

- **分离质量分析**：Cohen's d指标（>0.8表示易分离）
- **重叠度分析**：正常和异常样本分数的重叠程度
- **最优阈值建议**：基于F1分数的阈值
- **集成模型对比**：不同集成算法的性能对比

## 许可证

MIT License
