# 模型优化指南 - 解决假阳性（正常样本被误判为异常）

## 问题诊断

当前模型把**正常样本误判为异常**（假阳性），说明模型过于敏感，对正常模式的学习不够充分。

## 优化方案（按优先级排序）

### 🔥 方案1：调整模型参数（最快见效）

#### 1.1 降低 IsolationForest 的敏感度

在 `train_gmm_with_score_export.py` 第661行：

```python
# 当前（太严格）
iso_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)

# 优化后
iso_forest = IsolationForest(
    n_estimators=200,           # 增加树的数量，提高稳定性
    contamination=0.05,         # 降低到5%，减少误报
    max_samples=256,            # 限制每棵树的样本数
    random_state=42,
    n_jobs=-1
)
```

#### 1.2 调整 OneClassSVM 参数

在 `train_gmm_with_score_export.py` 第665行：

```python
# 当前（边界太紧）
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)

# 优化后
ocsvm = OneClassSVM(
    kernel='rbf',
    gamma='scale',              # 使用 scale 代替 auto，更稳定
    nu=0.05,                    # 降低到5%，放宽边界
    tol=1e-4                    # 提高容忍度
)
```

#### 1.3 增加 GMM 组件数

在 `train_gmm_with_score_export.py` 第652/672行：

```python
# 当前
best_n, best_cov = find_best_gmm_params(train_features_selected, val_features_selected, max_components=8)

# 优化后
best_n, best_cov = find_best_gmm_params(train_features_selected, val_features_selected, max_components=12)
```

更多GMM组件可以更好地捕捉正常数据的多样性。

---

### 🎯 方案2：增强训练数据多样性

#### 2.1 增加数据增强强度

在 `train_gmm_with_score_export.py` 第48-75行，增加更多增强方式：

```python
def augment_audio(audio, sr):
    """增强版数据增强"""
    augmented = [audio]

    # 1. 时间拉伸（更多变化）
    for rate in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
        try:
            audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
            augmented.append(audio_stretched)
        except:
            pass

    # 2. 音调偏移
    for n_steps in [-3, -2, -1, 1, 2, 3]:
        try:
            audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(audio_shifted)
        except:
            pass

    # 3. 添加噪声（多个强度）
    for noise_level in [0.003, 0.005, 0.008]:
        try:
            noise = np.random.normal(0, noise_level, len(audio))
            audio_noisy = audio + noise
            augmented.append(audio_noisy)
        except:
            pass

    # 4. 音量变化
    for gain in [0.8, 0.9, 1.1, 1.2]:
        try:
            audio_gain = audio * gain
            augmented.append(audio_gain)
        except:
            pass

    return augmented
```

#### 2.2 使用更多训练数据

在 `train_gmm_with_score_export.py` 第505行：

```python
# 当前（80%训练，20%测试）
train_val, normal_test = train_test_split(normal_data, test_size=0.2, random_state=42)

# 优化后（90%训练，10%测试）
train_val, normal_test = train_test_split(normal_data, test_size=0.1, random_state=42)
```

---

### 🧪 方案3：调整阈值策略

#### 3.1 使用更保守的阈值

在训练完成后，手动调整阈值。查看 `sample_scores.csv` 中的分数分布：

```python
import pandas as pd
import numpy as np

# 读取分数
df = pd.read_csv('models/saved_models/sample_scores.csv')

# 分析正常样本的分数分布
normal_scores = df[df['label'] == 0]['gmm_score']

# 使用更高的百分位数作为阈值（减少误报）
threshold_95 = np.percentile(normal_scores, 95)  # 当前可能用90%
threshold_98 = np.percentile(normal_scores, 98)  # 更保守
threshold_99 = np.percentile(normal_scores, 99)  # 非常保守

print(f"建议阈值范围: {threshold_95:.2f} - {threshold_99:.2f}")
```

#### 3.2 集成模型使用软投票

修改 `train_ensemble_from_scores.py` 第50行，使用概率阈值：

```python
# 当前（硬判断）
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    n_jobs=-1
)

# 优化后（增加深度，增加树的数量）
rf = RandomForestClassifier(
    n_estimators=300,           # 更多树
    max_depth=10,               # 更深的树
    min_samples_split=10,       # 需要更多样本才分裂
    min_samples_leaf=5,         # 叶节点最少样本数
    class_weight={0: 1, 1: 2},  # 增加异常类的权重（如果异常太少）
    random_state=42,
    n_jobs=-1
)
```

然后在预测时使用概率阈值：

```python
# 在 predict_with_ensemble.py 中
proba = model.predict_proba(X)[0]

# 当前：直接使用 argmax
prediction = model.predict(X)[0]  # 0.5阈值

# 优化后：使用更高的阈值判定为异常
ANOMALY_THRESHOLD = 0.7  # 异常概率要>70%才判为异常
prediction = 1 if proba[1] > ANOMALY_THRESHOLD else 0
```

---

### 🔧 方案4：特征工程优化

#### 4.1 增加特征鲁棒性

在 `train_gmm_with_score_export.py` 第576-580行，使用更鲁棒的缩放器：

```python
# 当前
scaler = RobustScaler()

# 优化后（多种缩放器测试）
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer

# 方案A：使用PowerTransformer处理偏态分布
scaler = PowerTransformer(method='yeo-johnson', standardize=True)

# 方案B：组合缩放器
from sklearn.pipeline import Pipeline
scaler = Pipeline([
    ('robust', RobustScaler()),
    ('power', PowerTransformer(method='yeo-johnson'))
])
```

#### 4.2 减少过拟合的特征选择

在 `train_gmm_with_score_export.py` 第588行：

```python
# 当前
k_features = min(args.k_features, train_features_scaled.shape[1])  # 默认60

# 优化后（选择更少但更稳定的特征）
k_features = min(40, train_features_scaled.shape[1])  # 减少到40

# 或者使用方差阈值预筛选
from sklearn.feature_selection import VarianceThreshold
var_selector = VarianceThreshold(threshold=0.1)  # 去除低方差特征
train_features_scaled = var_selector.fit_transform(train_features_scaled)
```

---

### 📊 方案5：模型集成策略优化

#### 5.1 加权集成（而非简单投票）

创建新文件 `train_weighted_ensemble.py`：

```python
def train_weighted_ensemble(df, score_cols):
    """使用加权平均，给假阳性高的模型降低权重"""
    X = df[score_cols].values
    y = df['label'].values

    # 分析每个模型的假阳性率
    weights = {}
    for col in score_cols:
        scores = df[col].values

        # 找最优阈值
        thresholds = np.linspace(scores.min(), scores.max(), 100)
        best_fpr = 1.0

        for thresh in thresholds:
            preds = (scores > thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y, preds).ravel()
            fpr = fp / (fp + tn)  # 假阳性率

            if fpr < best_fpr:
                best_fpr = fpr

        # 假阳性率越低，权重越高
        weights[col] = 1.0 / (best_fpr + 0.01)

    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}

    print("模型权重（基于假阳性率）:")
    for col, w in weights.items():
        print(f"  {col}: {w:.3f}")

    return weights
```

---

## 快速测试方案

### Step 1: 先调整参数再重新训练

```bash
# 修改 train_gmm_with_score_export.py 中的参数后重新训练
python src/train/train_gmm_with_score_export.py `
    --normal_train_dir data/normal `
    --anomaly_test_dir data/anomaly `
    --use_deep_features `
    --use_ensemble `
    --use_augmentation `
    --output_dir models/saved_models_optimized
```

### Step 2: 重新训练集成模型

```bash
python src/train/train_ensemble_from_scores.py --scores_csv models/saved_models_optimized/sample_scores.csv --output_dir models/saved_models_optimized
```

### Step 3: 测试并对比

```bash
# 测试新模型
python predict_with_ensemble.py \
    --base_model models/saved_models_optimized/gmm_with_scores.pkl \
    --ensemble_model models/saved_models_optimized/ensemble_model.pkl \
    --audio_dir dev_data/fan/train

# 对比旧模型
python predict_with_ensemble.py \
    --base_model models/saved_models/gmm_with_scores.pkl \
    --ensemble_model models/saved_models/ensemble_model.pkl \
    --audio_dir dev_data/fan/train
```

---

## 诊断工具

创建脚本 `diagnose_false_positives.py` 来分析误判样本：

```python
import pandas as pd
import numpy as np

# 读取预测结果
df = pd.read_csv('dev_data/prediction_results.csv')

# 筛选假阳性样本
false_positives = df[(df['true_label'] == 0) & (df['prediction'] == 1)]

print(f"假阳性样本数: {len(false_positives)}")
print(f"假阳性率: {len(false_positives) / len(df[df['true_label']==0]):.2%}")

# 分析分数分布
print("\n假阳性样本的分数特征:")
for col in ['gmm_score', 'iso_score', 'ocsvm_score']:
    if col in false_positives.columns:
        print(f"  {col}: 均值={false_positives[col].mean():.2f}, "
              f"标准差={false_positives[col].std():.2f}")

# 对比正确分类的正常样本
true_negatives = df[(df['true_label'] == 0) & (df['prediction'] == 0)]
print("\n正确分类的正常样本的分数特征:")
for col in ['gmm_score', 'iso_score', 'ocsvm_score']:
    if col in true_negatives.columns:
        print(f"  {col}: 均值={true_negatives[col].mean():.2f}, "
              f"标准差={true_negatives[col].std():.2f}")
```

---

## 推荐优化顺序

1. **立即执行**（30分钟内）：
   - 调整 IsolationForest `contamination=0.05`
   - 调整 OneClassSVM `nu=0.05`
   - 重新训练并测试

2. **短期优化**（1-2小时）：
   - 增强数据增强代码
   - 增加训练数据比例到90%
   - 调整集成模型参数

3. **中期优化**（半天）：
   - 实现加权集成
   - 优化特征选择
   - 交叉验证阈值

4. **长期优化**（1-2天）：
   - 收集更多正常样本
   - 尝试深度学习方法（VAE、Transformer）
   - 实现在线学习/持续学习

---

## 关键指标监控

训练后需要关注：

- **假阳性率（FPR）**：目标 < 5%
- **召回率（Recall）**：目标 > 85%（不能为了降低误报而漏掉真异常）
- **F1分数**：综合指标，目标 > 0.85
- **Cohen's d**：分离度指标，目标 > 0.8

运行 `sample_scores.csv` 分析时查看：
```
可分离性指标（Cohen's d）: X.XXXX
  > 0.8: 大效应（易分离）  ← 目标
  0.5-0.8: 中等效应
  < 0.5: 难分离  ← 当前可能在这里
```
