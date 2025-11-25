# 泛化能力提升完整方案

## 问题诊断

**核心问题**：模型对训练数据相似的样本准确率高（>85%），但对新数据准确率低（64%）。

**根本原因**：**分布偏移（Distribution Shift）** - 训练集和测试集来自不同的数据分布。

## 三大优化策略

### 🚀 策略1：渐进式训练

**原理**：逐步将新数据加入训练集，同时使用域适应技术对齐分布。

**脚本**：`train_progressive.py`

**使用方法**：

```bash
python train_progressive.py \
    --source_normal_dir dev_data/fan/train_old \
    --source_anomaly_dir dev_data/fan/anomaly_old \
    --target_normal_dir dev_data/fan/train_new \
    --target_anomaly_dir dev_data/fan/anomaly_new \
    --output_dir models/saved_models_progressive
```

**工作流程**：

1. **第0轮**：仅在源域（旧数据）训练
   - 建立基础模型

2. **第1轮**：加入20%目标域数据
   - 使用CORAL域适应对齐协方差
   - 重新训练所有模型

3. **第2轮**：加入50%目标域数据
   - 使用JDA（联合分布适应）
   - 进一步扩大模型容量

4. **自动选择最佳轮次**
   - 基于F1分数选择表现最好的模型

**预期效果**：

- ✅ 新数据准确率从64% → **75-80%**
- ✅ 保持旧数据性能不下降
- ✅ 更好的泛化能力

---

### 🎯 策略2：测试时增强（TTA）（推荐★★★★）

**原理**：对测试样本进行轻微变换，对多个版本分别预测，取平均结果。

**脚本**：`predict_with_tta.py`

**使用方法**：

```bash
# 使用TTA预测（5次增强）
python predict_with_tta.py \
    --base_model models/saved_models/gmm_with_scores.pkl \
    --ensemble_model models/saved_models/ensemble_model.pkl \
    --audio_dir dev_data/fan/test_new \
    --n_augmentations 5

# 更高精度（10次增强，但更慢）
python predict_with_tta.py \
    --base_model models/saved_models/gmm_with_scores.pkl \
    --ensemble_model models/saved_models/ensemble_model.pkl \
    --audio_dir dev_data/fan/test_new \
    --n_augmentations 10
```

**工作原理**：

对每个测试样本：
1. 生成N个轻微变换版本（时间拉伸、音高偏移、噪声）
2. 对每个版本独立预测
3. 平均所有预测概率
4. 根据平均概率做最终决策

**预期效果**：

- ✅ 提升准确率 **3-5%**（64% → 67-69%）
- ✅ 显著提升预测稳定性
- ✅ 无需重新训练，即插即用
- ⚠️ 预测速度变慢（N倍）

**适用场景**：
- 离线评估
- 高精度要求的场景
- 少量样本预测

---

### 🔧 策略3：组合优化（推荐★★★★★）

**最佳方案**：结合渐进式训练 + TTA

```bash
# Step 1: 渐进式训练
python train_progressive.py \
    --source_normal_dir dev_data/fan/train_old \
    --source_anomaly_dir dev_data/fan/anomaly_old \
    --target_normal_dir dev_data/fan/train_new \
    --target_anomaly_dir dev_data/fan/anomaly_new \
    --output_dir models/saved_models_progressive

# Step 2: 使用TTA测试
python predict_with_tta.py \
    --base_model models/saved_models_progressive/gmm_progressive.pkl \
    --ensemble_model models/saved_models_progressive/ensemble_progressive.pkl \
    --audio_dir dev_data/fan/test_new \
    --n_augmentations 5
```

**预期综合效果**：

| 指标 | 当前 | 渐进训练 | +TTA | 目标 |
|------|------|---------|------|------|
| 准确率 | 64% | 75-80% | **80-85%** | >85% |
| F1分数 | 63.5% | 73-78% | **78-83%** | >85% |
| 泛化能力 | 差 | 中 | **好** | 优秀 |

---

## 进阶优化方案

### 方案4：数据收集策略

如果上述方案仍达不到目标，说明**训练数据不足**。

#### 4.1 主动学习（Active Learning）

自动选择最不确定的样本请人工标注：

```python
# 在 predict_with_tta.py 的结果中筛选
import pandas as pd

results = pd.read_csv('prediction_results.csv')

# 筛选置信度低的样本（模型不确定）
uncertain = results[(results['confidence'] < 0.7) | (results['consistency'] < 0.8)]

print(f"需要人工标注的样本: {len(uncertain)}")
print(uncertain[['file', 'confidence', 'consistency']])
```

#### 4.2 半监督学习

使用大量无标签数据：

```python
# 对无标签数据进行聚类
from sklearn.cluster import KMeans

# 提取所有无标签数据的特征
unlabeled_features = extract_features(unlabeled_data)

# 聚类找到不同的模式
kmeans = KMeans(n_clusters=10)
clusters = kmeans.fit_predict(unlabeled_features)

# 从每个聚类中采样标注
for cluster_id in range(10):
    cluster_samples = unlabeled_data[clusters == cluster_id]
    # 选择靠近聚类中心的样本进行标注
```

---

### 方案5：模型架构升级

如果数据量足够（>5000样本），考虑深度学习：

#### 5.1 Variational Autoencoder (VAE)

```python
# 优势：
# - 学习正常数据的潜在分布
# - 生成新的正常样本进行数据增强
# - 更好的泛化能力

# 参考：models/autoencoder.py
# 需要修改为VAE架构
```

#### 5.2 Transformer-based模型

```python
# 优势：
# - 捕捉长时序依赖
# - 预训练+微调范式
# - SOTA性能

# 需要：
# - 更多数据（>10000样本）
# - GPU资源
```

---

## 快速决策树

```
当前准确率64%
    ↓
【立即执行】使用TTA预测测试集
    ↓
准确率>70%？
    ├─ 是 → 继续用TTA，定期重新训练
    └─ 否 ↓
        使用渐进式训练
            ↓
        准确率>75%？
            ├─ 是 → 结合TTA达到80-85%，满足要求
            └─ 否 ↓
                数据量>3000？
                    ├─ 是 → 尝试深度学习方案
                    └─ 否 → 执行主动学习收集更多数据
```

---

## 实施时间表

### 第1天：快速验证（2小时）

```bash
# 测试TTA效果
python predict_with_tta.py \
    --base_model models/saved_models/gmm_with_scores.pkl \
    --ensemble_model models/saved_models/ensemble_model.pkl \
    --audio_dir dev_data/fan/test_new \
    --n_augmentations 5
```

**预期**：准确率提升3-5%

---

### 第2-3天：渐进式训练（1天）

```bash
# 准备数据
# - 将数据分为源域（旧数据）和目标域（新数据）
# - 确保目录结构：
#   - source_normal_dir
#   - source_anomaly_dir
#   - target_normal_dir
#   - target_anomaly_dir

# 运行渐进式训练
python train_progressive.py \
    --source_normal_dir dev_data/fan/train_old \
    --source_anomaly_dir dev_data/fan/anomaly_old \
    --target_normal_dir dev_data/fan/train_new \
    --target_anomaly_dir dev_data/fan/anomaly_new
```

**预期**：准确率提升至75-80%

---

### 第4天：组合验证（0.5天）

```bash
# 使用新模型+TTA测试
python predict_with_tta.py \
    --base_model models/saved_models_progressive/gmm_progressive.pkl \
    --ensemble_model models/saved_models_progressive/ensemble_progressive.pkl \
    --audio_dir dev_data/fan/test_new \
    --n_augmentations 5
```

**预期**：准确率达到80-85%

---

### 第5天起：持续优化

如果仍未达标：
1. 分析错误样本特征
2. 收集更多相似数据
3. 考虑深度学习方案

---

## 关键指标监控

### 训练阶段监控

```python
# 在渐进式训练中
print(f"第0轮（源域）准确率: {acc_0:.2%}")
print(f"第1轮（+20%目标域）准确率: {acc_1:.2%}")  # 应该>acc_0
print(f"第2轮（+50%目标域）准确率: {acc_2:.2%}")  # 应该>acc_1

# 域距离（MMD）
print(f"域距离: {mmd:.4f}")  # 应该随训练轮次降低
```

### 测试阶段监控

```python
# 在TTA预测中
print(f"平均置信度: {avg_confidence:.2%}")  # >80%为佳
print(f"平均一致性: {avg_consistency:.2%}")  # >90%为佳

# 错误分类的置信度
print(f"错误样本平均置信度: {wrong_confidence:.2%}")  # 应该显著低于正确样本
```

---

## 常见问题

### Q1: 渐进式训练后，旧数据性能下降了怎么办？

A: **灾难性遗忘**问题。解决方法：
```python
# 在第2轮训练时增加源域样本权重
sample_weights = np.ones(len(combined_train))
sample_weights[:len(source_train)] = 1.5  # 源域权重1.5倍

rf.fit(X, y, sample_weight=sample_weights)
```

### Q2: TTA预测太慢，能加速吗？

A: 有几个方法：
1. 减少增强次数：`--n_augmentations 3`（准确率略降）
2. 只对不确定样本用TTA：
   ```python
   if base_confidence < 0.8:
       result = predict_with_tta(audio)
   else:
       result = predict_normal(audio)
   ```
3. 批量并行处理（需修改代码）

### Q3: 如何判断是否需要收集更多数据？

A: 看学习曲线：
```python
# 绘制不同数据量下的准确率
data_sizes = [100, 200, 500, 1000, 2000]
accuracies = []  # 对应的准确率

# 如果曲线已饱和（增加数据不再提升），说明是模型问题
# 如果曲线仍在上升，说明需要更多数据
```

---

## 总结

**最优路径**：

1. ✅ 立即使用 **TTA** 验证效果（2小时）
2. ✅ 运行 **渐进式训练**（1天）
3. ✅ 结合两者达到目标性能（80-85%）
4. 🔄 建立持续学习机制，定期用新数据重新训练

**投入产出比**：
- TTA: 投入低，收益中（+3-5%）
- 渐进式训练: 投入中，收益高（+10-15%）
- 深度学习: 投入高，收益不确定

**建议优先级**：TTA（立即） > 渐进式训练（1天内） > 数据收集（1周内） > 深度学习（1个月内）
