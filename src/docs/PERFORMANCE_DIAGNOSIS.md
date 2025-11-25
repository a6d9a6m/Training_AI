# 训练效果诊断报告

## 当前问题总结

经过多次尝试和诊断，确定了训练效果差的根本原因：

### 问题核心
**正常样本和异常样本在音频特征空间中几乎没有区别**

### 诊断数据
- ✅ **数据量充足**: 100个正常样本 + 100个异常样本（足够训练）
- ❌ **特征差异极小**: 归一化特征差异度 = 0.1824（应该 > 0.5）
- ❌ **类别高度重叠**: PCA空间分离比率 < 0.5
- ❌ **模型无法学习**: 准确率 ~52%，F1分数 ~0.52（接近随机猜测）

### 训练结果历史
| 方法 | 准确率 | F1分数 | AUC | 混淆矩阵 |
|------|--------|--------|-----|----------|
| 基础GMM | 50.0% | 0.50 | 0.5575 | 随机分布 |
| 改进GMM (RobustScaler + 特征选择) | 52.5% | 0.5175 | 0.5262 | [[8 12][7 13]] |

---

## 解决方案路径

### 方案1：验证数据标签（最重要！）

**手动听几对样本，确认标签是否正确：**

```bash
# 正常样本
data/normal/section_00_target_test_normal_0000.wav
data/normal/section_00_target_test_normal_0001.wav
data/normal/section_00_target_test_normal_0002.wav

# 异常样本
data/anomaly/section_00_target_test_anomaly_0000.wav
data/anomaly/section_00_target_test_anomaly_0001.wav
data/anomaly/section_00_target_test_anomaly_0002.wav
```

**检查清单：**
- [ ] 正常样本听起来都一样吗？
- [ ] 异常样本和正常样本有明显差异吗？
- [ ] 异常类型是什么？（噪声、频率异常、幅度异常、杂音？）
- [ ] 标签是否可能标注错误？

---

### 方案2：使用异常检测模式（推荐尝试）

如果你的"异常"是微妙的偏离正常模式，监督学习可能不适用。

**异常检测模式的优势：**
- 只学习"正常"的模式
- 任何偏离正常模式的都被视为异常
- 对微妙异常更敏感
- 不需要大量异常样本

**使用方法：**

```bash
# 方法1：只用正常样本训练
python train_anomaly_detection.py \
    --normal_train_dir data/normal \
    --epochs 100 \
    --encoding_dim 32

# 方法2：使用混合测试集评估
python train_anomaly_detection.py \
    --normal_train_dir data/normal \
    --mixed_test_dir data/anomaly \
    --epochs 100
```

**原理：**
- 自动编码器学习重构正常样本
- 异常样本重构误差会很大
- 根据重构误差判断是否异常

---

### 方案3：尝试不同的特征提取方法

当前使用的是标准音频特征（MFCC、梅尔频谱等），可能无法捕捉你的异常模式。

**已创建新的特征提取方法：**

#### 3.1 多分辨率特征
使用不同窗口大小捕捉不同时间尺度的信息：

```bash
python train_with_deep_features.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --feature_type multi_resolution \
    --n_components 5
```

#### 3.2 时间上下文特征
考虑相邻帧之间的关系和变化：

```bash
python train_with_deep_features.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --feature_type temporal_context \
    --n_components 5
```

#### 3.3 组合特征
同时使用两种特征：

```bash
python train_with_deep_features.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --feature_type both \
    --n_components 5
```

---

### 方案4：收集新数据

如果以上方法都不行，可能需要：

1. **确保异常样本确实异常**
   - 异常应该在听感上明显不同
   - 如果连人耳都难以区分，模型更不可能学会

2. **增加异常类型的多样性**
   - 当前异常样本是否类型单一？
   - 考虑收集多种异常情况

3. **提高数据质量**
   - 确保录音清晰
   - 减少背景噪声
   - 统一录音条件

---

## 推荐执行顺序

### 第1步：数据验证（必须做）
手动听 10-20 对样本，确认：
- 标签是否正确
- 正常和异常是否真的不同

### 第2步：尝试异常检测模式
```bash
python train_anomaly_detection.py \
    --normal_train_dir data/normal \
    --epochs 100
```

### 第3步：如果步骤2效果仍不好，尝试深度特征
```bash
python train_with_deep_features.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --feature_type both \
    --n_components 5
```

### 第4步：如果都不行
- 重新检查数据标签
- 考虑收集更明显的异常样本
- 或者重新定义"异常"的标准

---

## 已尝试的优化方法（无效）

✓ 调整GMM组件数量（2-15）
✓ 使用RobustScaler归一化
✓ 特征选择（SelectKBest）
✓ 不同协方差类型
✓ 阈值优化方法

**结论**：技术优化无法解决数据本身的问题。

---

## 诊断工具

### 查看数据诊断报告
```bash
python diagnose_data.py \
    --normal_dir data/normal \
    --anomaly_dir data/anomaly \
    --n_samples 20
```

会生成：
- `data_diagnosis.png` - 特征分布可视化
- 终端输出诊断报告和建议

---

## 需要帮助？

如果尝试了以上方法仍有问题，请提供：
1. 手动听样本后的结论（是否真的不同）
2. 异常的具体类型（噪声、频率、幅度？）
3. 数据的来源和录制方式
4. 实际应用场景的描述
