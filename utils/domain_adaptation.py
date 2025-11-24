import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import pairwise_distances, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, SelectPercentile, RFE, SelectFromModel, SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import joblib
import json
import os
import warnings


def handle_nan(features):
    """
    处理特征中的NaN值
    
    参数:
    features: 特征数组，可能包含NaN值
    
    返回:
    features_imputed: 处理后的特征数组，不含NaN值
    """
    # 创建一个副本以避免修改原始数据
    features_copy = features.copy()
    
    # 用均值填充NaN值
    imputer = SimpleImputer(strategy='mean')
    features_imputed = imputer.fit_transform(features_copy)
    
    # 检查是否还有NaN值（如果所有值都是NaN的情况）
    if np.isnan(features_imputed).any():
        # 用0填充剩余的NaN值
        features_imputed[np.isnan(features_imputed)] = 0
    
    return features_imputed

def calibrate_threshold(model, target_features, target_labels, source_threshold, method='dynamic'):
    """
    在目标域上重新校准分类阈值
    
    参数:
    model: 训练好的模型
    target_features: 目标域特征
    target_labels: 目标域标签
    source_threshold: 源域确定的阈值
    method: 校准方法，'dynamic', 'isotonic', 'f1_optimization', 'percentile'
    
    返回:
    calibrated_threshold: 校准后的阈值
    best_score: 校准后的最佳性能指标
    """
    if target_features is None or target_labels is None or len(target_features) == 0:
        print("警告: 目标域数据不足，无法进行阈值校准")
        return source_threshold, 0.0
    
    # 获取模型对目标域的预测分数
    try:
        # 假设模型有get_anomaly_scores方法
        if hasattr(model, 'get_anomaly_scores'):
            scores = model.get_anomaly_scores(target_features)
        elif hasattr(model, 'score_samples'):
            # 如果是sklearn风格的模型
            scores = -model.score_samples(target_features)  # 假设得分越低越异常
        else:
            # 如果都没有，尝试使用类别概率
            if hasattr(model, 'get_class_likelihood'):
                normal_probs = model.get_class_likelihood(target_features, 0)
                anomaly_probs = model.get_class_likelihood(target_features, 1)
                scores = anomaly_probs - normal_probs
            else:
                raise AttributeError("模型没有可用的分数计算方法")
    except Exception as e:
        print(f"计算目标域分数时出错: {e}")
        return source_threshold, 0.0
    
    # 根据不同的方法进行阈值校准
    if method == 'dynamic':
        # 动态阈值：根据目标域分数分布调整
        target_mean = np.mean(scores)
        target_std = np.std(scores)
        # 基于Z分数调整阈值
        source_mean = target_mean  # 这里简化处理，实际应使用源域统计信息
        source_std = target_std
        z_score = (source_threshold - source_mean) / (source_std + 1e-10)
        calibrated_threshold = target_mean + z_score * target_std
        
        # 使用校准后的阈值计算F1分数
        predictions = (scores > calibrated_threshold).astype(int)
        best_score = f1_score(target_labels, predictions) if len(np.unique(predictions)) > 1 else 0.0
        
    elif method == 'f1_optimization':
        # 在目标域上优化F1分数
        thresholds = np.linspace(min(scores) - 0.1, max(scores) + 0.1, 100)
        best_f1 = 0.0
        calibrated_threshold = source_threshold
        
        for t in thresholds:
            predictions = (scores > t).astype(int)
            # 确保有正样本和负样本
            if len(np.unique(predictions)) > 1:
                current_f1 = f1_score(target_labels, predictions)
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    calibrated_threshold = t
        
        best_score = best_f1
        
    elif method == 'percentile':
        # 基于百分位数的阈值校准
        # 假设异常样本占少数，使用高分位数
        percentile = 95  # 可以根据实际情况调整
        anomaly_percentile = np.percentile(scores, percentile)
        
        # 结合源域阈值和目标域百分位数
        calibrated_threshold = (source_threshold + anomaly_percentile) / 2
        
        predictions = (scores > calibrated_threshold).astype(int)
        best_score = f1_score(target_labels, predictions) if len(np.unique(predictions)) > 1 else 0.0
        
    elif method == 'isotonic':
        # 使用保序回归进行概率校准
        # 首先创建一个简单的分类器包装器
        class ScoreClassifier:
            def __init__(self, scores, labels):
                self.scores = scores
                self.labels = labels
                
            def fit(self, X, y):
                return self
                
            def predict_proba(self, X):
                # 这里应该使用模型对新样本的预测分数
                # 为了演示，我们假设X就是分数
                proba = np.zeros((len(X), 2))
                for i, score in enumerate(X):
                    proba[i, 1] = 1.0 / (1.0 + np.exp(-score))  # Sigmoid转换
                    proba[i, 0] = 1.0 - proba[i, 1]
                return proba
        
        try:
            # 校准分类器
            clf = ScoreClassifier(scores, target_labels)
            calibrated_clf = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
            calibrated_clf.fit(scores.reshape(-1, 1), target_labels)
            
            # 使用校准后的概率来确定阈值
            prob_pos = calibrated_clf.predict_proba(scores.reshape(-1, 1))[:, 1]
            
            # 寻找最优阈值
            thresholds = np.linspace(0, 1, 100)
            best_f1 = 0.0
            calibrated_threshold = 0.5
            
            for t in thresholds:
                predictions = (prob_pos > t).astype(int)
                if len(np.unique(predictions)) > 1:
                    current_f1 = f1_score(target_labels, predictions)
                    if current_f1 > best_f1:
                        best_f1 = current_f1
                        calibrated_threshold = t
            
            best_score = best_f1
        except Exception as e:
            print(f"保序回归校准失败: {e}")
            return source_threshold, 0.0
    else:
        raise ValueError(f"不支持的阈值校准方法: {method}")
    
    print(f"阈值校准: 源域阈值={source_threshold:.6f}, 校准后阈值={calibrated_threshold:.6f}, 最佳F1={best_score:.4f}")
    return calibrated_threshold, best_score

def transfer_threshold_with_unlabeled(source_model, unlabeled_target_features, source_threshold, method='distribution_matching'):
    """
    在只有未标记目标域数据的情况下迁移阈值
    
    参数:
    source_model: 源域训练的模型
    unlabeled_target_features: 未标记的目标域特征
    source_threshold: 源域确定的阈值
    method: 迁移方法，'distribution_matching', 'quantile_mapping', 'mean_adjustment'
    
    返回:
    transferred_threshold: 迁移后的阈值
    """
    if unlabeled_target_features is None or len(unlabeled_target_features) == 0:
        print("警告: 未标记目标域数据不足，无法进行阈值迁移")
        return source_threshold
    
    # 获取模型对未标记目标域的预测分数
    try:
        if hasattr(source_model, 'get_anomaly_scores'):
            target_scores = source_model.get_anomaly_scores(unlabeled_target_features)
        elif hasattr(source_model, 'score_samples'):
            target_scores = -source_model.score_samples(unlabeled_target_features)
        else:
            raise AttributeError("模型没有可用的分数计算方法")
    except Exception as e:
        print(f"计算未标记目标域分数时出错: {e}")
        return source_threshold
    
    # 根据不同方法迁移阈值
    if method == 'distribution_matching':
        # 匹配源域和目标域的分数分布
        # 这里假设我们知道源域的分数分布（通过源域阈值推断）
        # 我们将目标域的分数分布调整为与源域相似
        target_mean = np.mean(target_scores)
        target_std = np.std(target_scores)
        
        # 假设源域阈值对应某个分位数
        # 例如，假设源域阈值对应95%分位数
        percentile = 95
        
        # 在目标域中找到对应的分位数
        transferred_threshold = np.percentile(target_scores, percentile)
        
    elif method == 'quantile_mapping':
        # 分位数映射
        # 计算源域阈值在源域分数中的分位数
        # 这里我们假设源域阈值是基于某种分布的
        # 为简化，我们使用目标域的分位数
        quantiles = np.linspace(0, 100, 101)
        target_quantiles = np.percentile(target_scores, quantiles)
        
        # 找到与源域阈值最接近的分位数
        # 这里我们假设源域阈值对应95%分位数
        transferred_threshold = np.percentile(target_scores, 95)
        
    elif method == 'mean_adjustment':
        # 简单的均值调整
        # 假设目标域分数的均值偏移
        # 我们根据均值差异调整阈值
        # 这里我们需要源域分数的均值，但我们没有，所以使用目标域的统计信息
        target_mean = np.mean(target_scores)
        target_std = np.std(target_scores)
        
        # 简单策略：将阈值设置为目标域均值加上几个标准差
        # 这是一种启发式方法
        num_std = 2.0  # 可以根据经验调整
        transferred_threshold = target_mean + num_std * target_std
        
    else:
        raise ValueError(f"不支持的阈值迁移方法: {method}")
    
    print(f"阈值迁移: 源域阈值={source_threshold:.6f}, 迁移后阈值={transferred_threshold:.6f}")
    return transferred_threshold

def mean_shift_adaptation(source_features, target_features):
    """
    均值偏移适应方法
    将源域特征的均值调整为目标域特征的均值
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    
    返回:
    adapted_features: 适应后的源域特征
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 计算源域和目标域的均值
    source_mean = np.mean(source_features, axis=0)
    target_mean = np.mean(target_features, axis=0)
    
    # 执行均值偏移
    adapted_features = source_features - source_mean + target_mean
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features)

def robust_mean_shift_adaptation(source_features, target_features):
    """
    稳健的均值偏移适应方法
    使用中位数而不是均值，对异常值更不敏感
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    
    返回:
    adapted_features: 适应后的源域特征
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 使用中位数代替均值
    source_median = np.median(source_features, axis=0)
    target_median = np.median(target_features, axis=0)
    
    # 执行中位数偏移
    adapted_features = source_features - source_median + target_median
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features)


def standardization_adaptation(source_features, target_features):
    """
    标准化适应方法
    使用目标域的统计信息对源域特征进行标准化
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    
    返回:
    adapted_features: 适应后的源域特征
    scaler: 用于转换的标准化器
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 使用目标域数据训练标准化器
    scaler = StandardScaler()
    scaler.fit(target_features)
    
    # 对源域特征进行标准化
    adapted_features = scaler.transform(source_features)
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features), scaler

def robust_scaling_adaptation(source_features, target_features):
    """
    稳健缩放适应方法
    使用中位数和四分位距进行缩放，对异常值更不敏感
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    
    返回:
    adapted_features: 适应后的源域特征
    scaler: 用于转换的缩放器
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 使用目标域数据训练稳健缩放器
    scaler = RobustScaler()
    scaler.fit(target_features)
    
    # 对源域特征进行缩放
    adapted_features = scaler.transform(source_features)
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features), scaler


def minmax_normalization_adaptation(source_features, target_features):
    """
    最小-最大归一化适应方法
    使用目标域的统计信息对源域特征进行归一化
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    
    返回:
    adapted_features: 适应后的源域特征
    scaler: 用于转换的归一化器
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 使用目标域数据训练归一化器
    scaler = MinMaxScaler()
    scaler.fit(target_features)
    
    # 对源域特征进行归一化
    adapted_features = scaler.transform(source_features)
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features), scaler


def pca_alignment_adaptation(source_features, target_features, n_components=None):
    """
    PCA对齐适应方法
    将源域和目标域特征投影到公共的主成分空间
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    n_components: 主成分数量，如果为None，则保留所有成分
    
    返回:
    adapted_source: 适应后的源域特征
    adapted_target: 适应后的目标域特征
    pca: 训练好的PCA模型
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 合并源域和目标域特征以训练PCA
    combined_features = np.vstack((source_features, target_features))
    
    # 训练PCA模型
    pca = PCA(n_components=n_components)
    pca.fit(combined_features)
    
    # 对源域和目标域特征进行转换
    adapted_source = pca.transform(source_features)
    adapted_target = pca.transform(target_features)
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_source), handle_nan(adapted_target), pca

def kernel_pca_alignment_adaptation(source_features, target_features, n_components=None, kernel='rbf'):
    """
    核PCA对齐适应方法
    使用核方法将源域和目标域特征投影到非线性的主成分空间
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    n_components: 主成分数量，如果为None，则保留所有成分
    kernel: 核函数类型 ('rbf', 'poly', 'sigmoid', 'cosine')
    
    返回:
    adapted_source: 适应后的源域特征
    adapted_target: 适应后的目标域特征
    kpca: 训练好的核PCA模型
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 合并源域和目标域特征以训练核PCA
    combined_features = np.vstack((source_features, target_features))
    
    # 训练核PCA模型
    kpca = KernelPCA(n_components=n_components, kernel=kernel, gamma=None)
    kpca.fit(combined_features)
    
    # 对源域和目标域特征进行转换
    adapted_source = kpca.transform(source_features)
    adapted_target = kpca.transform(target_features)
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_source), handle_nan(adapted_target), kpca


def coral_adaptation(source_features, target_features, lam=0.1):
    """
    CORAL (Correlation Alignment) 适应方法
    对齐源域和目标域的二阶统计量
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    lam: 正则化参数
    
    返回:
    adapted_features: 适应后的源域特征
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 计算源域和目标域的协方差矩阵
    source_cov = np.cov(source_features, rowvar=False)
    target_cov = np.cov(target_features, rowvar=False)
    
    # 添加正则化
    source_cov = source_cov + lam * np.eye(source_cov.shape[0])
    target_cov = target_cov + lam * np.eye(target_cov.shape[0])
    
    # 计算平方根矩阵
    sqrt_source_cov = matrix_sqrt(source_cov)
    inv_sqrt_target_cov = matrix_sqrt(np.linalg.inv(target_cov))
    
    # 计算适应矩阵
    adapt_matrix = sqrt_source_cov @ inv_sqrt_target_cov
    
    # 执行适应
    adapted_features = source_features @ adapt_matrix
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features)

def joint_distribution_adaptation(source_features, target_features, alpha=0.2, kernel='rbf', gamma=None):
    """
    联合分布适应方法
    同时对齐一阶和二阶统计量，提供更好的领域适应效果
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    alpha: 平衡一阶和二阶统计量的权重 (0-1)
    kernel: 核函数类型 ('rbf', 'linear')
    gamma: RBF核的gamma参数，如果为None，则自动计算
    
    返回:
    adapted_features: 适应后的源域特征
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    # 先执行均值偏移
    mean_adapted = mean_shift_adaptation(source_features, target_features)
    
    # 再执行CORAL适应
    coral_adapted = coral_adaptation(source_features, target_features)
    
    # 加权融合两种适应方法
    adapted_features = alpha * mean_adapted + (1 - alpha) * coral_adapted
    
    # 处理可能产生的NaN值
    return handle_nan(adapted_features)


def calculate_mmd(source_features, target_features, kernel='rbf', gamma=None):
    """
    计算最大均值差异 (Maximum Mean Discrepancy, MMD)
    用于评估源域和目标域分布的差异
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    kernel: 核函数类型 ('rbf', 'linear')
    gamma: RBF核的gamma参数，如果为None，则自动计算
    
    返回:
    mmd_value: MMD值，值越小表示分布越接近
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    if kernel == 'rbf':
        if gamma is None:
            # 自动计算gamma值
            n_samples = min(source_features.shape[0], target_features.shape[0])
            X = np.vstack((source_features[:n_samples], target_features[:n_samples]))
            dists = pairwise_distances(X, X, metric='euclidean')
            gamma = 1.0 / (2 * np.median(dists[dists > 0])**2)
        
        # 计算核矩阵
        K_ss = np.exp(-gamma * pairwise_distances(source_features, source_features, metric='euclidean')**2)
        K_tt = np.exp(-gamma * pairwise_distances(target_features, target_features, metric='euclidean')**2)
        K_st = np.exp(-gamma * pairwise_distances(source_features, target_features, metric='euclidean')**2)
    
    elif kernel == 'linear':
        # 线性核
        K_ss = source_features @ source_features.T
        K_tt = target_features @ target_features.T
        K_st = source_features @ target_features.T
    
    else:
        raise ValueError(f"不支持的核函数类型: {kernel}")
    
    # 计算MMD
    m = source_features.shape[0]
    n = target_features.shape[0]
    
    mmd_value = np.mean(K_ss) - 2 * np.mean(K_st) + np.mean(K_tt)
    
    # 确保结果为非负数
    return max(0, mmd_value)

def select_domain_invariant_features(source_features, source_labels, target_features, method='f_statistic', k=0.8):
    """
    选择领域不变特征
    使用多种特征选择方法筛选在源域和目标域之间表现稳定的特征
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    source_labels: 源域标签数组，形状为 (n_samples,)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    method: 特征选择方法 ('f_statistic', 'mutual_info', 'mmd_based', 'correlation', 'random_forest', 'rfe', 'sequential')
    k: 选择的特征比例或数量，0.8表示选择前80%的特征
    
    返回:
    selected_source_features: 选择后的源域特征
    selected_target_features: 选择后的目标域特征
    selector: 特征选择器
    selected_indices: 选择的特征索引
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    try:
        print(f"使用 {method} 方法进行特征选择")
        
        if method == 'f_statistic':
            # 使用F统计量进行特征选择
            if isinstance(k, float):
                selector = SelectPercentile(f_classif, percentile=int(k * 100))
            else:
                selector = SelectKBest(f_classif, k=k)
            selector.fit(source_features, source_labels)
            
        elif method == 'mutual_info':
            # 使用互信息进行特征选择
            if isinstance(k, float):
                selector = SelectPercentile(mutual_info_classif, percentile=int(k * 100))
            else:
                selector = SelectKBest(mutual_info_classif, k=k)
            selector.fit(source_features, source_labels)
            
        elif method == 'mmd_based':
            # 基于MMD的特征选择，选择MMD值较小的特征
            n_features = source_features.shape[1]
            feature_weights = np.zeros(n_features)
            
            # 对每个特征计算MMD值
            for i in range(n_features):
                feat_source = source_features[:, i].reshape(-1, 1)
                feat_target = target_features[:, i].reshape(-1, 1)
                try:
                    mmd = calculate_mmd(feat_source, feat_target)
                    feature_weights[i] = 1.0 / (mmd + 1e-10)  # 特征的权重与MMD成反比
                except:
                    feature_weights[i] = 0.0
            
            # 排序并选择特征
            if isinstance(k, float):
                n_selected = max(1, int(n_features * k))
            else:
                n_selected = min(k, n_features)
            
            selected_indices = np.argsort(feature_weights)[::-1][:n_selected]
            selected_source_features = source_features[:, selected_indices]
            selected_target_features = target_features[:, selected_indices]
            
            print(f"已选择 {len(selected_indices)} 个领域不变特征")
            return selected_source_features, selected_target_features, None, selected_indices
            
        elif method == 'correlation':
            # 选择与标签相关但域间方差小的特征
            n_features = source_features.shape[1]
            
            # 计算特征与标签的相关性
            corr_with_label = np.zeros(n_features)
            for i in range(n_features):
                corr_with_label[i] = np.abs(np.corrcoef(source_features[:, i], source_labels)[0, 1])
            
            # 计算特征在域间的变化（标准差比率）
            domain_variance = np.zeros(n_features)
            for i in range(n_features):
                source_std = np.std(source_features[:, i]) if np.std(source_features[:, i]) > 0 else 1e-10
                target_std = np.std(target_features[:, i]) if np.std(target_features[:, i]) > 0 else 1e-10
                domain_variance[i] = max(source_std, target_std) / (min(source_std, target_std) + 1e-10)
            
            # 综合评分
            feature_scores = corr_with_label / (domain_variance + 1e-10)
            
            # 排序并选择特征
            if isinstance(k, float):
                n_selected = max(1, int(n_features * k))
            else:
                n_selected = min(k, n_features)
            
            selected_indices = np.argsort(feature_scores)[::-1][:n_selected]
            selected_source_features = source_features[:, selected_indices]
            selected_target_features = target_features[:, selected_indices]
            
            print(f"已选择 {len(selected_indices)} 个领域不变特征")
            return selected_source_features, selected_target_features, None, selected_indices
            
        elif method == 'random_forest':
            # 使用随机森林的特征重要性进行选择
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(source_features, source_labels)
            
            if isinstance(k, float):
                selector = SelectPercentile(score_func=lambda X, y: clf.feature_importances_, 
                                         percentile=int(k * 100))
            else:
                selector = SelectFromModel(clf, max_features=k, prefit=True)
            
        elif method == 'rfe':
            # 递归特征消除
            estimator = SVC(kernel="linear")
            n_selected = int(source_features.shape[1] * k) if isinstance(k, float) else k
            selector = RFE(estimator, n_features_to_select=n_selected, step=1)
            selector.fit(source_features, source_labels)
            
        elif method == 'sequential':
            # 序列特征选择
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            n_selected = int(source_features.shape[1] * k) if isinstance(k, float) else k
            selector = SequentialFeatureSelector(estimator, n_features_to_select=n_selected, direction='forward')
            selector.fit(source_features, source_labels)
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")
        
        # 获取选择的特征索引
        selected_indices = selector.get_support(indices=True)
        
        # 对源域和目标域特征进行选择
        selected_source_features = selector.transform(source_features)
        selected_target_features = selector.transform(target_features)
        
        print(f"已选择 {len(selected_indices)} 个领域不变特征")
        return selected_source_features, selected_target_features, selector, selected_indices
    except Exception as e:
        print(f"特征选择失败，使用原始特征: {e}")
        # 如果特征选择失败，返回原始特征
        return source_features, target_features, None, np.arange(source_features.shape[1])

def ensemble_feature_selection(source_features, source_labels, target_features, methods=['f_statistic', 'mutual_info', 'random_forest'], k=0.8):
    """
    集成多种特征选择方法
    使用投票机制从多个特征选择器中选择最稳定的特征
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    source_labels: 源域标签数组，形状为 (n_samples,)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    methods: 要使用的特征选择方法列表
    k: 每种方法选择的特征比例
    
    返回:
    selected_source_features: 选择后的源域特征
    selected_target_features: 选择后的目标域特征
    selected_indices: 选择的特征索引
    selection_counts: 每个特征被选中的次数
    """
    n_features = source_features.shape[1]
    selection_counts = np.zeros(n_features)
    
    # 对每种方法进行特征选择
    for method in methods:
        try:
            _, _, _, selected_indices = select_domain_invariant_features(
                source_features, source_labels, target_features, method=method, k=k
            )
            selection_counts[selected_indices] += 1
        except Exception as e:
            print(f"方法 {method} 特征选择失败: {e}")
    
    # 确定最终选择的特征（至少被一种方法选中）
    min_selections = max(1, len(methods) // 2)  # 至少需要被一半的方法选中
    selected_indices = np.where(selection_counts >= min_selections)[0]
    
    # 如果没有足够的特征，选择得票最多的特征
    if len(selected_indices) == 0:
        n_selected = max(1, int(n_features * k))
        selected_indices = np.argsort(selection_counts)[::-1][:n_selected]
    
    print(f"集成特征选择: 从 {len(methods)} 种方法中选择了 {len(selected_indices)} 个最稳定的特征")
    
    # 选择特征
    selected_source_features = source_features[:, selected_indices]
    selected_target_features = target_features[:, selected_indices]
    
    return selected_source_features, selected_target_features, selected_indices, selection_counts

def calibration_transfer(source_model, target_unlabeled_features, target_labels=None, method='shift'):
    """
    阈值校准方法
    在目标域未标记或少量标记数据上重新校准决策阈值
    
    参数:
    source_model: 在源域训练好的模型
    target_unlabeled_features: 目标域未标记特征
    target_labels: 目标域标签（如果有）
    method: 校准方法 ('shift', 'scale', 'both')
    
    返回:
    calibration_params: 校准参数
    """
    # 处理NaN值
    target_unlabeled_features = handle_nan(target_unlabeled_features)
    
    # 获取目标域上的预测分数
    if hasattr(source_model, 'get_class_likelihood'):
        try:
            normal_scores = source_model.get_class_likelihood(target_unlabeled_features, 0)
            scores = -normal_scores  # 假设低正常概率表示异常
        except Exception as e:
            print(f"无法获取类别似然，使用默认校准: {e}")
            return {'type': 'none'}
    else:
        # 如果没有get_class_likelihood方法，返回默认校准
        return {'type': 'none'}
    
    # 计算校准参数
    scores_mean = np.mean(scores)
    scores_std = np.std(scores) if len(scores) > 1 else 1.0
    
    calibration_params = {
        'type': method,
        'mean': scores_mean,
        'std': scores_std
    }
    
    print(f"阈值校准参数: {calibration_params}")
    return calibration_params

def apply_calibration(anomaly_scores, calibration_params):
    """
    应用校准参数到异常分数
    
    参数:
    anomaly_scores: 原始异常分数
    calibration_params: 校准参数
    
    返回:
    calibrated_scores: 校准后的异常分数
    """
    if calibration_params['type'] == 'none':
        return anomaly_scores
    
    scores = np.array(anomaly_scores)
    
    if calibration_params['type'] == 'shift':
        # 仅平移
        return scores - calibration_params['mean']
    elif calibration_params['type'] == 'scale':
        # 仅缩放
        return scores / (calibration_params['std'] + 1e-10)
    elif calibration_params['type'] == 'both':
        # 平移和缩放
        return (scores - calibration_params['mean']) / (calibration_params['std'] + 1e-10)
    else:
        return anomaly_scores


def matrix_sqrt(mat):
    """
    计算矩阵的平方根
    使用特征值分解
    
    参数:
    mat: 对称正定矩阵
    
    返回:
    sqrt_mat: 矩阵的平方根
    """
    # 处理NaN值
    mat = handle_nan(mat)
    
    # 进行特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(mat)
    
    # 确保特征值为正数
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    
    # 计算平方根矩阵
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    sqrt_mat = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
    
    # 处理可能产生的NaN值
    return handle_nan(sqrt_mat)


def get_adaptation_method(method_name):
    """
    获取指定名称的适应方法函数
    
    参数:
    method_name: 方法名称
    
    返回:
    adaptation_func: 适应方法函数
    """
    methods = {
        'mean_shift': mean_shift_adaptation,
        'robust_mean_shift': robust_mean_shift_adaptation,
        'standardization': standardization_adaptation,
        'robust_scaling': robust_scaling_adaptation,
        'minmax': minmax_normalization_adaptation,
        'pca': pca_alignment_adaptation,
        'kernel_pca': kernel_pca_alignment_adaptation,
        'coral': coral_adaptation,
        'joint_distribution': joint_distribution_adaptation
    }
    
    if method_name not in methods:
        raise ValueError(f"不支持的适应方法: {method_name}")
    
    return methods[method_name]

def ensemble_adaptation(source_features, target_features, methods=None, weights=None, strategy='weighted_average', adaptation_config=None):
    """
    集成多种领域适应方法的混合策略
    
    Args:
        source_features: 源域特征
        target_features: 目标域特征
        methods: 要使用的适应方法列表
        weights: 各方法的权重，如果为None则根据MMD改善自动计算
        strategy: 集成策略 ('weighted_average', 'dynamic_weighting', 'stacking', 'voting', 'feature_wise')
        adaptation_config: 各方法的配置参数字典
    
    Returns:
        集成后的源域特征
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    if methods is None:
        methods = ['standardization', 'coral', 'mean_shift']
    
    if adaptation_config is None:
        adaptation_config = {}
    
    # 保存每个方法的结果和MMD改善率
    adapted_results = []
    mmd_improvements = []
    method_results = {}
    
    # 计算原始MMD
    original_mmd = calculate_mmd(source_features, target_features, kernel='rbf')
    
    for method_name in methods:
        try:
            method_func = get_adaptation_method(method_name)
            config = adaptation_config.get(method_name, {})
            
            # 应用每个方法
            if method_name == 'pca':
                n_components = config.get('n_components', None)
                adapted_features, _, model = method_func(source_features, target_features, n_components=n_components)
                method_results[method_name] = {'features': adapted_features, 'model': model}
            elif method_name in ['standardization', 'minmax', 'robust_scaling']:
                adapted_features, model = method_func(source_features, target_features)
                method_results[method_name] = {'features': adapted_features, 'model': model}
            else:
                adapted_features = method_func(source_features, target_features)
                method_results[method_name] = {'features': adapted_features}
            
            adapted_results.append(adapted_features)
            
            # 计算MMD改善率
            new_mmd = calculate_mmd(adapted_features, target_features, kernel='rbf')
            improvement = (original_mmd - new_mmd) / original_mmd if original_mmd > 0 else 0
            mmd_improvements.append(max(improvement, 0))  # 确保非负
            
        except Exception as e:
            print(f"应用方法 {method_name} 时出错: {e}")
            adapted_results.append(source_features)  # 失败时使用原始特征
            mmd_improvements.append(0)  # 无改善
    
    if not adapted_results:
        return source_features, {}
    
    # 根据策略进行集成
    if strategy == 'weighted_average':
        # 加权平均策略
        if weights is None:
            # 如果没有提供权重，使用等权重
            weights = np.ones(len(methods)) / len(methods)
        else:
            # 归一化权重
            weights = np.array(weights) / np.sum(weights)
        
        final_features = np.zeros_like(source_features)
        for i, adapted in enumerate(adapted_results):
            final_features += adapted * weights[i]
    
    elif strategy == 'dynamic_weighting':
        # 动态权重策略：根据MMD改善率动态调整权重
        if np.sum(mmd_improvements) > 0:
            weights = np.array(mmd_improvements) / np.sum(mmd_improvements)
        else:
            weights = np.ones(len(methods)) / len(methods)
        
        final_features = np.zeros_like(source_features)
        for i, adapted in enumerate(adapted_results):
            final_features += adapted * weights[i]
    
    elif strategy == 'stacking':
        # 堆叠策略：在不同适应方法的结果上应用第二层变换
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        
        # 水平堆叠所有适应方法的结果
        stacked_features = np.hstack(adapted_results)
        
        # 应用PCA降维到原始维度
        pca = PCA(n_components=min(source_features.shape[1], stacked_features.shape[1]))
        final_features = pca.fit_transform(stacked_features)
    
    elif strategy == 'voting':
        # 投票策略：选择MMD改善最大的前k个方法的平均
        k = min(3, len(methods))  # 选择改善最大的前3个方法
        top_indices = np.argsort(mmd_improvements)[-k:]
        
        final_features = np.mean([adapted_results[i] for i in top_indices], axis=0)
    
    elif strategy == 'feature_wise':
        # 特征级集成：对每个特征维度选择最佳的适应方法
        final_features = np.zeros_like(source_features)
        
        # 计算每个特征维度的差异（领域偏移程度）
        source_mean = np.mean(source_features, axis=0)
        target_mean = np.mean(target_features, axis=0)
        feature_diff = np.abs(source_mean - target_mean)
        
        for i in range(source_features.shape[1]):
            # 对于每个特征，选择在该特征上表现最好的方法
            best_method_idx = 0
            best_feature_improvement = -np.inf
            
            for j, adapted in enumerate(adapted_results):
                # 计算该特征的改善
                adapted_mean = np.mean(adapted, axis=0)
                new_diff = np.abs(adapted_mean[i] - target_mean[i])
                feature_improvement = (feature_diff[i] - new_diff) / (feature_diff[i] + 1e-10)
                
                if feature_improvement > best_feature_improvement:
                    best_feature_improvement = feature_improvement
                    best_method_idx = j
            
            # 使用最佳方法的该特征
            final_features[:, i] = adapted_results[best_method_idx][:, i]
    
    else:
        raise ValueError(f"不支持的集成策略: {strategy}")
    
    return handle_nan(final_features), method_results

def get_best_adaptation_method(source_features, source_labels, target_features, target_labels=None, methods=None):
    """
    根据验证性能自动选择最佳的适应方法
    
    参数:
        source_features: 源域特征
        source_labels: 源域标签
        target_features: 目标域特征
        target_labels: 目标域标签（用于有监督评估）
        methods: 要评估的方法列表
    
    返回:
        best_method: 最佳方法名称
        best_score: 最佳得分
    """
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score
    
    if methods is None:
        methods = ['standardization', 'coral', 'mean_shift', 'pca', 'joint_distribution']
    
    scores = []
    
    # 在源域上训练一个简单的分类器作为评估器
    clf = SVC(kernel='rbf', probability=True)
    clf.fit(source_features, source_labels)
    
    for method_name in methods:
        try:
            method_func = get_adaptation_method(method_name)
            
            # 应用适应方法
            if method_name == 'pca':
                adapted_features, _, _ = method_func(source_features, target_features)
                adapted_target = method_func(target_features, target_features)[0]  # 自转换目标域
            elif method_name in ['standardization', 'minmax', 'robust_scaling']:
                adapted_features, model = method_func(source_features, target_features)
                adapted_target = model.transform(target_features)
            else:
                adapted_features = method_func(source_features, target_features)
                adapted_target = method_func(target_features, target_features)
            
            # 评估MMD改善
            original_mmd = calculate_mmd(source_features, target_features)
            adapted_mmd = calculate_mmd(adapted_features, adapted_target)
            mmd_improvement = (original_mmd - adapted_mmd) / original_mmd if original_mmd > 0 else 0
            
            # 如果有目标域标签，使用分类性能作为评分
            if target_labels is not None and len(target_labels) > 0:
                # 在适应后的特征上重新训练分类器
                adapted_clf = SVC(kernel='rbf', probability=True)
                adapted_clf.fit(adapted_features, source_labels)
                
                # 在目标域上评估
                y_pred = adapted_clf.predict(adapted_target)
                f1 = f1_score(target_labels, y_pred, average='weighted') if len(np.unique(y_pred)) > 1 else 0
                
                # 综合评分：70% F1 + 30% MMD改善
                score = 0.7 * f1 + 0.3 * mmd_improvement
            else:
                # 无监督情况下使用MMD改善作为评分
                score = mmd_improvement
            
            scores.append((method_name, score))
            
        except Exception as e:
            print(f"评估方法 {method_name} 时出错: {e}")
            scores.append((method_name, -np.inf))
    
    if scores:
        best_method, best_score = max(scores, key=lambda x: x[1])
        return best_method, best_score
    else:
        return 'none', -np.inf

# 更新方法字典以包含集成方法
def get_adaptation_method(method_name):
    """
    获取指定名称的适应方法函数
    
    参数:
    method_name: 方法名称
    
    返回:
    adaptation_func: 适应方法函数
    """
    methods = {
        'mean_shift': mean_shift_adaptation,
        'robust_mean_shift': robust_mean_shift_adaptation,
        'standardization': standardization_adaptation,
        'robust_scaling': robust_scaling_adaptation,
        'minmax': minmax_normalization_adaptation,
        'pca': pca_alignment_adaptation,
        'kernel_pca': kernel_pca_alignment_adaptation,
        'coral': coral_adaptation,
        'joint_distribution': joint_distribution_adaptation,
        'ensemble': ensemble_adaptation,
        'threshold_calibration': calibrate_threshold,
        'threshold_transfer': transfer_threshold_with_unlabeled
    }
    
    if method_name not in methods:
        raise ValueError(f"不支持的适应方法: {method_name}")
    
    return methods[method_name]

# 导出阈值校准函数，使其在main.py中可用
class MMDAdaptationTrainer:
    """
    结合MMD领域适应的自动编码器训练器
    用于训练能对齐源域和目标域数据分布的自动编码器
    """
    
    def __init__(self, model, lambda_mmd=0.1):
        """
        初始化MMD适应训练器
        
        参数:
        model: 自动编码器模型
        lambda_mmd: MMD损失的权重系数
        """
        self.model = model
        self.lambda_mmd = lambda_mmd
        self.recon_criterion = nn.MSELoss()
    
    def train_epoch(self, source_loader, target_loader, optimizer, epoch, verbose=True):
        """
        训练一个epoch
        
        参数:
        source_loader: 源域数据加载器（仅正常样本）
        target_loader: 目标域数据加载器（仅正常样本）
        optimizer: 优化器
        epoch: 当前epoch数
        verbose: 是否打印训练信息
        
        返回:
        epoch_loss: 当前epoch的总损失
        recon_loss: 重构损失
        mmd_loss: MMD损失
        """
        self.model.train()
        total_loss = 0
        total_recon_loss = 0
        total_mmd_loss = 0
        
        # 确定迭代次数（取两者中的较大值）
        max_iter = max(len(source_loader), len(target_loader))
        source_iter = iter(source_loader)
        target_iter = iter(target_loader)
        
        for i in range(max_iter):
            try:
                source_batch = next(source_iter)
            except StopIteration:
                source_iter = iter(source_loader)
                source_batch = next(source_iter)
            
            try:
                target_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                target_batch = next(target_iter)
            
            # 提取数据并移动到设备
            source_features = source_batch[0].to(self.model.device)
            target_features = target_batch[0].to(self.model.device)
            
            # 梯度清零
            optimizer.zero_grad()
            
            # 前向传播：源域
            source_encoded, source_recon = self.model(source_features)
            
            # 前向传播：目标域
            target_encoded, target_recon = self.model(target_features)
            
            # 计算重构损失
            recon_loss_source = self.recon_criterion(source_recon, source_features)
            recon_loss_target = self.recon_criterion(target_recon, target_features)
            recon_loss = (recon_loss_source + recon_loss_target) / 2
            
            # 计算MMD损失（在潜在空间中）
            # 使用numpy版本的calculate_mmd函数
            source_encoded_np = source_encoded.detach().cpu().numpy()
            target_encoded_np = target_encoded.detach().cpu().numpy()
            mmd_loss_np = calculate_mmd(source_encoded_np, target_encoded_np, kernel='rbf')
            mmd_loss = torch.tensor(mmd_loss_np, requires_grad=True, device=self.model.device)
            
            # 总损失 = 重构损失 + lambda * MMD损失
            total_batch_loss = recon_loss + self.lambda_mmd * mmd_loss
            
            # 反向传播和优化
            total_batch_loss.backward()
            optimizer.step()
            
            # 累计损失
            batch_size = source_features.size(0) + target_features.size(0)
            total_loss += total_batch_loss.item() * batch_size
            total_recon_loss += recon_loss.item() * batch_size
            total_mmd_loss += mmd_loss.item() * batch_size
        
        # 计算平均损失
        total_samples = len(source_loader.dataset) + len(target_loader.dataset)
        epoch_loss = total_loss / total_samples
        epoch_recon_loss = total_recon_loss / total_samples
        epoch_mmd_loss = total_mmd_loss / total_samples
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}], Total Loss: {epoch_loss:.6f}, '\
                  f'Recon Loss: {epoch_recon_loss:.6f}, '\
                  f'MMD Loss: {epoch_mmd_loss:.6f}')
        
        return epoch_loss, epoch_recon_loss, epoch_mmd_loss
    
    def train(self, source_loader, target_loader, optimizer, epochs=100, verbose=True):
        """
        完整训练过程
        
        参数:
        source_loader: 源域数据加载器（仅正常样本）
        target_loader: 目标域数据加载器（仅正常样本）
        optimizer: 优化器
        epochs: 训练轮数
        verbose: 是否打印训练信息
        
        返回:
        losses: 训练过程中的损失记录
        """
        losses = {
            'total_loss': [],
            'recon_loss': [],
            'mmd_loss': []
        }
        
        for epoch in range(epochs):
            epoch_loss, epoch_recon_loss, epoch_mmd_loss = self.train_epoch(
                source_loader, target_loader, optimizer, epoch, verbose
            )
            
            # 记录损失
            losses['total_loss'].append(epoch_loss)
            losses['recon_loss'].append(epoch_recon_loss)
            losses['mmd_loss'].append(epoch_mmd_loss)
        
        if verbose:
            print('训练完成')
        
        return losses
    
    def set_lambda_mmd(self, lambda_mmd):
        """
        调整MMD损失的权重系数
        
        参数:
        lambda_mmd: 新的MMD损失权重
        """
        self.lambda_mmd = lambda_mmd
        print(f"MMD损失权重已设置为: {lambda_mmd}")

def evaluate_domain_adaptation(model, source_loader, target_loader, layer='encoded'):
    """
    评估域适应的效果
    
    参数:
    model: 训练好的模型
    source_loader: 源域数据加载器
    target_loader: 目标域数据加载器
    layer: 要评估的层，可选 'encoded' (潜在空间) 或 'recon' (重构空间)
    
    返回:
    distance: 适应后的域间距离
    """
    model.eval()
    source_features_list = []
    target_features_list = []
    
    with torch.no_grad():
        # 收集源域特征
        for batch_features, _ in source_loader:
            batch_features = batch_features.to(model.device)
            
            if layer == 'encoded':
                features = model.encode(batch_features)
            elif layer == 'recon':
                _, features = model(batch_features)
            else:
                raise ValueError(f"不支持的层类型: {layer}")
            
            source_features_list.append(features.cpu().numpy())
        
        # 收集目标域特征
        for batch_features, _ in target_loader:
            batch_features = batch_features.to(model.device)
            
            if layer == 'encoded':
                features = model.encode(batch_features)
            elif layer == 'recon':
                _, features = model(batch_features)
            else:
                raise ValueError(f"不支持的层类型: {layer}")
            
            target_features_list.append(features.cpu().numpy())
    
    # 合并特征
    source_features = np.vstack(source_features_list)
    target_features = np.vstack(target_features_list)
    
    # 计算域间距离
    distance = calculate_mmd(source_features, target_features)
    
    print(f"域适应评估结果 - {layer}空间的MMD距离: {distance:.6f}")
    
    return distance

__all__ = ['handle_nan', 'calculate_mmd', 'matrix_sqrt', 'get_adaptation_method', 'MMDAdaptationTrainer', 'evaluate_domain_adaptation',
           'mean_shift_adaptation', 'standardization_adaptation', 'minmax_normalization_adaptation',
           'pca_alignment_adaptation', 'coral_adaptation', 'robust_mean_shift_adaptation',
           'robust_scaling_adaptation', 'kernel_pca_alignment_adaptation', 'joint_distribution_adaptation',
           'select_domain_invariant_features', 'ensemble_feature_selection', 'ensemble_adaptation', 'calibration_transfer',
           'calibrate_threshold', 'transfer_threshold_with_unlabeled', 'get_best_adaptation_method']