import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.metrics import pairwise_distances, f1_score
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibratedClassifierCV


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

def select_domain_invariant_features(source_features, source_labels, target_features, k='all'):
    """
    选择领域不变特征
    使用特征选择方法筛选在源域和目标域之间表现稳定的特征
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    source_labels: 源域标签数组，形状为 (n_samples,)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    k: 选择的特征数量，'all'表示选择所有特征
    
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
        # 使用F统计量进行特征选择
        selector = SelectKBest(f_classif, k=k)
        selector.fit(source_features, source_labels)
        
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

def ensemble_adaptation(source_features, target_features, methods=None):
    """
    集成多种适应方法的混合策略
    对多种适应方法的结果进行加权融合
    
    参数:
    source_features: 源域特征数组，形状为 (n_samples, n_features)
    target_features: 目标域特征数组，形状为 (n_samples, n_features)
    methods: 要集成的方法列表，如果为None，使用默认方法集
    
    返回:
    ensemble_features: 集成适应后的源域特征
    adaptations: 各方法的适应结果字典
    """
    # 处理NaN值
    source_features = handle_nan(source_features)
    target_features = handle_nan(target_features)
    
    if methods is None:
        # 默认集成方法
        methods = ['mean_shift', 'coral', 'joint_distribution']
    
    adaptations = {}
    
    try:
        # 应用各种适应方法
        for method_name in methods:
            try:
                adapt_func = get_adaptation_method(method_name)
                
                # 处理不同返回值的适应方法
                if method_name in ['standardization', 'minmax', 'robust_scaling']:
                    adapted, _ = adapt_func(source_features, target_features)
                elif method_name in ['pca', 'kernel_pca']:
                    adapted, _, _ = adapt_func(source_features, target_features)
                else:
                    adapted = adapt_func(source_features, target_features)
                
                adaptations[method_name] = adapted
                print(f"已应用 {method_name} 适应方法")
            except Exception as e:
                print(f"应用 {method_name} 时出错: {e}")
        
        # 如果没有成功的适应方法，返回原始特征
        if not adaptations:
            print("所有适应方法都失败，返回原始特征")
            return source_features, {}
        
        # 计算MMD并选择最佳权重
        weights = {}
        total_mmd = 0
        
        for method_name, adapted_features in adaptations.items():
            try:
                # 计算MMD，使用MMD的倒数作为权重
                mmd = calculate_mmd(adapted_features, target_features)
                # 避免除零错误
                weights[method_name] = 1.0 / (mmd + 1e-10)
                total_mmd += weights[method_name]
            except Exception:
                weights[method_name] = 1.0  # 默认权重
                total_mmd += 1.0
        
        # 归一化权重
        for method_name in weights:
            weights[method_name] /= total_mmd
        
        print(f"适应方法权重: {weights}")
        
        # 加权融合
        ensemble_features = np.zeros_like(source_features)
        for method_name, adapted_features in adaptations.items():
            ensemble_features += weights[method_name] * adapted_features
        
        return handle_nan(ensemble_features), adaptations
    except Exception as e:
        print(f"集成适应失败: {e}")
        return source_features, {}

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
__all__ = ['handle_nan', 'calculate_mmd', 'matrix_sqrt', 'get_adaptation_method',
           'mean_shift_adaptation', 'standardization_adaptation', 'minmax_normalization_adaptation',
           'pca_alignment_adaptation', 'coral_adaptation', 'robust_mean_shift_adaptation',
           'robust_scaling_adaptation', 'kernel_pca_alignment_adaptation', 'joint_distribution_adaptation',
           'select_domain_invariant_features', 'ensemble_adaptation', 'calibration_transfer',
           'calibrate_threshold', 'transfer_threshold_with_unlabeled']