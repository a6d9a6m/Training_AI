import os
import numpy as np
import pickle
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class GMMModel:
    """
    高斯混合模型(GMM)分类器，用于声音分类
    
    该模型为每个类别训练一个GMM，通过比较样本在不同模型中的似然概率进行分类
    """
    
    def __init__(self, n_components=8, covariance_type='diag', random_state=42):
        """
        初始化GMM模型
        
        参数:
        n_components: 每个GMM的组件数量
        covariance_type: 协方差矩阵类型，可选'diag', 'full', 'tied', 'spherical'
        random_state: 随机种子
        """
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.classes = []
        self.trained = False
    
    def fit(self, X, y):
        """
        训练GMM模型
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        y: 标签数组，形状为 (n_samples,)
        """
        # 获取唯一类别
        self.classes = np.unique(y)
        
        # 标准化特征
        X_scaled = self.scaler.fit_transform(X)
        
        # 为每个类别训练一个GMM模型
        for class_label in self.classes:
            # 获取当前类别的样本
            class_samples = X_scaled[y == class_label]
            
            if len(class_samples) == 0:
                raise ValueError(f"类别 {class_label} 没有样本")
            
            # 创建并训练GMM
            gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type=self.covariance_type,
                random_state=self.random_state,
                max_iter=200,
                n_init=3
            )
            gmm.fit(class_samples)
            
            # 保存模型
            self.models[class_label] = gmm
            print(f"类别 {class_label} 的GMM模型训练完成，样本数: {len(class_samples)}")
        
        self.trained = True
        return self
    
    def predict(self, X):
        """
        预测样本类别
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
        y_pred: 预测标签数组
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 预测每个样本的类别
        y_pred = []
        for sample in X_scaled:
            # 计算样本在每个模型中的对数似然概率
            log_likelihoods = {}
            for class_label, gmm in self.models.items():
                log_likelihood = gmm.score_samples(sample.reshape(1, -1))[0]
                log_likelihoods[class_label] = log_likelihood
            
            # 选择对数似然概率最大的类别
            predicted_class = max(log_likelihoods, key=log_likelihoods.get)
            y_pred.append(predicted_class)
        
        return np.array(y_pred)
    
    def predict_proba(self, X):
        """
        预测样本属于每个类别的概率
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        
        返回:
        proba: 概率矩阵，形状为 (n_samples, n_classes)
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 计算每个样本的对数似然概率
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        log_likelihoods = np.zeros((n_samples, n_classes))
        
        for i, sample in enumerate(X_scaled):
            for j, class_label in enumerate(self.classes):
                gmm = self.models[class_label]
                log_likelihoods[i, j] = gmm.score_samples(sample.reshape(1, -1))[0]
        
        # 将对数似然概率转换为概率（使用softmax）
        # 为了数值稳定性，先减去最大值
        log_likelihoods = log_likelihoods - np.max(log_likelihoods, axis=1, keepdims=True)
        exp_likelihoods = np.exp(log_likelihoods)
        proba = exp_likelihoods / np.sum(exp_likelihoods, axis=1, keepdims=True)
        
        return proba
    
    def get_class_likelihood(self, X, class_label):
        """
        获取样本属于指定类别的对数似然概率
        
        参数:
        X: 特征矩阵，形状为 (n_samples, n_features)
        class_label: 类别标签
        
        返回:
        likelihoods: 对数似然概率数组
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        if class_label not in self.models:
            raise ValueError(f"类别 {class_label} 不存在")
        
        # 标准化特征
        X_scaled = self.scaler.transform(X)
        
        # 获取指定类别的GMM模型
        gmm = self.models[class_label]
        
        # 计算对数似然概率
        likelihoods = gmm.score_samples(X_scaled)
        
        return likelihoods
    
    def save(self, save_dir):
        """
        保存模型到文件
        
        参数:
        save_dir: 保存目录
        """
        if not self.trained:
            raise ValueError("模型尚未训练")
        
        # 确保保存目录存在
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型参数
        model_params = {
            'n_components': self.n_components,
            'covariance_type': self.covariance_type,
            'random_state': self.random_state,
            'classes': self.classes,
            'trained': self.trained
        }
        
        with open(os.path.join(save_dir, 'model_params.pkl'), 'wb') as f:
            pickle.dump(model_params, f)
        
        # 保存scaler
        with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # 保存每个类别的GMM模型
        for class_label, gmm in self.models.items():
            with open(os.path.join(save_dir, f'gmm_{class_label}.pkl'), 'wb') as f:
                pickle.dump(gmm, f)
        
        print(f"模型已保存到 {save_dir}")
    
    @classmethod
    def load(cls, load_dir):
        """
        从文件加载模型
        
        参数:
        load_dir: 加载目录
        
        返回:
        model: GMMModel实例
        """
        # 加载模型参数
        with open(os.path.join(load_dir, 'model_params.pkl'), 'rb') as f:
            model_params = pickle.load(f)
        
        # 创建模型实例
        model = cls(
            n_components=model_params['n_components'],
            covariance_type=model_params['covariance_type'],
            random_state=model_params['random_state']
        )
        
        # 加载scaler
        with open(os.path.join(load_dir, 'scaler.pkl'), 'rb') as f:
            model.scaler = pickle.load(f)
        
        # 加载每个类别的GMM模型
        model.models = {}
        for class_label in model_params['classes']:
            with open(os.path.join(load_dir, f'gmm_{class_label}.pkl'), 'rb') as f:
                model.models[class_label] = pickle.load(f)
        
        model.classes = model_params['classes']
        model.trained = model_params['trained']
        
        print(f"模型已从 {load_dir} 加载")
        return model


def train_gmm_model(X_train, y_train, n_components=8, covariance_type='diag', random_state=42):
    """
    训练GMM模型的便捷函数
    
    参数:
    X_train: 训练特征
    y_train: 训练标签
    n_components: GMM组件数量
    covariance_type: 协方差类型
    random_state: 随机种子
    
    返回:
    model: 训练好的GMM模型
    """
    model = GMMModel(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def find_optimal_components(X_train, y_train, min_components=2, max_components=16, cv=3):
    """
    通过交叉验证找到最佳的GMM组件数量
    
    参数:
    X_train: 训练特征
    y_train: 训练标签
    min_components: 最小组件数量
    max_components: 最大组件数量
    cv: 交叉验证折数
    
    返回:
    best_n_components: 最佳组件数量
    best_score: 最佳得分
    scores: 各组件数量对应的得分
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = {}
    
    for n_components in range(min_components, max_components + 1):
        fold_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # 训练模型
            model = GMMModel(n_components=n_components)
            model.fit(X_fold_train, y_fold_train)
            
            # 在验证集上评估
            y_pred = model.predict(X_fold_val)
            accuracy = np.mean(y_pred == y_fold_val)
            fold_scores.append(accuracy)
        
        # 计算平均准确率
        avg_score = np.mean(fold_scores)
        scores[n_components] = avg_score
        print(f"组件数量: {n_components}, 平均准确率: {avg_score:.4f}")
    
    # 找到最佳组件数量
    best_n_components = max(scores, key=scores.get)
    best_score = scores[best_n_components]
    
    print(f"最佳组件数量: {best_n_components}, 最佳准确率: {best_score:.4f}")
    
    return best_n_components, best_score, scores