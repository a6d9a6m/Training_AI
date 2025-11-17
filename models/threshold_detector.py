import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score, roc_curve, auc


class ThresholdDetector:
    """
    阈值检测器，用于确定声音分类的最佳阈值
    
    支持基于验证集的F1分数优化，ROC曲线分析等方法
    """
    
    def __init__(self, model=None):
        """
        初始化阈值检测器
        
        参数:
        model: 训练好的模型，需要支持get_class_likelihood方法
        """
        self.model = model
        self.best_threshold = None
        self.best_score = None
        self.threshold_scores = None
    
    def set_model(self, model):
        """
        设置模型
        
        参数:
        model: 训练好的模型
        """
        self.model = model
        return self
    
    def optimize_threshold_f1(self, X_val, y_val, n_thresholds=100, normal_class=0, anomaly_class=1):
        """
        基于F1分数优化阈值
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        n_thresholds: 尝试的阈值数量
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        
        返回:
        best_threshold: 最佳阈值
        best_f1: 最佳F1分数
        thresholds: 尝试的阈值列表
        f1_scores: 对应阈值的F1分数列表
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        # 获取正常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X_val, normal_class)
        
        # 检查模型是否包含异常类
        try:
            # 尝试获取异常类的似然概率
            anomaly_likelihoods = self.model.get_class_likelihood(X_val, anomaly_class)
            # 计算异常分数 (异常类似然 - 正常类似然)
            anomaly_scores = anomaly_likelihoods - normal_likelihoods
        except ValueError:
            # 如果模型只有正常类，使用负的正常类似然作为异常分数
            print(f"警告: 模型中不存在异常类 {anomaly_class}，使用单类别模式")
            anomaly_scores = -normal_likelihoods
        
        # 生成候选阈值
        min_score = np.min(anomaly_scores)
        max_score = np.max(anomaly_scores)
        thresholds = np.linspace(min_score, max_score, n_thresholds)
        
        # 计算每个阈值的F1分数
        f1_scores = []
        for threshold in thresholds:
            # 应用阈值进行预测
            y_pred = (anomaly_scores >= threshold).astype(int)
            # 计算F1分数
            f1 = f1_score(y_val, y_pred, pos_label=anomaly_class)
            f1_scores.append(f1)
        
        # 找到最佳阈值
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        
        # 保存结果
        self.best_threshold = best_threshold
        self.best_score = best_f1
        self.threshold_scores = {
            'thresholds': thresholds,
            'f1_scores': f1_scores,
            'anomaly_scores': anomaly_scores
        }
        
        print(f"最佳阈值: {best_threshold:.4f}")
        print(f"最佳F1分数: {best_f1:.4f}")
        
        return best_threshold, best_f1, thresholds, f1_scores
    
    def optimize_threshold_precision_recall(self, X_val, y_val, target_metric='f1', 
                                           normal_class=0, anomaly_class=1):
        """
        基于精确率-召回率曲线优化阈值
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        target_metric: 目标指标，可选'f1', 'precision', 'recall'
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        
        返回:
        best_threshold: 最佳阈值
        best_metric: 最佳指标值
        precision: 精确率数组
        recall: 召回率数组
        thresholds: 阈值数组
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        # 获取正常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X_val, normal_class)
        
        # 获取异常类的似然概率
        anomaly_likelihoods = self.model.get_class_likelihood(X_val, anomaly_class)
        
        # 计算异常分数
        anomaly_scores = anomaly_likelihoods - normal_likelihoods
        
        # 计算精确率-召回率曲线
        precision, recall, thresholds = precision_recall_curve(
            y_val, anomaly_scores, pos_label=anomaly_class
        )
        
        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 根据目标指标选择最佳阈值
        if target_metric == 'f1':
            best_idx = np.argmax(f1_scores)
            best_metric = f1_scores[best_idx]
        elif target_metric == 'precision':
            best_idx = np.argmax(precision)
            best_metric = precision[best_idx]
        elif target_metric == 'recall':
            best_idx = np.argmax(recall)
            best_metric = recall[best_idx]
        else:
            raise ValueError(f"不支持的目标指标: {target_metric}")
        
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        # 保存结果
        self.best_threshold = best_threshold
        self.best_score = best_metric
        
        print(f"基于{target_metric}优化的最佳阈值: {best_threshold:.4f}")
        print(f"最佳{target_metric}值: {best_metric:.4f}")
        
        return best_threshold, best_metric, precision, recall, thresholds
    
    def optimize_threshold_roc(self, X_val, y_val, target='youden', 
                             normal_class=0, anomaly_class=1):
        """
        基于ROC曲线优化阈值
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        target: 目标优化方法，可选'youden'(约登指数), 'closest_to_100'(最接近左上角)
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        
        返回:
        best_threshold: 最佳阈值
        best_metric: 最佳指标值
        fpr: 假阳性率
        tpr: 真阳性率
        thresholds: 阈值数组
        roc_auc: ROC曲线下面积
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        # 获取正常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X_val, normal_class)
        
        # 获取异常类的似然概率
        anomaly_likelihoods = self.model.get_class_likelihood(X_val, anomaly_class)
        
        # 计算异常分数
        anomaly_scores = anomaly_likelihoods - normal_likelihoods
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_val, anomaly_scores, pos_label=anomaly_class)
        roc_auc = auc(fpr, tpr)
        
        # 根据目标选择最佳阈值
        if target == 'youden':
            # 约登指数 = 敏感性 + 特异性 - 1 = TPR - FPR
            youden = tpr - fpr
            best_idx = np.argmax(youden)
            best_metric = youden[best_idx]
        elif target == 'closest_to_100':
            # 找到最接近(0,1)的点
            distances = np.sqrt(fpr**2 + (1 - tpr)** 2)
            best_idx = np.argmin(distances)
            best_metric = 1 - distances[best_idx]
        else:
            raise ValueError(f"不支持的目标方法: {target}")
        
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
        
        # 保存结果
        self.best_threshold = best_threshold
        self.best_score = best_metric
        
        print(f"基于ROC优化的最佳阈值: {best_threshold:.4f}")
        print(f"最佳{target}指标值: {best_metric:.4f}")
        print(f"ROC曲线下面积(AUC): {roc_auc:.4f}")
        
        return best_threshold, best_metric, fpr, tpr, thresholds, roc_auc
    
    def apply_threshold(self, X, threshold=None, normal_class=0, anomaly_class=1):
        """
        应用阈值进行预测
        
        参数:
        X: 输入特征
        threshold: 使用的阈值，如果为None则使用已确定的最佳阈值
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        
        返回:
        y_pred: 预测标签
        anomaly_scores: 异常分数
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        if threshold is None:
            threshold = self.best_threshold
            if threshold is None:
                raise ValueError("阈值尚未确定，请先运行优化方法")
        
        # 获取正常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X, normal_class)
        
        # 获取异常类的似然概率
        anomaly_likelihoods = self.model.get_class_likelihood(X, anomaly_class)
        
        # 计算异常分数
        anomaly_scores = anomaly_likelihoods - normal_likelihoods
        
        # 应用阈值进行预测
        y_pred = np.where(anomaly_scores >= threshold, anomaly_class, normal_class)
        
        return y_pred, anomaly_scores
    
    def plot_threshold_performance(self, save_path=None):
        """
        绘制阈值性能曲线
        
        参数:
        save_path: 保存路径，如果为None则显示图形
        """
        if self.threshold_scores is None:
            raise ValueError("尚未运行阈值优化，请先调用optimize_threshold_f1方法")
        
        thresholds = self.threshold_scores['thresholds']
        f1_scores = self.threshold_scores['f1_scores']
        
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, f1_scores, 'b-', linewidth=2)
        
        # 标记最佳阈值
        if self.best_threshold is not None:
            plt.axvline(x=self.best_threshold, color='r', linestyle='--', 
                       label=f'最佳阈值: {self.best_threshold:.4f}')
            plt.axhline(y=self.best_score, color='g', linestyle='--',
                       label=f'最佳F1: {self.best_score:.4f}')
        
        plt.xlabel('阈值')
        plt.ylabel('F1分数')
        plt.title('阈值与F1分数关系图')
        plt.grid(True)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图形已保存到 {save_path}")
        else:
            plt.show()
    
    def plot_roc_curve(self, X_val, y_val, save_path=None, normal_class=0, anomaly_class=1):
        """
        绘制ROC曲线
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        save_path: 保存路径，如果为None则显示图形
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        # 获取正常类和异常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X_val, normal_class)
        anomaly_likelihoods = self.model.get_class_likelihood(X_val, anomaly_class)
        
        # 计算异常分数
        anomaly_scores = anomaly_likelihoods - normal_likelihoods
        
        # 计算ROC曲线
        fpr, tpr, thresholds = roc_curve(y_val, anomaly_scores, pos_label=anomaly_class)
        roc_auc = auc(fpr, tpr)
        
        # 绘制ROC曲线
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        
        # 标记最佳阈值对应的点
        if self.best_threshold is not None:
            # 找到最佳阈值对应的fpr和tpr
            best_idx = np.argmin(np.abs(thresholds - self.best_threshold))
            if best_idx < len(fpr):
                plt.plot(fpr[best_idx], tpr[best_idx], 'ro', 
                        label=f'最佳阈值点 ({fpr[best_idx]:.4f}, {tpr[best_idx]:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假阳性率 (FPR)')
        plt.ylabel('真阳性率 (TPR)')
        plt.title('接收者操作特征 (ROC) 曲线')
        plt.grid(True)
        plt.legend(loc="lower right")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存到 {save_path}")
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, X_val, y_val, save_path=None, 
                                  normal_class=0, anomaly_class=1):
        """
        绘制精确率-召回率曲线
        
        参数:
        X_val: 验证集特征
        y_val: 验证集标签
        save_path: 保存路径，如果为None则显示图形
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        """
        if self.model is None:
            raise ValueError("模型尚未设置")
        
        # 获取正常类和异常类的似然概率
        normal_likelihoods = self.model.get_class_likelihood(X_val, normal_class)
        anomaly_likelihoods = self.model.get_class_likelihood(X_val, anomaly_class)
        
        # 计算异常分数
        anomaly_scores = anomaly_likelihoods - normal_likelihoods
        
        # 计算精确率-召回率曲线
        precision, recall, thresholds = precision_recall_curve(
            y_val, anomaly_scores, pos_label=anomaly_class
        )
        
        # 绘制精确率-召回率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, color='blue', lw=2, label='精确率-召回率曲线')
        
        # 标记最佳阈值对应的点
        if self.best_threshold is not None:
            # 计算每个阈值对应的F1分数
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            best_idx = np.argmin(np.abs(thresholds - self.best_threshold))
            if best_idx < len(precision[:-1]):
                plt.plot(recall[best_idx], precision[best_idx], 'ro', 
                        label=f'最佳阈值点 ({recall[best_idx]:.4f}, {precision[best_idx]:.4f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('召回率 (Recall)')
        plt.ylabel('精确率 (Precision)')
        plt.title('精确率-召回率曲线')
        plt.grid(True)
        plt.legend(loc="best")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"精确率-召回率曲线已保存到 {save_path}")
        else:
            plt.show()


def find_optimal_threshold(model, X_val, y_val, method='f1', **kwargs):
    """
    找到最佳阈值的便捷函数
    
    参数:
    model: 训练好的模型
    X_val: 验证集特征
    y_val: 验证集标签
    method: 优化方法，可选'f1', 'precision_recall', 'roc'
    **kwargs: 传递给相应优化方法的参数
    
    返回:
    detector: 配置好的ThresholdDetector实例
    """
    detector = ThresholdDetector(model)
    
    if method == 'f1':
        detector.optimize_threshold_f1(X_val, y_val, **kwargs)
    elif method == 'precision_recall':
        detector.optimize_threshold_precision_recall(X_val, y_val, **kwargs)
    elif method == 'roc':
        detector.optimize_threshold_roc(X_val, y_val, **kwargs)
    else:
        raise ValueError(f"不支持的优化方法: {method}")
    
    return detector