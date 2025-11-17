import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)


class ModelEvaluator:
    """
    模型评估器，用于评估分类模型的性能
    
    支持多种评估指标计算和可视化展示
    """
    
    def __init__(self, model=None, threshold_detector=None):
        """
        初始化模型评估器
        
        参数:
        model: 要评估的模型
        threshold_detector: 阈值检测器
        """
        self.model = model
        self.threshold_detector = threshold_detector
        self.y_true = None
        self.y_pred = None
        self.y_score = None
    
    def set_model(self, model):
        """
        设置要评估的模型
        
        参数:
        model: 要评估的模型
        """
        self.model = model
        return self
    
    def set_threshold_detector(self, threshold_detector):
        """
        设置阈值检测器
        
        参数:
        threshold_detector: 阈值检测器
        """
        self.threshold_detector = threshold_detector
        return self
    
    def predict(self, X, y_true, threshold=None, normal_class=0, anomaly_class=1):
        """
        使用模型进行预测并保存结果
        
        参数:
        X: 输入特征
        y_true: 真实标签
        threshold: 阈值，如果为None则使用阈值检测器的最佳阈值
        normal_class: 正常类的标签
        anomaly_class: 异常类的标签
        
        返回:
        y_pred: 预测标签
        y_score: 异常分数
        """
        if self.threshold_detector is not None:
            y_pred, y_score = self.threshold_detector.apply_threshold(
                X, threshold, normal_class, anomaly_class
            )
        elif self.model is not None:
            # 如果只有模型，直接使用模型的预测功能
            y_pred = self.model.predict(X)
            # 尝试获取分数
            if hasattr(self.model, 'predict_proba'):
                try:
                    # 尝试获取异常类的概率
                    proba = self.model.predict_proba(X)
                    # 检查是否有足够的类别
                    if proba.shape[1] > anomaly_class:
                        y_score = proba[:, anomaly_class]
                    else:
                        # 单类别情况，使用负的正常类概率作为异常分数
                        print(f"警告: 模型只有{proba.shape[1]}个类别，使用单类别模式")
                        y_score = -proba[:, normal_class]
                except Exception as e:
                    print(f"获取预测概率时出错: {e}")
                    y_score = None
            elif hasattr(self.model, 'get_class_likelihood'):
                try:
                    normal_likelihoods = self.model.get_class_likelihood(X, normal_class)
                    # 尝试获取异常类的似然概率
                    anomaly_likelihoods = self.model.get_class_likelihood(X, anomaly_class)
                    y_score = anomaly_likelihoods - normal_likelihoods
                except ValueError:
                    # 单类别情况，使用负的正常类似然作为异常分数
                    print(f"警告: 模型中不存在异常类 {anomaly_class}，使用单类别模式")
                    y_score = -normal_likelihoods
            else:
                y_score = None
        else:
            raise ValueError("请设置模型或阈值检测器")
        
        # 保存结果
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_score = y_score
        
        return y_pred, y_score
    
    def calculate_metrics(self, target_names=None):
        """
        计算评估指标
        
        参数:
        target_names: 类别名称列表
        
        返回:
        metrics: 评估指标字典
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先运行predict方法")
        
        metrics = {}
        
        # 基本分类指标
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='macro')
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='macro')
        metrics['f1'] = f1_score(self.y_true, self.y_pred, average='macro')
        
        # 类别的具体指标
        metrics['class_precision'] = precision_score(self.y_true, self.y_pred, average=None)
        metrics['class_recall'] = recall_score(self.y_true, self.y_pred, average=None)
        metrics['class_f1'] = f1_score(self.y_true, self.y_pred, average=None)
        
        # 混淆矩阵
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred)
        
        # 如果有分数，计算ROC AUC
        if self.y_score is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_score)
            except ValueError:
                # 如果只有一个类别，无法计算ROC AUC
                metrics['roc_auc'] = None
        
        # 分类报告
        metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred, target_names=target_names
        )
        
        return metrics
    
    def print_metrics(self, target_names=None):
        """
        打印评估指标
        
        参数:
        target_names: 类别名称列表
        """
        metrics = self.calculate_metrics(target_names)
        
        print("===== 模型评估结果 =====")
        print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision): {metrics['precision']:.4f}")
        print(f"召回率 (Recall): {metrics['recall']:.4f}")
        print(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
        
        if metrics['roc_auc'] is not None:
            print(f"ROC AUC: {metrics['roc_auc']:.4f}")
        
        print("\n各类别指标:")
        classes = np.unique(self.y_true)
        for i, cls in enumerate(classes):
            class_name = target_names[i] if target_names and i < len(target_names) else f"类别 {cls}"
            print(f"{class_name}:")
            print(f"  精确率: {metrics['class_precision'][i]:.4f}")
            print(f"  召回率: {metrics['class_recall'][i]:.4f}")
            print(f"  F1分数: {metrics['class_f1'][i]:.4f}")
        
        print("\n分类报告:")
        print(metrics['classification_report'])
        
        print("\n混淆矩阵:")
        print(metrics['confusion_matrix'])
    
    def plot_confusion_matrix(self, normalize=False, target_names=None, save_path=None):
        """
        绘制混淆矩阵
        
        参数:
        normalize: 是否归一化
        target_names: 类别名称列表
        save_path: 保存路径，如果为None则显示图形
        """
        if self.y_true is None or self.y_pred is None:
            raise ValueError("请先运行predict方法")
        
        cm = confusion_matrix(self.y_true, self.y_pred)
        
        # 归一化
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵' + (' (归一化)' if normalize else ''))
        plt.colorbar()
        
        # 设置类别标签
        classes = np.unique(self.y_true)
        if target_names and len(target_names) == len(classes):
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, target_names, rotation=45)
            plt.yticks(tick_marks, target_names)
        
        # 在矩阵中显示数值
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到 {save_path}")
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path=None, target_names=None):
        """
        绘制ROC曲线
        
        参数:
        save_path: 保存路径，如果为None则显示图形
        target_names: 类别名称列表
        """
        if self.y_true is None or self.y_score is None:
            raise ValueError("请确保运行了predict方法并获得了预测分数")
        
        plt.figure(figsize=(10, 6))
        
        # 检查是否是多类别
        classes = np.unique(self.y_true)
        if len(classes) == 2:
            # 二分类情况
            fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
            roc_auc = roc_auc_score(self.y_true, self.y_score)
            
            plt.plot(fpr, tpr, lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        else:
            # 多类别情况，使用一对多方法
            for i, cls in enumerate(classes):
                # 二值化标签
                y_true_binary = (self.y_true == cls).astype(int)
                
                # 如果有概率分数，使用对应类别的概率
                if len(self.y_score.shape) > 1 and self.y_score.shape[1] == len(classes):
                    y_score_binary = self.y_score[:, i]
                else:
                    # 否则使用模型特定的方式获取分数
                    try:
                        y_score_binary = self.y_score
                    except:
                        print(f"无法为类别 {cls} 计算ROC曲线")
                        continue
                
                # 计算ROC曲线
                fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
                roc_auc = roc_auc_score(y_true_binary, y_score_binary)
                
                label = target_names[i] if target_names and i < len(target_names) else f'类别 {cls}'
                plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.4f})')
        
        # 绘制随机猜测线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='随机猜测 (AUC = 0.5)')
        
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
    
    def plot_precision_recall_curve(self, save_path=None, target_names=None):
        """
        绘制精确率-召回率曲线
        
        参数:
        save_path: 保存路径，如果为None则显示图形
        target_names: 类别名称列表
        """
        if self.y_true is None or self.y_score is None:
            raise ValueError("请确保运行了predict方法并获得了预测分数")
        
        plt.figure(figsize=(10, 6))
        
        # 检查是否是多类别
        classes = np.unique(self.y_true)
        if len(classes) == 2:
            # 二分类情况
            precision, recall, _ = precision_recall_curve(self.y_true, self.y_score)
            
            plt.plot(recall, precision, lw=2, label='精确率-召回率曲线')
            
            # 计算并显示平均精确度
            ap = np.mean(precision)
            plt.axhline(y=ap, color='r', linestyle='--', 
                       label=f'平均精确度 (AP) = {ap:.4f}')
        else:
            # 多类别情况，使用一对多方法
            for i, cls in enumerate(classes):
                # 二值化标签
                y_true_binary = (self.y_true == cls).astype(int)
                
                # 如果有概率分数，使用对应类别的概率
                if len(self.y_score.shape) > 1 and self.y_score.shape[1] == len(classes):
                    y_score_binary = self.y_score[:, i]
                else:
                    # 否则使用模型特定的方式获取分数
                    try:
                        y_score_binary = self.y_score
                    except:
                        print(f"无法为类别 {cls} 计算精确率-召回率曲线")
                        continue
                
                # 计算精确率-召回率曲线
                precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
                
                label = target_names[i] if target_names and i < len(target_names) else f'类别 {cls}'
                plt.plot(recall, precision, lw=2, label=label)
        
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
    
    def plot_performance_summary(self, save_path=None):
        """
        绘制性能汇总图
        
        参数:
        save_path: 保存路径，如果为None则显示图形
        """
        metrics = self.calculate_metrics()
        
        plt.figure(figsize=(12, 8))
        
        # 创建子图
        plt.subplot(2, 2, 1)
        # 绘制主要指标条形图
        metric_names = ['准确率', '精确率', '召回率', 'F1分数']
        metric_values = [metrics['accuracy'], metrics['precision'], 
                        metrics['recall'], metrics['f1']]
        
        bars = plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'purple'])
        plt.ylim([0, 1])
        plt.title('主要评估指标')
        
        # 在条形图上显示数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{height:.4f}', ha='center', va='bottom')
        
        plt.subplot(2, 2, 2)
        # 绘制各类别的F1分数
        classes = np.unique(self.y_true)
        plt.bar(range(len(classes)), metrics['class_f1'], color='orange')
        plt.xticks(range(len(classes)), [f'类别 {cls}' for cls in classes])
        plt.ylim([0, 1])
        plt.title('各类别F1分数')
        
        plt.subplot(2, 2, 3)
        # 绘制混淆矩阵
        cm = metrics['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        
        # 在矩阵中显示数值
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         horizontalalignment="center",
                         color="white" if cm[i, j] > cm.max()/2. else "black")
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        
        plt.subplot(2, 2, 4)
        # 绘制ROC曲线（如果有分数）
        if self.y_score is not None and metrics['roc_auc'] is not None:
            fpr, tpr, _ = roc_curve(self.y_true, self.y_score)
            plt.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {metrics["roc_auc"]:.4f})')
            plt.plot([0, 1], [0, 1], 'k--', lw=2)
            plt.title('ROC曲线')
            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.legend()
        else:
            plt.text(0.5, 0.5, '无法绘制ROC曲线', ha='center', va='center')
            plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能汇总图已保存到 {save_path}")
        else:
            plt.show()


def evaluate_model(model, X_test, y_test, threshold=None, threshold_detector=None, 
                  target_names=['normal', 'anomaly'], output_dir=None):
    """
    评估模型性能的便捷函数
    
    参数:
    model: 要评估的模型
    X_test: 测试集特征
    y_test: 测试集标签
    threshold: 阈值
    threshold_detector: 阈值检测器
    target_names: 类别名称列表
    output_dir: 输出目录，如果不为None则保存图表
    
    返回:
    evaluator: 配置好的ModelEvaluator实例
    metrics: 评估指标字典
    """
    evaluator = ModelEvaluator(model, threshold_detector)
    
    # 进行预测
    evaluator.predict(X_test, y_test, threshold)
    
    # 计算指标
    metrics = evaluator.calculate_metrics(target_names)
    
    # 打印评估结果
    evaluator.print_metrics(target_names)
    
    # 如果指定了输出目录，保存图表
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存混淆矩阵
        evaluator.plot_confusion_matrix(save_path=os.path.join(output_dir, 'confusion_matrix.png'),
                                      target_names=target_names)
        
        # 保存归一化混淆矩阵
        evaluator.plot_confusion_matrix(normalize=True,
                                      save_path=os.path.join(output_dir, 'confusion_matrix_normalized.png'),
                                      target_names=target_names)
        
        # 保存ROC曲线
        if evaluator.y_score is not None:
            evaluator.plot_roc_curve(save_path=os.path.join(output_dir, 'roc_curve.png'),
                                   target_names=target_names)
            
            # 保存精确率-召回率曲线
            evaluator.plot_precision_recall_curve(save_path=os.path.join(output_dir, 'precision_recall_curve.png'),
                                               target_names=target_names)
        
        # 保存性能汇总图
        evaluator.plot_performance_summary(save_path=os.path.join(output_dir, 'performance_summary.png'))
    
    return evaluator, metrics