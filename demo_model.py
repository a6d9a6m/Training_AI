#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型功能演示脚本

此脚本用于展示训练好的GMM模型的功能，包括：
1. 加载训练好的模型
2. 生成随机测试数据
3. 展示模型预测
4. 显示模型评估结果
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from models.gmm_model import GMMModel
from models.threshold_detector import ThresholdDetector
from utils.evaluator import ModelEvaluator, evaluate_model
from utils.config_manager import load_config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_trained_model(model_path):
    """
    加载训练好的模型
    """
    print(f"正在加载模型: {model_path}")
    try:
        model = GMMModel.load(model_path)
        print("✓ 模型加载成功")
        print(f"  - 模型类型: {type(model).__name__}")
        
        # 检查模型属性
        if hasattr(model, 'models'):
            print(f"  - 类别数: {len(model.models)}")
            # 尝试获取组件数信息（更安全的方式）
            if all(hasattr(m, 'n_components') for m in model.models):
                print(f"  - 组件数: {[m.n_components for m in model.models]}")
            else:
                print(f"  - 模型详情: {model.models}")
        
        # 显示其他可用属性
        print(f"  - 可用方法: {[method for method in dir(model) if not method.startswith('_') and callable(getattr(model, method))]}")
        
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_demo_data(model, num_samples=100):
    """
    生成演示用的测试数据
    """
    # 尝试确定特征维度（通用方式）
    feature_dim = 10  # 默认维度
    try:
        # 尝试从模型中推断特征维度
        if hasattr(model, 'models') and len(model.models) > 0:
            if hasattr(model.models[0], 'means_'):
                feature_dim = model.models[0].means_.shape[1]
            elif isinstance(model.models[0], (int, float)):
                # 如果模型是简单的数字类型，使用默认维度
                pass
    except:
        pass
    
    print(f"生成随机测试数据 ({num_samples}个样本，特征维度: {feature_dim})")
    
    # 生成随机高斯分布数据
    X_demo = np.random.randn(num_samples, feature_dim)
    
    # 生成二分类标签
    y_demo = np.random.randint(0, 2, num_samples)
    
    print("✓ 测试数据生成完成")
    return X_demo, y_demo

def demonstrate_prediction(model, X_demo):
    """
    展示模型预测功能
    """
    print("\n=== 展示模型预测功能 ===")
    
    # 预测前5个样本
    sample_indices = np.random.choice(len(X_demo), min(5, len(X_demo)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        sample = X_demo[idx:idx+1]
        
        try:
            # 获取预测结果
            if hasattr(model, 'predict'):
                prediction = model.predict(sample)[0]
                print(f"\n样本 {i+1}:")
                print(f"  预测类别: {'异常' if prediction == 1 else '正常'}")
                
                # 尝试获取概率信息
                if hasattr(model, 'get_class_likelihood'):
                    try:
                        class_likelihoods = model.get_class_likelihood(sample)
                        # 安全地计算概率
                        try:
                            probs = np.exp(class_likelihoods) / np.sum(np.exp(class_likelihoods))
                            for class_idx, prob in enumerate(probs):
                                if hasattr(prob, '__getitem__'):
                                    print(f"  类别 {class_idx} 概率: {prob[0]:.4f}")
                                else:
                                    print(f"  类别 {class_idx} 概率: {prob:.4f}")
                        except:
                            print(f"  原始似然值: {class_likelihoods}")
                    except Exception as e:
                        print(f"  无法获取概率: {e}")
        except Exception as e:
            print(f"\n样本 {i+1} 预测失败: {e}")
    
    # 展示批量预测
    try:
        if hasattr(model, 'predict'):
            batch_predictions = model.predict(X_demo[:10])
            unique_classes, counts = np.unique(batch_predictions, return_counts=True)
            print(f"\n批量预测统计（前10个样本）:")
            for cls, count in zip(unique_classes, counts):
                print(f"  类别 {'异常' if cls == 1 else '正常'}: {count}个样本")
    except Exception as e:
        print(f"批量预测失败: {e}")

def demonstrate_evaluation(model, X_demo, y_demo):
    """
    展示模型评估结果
    """
    print("\n=== 展示模型评估结果 ===")
    
    # 使用评估函数
    try:
        evaluator, metrics = evaluate_model(
            model, X_demo, y_demo,
            threshold=0.5,  # 使用默认阈值
            target_names=['正常', '异常'],
            output_dir=None  # 不保存图表
        )
        print("\n✓ 模型评估完成")
        
        # 可视化混淆矩阵
        plt.figure(figsize=(8, 6))
        cm = metrics['confusion_matrix']
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('模型混淆矩阵')
        plt.colorbar()
        
        # 添加标签
        classes = ['正常', '异常']
        plt.xticks(np.arange(len(classes)), classes)
        plt.yticks(np.arange(len(classes)), classes)
        
        # 在矩阵中显示数值
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('真实标签')
        plt.xlabel('预测标签')
        plt.tight_layout()
        
        # 显示图表
        print("显示混淆矩阵...")
        plt.show()
        
    except Exception as e:
        print(f"✗ 模型评估失败: {e}")

def main():
    """
    主函数
    """
    print("声音异常检测系统 - 模型演示")
    print("=" * 50)
    
    # 模型路径
    model_path = "models/saved_models/gmm_model.pkl"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        print("请先运行训练脚本: python main.py --mode train")
        return
    
    # 加载模型
    model = load_trained_model(model_path)
    if model is None:
        return
    
    # 生成演示数据
    X_demo, y_demo = generate_demo_data(model)
    
    # 展示预测功能
    demonstrate_prediction(model, X_demo)
    
    # 展示评估结果
    demonstrate_evaluation(model, X_demo, y_demo)
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()