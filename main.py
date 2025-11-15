#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
声音异常检测系统 - 主程序

此脚本提供了训练、评估和推理的完整流程：
1. 加载配置
2. 准备数据
3. 提取特征
4. 训练模型
5. 确定最佳阈值
6. 评估模型性能
7. 保存模型和结果
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

# 导入项目模块
from utils.config_manager import load_config, create_default_config
from utils.data_loader import load_dataset, load_audio
from features.extract_features import extract_all_features, extract_features_from_files
from models.gmm_model import GMMModel, train_gmm_model, find_optimal_components
from models.threshold_detector import ThresholdDetector, find_optimal_threshold
from utils.evaluator import ModelEvaluator, evaluate_model


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='声音异常检测系统')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='运行模式：train（训练）、evaluate（评估）、predict（预测）')
    parser.add_argument('--audio_file', type=str,
                        help='在predict模式下需要预测的音频文件路径')
    parser.add_argument('--model_path', type=str,
                        help='预训练模型路径（用于evaluate或predict模式）')
    parser.add_argument('--threshold', type=float,
                        help='分类阈值（如果不提供，将使用验证集确定）')
    return parser.parse_args()


def prepare_data(config):
    """
    准备数据：加载并分割数据集
    """
    print("正在准备数据...")
    
    sample_rate = config.get('data.sample_rate')
    
    # 检查是否使用设备类型加载数据
    if config.get('data.use_device_type', False):
        device_type = config.get('data.device_type', 'fan')  # 默认使用fan
        base_data_dir = config.get('paths.base_data_dir', 'dev_data')
        
        print(f"使用设备类型 '{device_type}' 加载数据")
        
        # 加载数据集 - 使用设备类型方式
        train_data, val_data, test_data = load_dataset(
            device_type=device_type,
            base_data_dir=base_data_dir,
            sr=sample_rate,
            test_size=config.get('data.test_size', 0.2),
            val_size=config.get('data.val_size', 0.2),
            random_state=config.get('data.random_state', 42)
        )
    else:
        # 传统方式：分别指定正常和异常数据目录
        normal_dir = config.get('paths.normal_data_dir')
        anomaly_dir = config.get('paths.anomaly_data_dir')
        
        # 检查数据目录是否存在
        if not os.path.exists(normal_dir):
            raise FileNotFoundError(f"正常数据目录不存在: {normal_dir}")
        if not os.path.exists(anomaly_dir):
            raise FileNotFoundError(f"异常数据目录不存在: {anomaly_dir}")
        
        # 加载数据集
        train_data, val_data, test_data = load_dataset(
            normal_dir=normal_dir,
            anomaly_dir=anomaly_dir,
            sr=sample_rate,
            test_size=config.get('data.test_size', 0.2),
            val_size=config.get('data.val_size', 0.2),
            random_state=config.get('data.random_state', 42)
        )
    
    print(f"数据集加载完成: 训练集 {len(train_data)} 个样本, "
          f"验证集 {len(val_data)} 个样本, 测试集 {len(test_data)} 个样本")
    
    return train_data, val_data, test_data


def extract_features_wrapper(audio_data, config):
    """
    特征提取包装函数
    """
    audio, label = audio_data
    features = extract_all_features(
        audio, 
        sample_rate=config.get('data.sample_rate'),
        n_mfcc=config.get('features.n_mfcc', 13),
        n_fft=config.get('features.n_fft', 2048),
        hop_length=config.get('features.hop_length', 512),
        win_length=config.get('features.win_length', None),
        n_mels=config.get('features.n_mels', 128)
    )
    return features, label


def extract_dataset_features(dataset, config):
    """
    提取数据集的特征
    """
    print(f"正在提取特征，共 {len(dataset)} 个样本...")
    
    features_list = []
    labels_list = []
    
    start_time = time.time()
    
    for i, data in enumerate(dataset):
        try:
            features, label = extract_features_wrapper(data, config)
            features_list.append(features)
            labels_list.append(label)
            
            if (i + 1) % 10 == 0 or i == len(dataset) - 1:
                print(f"进度: {i + 1}/{len(dataset)}")
        except Exception as e:
            print(f"处理样本 {i} 时出错: {e}")
    
    end_time = time.time()
    
    print(f"特征提取完成，耗时: {end_time - start_time:.2f} 秒")
    print(f"成功提取: {len(features_list)} 个样本的特征")
    
    return features_list, labels_list


def train_model(train_features, train_labels, config, val_features=None, val_labels=None):
    """
    训练GMM模型
    """
    print("开始训练模型...")
    
    # 检查是否需要寻找最佳组件数量
    if config.get('model.auto_find_components', True):
        print("正在寻找最佳组件数量...")
        min_components = config.get('model.min_components', 1)
        max_components = config.get('model.max_components', 10)
        cv_folds = config.get('model.cv_folds', 5)
        
        optimal_components = find_optimal_components(
            train_features, train_labels,
            min_components=min_components,
            max_components=max_components,
            cv_folds=cv_folds
        )
        print(f"最佳组件数量: {optimal_components}")
    else:
        optimal_components = config.get('model.n_components', 5)
    
    # 训练模型
    model = train_gmm_model(
        train_features, train_labels,
        n_components=optimal_components,
        covariance_type=config.get('model.covariance_type', 'diag'),
        reg_covar=config.get('model.reg_covar', 1e-6),
        max_iter=config.get('model.max_iter', 100)
    )
    
    print("模型训练完成")
    return model, optimal_components


def determine_threshold(model, val_features, val_labels, config, threshold=None):
    """
    确定最佳分类阈值
    """
    if threshold is not None:
        print(f"使用指定的阈值: {threshold}")
        return threshold
    
    if val_features is None or val_labels is None:
        print("没有验证集，使用默认阈值: 0.0")
        return 0.0
    
    print("正在确定最佳阈值...")
    method = config.get('evaluation.threshold_method', 'f1_score')
    
    optimal_threshold = find_optimal_threshold(
        model, val_features, val_labels,
        method=method,
        plot=config.get('evaluation.plot_threshold_curves', True)
    )
    
    print(f"最佳阈值 ({method}): {optimal_threshold}")
    return optimal_threshold


def evaluate_final_model(model, test_features, test_labels, threshold, config):
    """
    评估最终模型性能
    """
    print("\n正在评估模型性能...")
    
    # 使用评估器
    evaluator = ModelEvaluator(model, threshold)
    metrics = evaluator.evaluate(
        test_features, test_labels,
        plot=config.get('evaluation.plot_results', True),
        plot_save_dir=config.get('paths.plots_dir', None)
    )
    
    # 打印评估指标
    print("\n评估指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1 Score): {metrics['f1_score']:.4f}")
    print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"精确率-召回率 AUC: {metrics['pr_auc']:.4f}")
    print(f"混淆矩阵:")
    print(metrics['confusion_matrix'])
    
    return metrics


def save_results(model, threshold, optimal_components, config):
    """
    保存模型和结果
    """
    model_dir = config.get('paths.model_dir')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存模型
    model_path = os.path.join(model_dir, 'gmm_model.pkl')
    model.save(model_path)
    print(f"模型已保存至: {model_path}")
    
    # 保存配置信息
    import json
    results = {
        'threshold': threshold,
        'optimal_components': optimal_components,
        'config': dict(config.config)
    }
    
    results_path = os.path.join(model_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"训练结果已保存至: {results_path}")


def predict_audio_file(audio_path, model, config, threshold):
    """
    预测单个音频文件
    """
    print(f"预测音频文件: {audio_path}")
    
    # 加载并预处理音频
    audio, sr = load_audio(audio_path, sr=config.get('data.sample_rate'))
    
    # 提取特征
    features = extract_all_features(
        audio, 
        sample_rate=sr,
        n_mfcc=config.get('features.n_mfcc', 13),
        n_fft=config.get('features.n_fft', 2048),
        hop_length=config.get('features.hop_length', 512),
        win_length=config.get('features.win_length', None),
        n_mels=config.get('features.n_mels', 128)
    )
    
    # 预测
    detector = ThresholdDetector(model)
    prediction, score = detector.apply_threshold([features], threshold)
    
    # 获取更详细的概率信息
    if hasattr(model, 'get_class_likelihood'):
        normal_prob = model.get_class_likelihood([features], 0)[0]
        anomaly_prob = model.get_class_likelihood([features], 1)[0]
    else:
        normal_prob, anomaly_prob = None, None
    
    # 显示结果
    result = '异常' if prediction[0] == 1 else '正常'
    print(f"\n预测结果: {result}")
    print(f"分类分数: {score[0]:.4f}")
    print(f"使用阈值: {threshold:.4f}")
    
    if normal_prob is not None and anomaly_prob is not None:
        print(f"正常类别概率: {normal_prob:.4f}")
        print(f"异常类别概率: {anomaly_prob:.4f}")
    
    return {
        'prediction': result,
        'prediction_code': prediction[0],
        'score': score[0],
        'threshold': threshold,
        'normal_probability': normal_prob,
        'anomaly_probability': anomaly_prob
    }


def train_mode(config, args):
    """
    训练模式
    """
    # 准备数据
    train_data, val_data, test_data = prepare_data(config)
    
    # 提取特征
    train_features, train_labels = extract_dataset_features(train_data, config)
    val_features, val_labels = extract_dataset_features(val_data, config)
    test_features, test_labels = extract_dataset_features(test_data, config)
    
    # 训练模型
    model, optimal_components = train_model(
        train_features, train_labels, config,
        val_features=val_features, val_labels=val_labels
    )
    
    # 确定阈值
    threshold = determine_threshold(
        model, val_features, val_labels, config,
        threshold=args.threshold
    )
    
    # 评估模型
    metrics = evaluate_final_model(model, test_features, test_labels, threshold, config)
    
    # 保存结果
    save_results(model, threshold, optimal_components, config)
    
    return model, threshold


def evaluate_mode(config, args):
    """
    评估模式
    """
    # 加载模型
    if not args.model_path:
        raise ValueError("请提供模型路径 --model_path")
    
    model = GMMModel.load(args.model_path)
    print(f"已加载模型: {args.model_path}")
    
    # 准备数据（只需要测试集）
    _, _, test_data = prepare_data(config)
    
    # 提取特征
    test_features, test_labels = extract_dataset_features(test_data, config)
    
    # 确定阈值
    threshold = determine_threshold(
        model, None, None, config, threshold=args.threshold
    )
    
    # 评估模型
    evaluate_final_model(model, test_features, test_labels, threshold, config)


def predict_mode(config, args):
    """
    预测模式
    """
    # 加载模型
    if not args.model_path:
        raise ValueError("请提供模型路径 --model_path")
    
    model = GMMModel.load(args.model_path)
    print(f"已加载模型: {args.model_path}")
    
    # 确定阈值
    threshold = determine_threshold(
        model, None, None, config, threshold=args.threshold
    )
    
    # 预测音频文件
    if not args.audio_file:
        raise ValueError("请提供音频文件路径 --audio_file")
    
    predict_audio_file(args.audio_file, model, config, threshold)


def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    print("声音异常检测系统")
    print("=" * 50)
    
    # 加载配置
    try:
        config = load_config(args.config)
        print(f"已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"配置文件不存在: {args.config}")
        print("正在创建默认配置文件...")
        create_default_config(args.config)
        config = load_config(args.config)
        print(f"已创建并加载默认配置文件: {args.config}")
    
    # 确保必要的目录存在
    os.makedirs(config.get('paths.model_dir'), exist_ok=True)
    if config.get('evaluation.plot_results', True):
        os.makedirs(config.get('paths.plots_dir'), exist_ok=True)
    
    # 根据模式执行不同的功能
    if args.mode == 'train':
        train_mode(config, args)
    elif args.mode == 'evaluate':
        evaluate_mode(config, args)
    elif args.mode == 'predict':
        predict_mode(config, args)
    
    print("\n程序执行完成")


if __name__ == "__main__":
    main()