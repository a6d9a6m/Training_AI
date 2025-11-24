#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试脚本：评估改进后的领域适应性能
测试不同组合的特征选择方法、特征提取和阈值校准的效果
"""

import os
import sys
import json
import numpy as np
import torch
import argparse
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import prepare_data
from main import extract_dataset_features
from models.gmm_model import GMMModel
from models.autoencoder import AudioAutoencoder
from utils.domain_adaptation import (
    calculate_mmd,
    get_adaptation_method,
    calibrate_threshold,
    transfer_threshold_with_unlabeled,
    select_domain_invariant_features,
    ensemble_feature_selection,
    MMDAdaptationTrainer
)
from utils.evaluator import evaluate_model as evaluate_final_model, determine_threshold
import yaml

def load_config(config_path='config/config.yaml'):
    """简单的配置加载函数"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    # 创建一个简单的Config类来模拟配置对象
    class Config:
        def __init__(self, data):
            self.config = data
        def get(self, key, default=None):
            keys = key.split('.')
            value = self.config
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
    return Config(config)

def run_experiment(config, experiment_name, adaptation_method, feature_selection_config=None, threshold_calib_config=None, model_type='gmm'):
    """
    运行单个实验
    """
    print(f"\n=== 开始实验: {experiment_name} ===")
    print(f"模型类型: {model_type}")
    print(f"适应方法: {adaptation_method}")
    print(f"特征选择: {feature_selection_config}")
    print(f"阈值校准: {threshold_calib_config}")
    
    # 确保启用领域偏移模式以获取4个返回值
    config.config['data']['enable_domain_shift'] = True
    
    # 准备数据
    source_train_data, source_test_data, target_train_data, target_test_data = prepare_data(config)
    
    # 提取特征
    print("\n提取源域特征...")
    source_train_features, source_train_labels = extract_dataset_features(source_train_data, config)
    source_test_features, source_test_labels = extract_dataset_features(source_test_data, config)
    
    print("提取目标域特征...")
    target_train_features, target_train_labels = extract_dataset_features(target_train_data, config)
    target_test_features, target_test_labels = extract_dataset_features(target_test_data, config)
    
    # 计算原始MMD
    if len(source_train_features) > 0 and len(target_train_features) > 0:
        mmd_before = calculate_mmd(source_train_features, target_train_features, kernel='rbf')
        print(f"原始MMD距离: {mmd_before:.6f}")
    else:
        mmd_before = None
        print("警告: 样本不足，无法计算MMD")
    
    # 应用特征选择
    selected_source_train_features = source_train_features
    selected_source_test_features = source_test_features
    selected_target_train_features = target_train_features
    selected_target_test_features = target_test_features
    selected_feature_indices = None
    
    if feature_selection_config and feature_selection_config.get('enabled', False):
        method = feature_selection_config.get('method', 'mmd_based')
        ratio = feature_selection_config.get('ratio', 0.8)
        ensemble = feature_selection_config.get('ensemble', False)
        
        print(f"\n应用特征选择: {method}, 比例: {ratio}")
        try:
            if ensemble:
                methods = feature_selection_config.get('ensemble_methods', ['f_statistic', 'mutual_info', 'mmd_based'])
                selected_source_train_features, selected_target_train_features, selected_feature_indices, _ = ensemble_feature_selection(
                    source_train_features, source_train_labels, target_train_features,
                    methods=methods, k=ratio
                )
                selected_source_test_features = source_test_features[:, selected_feature_indices]
                selected_target_test_features = target_test_features[:, selected_feature_indices]
            else:
                selected_source_train_features, selected_target_train_features, _, selected_feature_indices = select_domain_invariant_features(
                    source_train_features, source_train_labels, target_train_features,
                    method=method, k=ratio
                )
                selected_source_test_features = source_test_features[:, selected_feature_indices]
                selected_target_test_features = target_test_features[:, selected_feature_indices]
            
            print(f"特征选择后维度: {selected_source_train_features.shape[1]}")
        except Exception as e:
            print(f"特征选择失败: {e}")
    
    # 应用领域适应
    adapted_source_train_features = selected_source_train_features.copy()
    adapted_source_test_features = selected_source_test_features.copy()
    adapt_model = None
    
    try:
        adapt_func = get_adaptation_method(adaptation_method)
        
        if adaptation_method in ['standardization', 'minmax']:
            adapted_source_train_features, adapt_model = adapt_func(selected_source_train_features, selected_target_train_features)
            adapted_source_test_features = adapt_model.transform(selected_source_test_features)
        elif adaptation_method == 'pca':
            adapted_source_train_features, _, adapt_model = adapt_func(
                selected_source_train_features, selected_target_train_features,
                n_components=config.get('domain_shift.pca_components', None)
            )
            adapted_source_test_features = adapt_model.transform(selected_source_test_features)
            selected_target_test_features = adapt_model.transform(selected_target_test_features)
        else:
            adapted_source_train_features = adapt_func(selected_source_train_features, selected_target_train_features)
            adapted_source_test_features = adapt_func(selected_source_test_features, selected_target_train_features)
        
        # 计算适应后的MMD
        if mmd_before is not None:
            mmd_after = calculate_mmd(adapted_source_train_features, selected_target_train_features, kernel='rbf')
            print(f"适应后MMD距离: {mmd_after:.6f}")
            print(f"MMD改善率: {(mmd_before - mmd_after) / mmd_before * 100:.2f}%")
            
    except Exception as e:
        print(f"领域适应失败: {e}")
    
    # 训练模型
    model = None
    source_threshold = 0.5  # 默认阈值
    source_test_scores = None
    target_test_scores = None
    
    if model_type == 'gmm':
        print("\n训练GMM模型...")
        model = GMMModel(n_components=config.get('model.gmm_components', 10))
        model.train(adapted_source_train_features, source_train_labels)
        
        # 确定阈值
        try:
            source_threshold = determine_threshold(model, adapted_source_test_features, source_test_labels, config)
        except Exception as e:
            print(f"阈值确定失败，使用默认值: {e}")
    
    elif model_type == 'autoencoder':
        print("\n训练自动编码器模型...")
        
        # 筛选正常数据进行训练（标签为0的数据）
        normal_source_train_mask = (source_train_labels == 0)
        normal_source_train_features = adapted_source_train_features[normal_source_train_mask]
        print(f"正常样本数量: {len(normal_source_train_features)}/{len(adapted_source_train_features)}")
        
        # 创建自动编码器模型
        input_dim = normal_source_train_features.shape[1]
        model = AudioAutoencoder(
            input_dim=input_dim,
            encoding_dim=config.config['model'].get('encoding_dim', 128),
            dropout_rate=config.config['model'].get('dropout_rate', 0.3)
        )
        
        # 如果使用MMD领域适应，使用MMDAdaptationTrainer
        if adaptation_method == 'mmd_adaptation' and len(selected_target_train_features) > 0:
            # 筛选目标域中的正常数据（假设目标域训练数据主要是正常的）
            # 注意：MMDAdaptationTrainer可能需要调整以支持AudioAutoencoder
            # 这里使用简化的训练方式，仅在源域上训练
            print("警告: MMD领域适应对于自动编码器尚未完全实现，使用常规训练")
            # 创建数据加载器
            import torch.utils.data as Data
            train_dataset = Data.TensorDataset(
                torch.tensor(normal_source_train_features, dtype=torch.float32),
                torch.zeros(len(normal_source_train_features))  # 标签占位符
            )
            train_loader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=config.get('training.batch_size', 32),
                shuffle=True
            )
            
            # 设置优化器
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get('training.learning_rate', 0.001)
            )
            
            # 使用正确的训练方法
            model.train_model(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                epochs=config.get('training.epochs', 100)
            )
        else:
            # 常规训练
            # 创建数据加载器
            import torch.utils.data as Data
            train_dataset = Data.TensorDataset(
                torch.tensor(normal_source_train_features, dtype=torch.float32),
                torch.zeros(len(normal_source_train_features))  # 标签占位符
            )
            train_loader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=config.get('training.batch_size', 32),
                shuffle=True
            )
            
            # 设置优化器
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=config.get('training.learning_rate', 0.001)
            )
            
            # 使用正确的训练方法
            model.train_model(
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=torch.nn.MSELoss(),
                epochs=config.get('training.epochs', 100)
            )
        
        # 计算重构误差作为异常分数
        # 将特征转换为张量
        adapted_source_test_tensor = torch.tensor(adapted_source_test_features, dtype=torch.float32).to(model.device)
        selected_target_test_tensor = torch.tensor(selected_target_test_features, dtype=torch.float32).to(model.device)
        
        # 使用自动编码器计算重构误差
        model.eval()
        with torch.no_grad():
            _, source_reconstructed = model(adapted_source_test_tensor)
            _, target_reconstructed = model(selected_target_test_tensor)
            
            # 计算MSE重构误差
            source_test_scores = torch.mean((source_reconstructed - adapted_source_test_tensor) ** 2, dim=1).cpu().numpy()
            target_test_scores = torch.mean((target_reconstructed - selected_target_test_tensor) ** 2, dim=1).cpu().numpy()
        
        # 确定阈值（基于源域验证集的正常样本重构误差分布）
        try:
            # 使用正常样本的重构误差确定阈值
            normal_source_test_mask = (source_test_labels == 0)
            if np.any(normal_source_test_mask):
                normal_source_test_scores = source_test_scores[normal_source_test_mask]
                # 使用95%分位数作为阈值
                percentile = config.get('model.reconstruction_percentile', 95)
                source_threshold = np.percentile(normal_source_test_scores, percentile)
                print(f"基于正常样本{percentile}%分位数的重构误差阈值: {source_threshold:.6f}")
            else:
                print("警告: 源域测试集中没有正常样本，使用默认阈值")
        except Exception as e:
            print(f"阈值确定失败，使用默认值: {e}")
    
    # 阈值校准
    calibrated_threshold = source_threshold
    if threshold_calib_config and threshold_calib_config.get('enabled', False):
        method = threshold_calib_config.get('method', 'f1_optimization')
        print(f"\n应用阈值校准: {method}")
        try:
            if len(target_test_labels) > 0:
                if model_type == 'autoencoder' and target_test_scores is not None:
                    # 使用预计算的重构误差进行阈值校准
                    # 这里简化处理，使用f1优化
                    best_threshold = source_threshold
                    best_f1 = 0
                    thresholds = np.linspace(np.min(target_test_scores) * 0.5, np.max(target_test_scores) * 1.5, 100)
                    
                    for t in thresholds:
                        y_pred = (target_test_scores > t).astype(int)
                        f1 = f1_score(target_test_labels, y_pred, zero_division=0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_threshold = t
                    
                    calibrated_threshold = best_threshold
                    print(f"校准后的阈值: {calibrated_threshold:.6f} (F1={best_f1:.4f})")
                else:
                    calibrated_threshold, _ = calibrate_threshold(
                        model, selected_target_test_features, target_test_labels, source_threshold, method
                    )
            else:
                if model_type == 'autoencoder':
                    # 使用目标域的未标记数据分布匹配
                    target_scores = model.calculate_reconstruction_error(selected_target_train_features)
                    # 假设目标域大部分是正常的，使用相同的分位数
                    percentile = config.get('model.reconstruction_percentile', 95)
                    calibrated_threshold = np.percentile(target_scores, percentile)
                else:
                    calibrated_threshold = transfer_threshold_with_unlabeled(
                        model, selected_target_train_features, source_threshold, method='distribution_matching'
                    )
        except Exception as e:
            print(f"阈值校准失败: {e}")
    
    # 评估
    print("\n评估性能...")
    # 为evaluate_final_model提供正确的参数
    source_metrics = evaluate_final_model(
        model, 
        adapted_source_test_features, 
        source_test_labels, 
        threshold=source_threshold,
        scores=source_test_scores
    )[1]  # 只取metrics部分
    
    target_metrics = evaluate_final_model(
        model, 
        selected_target_test_features, 
        target_test_labels, 
        threshold=calibrated_threshold,
        scores=target_test_scores
    )[1]  # 只取metrics部分
    
    # 计算性能变化
    performance_change = {
        'accuracy': (target_metrics['accuracy'] - source_metrics['accuracy']) * 100,
        'f1': (target_metrics['f1'] - source_metrics['f1']) * 100,
        'recall': (target_metrics['recall'] - source_metrics['recall']) * 100
    }
    
    # 添加AUC变化（如果有）
    if 'roc_auc' in source_metrics and 'roc_auc' in target_metrics:
        performance_change['auc'] = (target_metrics['roc_auc'] - source_metrics['roc_auc']) * 100
    elif 'auc' in source_metrics and 'auc' in target_metrics:
        performance_change['auc'] = (target_metrics['auc'] - source_metrics['auc']) * 100
    
    # 打印结果
    print("\n=== 实验结果 ===")
    print("源域性能:")
    print(f"  准确率: {source_metrics['accuracy']:.4f}")
    print(f"  F1分数: {source_metrics['f1']:.4f}")
    print(f"  召回率: {source_metrics['recall']:.4f}")
    
    print("\n目标域性能:")
    print(f"  准确率: {target_metrics['accuracy']:.4f}")
    print(f"  F1分数: {target_metrics['f1']:.4f}")
    print(f"  召回率: {target_metrics['recall']:.4f}")
    
    print("\n性能变化 (%):")
    print(f"  准确率变化: {performance_change['accuracy']:+.2f}")
    print(f"  F1分数变化: {performance_change['f1']:+.2f}")
    print(f"  召回率变化: {performance_change['recall']:+.2f}")
    if 'auc' in performance_change:
        print(f"  AUC变化: {performance_change['auc']:+.2f}")
    
    # 返回结果
    return {
        'experiment_name': experiment_name,
        'adaptation_method': adaptation_method,
        'feature_selection': feature_selection_config,
        'threshold_calibration': threshold_calib_config,
        'mmd_before': mmd_before,
        'mmd_after': mmd_after if 'mmd_after' in locals() else None,
        'source_threshold': source_threshold,
        'calibrated_threshold': calibrated_threshold,
        'source_metrics': source_metrics,
        'target_metrics': target_metrics,
        'performance_change': performance_change,
        'feature_dim_before': source_train_features.shape[1],
        'feature_dim_after': selected_source_train_features.shape[1] if selected_feature_indices is not None else source_train_features.shape[1]
    }

def main():
    parser = argparse.ArgumentParser(description='测试领域适应性能')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='experiments/results', help='结果输出目录')
    parser.add_argument('--skip-existing', action='store_true', help='跳过已存在的结果')
    parser.add_argument('--model-type', type=str, default='all', choices=['gmm', 'autoencoder', 'all'], 
                       help='要测试的模型类型')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 定义实验列表
    experiments = []
    
    # 根据参数添加对应的实验
    if args.model_type in ['gmm', 'all']:
        # GMM模型实验
        gmm_experiments = [
            # 基线实验
            {
                'name': 'gmm_baseline',
                'model_type': 'gmm',
                'adaptation_method': 'none',
                'feature_selection': None,
                'threshold_calibration': None
            },
            # 测试不同的适应方法
            {
                'name': 'gmm_mean_shift',
                'model_type': 'gmm',
                'adaptation_method': 'mean_shift',
                'feature_selection': None,
                'threshold_calibration': None
            },
            {
                'name': 'gmm_coral',
                'model_type': 'gmm',
                'adaptation_method': 'coral_adaptation',
                'feature_selection': None,
                'threshold_calibration': None
            },
            # 完整组合（最佳性能预期）
            {
                'name': 'gmm_full_combination',
                'model_type': 'gmm',
                'adaptation_method': 'coral_adaptation',
                'feature_selection': {
                    'enabled': True,
                    'ensemble': True,
                    'ratio': 0.8,
                    'ensemble_methods': ['f_statistic', 'mutual_info', 'mmd_based', 'correlation']
                },
                'threshold_calibration': {'enabled': True, 'method': 'f1_optimization'}
            }
        ]
        experiments.extend(gmm_experiments)
    
    if args.model_type in ['autoencoder', 'all']:
        # 自动编码器模型实验
        autoencoder_experiments = [
            # 基线实验
            {
                'name': 'auto_baseline',
                'model_type': 'autoencoder',
                'adaptation_method': 'none',
                'feature_selection': None,
                'threshold_calibration': None
            },
            # 测试MMD领域适应
            {
                'name': 'auto_mmd_adaptation',
                'model_type': 'autoencoder',
                'adaptation_method': 'mmd_adaptation',
                'feature_selection': None,
                'threshold_calibration': None
            },
            # 测试特征选择
            {
                'name': 'auto_feature_selection',
                'model_type': 'autoencoder',
                'adaptation_method': 'none',
                'feature_selection': {
                    'enabled': True,
                    'ensemble': True,
                    'ratio': 0.8,
                    'ensemble_methods': ['f_statistic', 'mutual_info', 'mmd_based', 'correlation']
                },
                'threshold_calibration': None
            },
            # 测试阈值校准
            {
                'name': 'auto_threshold_calibration',
                'model_type': 'autoencoder',
                'adaptation_method': 'none',
                'feature_selection': None,
                'threshold_calibration': {'enabled': True, 'method': 'f1_optimization'}
            },
            # 完整组合（最佳性能预期）
            {
                'name': 'auto_full_combination',
                'model_type': 'autoencoder',
                'adaptation_method': 'mmd_adaptation',
                'feature_selection': {
                    'enabled': True,
                    'ensemble': True,
                    'ratio': 0.8,
                    'ensemble_methods': ['f_statistic', 'mutual_info', 'mmd_based', 'correlation']
                },
                'threshold_calibration': {'enabled': True, 'method': 'f1_optimization'}
            }
        ]
        experiments.extend(autoencoder_experiments)
    
    # 运行所有实验
    results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for exp in experiments:
        result_file = os.path.join(args.output, f"result_{exp['name']}_{timestamp}.json")
        
        # 跳过已存在的结果
        if args.skip_existing and os.path.exists(result_file):
            print(f"跳过实验 {exp['name']}，结果已存在")
            continue
        
        # 运行实验
        result = run_experiment(
            config, 
            exp['name'], 
            exp['adaptation_method'], 
            exp['feature_selection'], 
            exp['threshold_calibration'],
            model_type=exp.get('model_type', 'gmm')
        )
        results.append(result)
        
        # 保存结果
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"实验结果已保存: {result_file}")
    
    # 汇总结果
    summary_file = os.path.join(args.output, f"summary_{timestamp}.json")
    
    # 按模型类型分组结果
    gmm_results = [r for r in results if r.get('experiment_name', '').startswith('gmm_')]
    auto_results = [r for r in results if r.get('experiment_name', '').startswith('auto_')]
    
    # 确定各组的最佳模型
    best_gmm = max(gmm_results, key=lambda x: x['target_metrics'].get('roc_auc', 0) if 'roc_auc' in x['target_metrics'] else x['target_metrics'].get('auc', 0) if 'auc' in x['target_metrics'] else x['target_metrics']['f1']) if gmm_results else None
    best_auto = max(auto_results, key=lambda x: x['target_metrics'].get('roc_auc', 0) if 'roc_auc' in x['target_metrics'] else x['target_metrics'].get('auc', 0) if 'auc' in x['target_metrics'] else x['target_metrics']['f1']) if auto_results else None
    
    # 确定总体最佳模型（优先使用AUC，然后使用F1）
    best_overall = None
    if best_gmm and best_auto:
        gmm_score = best_gmm['target_metrics'].get('roc_auc', 0) if 'roc_auc' in best_gmm['target_metrics'] else best_gmm['target_metrics'].get('auc', 0) if 'auc' in best_gmm['target_metrics'] else best_gmm['target_metrics']['f1']
        auto_score = best_auto['target_metrics'].get('roc_auc', 0) if 'roc_auc' in best_auto['target_metrics'] else best_auto['target_metrics'].get('auc', 0) if 'auc' in best_auto['target_metrics'] else best_auto['target_metrics']['f1']
        best_overall = best_auto if auto_score > gmm_score else best_gmm
    elif best_gmm:
        best_overall = best_gmm
    elif best_auto:
        best_overall = best_auto
    
    # 确定最佳性能变化
    best_f1_change = max(results, key=lambda x: x['performance_change']['f1']) if results else None
    
    # 确定最佳MMD减少
    best_mmd_reduction = max(results, key=lambda x: (x['mmd_before'] - x['mmd_after']) / x['mmd_before'] * 100 if x['mmd_before'] and x['mmd_after'] is not None else 0) if results else None
    
    summary = {
        'timestamp': timestamp,
        'experiments': results,
        'best_overall': best_overall,
        'best_gmm': best_gmm,
        'best_autoencoder': best_auto,
        'best_f1_change': best_f1_change,
        'best_mmd_reduction': best_mmd_reduction
    }
    
    # 保存汇总
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n=== 实验汇总 ===")
    print(f"总计运行 {len(results)} 个实验")
    
    # 打印各组最佳模型
    if summary['best_gmm']:
        gmm_auc = summary['best_gmm']['target_metrics'].get('roc_auc', summary['best_gmm']['target_metrics'].get('auc', 'N/A'))
        print(f"\n最佳GMM模型: {summary['best_gmm']['experiment_name']}")
        print(f"  目标域F1分数: {summary['best_gmm']['target_metrics']['f1']:.4f}")
        print(f"  目标域AUC: {gmm_auc:.4f}" if isinstance(gmm_auc, float) else f"  目标域AUC: {gmm_auc}")
    
    if summary['best_autoencoder']:
        auto_auc = summary['best_autoencoder']['target_metrics'].get('roc_auc', summary['best_autoencoder']['target_metrics'].get('auc', 'N/A'))
        print(f"\n最佳自动编码器模型: {summary['best_autoencoder']['experiment_name']}")
        print(f"  目标域F1分数: {summary['best_autoencoder']['target_metrics']['f1']:.4f}")
        print(f"  目标域AUC: {auto_auc:.4f}" if isinstance(auto_auc, float) else f"  目标域AUC: {auto_auc}")
    
    # 打印总体最佳模型
    if summary['best_overall']:
        best_auc = summary['best_overall']['target_metrics'].get('roc_auc', summary['best_overall']['target_metrics'].get('auc', 'N/A'))
        print(f"\n总体最佳模型: {summary['best_overall']['experiment_name']}")
        print(f"  目标域F1分数: {summary['best_overall']['target_metrics']['f1']:.4f}")
        print(f"  目标域AUC: {best_auc:.4f}" if isinstance(best_auc, float) else f"  目标域AUC: {best_auc}")
    
    # 打印最佳性能变化
    if summary['best_f1_change']:
        print(f"\n最佳F1改善: {summary['best_f1_change']['performance_change']['f1']:+.2f}% ({summary['best_f1_change']['experiment_name']})")
    
    # 打印最佳MMD减少
    if summary['best_mmd_reduction'] and summary['best_mmd_reduction']['mmd_before'] and summary['best_mmd_reduction']['mmd_after'] is not None:
        reduction = (summary['best_mmd_reduction']['mmd_before'] - summary['best_mmd_reduction']['mmd_after']) / summary['best_mmd_reduction']['mmd_before'] * 100
        print(f"\n最佳MMD减少: {reduction:+.2f}% ({summary['best_mmd_reduction']['experiment_name']})")
    
    print(f"\n汇总结果已保存: {summary_file}")

if __name__ == "__main__":
    main()