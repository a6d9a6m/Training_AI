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
import hashlib

# 导入项目模块
from utils.config_manager import load_config, create_default_config
from utils.data_loader import load_dataset, load_audio
from features.extract_features import extract_all_features, extract_features_from_files
from models.gmm_model import GMMModel, train_gmm_model, find_optimal_components
from models.threshold_detector import ThresholdDetector, find_optimal_threshold
from models.autoencoder import AudioAutoencoder
from utils.evaluator import ModelEvaluator, evaluate_model
from utils.domain_adaptation import get_adaptation_method, calculate_mmd, calibrate_threshold, transfer_threshold_with_unlabeled, select_domain_invariant_features, ensemble_feature_selection, ensemble_adaptation, get_best_adaptation_method, MMDAdaptationTrainer


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='声音异常检测系统')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict', 'domain_shift'],
                        help='运行模式：train（训练）、evaluate（评估）、predict（预测）、domain_shift（领域偏移）')
    parser.add_argument('--audio_file', type=str,
                        help='在predict模式下需要预测的音频文件路径')
    parser.add_argument('--model_path', type=str,
                        help='预训练模型路径（用于evaluate或predict模式）')
    parser.add_argument('--threshold', type=float,
                        help='分类阈值（如果不提供，将使用验证集确定）')
    parser.add_argument('--adaptation_method', type=str,
                        help='领域适应方法（在domain_shift模式下使用）')
    return parser.parse_args()


def prepare_data(config):
    """
    准备数据：加载并分割数据集
    """
    print("正在准备数据...")
    
    sample_rate = config.get('data.sample_rate')
    enable_domain_shift = config.get('data.enable_domain_shift', False)
    
    # 检查是否使用设备类型加载数据
    if config.get('data.use_device_type', False):
        device_type = config.get('data.device_type', 'fan')  # 默认使用fan
        base_data_dir = config.get('paths.base_data_dir', 'dev_data')
        
        print(f"使用设备类型 '{device_type}' 加载数据")
        
        # 领域偏移模式
        if enable_domain_shift:
            print("启用领域偏移模式")
            source_train_data, source_test_data, target_train_data, target_test_data = load_dataset(
                device_type=device_type,
                base_data_dir=base_data_dir,
                sr=sample_rate,
                enable_domain_shift=True
            )
            
            print(f"领域偏移数据加载完成:")
            print(f"- 源域训练: {len(source_train_data)} 个样本")
            print(f"- 源域测试: {len(source_test_data)} 个样本")
            print(f"- 目标域训练: {len(target_train_data)} 个样本")
            print(f"- 目标域测试: {len(target_test_data)} 个样本")
            
            return source_train_data, source_test_data, target_train_data, target_test_data
        else:
            # 标准模式
            # 加载数据集 - 使用设备类型方式
            train_data, val_data, test_data = load_dataset(
                device_type=device_type,
                base_data_dir=base_data_dir,
                sr=sample_rate,
                test_size=config.get('data.test_size', 0.2),
                val_size=config.get('data.val_size', 0.2),
                random_state=config.get('data.random_state', 42),
                enable_domain_shift=False
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
            random_state=config.get('data.random_state', 42),
            enable_domain_shift=False
        )
    
    print(f"数据集加载完成: 训练集 {len(train_data)} 个样本, "
          f"验证集 {len(val_data)} 个样本, 测试集 {len(test_data)} 个样本")
    
    return train_data, val_data, test_data


def extract_features_wrapper(audio_data, config):
    """
    特征提取包装函数
    """
    audio, label = audio_data
    # 创建特征配置字典
    feature_config = {
        'mfcc': {'n_mfcc': config.get('features.n_mfcc', 13)},
        'melspectrogram': {'n_mels': config.get('features.n_mels', 128)},
        'chroma': {},
        'spectral_contrast': {},
        'zero_crossing_rate': {},
        'rms_energy': {}
    }
    # extract_all_features返回(combined_features, feature_dict)，我们只需要第一个
    combined_features, _ = extract_all_features(audio, config.get('data.sample_rate'), feature_config)
    return combined_features, label


def extract_dataset_features(dataset, config, dataset_name="dataset"):
    """
    从数据集中提取特征，支持特征缓存机制
    
    参数:
    dataset: 数据集，包含音频数据和标签
    config: 配置字典
    dataset_name: 数据集名称，用于生成缓存文件名
    
    返回:
    features_array: 特征数组
    labels_array: 标签数组
    """
    # 检查是否使用特征缓存
    use_cache = config.get('features.use_cache', True)
    features_dir = config.get('paths.features_save_path', './features/saved_features')
    
    # 创建缓存目录（如果不存在）
    if use_cache:
        os.makedirs(features_dir, exist_ok=True)
        
        # 生成缓存文件名（包含配置的哈希值，确保配置变化时重新生成特征）
        config_str = str(config.get('features', {}))
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        cache_file = os.path.join(features_dir, f"features_{dataset_name}_{len(dataset)}samples_{config_hash}.npz")
        
        # 尝试加载缓存特征
        if os.path.exists(cache_file):
            print(f"正在从缓存加载特征: {cache_file}")
            try:
                data = np.load(cache_file)
                features_array = data['features']
                labels_array = data['labels']
                print(f"特征加载成功，共 {len(features_array)} 个样本")
                return features_array, labels_array
            except Exception as e:
                print(f"加载缓存特征失败: {e}，将重新提取特征")
    
    # 如果没有缓存或加载失败，提取特征
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
    
    # 转换为NumPy数组
    features_array = np.array(features_list)
    labels_array = np.array(labels_list)
    
    # 保存特征到缓存
    if use_cache and len(features_list) > 0:
        try:
            np.savez_compressed(cache_file, features=features_array, labels=labels_array)
            print(f"特征已保存到缓存: {cache_file}")
        except Exception as e:
            print(f"保存特征缓存失败: {e}")
    
    return features_array, labels_array


def train_model(train_features, train_labels, config, val_features=None, val_labels=None, target_features=None):
    """
    训练模型
    支持自动编码器和GMM模型
    """
    print("开始训练模型...")
    
    # 获取模型类型
    model_type = config.get('model.type', 'autoencoder')
    
    if model_type == 'autoencoder':
        print("使用自动编码器模型进行无监督异常检测")
        
        # 只选择正常数据进行训练（标签为0的样本）
        if train_labels is not None:
            normal_indices = train_labels == 0
            if np.sum(normal_indices) > 0:
                train_features = train_features[normal_indices]
                print(f"使用 {np.sum(normal_indices)} 个正常样本进行训练")
            else:
                print("警告: 没有正常样本用于训练")
        
        # 创建自动编码器模型
        input_dim = train_features.shape[1]
        model = AudioAutoencoder(
            input_dim=input_dim,
            hidden_dims=config.get('model.autoencoder.hidden_dims', [64, 32, 64]),
            dropout=config.get('model.autoencoder.dropout', 0.3)
        )
        
        # 准备验证数据（只使用正常样本）
        val_normal_features = None
        if val_features is not None and val_labels is not None:
            val_normal_indices = val_labels == 0
            if np.sum(val_normal_indices) > 0:
                val_normal_features = val_features[val_normal_indices]
                print(f"使用 {np.sum(val_normal_indices)} 个正常样本进行验证")
        
        # 检查是否使用MMD领域适应
        use_mmd_adaptation = config.get('model.use_mmd_adaptation', False)
        
        if use_mmd_adaptation and target_features is not None:
            print("使用MMD领域适应进行训练")
            # 创建MMD适应训练器
            mmd_trainer = MMDAdaptationTrainer(
                model=model,
                lambda_mmd=config.get('model.mmd_lambda', 0.1),
                kernel_type=config.get('model.mmd_kernel', 'rbf'),
                learning_rate=config.get('model.learning_rate', 0.001)
            )
            
            # 训练模型
            mmd_trainer.train(
                source_features=train_features,
                target_features=target_features,
                val_source_features=val_normal_features,
                epochs=config.get('model.epochs', 100),
                batch_size=config.get('model.batch_size', 32),
                patience=config.get('model.patience', 10)
            )
        else:
            # 常规训练
            model.train(
                train_features=train_features,
                val_features=val_normal_features,
                epochs=config.get('model.epochs', 100),
                batch_size=config.get('model.batch_size', 32),
                learning_rate=config.get('model.learning_rate', 0.001),
                patience=config.get('model.patience', 10)
            )
        
        # 设置重构误差阈值
        if val_normal_features is not None:
            val_errors = model.calculate_reconstruction_error(val_normal_features)
            model.set_threshold(val_errors, percentile=config.get('model.threshold_percentile', 95))
        
        print("自动编码器模型训练完成")
        return model, None
    
    else:  # GMM模型
        print("使用GMM模型进行训练")
        
        # 检查是否需要寻找最佳组件数量
        if config.get('model.auto_find_components', True):
            print("正在寻找最佳组件数量...")
            min_components = config.get('model.min_components', 1)
            max_components = config.get('model.max_components', 10)
            cv_folds = config.get('model.cv_folds', 5)
            
            optimal_components, best_score, _ = find_optimal_components(
                train_features, train_labels,
                min_components=min_components,
                max_components=max_components,
                cv=cv_folds
            )
            print(f"最佳组件数量: {optimal_components}")
        else:
            optimal_components = config.get('model.n_components', 5)
        
        # 训练模型
        model = train_gmm_model(
            train_features, train_labels,
            n_components=optimal_components,
            covariance_type=config.get('model.covariance_type', 'diag')
        )
        
        print("GMM模型训练完成")
        return model, optimal_components


def determine_threshold(model, val_features, val_labels, config, threshold=None):
    """
    确定最佳分类阈值
    支持自动编码器和GMM模型
    """
    # 检查是否为自动编码器模型
    if hasattr(model, 'calculate_reconstruction_error'):
        print("自动编码器模型: 使用重构误差阈值")
        
        if threshold is not None:
            print(f"使用指定的阈值: {threshold}")
            model.threshold = threshold
            return threshold
        
        # 如果模型已经设置了阈值，直接返回
        if hasattr(model, 'threshold') and model.threshold is not None:
            print(f"使用模型已设置的阈值: {model.threshold}")
            return model.threshold
        
        if val_features is None or val_labels is None:
            # 如果有验证集但没有标签，使用验证集的正常样本重构误差设置阈值
            if val_features is not None:
                print("使用验证集的重构误差设置阈值（假设全部为正常样本）")
                errors = model.calculate_reconstruction_error(val_features)
                percentile = config.get('model.threshold_percentile', 95)
                optimal_threshold = np.percentile(errors, percentile)
                model.threshold = optimal_threshold
                print(f"基于验证集设置阈值 (第{percentile}百分位): {optimal_threshold}")
                return optimal_threshold
            else:
                print("没有验证集，使用默认阈值: 1.0")
                return 1.0
        
        # 使用正常样本的重构误差设置阈值
        normal_indices = val_labels == 0
        if np.sum(normal_indices) > 0:
            normal_features = val_features[normal_indices]
            print(f"使用 {np.sum(normal_indices)} 个正常样本设置阈值")
            errors = model.calculate_reconstruction_error(normal_features)
            percentile = config.get('model.threshold_percentile', 95)
            optimal_threshold = np.percentile(errors, percentile)
            model.threshold = optimal_threshold
            print(f"基于正常样本设置阈值 (第{percentile}百分位): {optimal_threshold}")
            return optimal_threshold
        else:
            print("验证集中没有正常样本，使用所有样本设置阈值")
            errors = model.calculate_reconstruction_error(val_features)
            percentile = config.get('model.threshold_percentile', 95)
            optimal_threshold = np.percentile(errors, percentile)
            model.threshold = optimal_threshold
            print(f"基于所有样本设置阈值 (第{percentile}百分位): {optimal_threshold}")
            return optimal_threshold
    
    # GMM模型阈值设置
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
        method=method
    )
    
    print(f"最佳阈值 ({method}): {optimal_threshold}")
    return optimal_threshold


def evaluate_final_model(model, test_features, test_labels, threshold, config):
    """
    评估最终模型性能
    支持自动编码器和GMM模型
    """
    print("\n正在评估模型性能...")
    
    # 使用评估函数
    output_dir = config.get('paths.plots_dir', None) if config.get('evaluation.plot_results', True) else None
    
    # 针对自动编码器模型的特殊处理
    if hasattr(model, 'calculate_reconstruction_error'):
        print("使用自动编码器模型评估")
        
        # 设置阈值
        if threshold is not None:
            model.threshold = threshold
        
        # 计算所有样本的重构误差
        all_errors = model.calculate_reconstruction_error(test_features)
        
        # 预测结果
        predictions = (all_errors > model.threshold).astype(int)
        
        # 使用评估函数
        evaluator, metrics = evaluate_model(
            model, test_features, test_labels, 
            threshold=threshold,
            target_names=['normal', 'anomaly'],
            output_dir=output_dir,
            # 传递重构误差作为分数
            scores=all_errors
        )
        
        # 额外打印重构误差统计信息
        print("\n重构误差统计:")
        if test_labels is not None:
            normal_indices = test_labels == 0
            anomaly_indices = test_labels == 1
            if np.sum(normal_indices) > 0:
                normal_errors = all_errors[normal_indices]
                print(f"正常样本平均误差: {np.mean(normal_errors):.4f}")
                print(f"正常样本误差标准差: {np.std(normal_errors):.4f}")
            if np.sum(anomaly_indices) > 0:
                anomaly_errors = all_errors[anomaly_indices]
                print(f"异常样本平均误差: {np.mean(anomaly_errors):.4f}")
                print(f"异常样本误差标准差: {np.std(anomaly_errors):.4f}")
        print(f"使用的阈值: {model.threshold:.4f}")
    else:
        # GMM模型评估
        evaluator, metrics = evaluate_model(
            model, test_features, test_labels, 
            threshold=threshold,
            target_names=['normal', 'anomaly'],
            output_dir=output_dir
        )
    
    # 打印评估指标
    print("\n评估指标:")
    print(f"准确率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"精确率 (Precision): {metrics['precision']:.4f}")
    print(f"召回率 (Recall): {metrics['recall']:.4f}")
    print(f"F1分数 (F1 Score): {metrics['f1']:.4f}")
    print(f"ROC AUC: {metrics.get('roc_auc', 'N/A')}")
    print(f"混淆矩阵:")
    print(metrics['confusion_matrix'])
    
    return metrics


def save_results(model, threshold, optimal_components, config):
    """
    保存模型和结果
    支持自动编码器和GMM模型
    """
    model_dir = config.get('paths.model_dir')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 根据模型类型保存
    if hasattr(model, 'calculate_reconstruction_error'):
        # 自动编码器模型
        model_path = os.path.join(model_dir, 'autoencoder_model.pth')
        model.save(model_path)
        print(f"自动编码器模型已保存至: {model_path}")
        
        # 保存配置信息
        import json
        results = {
            'threshold': threshold,
            'model_type': 'autoencoder',
            'config': dict(config.config)
        }
    else:
        # GMM模型
        model_path = os.path.join(model_dir, 'gmm_model.pkl')
        model.save(model_path)
        print(f"GMM模型已保存至: {model_path}")
        
        # 保存配置信息
        import json
        results = {
            'threshold': threshold,
            'optimal_components': optimal_components,
            'model_type': 'gmm',
            'config': dict(config.config)
        }
    
    results_path = os.path.join(model_dir, 'training_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"训练结果已保存至: {results_path}")


def predict_audio_file(audio_path, model, config, threshold):
    """
    预测单个音频文件
    支持自动编码器和GMM模型
    """
    print(f"预测音频文件: {audio_path}")
    
    # 加载并预处理音频
    audio, sr = load_audio(audio_path, sr=config.get('data.sample_rate'))
    
    # 提取特征
    features, _ = extract_all_features(
        audio, 
        sr=sr,
        feature_config={
            'mfcc': {'n_mfcc': config.get('features.n_mfcc', 13)},
            'melspectrogram': {'n_mels': config.get('features.n_mels', 128)},
            'chroma': {},
            'spectral_contrast': {},
            'zero_crossing_rate': {},
            'rms_energy': {}
        }
    )
    
    # 根据模型类型进行预测
    if hasattr(model, 'calculate_reconstruction_error'):
        # 自动编码器模型预测
        print("使用自动编码器模型进行预测")
        
        # 设置阈值
        if threshold is not None:
            model.threshold = threshold
        
        # 计算重构误差
        error = model.calculate_reconstruction_error(features.reshape(1, -1))[0]
        
        # 判断是否异常
        prediction = 1 if error > model.threshold else 0
        
        # 显示结果
        result = '异常' if prediction == 1 else '正常'
        print(f"\n预测结果: {result}")
        print(f"重构误差: {error:.4f}")
        print(f"使用阈值: {model.threshold:.4f}")
        
        return {
            'prediction': result,
            'prediction_code': prediction,
            'reconstruction_error': error,
            'threshold': model.threshold
        }
    else:
        # GMM模型预测
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
    train_features, train_labels = extract_dataset_features(train_data, config, dataset_name="train")
    val_features, val_labels = extract_dataset_features(val_data, config, dataset_name="val")
    test_features, test_labels = extract_dataset_features(test_data, config, dataset_name="test")
    
    # 训练模型（只使用正常数据，autoencoder模型会在内部筛选）
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

def domain_shift_mode(config, args):
    """
    领域偏移模式：在源域训练模型，在目标域测试
    """
    print("\n=== 领域偏移训练与评估 ===")
    
    # 准备数据 - 领域偏移模式
    source_train_data, source_test_data, target_train_data, target_test_data = prepare_data(config)
    
    # 提取特征
    print("\n提取源域特征...")
    source_train_features, source_train_labels = extract_dataset_features(source_train_data, config, dataset_name="source_train")
    source_test_features, source_test_labels = extract_dataset_features(source_test_data, config, dataset_name="source_test")
    
    print("\n提取目标域特征...")
    target_train_features, target_train_labels = extract_dataset_features(target_train_data, config, dataset_name="target_train")
    target_test_features, target_test_labels = extract_dataset_features(target_test_data, config, dataset_name="target_test")
    
    # 计算领域差异
    print("\n计算领域差异 (MMD)...")
    if len(source_train_features) > 0 and len(target_train_features) > 0:
        mmd_before = calculate_mmd(source_train_features, target_train_features, kernel='rbf')
        print(f"源域与目标域的MMD距离 (适应前): {mmd_before:.6f}")
    else:
        mmd_before = None
        print("警告: 源域或目标域训练样本不足，无法计算MMD")
    
    # 应用特征选择
    enable_feature_selection = config.get('domain_shift.feature_selection', {}).get('enabled', False)
    feature_selection_method = config.get('domain_shift.feature_selection', {}).get('method', 'mmd_based')
    feature_selection_ratio = config.get('domain_shift.feature_selection', {}).get('ratio', 0.8)
    use_ensemble_selection = config.get('domain_shift.feature_selection', {}).get('ensemble', False)
    
    selected_source_train_features = source_train_features
    selected_source_test_features = source_test_features
    selected_target_train_features = target_train_features
    selected_target_test_features = target_test_features
    feature_selector = None
    selected_feature_indices = None
    
    if enable_feature_selection:
        print(f"\n应用特征选择 (方法: {feature_selection_method}, 比例: {feature_selection_ratio})...")
        try:
            if use_ensemble_selection:
                # 使用集成特征选择
                methods = config.get('domain_shift.feature_selection', {}).get('ensemble_methods', 
                                                                          ['f_statistic', 'mutual_info', 'mmd_based'])
                selected_source_train_features, selected_target_train_features, selected_feature_indices, _ = ensemble_feature_selection(
                    source_train_features, source_train_labels, target_train_features,
                    methods=methods, k=feature_selection_ratio
                )
                # 对测试集应用相同的特征选择
                selected_source_test_features = source_test_features[:, selected_feature_indices]
                selected_target_test_features = target_test_features[:, selected_feature_indices]
            else:
                # 使用单一特征选择方法
                selected_source_train_features, selected_target_train_features, feature_selector, selected_feature_indices = select_domain_invariant_features(
                    source_train_features, source_train_labels, target_train_features,
                    method=feature_selection_method, k=feature_selection_ratio
                )
                # 对测试集应用相同的特征选择
                if feature_selector:
                    selected_source_test_features = feature_selector.transform(source_test_features)
                    selected_target_test_features = feature_selector.transform(target_test_features)
                else:
                    selected_source_test_features = source_test_features[:, selected_feature_indices]
                    selected_target_test_features = target_test_features[:, selected_feature_indices]
            
            print(f"特征选择后: 源域特征维度 {selected_source_train_features.shape[1]}, 目标域特征维度 {selected_target_train_features.shape[1]}")
        except Exception as e:
            print(f"特征选择失败: {e}")
            print("继续使用原始特征")
            selected_source_train_features = source_train_features
            selected_source_test_features = source_test_features
            selected_target_train_features = target_train_features
            selected_target_test_features = target_test_features
    else:
        print("未启用特征选择，使用原始特征")
    
    # 应用领域适应
    adaptation_method = args.adaptation_method or config.get('domain_shift.adaptation_method', 'mean_shift')
    print(f"\n应用领域适应方法: {adaptation_method}")
    
    adapted_source_train_features = selected_source_train_features.copy()
    adapted_source_test_features = selected_source_test_features.copy()
    adapt_model = None
    domain_shift_config = config.get('domain_shift', {})
    
    try:
        # 处理集成适应或自动方法选择
        if adaptation_method == 'ensemble':
            # 集成多种适应方法
            ensemble_config = domain_shift_config.get('ensemble_config', {})
            methods = ensemble_config.get('methods', ['standardization', 'coral_adaptation', 'mean_shift'])
            strategy = ensemble_config.get('strategy', 'weighted_average')
            weights = ensemble_config.get('weights', None)
            adapt_config = ensemble_config.get('adaptation_config', {})
            
            print(f"使用集成策略: {strategy}")
            print(f"集成方法: {', '.join(methods)}")
            
            adapted_source_train_features, method_results = ensemble_adaptation(
                selected_source_train_features, selected_target_train_features,
                methods=methods, weights=weights, strategy=strategy, adaptation_config=adapt_config
            )
            
            # 转换测试集
            if len(adapted_source_test_features) > 0:
                test_results = []
                for method_name in methods:
                    if method_name in method_results:
                        if 'model' in method_results[method_name]:
                            # 对于有model的方法（如标准化）
                            test_result = method_results[method_name]['model'].transform(adapted_source_test_features)
                        elif method_name in ['coral_adaptation', 'joint_distribution_adaptation']:
                            # 对于需要参考目标域的方法
                            adapt_func = get_adaptation_method(method_name)
                            test_result = adapt_func(adapted_source_test_features, selected_target_train_features)
                        else:
                            # 其他方法
                            adapt_func = get_adaptation_method(method_name)
                            test_result = adapt_func(adapted_source_test_features, adapted_source_test_features)
                        test_results.append(test_result)
                
                # 对测试集应用相同的集成策略
                if strategy == 'weighted_average':
                    if weights is None:
                        weights = np.ones(len(methods)) / len(methods)
                    else:
                        weights = np.array(weights) / np.sum(weights)
                    
                    adapted_source_test_features = np.zeros_like(adapted_source_test_features)
                    for i, result in enumerate(test_results):
                        adapted_source_test_features += result * weights[i]
                elif strategy == 'dynamic_weighting':
                    # 对于动态权重，使用与训练相同的权重策略
                    # 这里简化处理，使用等权重
                    adapted_source_test_features = np.mean(test_results, axis=0)
                else:
                    # 其他策略简化处理
                    adapted_source_test_features = np.mean(test_results, axis=0)
            
            adapted_target_train_features = selected_target_train_features  # 集成适应后目标域特征保持不变
            
        elif adaptation_method == 'auto':
            # 自动选择最佳适应方法
            print("自动选择最佳适应方法...")
            methods_to_try = domain_shift_config.get('methods_to_try', 
                ['standardization', 'minmax', 'coral_adaptation', 'mean_shift', 'joint_distribution_adaptation', 'pca'])
            
            best_method, best_score = get_best_adaptation_method(
                selected_source_train_features, source_train_labels,
                selected_target_train_features, 
                target_labels=target_train_labels if len(target_train_labels) > 0 else None,
                methods=methods_to_try
            )
            
            print(f"最佳方法: {best_method}, 得分: {best_score:.4f}")
            
            # 应用最佳方法
            adaptation_method = best_method
            adapt_func = get_adaptation_method(adaptation_method)
            
            # 根据方法类型应用不同的领域适应
            if adaptation_method in ['standardization', 'minmax', 'robust_scaling']:
                adapted_source_train_features, adapt_model = adapt_func(selected_source_train_features, selected_target_train_features)
                if len(adapted_source_test_features) > 0:
                    adapted_source_test_features = adapt_model.transform(adapted_source_test_features)
            elif adaptation_method == 'pca':
                pca_components = domain_shift_config.get('pca_components', None)
                adapted_source_train_features, adapted_target_train_features, adapt_model = adapt_func(
                    selected_source_train_features, selected_target_train_features, n_components=pca_components
                )
                if len(adapted_source_test_features) > 0:
                    adapted_source_test_features = adapt_model.transform(adapted_source_test_features)
                    selected_target_test_features = adapt_model.transform(selected_target_test_features)
            else:
                adapted_source_train_features = adapt_func(selected_source_train_features, selected_target_train_features)
                if adaptation_method in ['coral', 'joint_distribution', 'coral_adaptation', 'joint_distribution_adaptation']:
                    adapted_target_train_features = adapt_func(selected_target_train_features, selected_target_train_features)
                if len(adapted_source_test_features) > 0:
                    adapted_source_test_features = adapt_func(adapted_source_test_features, selected_target_train_features)
        
        else:
            # 单一适应方法
            adapt_func = get_adaptation_method(adaptation_method)
            
            # 根据方法类型应用不同的领域适应
            if adaptation_method in ['standardization', 'minmax']:
                adapted_source_train_features, adapt_model = adapt_func(selected_source_train_features, selected_target_train_features)
                # 使用相同的转换模型处理测试集
                if adaptation_method == 'standardization':
                    adapted_source_test_features = adapt_model.transform(adapted_source_test_features)
                else:  # minmax
                    adapted_source_test_features = adapt_model.transform(adapted_source_test_features)
            elif adaptation_method == 'pca':
                adapted_source_train_features, adapted_target_train_features, adapt_model = adapt_func(
                    selected_source_train_features, selected_target_train_features,
                    n_components=config.get('domain_shift.pca_components', None)
                )
                # 使用相同的PCA模型处理测试集
                adapted_source_test_features = adapt_model.transform(adapted_source_test_features)
                selected_target_test_features = adapt_model.transform(selected_target_test_features)
            else:
                # mean_shift, coral等方法
                adapted_source_train_features = adapt_func(selected_source_train_features, selected_target_train_features)
                adapted_source_test_features = adapt_func(selected_source_test_features, selected_target_train_features)
        
        # 重新计算MMD
        if mmd_before is not None:
            mmd_after = calculate_mmd(adapted_source_train_features, selected_target_train_features, kernel='rbf')
            print(f"源域与目标域的MMD距离 (适应后): {mmd_after:.6f}")
            print(f"MMD改善率: {(mmd_before - mmd_after) / mmd_before * 100:.2f}%")
            
    except Exception as e:
        print(f"应用领域适应时出错: {e}")
        print("继续使用原始特征")
    
    # 在源域训练模型（只使用正常数据）
    print("\n在源域训练模型...")
    
    # 准备目标域正常数据（用于MMD领域适应）
    target_normal_features = None
    if len(target_train_features) > 0 and len(target_train_labels) > 0:
        target_normal_indices = target_train_labels == 0
        if np.sum(target_normal_indices) > 0:
            target_normal_features = target_train_features[target_normal_indices]
            print(f"目标域正常样本数量: {np.sum(target_normal_indices)}")
    
    # 训练模型
    model, optimal_components = train_model(
        adapted_source_train_features, source_train_labels, config,
        val_features=adapted_source_test_features,
        val_labels=source_test_labels,
        target_features=target_normal_features
    )
    
    # 确定阈值（使用源域验证集或测试集）
    print("\n确定最佳阈值...")
    source_threshold = determine_threshold(
        model, adapted_source_test_features, source_test_labels, config,
        threshold=args.threshold
    )
    
    # 阈值校准设置
    enable_threshold_calibration = config.get('domain_shift.threshold_calibration', {}).get('enabled', True)
    calibration_method = config.get('domain_shift.threshold_calibration', {}).get('method', 'f1_optimization')
    
    # 应用阈值校准
    calibrated_threshold = source_threshold
    calibration_score = 0.0
    
    if enable_threshold_calibration:
        print(f"\n在目标域上校准阈值 (方法: {calibration_method})...")
        try:
            # 检查是否有目标域标签可用于校准
            if len(target_test_labels) > 0:
                # 使用有标签的目标域数据进行校准
                calibrated_threshold, calibration_score = calibrate_threshold(
                    model, target_test_features, target_test_labels, source_threshold, calibration_method
                )
            else:
                # 使用未标记的目标域数据进行阈值迁移
                print("目标域没有标签，使用未标记数据进行阈值迁移...")
                calibrated_threshold = transfer_threshold_with_unlabeled(
                    model, target_train_features, source_threshold, method='distribution_matching'
                )
        except Exception as e:
            print(f"阈值校准失败: {e}")
            print("继续使用源域阈值...")
            calibrated_threshold = source_threshold
    else:
        print("未启用阈值校准，使用源域阈值")
    
    # 在源域评估
    print("\n在源域评估模型性能...")
    source_metrics = evaluate_final_model(
        model, adapted_source_test_features, source_test_labels, source_threshold, config
    )
    
    # 在目标域评估
    print("\n在目标域评估模型性能...")
    target_metrics = evaluate_final_model(
        model, selected_target_test_features, target_test_labels, calibrated_threshold, config
    )
    
    # 保存结果
    print("\n保存结果...")
    model_dir = config.get('paths.model_dir')
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型
    model_path = os.path.join(model_dir, f'gmm_domain_shift_model_{adaptation_method}.pkl')
    model.save(model_path)
    print(f"领域偏移模型已保存至: {model_path}")
    
    # 保存领域适应模型（如果有）
    if adapt_model is not None:
        import joblib
        adapt_model_path = os.path.join(model_dir, f'adapt_model_{adaptation_method}.pkl')
        joblib.dump(adapt_model, adapt_model_path)
        print(f"领域适应模型已保存至: {adapt_model_path}")
    
    # 保存配置和结果
    import json
    
    # 辅助函数：将numpy对象转换为可JSON序列化的类型
    def convert_numpy_to_json_serializable(obj):
        import numpy as np
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy_to_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_to_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_to_json_serializable(item) for item in obj)
        else:
            return obj
    
    results = {
        'source_threshold': source_threshold,
        'calibrated_threshold': calibrated_threshold,
        'optimal_components': optimal_components,
        'adaptation_method': adaptation_method,
        'mmd_before': mmd_before,
        'mmd_after': mmd_after if 'mmd_after' in locals() else None,
        'source_domain_metrics': source_metrics,
        'target_domain_metrics': target_metrics,
        'threshold_calibration_enabled': enable_threshold_calibration,
        'threshold_calibration_method': calibration_method,
        'calibration_score': calibration_score,
        'feature_selection_enabled': enable_feature_selection,
        'feature_selection_method': feature_selection_method,
        'feature_selection_ratio': feature_selection_ratio,
        'use_ensemble_selection': use_ensemble_selection,
        'selected_feature_count': len(selected_feature_indices) if selected_feature_indices is not None else None,
        'original_feature_count': source_train_features.shape[1],
        'config': dict(config.config)
    }
    
    results_path = os.path.join(model_dir, f'domain_shift_results_{adaptation_method}.json')
    
    # 转换结果字典中的numpy对象为可JSON序列化的类型
    serializable_results = convert_numpy_to_json_serializable(results)
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"领域偏移训练结果已保存至: {results_path}")
    
    # 打印对比结果
    print("\n=== 领域偏移结果对比 ===")
    print("源域性能:")
    print(f"  准确率: {source_metrics['accuracy']:.4f}")
    print(f"  F1分数: {source_metrics['f1']:.4f}")
    
    print("\n目标域性能:")
    print(f"  准确率: {target_metrics['accuracy']:.4f}")
    print(f"  F1分数: {target_metrics['f1']:.4f}")
    print(f"\n阈值:")
    print(f"  源域: {source_threshold:.4f}")
    print(f"  校准后: {calibrated_threshold:.4f}")
    
    # 计算阈值校准带来的改善
    if enable_threshold_calibration:
        try:
            # 获取模型在目标域上使用源域阈值的性能
            detector = ThresholdDetector(model)
            _, source_threshold_metrics = evaluate_model(
                model, target_test_features, target_test_labels,
                threshold=source_threshold,
                target_names=['normal', 'anomaly']
            )
            
            f1_before = source_threshold_metrics['f1']
            f1_after = target_metrics['f1']
            
            if f1_before > 0:
                f1_improvement = (f1_after - f1_before) / f1_before * 100
                print(f"\n阈值校准带来的F1改善: {f1_improvement:.2f}%")
        except Exception as e:
            print(f"\n计算阈值校准改善时出错: {e}")
    
    if target_metrics['f1'] > source_metrics['f1']:
        print("\n✅ 领域适应成功: 目标域性能优于源域")
    elif target_metrics['f1'] < source_metrics['f1']:
        print("\n⚠️ 领域适应效果不佳: 目标域性能低于源域")
    else:
        print("\n➡️ 领域适应中性: 目标域性能与源域相当")
    
    return model, calibrated_threshold, source_metrics, target_metrics


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
    test_features, test_labels = extract_dataset_features(test_data, config, dataset_name="test")
    
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
    elif args.mode == 'domain_shift':
        # 设置领域偏移模式标志
        config.config['data']['enable_domain_shift'] = True
        domain_shift_mode(config, args)
    
    print("\n程序执行完成")


if __name__ == "__main__":
    main()