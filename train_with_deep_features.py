#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用深度特征的训练脚本 - 尝试不同的特征表示
"""

import os
import sys
import argparse
import numpy as np
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import load_supervised_data, get_audio_and_labels
from features.deep_features import extract_multi_resolution_features, extract_temporal_context_features
from models.gmm_model import GMMModel
from models.threshold_detector import find_optimal_threshold
from utils.evaluator import evaluate_model
from sklearn.preprocessing import StandardScaler


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用深度特征的监督学习训练')
    parser.add_argument('--normal_dir', type=str, required=True,
                        help='正常样本目录')
    parser.add_argument('--anomaly_dir', type=str, required=True,
                        help='异常样本目录')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--n_components', type=int, default=5,
                        help='GMM组件数量')
    parser.add_argument('--feature_type', type=str, default='multi_resolution',
                        choices=['multi_resolution', 'temporal_context', 'both'],
                        help='特征类型')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    return parser.parse_args()


def extract_features_from_data(data, sr, feature_type, desc="数据"):
    """从数据中提取深度特征"""
    print(f"正在从{desc}提取 {feature_type} 特征...")
    audios, labels = get_audio_and_labels(data)

    features_list = []
    for i, audio in enumerate(audios):
        if (i + 1) % 10 == 0 or i == len(audios) - 1:
            print(f"  进度: {i + 1}/{len(audios)}")

        if feature_type == 'multi_resolution':
            features = extract_multi_resolution_features(audio, sr)
        elif feature_type == 'temporal_context':
            features = extract_temporal_context_features(audio, sr)
        elif feature_type == 'both':
            feat1 = extract_multi_resolution_features(audio, sr)
            feat2 = extract_temporal_context_features(audio, sr)
            features = np.hstack([feat1, feat2])
        else:
            raise ValueError(f"未知的特征类型: {feature_type}")

        features_list.append(features)

    features_array = np.array(features_list)
    labels_array = np.array(labels)

    print(f"{desc}特征提取完成: {features_array.shape}")
    return features_array, labels_array


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 60)
    print("深度特征训练 - GMM分类模型")
    print(f"特征类型: {args.feature_type}")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_data, val_data, test_data = load_supervised_data(
        normal_dir=args.normal_dir,
        anomaly_dir=args.anomaly_dir,
        sr=args.sr,
        test_size=args.test_size,
        val_size=args.val_size
    )

    # 2. 提取特征
    print("\n[2/6] 提取深度特征...")
    start_time = time.time()

    train_features, train_labels = extract_features_from_data(
        train_data, args.sr, args.feature_type, "训练集"
    )
    val_features, val_labels = extract_features_from_data(
        val_data, args.sr, args.feature_type, "验证集"
    )
    test_features, test_labels = extract_features_from_data(
        test_data, args.sr, args.feature_type, "测试集"
    )

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 特征标准化
    print("\n[3/6] 特征标准化...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 4. 训练模型
    print("\n[4/6] 训练GMM模型...")
    print(f"  组件数量: {args.n_components}")

    model = GMMModel(n_components=args.n_components, covariance_type='diag')
    model.fit(train_features_scaled, train_labels)

    print("模型训练完成")

    # 5. 确定最佳阈值
    print("\n[5/6] 在验证集上确定最佳阈值...")
    optimal_threshold = find_optimal_threshold(
        model, val_features_scaled, val_labels,
        method='f1'
    )
    print(f"最佳阈值: {optimal_threshold:.4f}")

    # 6. 在测试集上评估
    print("\n[6/6] 在测试集上评估模型...")
    evaluator, metrics = evaluate_model(
        model, test_features_scaled, test_labels,
        threshold=optimal_threshold,
        target_names=['正常', '异常']
    )

    print("\n评估结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"  AUC: {metrics['roc_auc']:.4f}")
    print(f"混淆矩阵:")
    print(metrics['confusion_matrix'])

    # 7. 保存模型
    print("\n保存模型和预处理器...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(args.output_dir, 'deep_features_gmm_model.pkl')
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存预处理器
    import pickle
    preprocessor_path = os.path.join(args.output_dir, 'deep_features_preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'feature_type': args.feature_type
        }, f)
    print(f"预处理器已保存: {preprocessor_path}")

    # 保存训练信息
    info = {
        'model_type': 'deep_features_gmm',
        'feature_type': args.feature_type,
        'n_components': args.n_components,
        'threshold': optimal_threshold,
        'sample_rate': args.sr,
        'n_features': train_features_scaled.shape[1],
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['roc_auc']) if 'roc_auc' in metrics and metrics['roc_auc'] is not None else None
        },
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data)
    }

    info_path = os.path.join(args.output_dir, 'deep_features_gmm_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 性能评估
    if metrics['f1'] >= 0.8:
        print("\n性能评估: 优秀 (F1 >= 0.8)")
    elif metrics['f1'] >= 0.7:
        print("\n性能评估: 良好 (F1 >= 0.7)")
    elif metrics['f1'] >= 0.6:
        print("\n性能评估: 一般 (F1 >= 0.6)")
    else:
        print("\n性能评估: 需要改进 (F1 < 0.6)")
        print("\n这表明数据本身可能存在问题:")
        print("  1. 正常和异常样本在听感上没有明显差异")
        print("  2. 数据标签可能有误")
        print("  3. 建议使用异常检测模式: python train_anomaly_detection.py")


if __name__ == "__main__":
    main()
