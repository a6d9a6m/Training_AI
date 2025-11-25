#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
监督学习训练脚本

使用有标签的正常和异常样本训练GMM分类模型
"""

import os
import sys
import argparse
import numpy as np
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import load_supervised_data, get_audio_and_labels
from features.extract_features import extract_all_features
from models.gmm_model import GMMModel
from models.threshold_detector import find_optimal_threshold
from utils.evaluator import evaluate_model


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='监督学习训练 - 使用有标签的正常和异常样本')
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
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='测试集比例')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='验证集比例')
    return parser.parse_args()


def extract_features_from_data(data, sr, desc="数据"):
    """从数据中提取特征"""
    print(f"正在从{desc}提取特征...")
    audios, labels = get_audio_and_labels(data)

    features_list = []
    for i, audio in enumerate(audios):
        if (i + 1) % 10 == 0 or i == len(audios) - 1:
            print(f"  进度: {i + 1}/{len(audios)}")

        features, _ = extract_all_features(audio, sr)
        features_list.append(features)

    features_array = np.array(features_list)
    labels_array = np.array(labels)

    print(f"{desc}特征提取完成: {features_array.shape}")
    return features_array, labels_array


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 60)
    print("监督学习训练 - GMM分类模型")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_data, val_data, test_data = load_supervised_data(
        normal_dir=args.normal_dir,
        anomaly_dir=args.anomaly_dir,
        sr=args.sr,
        test_size=args.test_size,
        val_size=args.val_size
    )

    # 2. 提取特征
    print("\n[2/5] 提取特征...")
    start_time = time.time()

    train_features, train_labels = extract_features_from_data(train_data, args.sr, "训练集")
    val_features, val_labels = extract_features_from_data(val_data, args.sr, "验证集")
    test_features, test_labels = extract_features_from_data(test_data, args.sr, "测试集")

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 训练模型
    print("\n[3/5] 训练GMM模型...")
    print(f"  组件数量: {args.n_components}")

    model = GMMModel(n_components=args.n_components, covariance_type='diag')
    model.fit(train_features, train_labels)

    print("模型训练完成")

    # 4. 确定最佳阈值
    print("\n[4/5] 在验证集上确定最佳阈值...")
    optimal_threshold = find_optimal_threshold(
        model, val_features, val_labels,
        method='f1'
    )
    print(f"最佳阈值: {optimal_threshold:.4f}")

    # 5. 在测试集上评估
    print("\n[5/5] 在测试集上评估模型...")
    evaluator, metrics = evaluate_model(
        model, test_features, test_labels,
        threshold=optimal_threshold,
        target_names=['正常', '异常']
    )

    print("\n评估结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    print(f"混淆矩阵:")
    print(metrics['confusion_matrix'])

    # 6. 保存模型
    print("\n保存模型...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'supervised_gmm_model.pkl')
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存训练信息
    info = {
        'model_type': 'supervised_gmm',
        'n_components': args.n_components,
        'threshold': optimal_threshold,
        'sample_rate': args.sr,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        },
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data)
    }

    info_path = os.path.join(args.output_dir, 'supervised_gmm_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
