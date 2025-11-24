#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
异常检测训练脚本

只使用正常样本训练，学习正常模式，自动识别异常
"""

import os
import sys
import argparse
import numpy as np
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import load_anomaly_detection_data, get_audio_and_labels
from features.extract_features import extract_all_features
from models.autoencoder import AudioAutoencoder
from utils.evaluator import evaluate_model


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='异常检测训练 - 只使用正常样本训练')
    parser.add_argument('--normal_train_dir', type=str, required=True,
                        help='纯净正常样本目录（用于训练）')
    parser.add_argument('--mixed_test_dir', type=str,
                        help='混合样本目录（大部分正常，少量异常，用于测试）')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--encoding_dim', type=int, default=64,
                        help='编码器输出维度')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                        help='阈值百分位数（基于正常样本的重构误差）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
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
    print("异常检测训练 - 自动编码器模型")
    print("只使用正常样本训练，学习正常模式")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_data, val_data, test_data = load_anomaly_detection_data(
        normal_train_dir=args.normal_train_dir,
        mixed_test_dir=args.mixed_test_dir,
        sr=args.sr,
        val_ratio=args.val_ratio
    )

    # 2. 提取特征
    print("\n[2/5] 提取特征...")
    start_time = time.time()

    train_features, train_labels = extract_features_from_data(train_data, args.sr, "训练集")
    val_features, val_labels = extract_features_from_data(val_data, args.sr, "验证集")

    if len(test_data) > 0:
        test_features, test_labels = extract_features_from_data(test_data, args.sr, "测试集")
    else:
        print("未提供测试集，将使用验证集进行评估")
        test_features, test_labels = val_features, val_labels

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 训练模型
    print("\n[3/5] 训练自动编码器...")
    print(f"  输入维度: {train_features.shape[1]}")
    print(f"  编码维度: {args.encoding_dim}")
    print(f"  训练轮数: {args.epochs}")

    model = AudioAutoencoder(
        input_dim=train_features.shape[1],
        encoding_dim=args.encoding_dim,
        dropout_rate=0.3
    )

    # 训练（只使用正常样本）
    model.train(
        train_features=train_features,  # 全部是正常样本
        val_features=val_features,      # 全部是正常样本
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=0.001,
        patience=15
    )

    print("模型训练完成")

    # 4. 设置阈值
    print(f"\n[4/5] 基于正常样本设置阈值（第{args.threshold_percentile}百分位）...")

    # 使用验证集的正常样本设置阈值
    val_errors = model.calculate_reconstruction_error(val_features)
    threshold = np.percentile(val_errors, args.threshold_percentile)
    model.threshold = threshold

    print(f"阈值设置为: {threshold:.4f}")
    print(f"  正常样本平均误差: {np.mean(val_errors):.4f}")
    print(f"  正常样本误差标准差: {np.std(val_errors):.4f}")

    # 5. 在测试集上评估
    print("\n[5/5] 在测试集上评估模型...")

    # 计算测试集的重构误差
    test_errors = model.calculate_reconstruction_error(test_features)

    # 预测
    predictions = (test_errors > threshold).astype(int)

    # 计算指标
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    print("\n评估结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"混淆矩阵:")
    print(cm)

    # 打印误差统计
    if len(test_labels) > 0 and np.sum(test_labels == 1) > 0:
        normal_indices = test_labels == 0
        anomaly_indices = test_labels == 1

        if np.sum(normal_indices) > 0:
            normal_errors = test_errors[normal_indices]
            print(f"\n正常样本误差统计:")
            print(f"  平均: {np.mean(normal_errors):.4f}")
            print(f"  标准差: {np.std(normal_errors):.4f}")
            print(f"  最大: {np.max(normal_errors):.4f}")

        if np.sum(anomaly_indices) > 0:
            anomaly_errors = test_errors[anomaly_indices]
            print(f"\n异常样本误差统计:")
            print(f"  平均: {np.mean(anomaly_errors):.4f}")
            print(f"  标准差: {np.std(anomaly_errors):.4f}")
            print(f"  最小: {np.min(anomaly_errors):.4f}")

    # 6. 保存模型
    print("\n保存模型...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'anomaly_detection_model.pth')
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存训练信息
    info = {
        'model_type': 'anomaly_detection_autoencoder',
        'encoding_dim': args.encoding_dim,
        'threshold': float(threshold),
        'threshold_percentile': args.threshold_percentile,
        'sample_rate': args.sr,
        'epochs_trained': args.epochs,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data)
    }

    info_path = os.path.join(args.output_dir, 'anomaly_detection_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print("\n使用提示:")
    print(f"- 模型已学习正常声音的模式")
    print(f"- 重构误差 > {threshold:.4f} 的样本将被判定为异常")
    print(f"- 可以使用 realtime_detection.py 进行实时检测")


if __name__ == "__main__":
    main()
