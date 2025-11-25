#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于评分的异常检测 - 简单直观的异常检测方法

思路：
1. 对每个声音计算异常评分
2. 在正常样本上统计得分分布，确定正常区间
3. 超出正常区间的判定为异常
"""

import os
import sys
import argparse
import numpy as np
import json
import pickle
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from features.industrial_features import extract_industrial_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import librosa


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于评分的异常检测')
    parser.add_argument('--normal_train_dir', type=str, required=True,
                        help='正常样本目录（用于训练）')
    parser.add_argument('--anomaly_test_dir', type=str, required=True,
                        help='异常测试样本目录')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--denoise', action='store_true', default=True,
                        help='是否进行背景噪声抑制')
    parser.add_argument('--no_denoise', action='store_false', dest='denoise',
                        help='不进行背景噪声抑制')
    parser.add_argument('--score_method', type=str, default='mahalanobis',
                        choices=['mahalanobis', 'euclidean', 'pca_reconstruction'],
                        help='评分方法')
    parser.add_argument('--sigma_threshold', type=float, default=2.0,
                        help='正常区间：均值 ± sigma_threshold * 标准差')
    return parser.parse_args()


def calculate_mahalanobis_score(X, mean, cov_inv):
    """
    计算马氏距离分数

    马氏距离考虑了特征之间的相关性，比欧氏距离更适合多维数据
    分数越大 = 越异常
    """
    diff = X - mean
    score = np.sqrt(np.sum(np.dot(diff, cov_inv) * diff, axis=1))
    return score


def calculate_euclidean_score(X, mean):
    """
    计算欧氏距离分数

    简单的距离度量
    分数越大 = 越异常
    """
    diff = X - mean
    score = np.sqrt(np.sum(diff ** 2, axis=1))
    return score


def calculate_pca_reconstruction_score(X, pca, mean):
    """
    计算PCA重构误差分数

    通过降维再重构，异常样本的重构误差会更大
    分数越大 = 越异常
    """
    # 投影到主成分空间
    X_transformed = pca.transform(X)

    # 重构
    X_reconstructed = pca.inverse_transform(X_transformed)

    # 重构误差
    score = np.sqrt(np.sum((X - X_reconstructed) ** 2, axis=1))

    return score


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 70)
    print("基于评分的异常检测")
    print("=" * 70)
    print(f"评分方法: {args.score_method}")
    print(f"正常区间: 均值 ± {args.sigma_threshold} * 标准差")
    print(f"降噪: {'启用' if args.denoise else '禁用'}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[步骤 1/6] 加载数据...")

    # 加载正常训练样本
    normal_files = glob(os.path.join(args.normal_train_dir, "*.wav"))
    print(f"  正常训练样本: {len(normal_files)} 个")

    normal_train_data = []
    for i, file_path in enumerate(normal_files):
        if (i + 1) % 20 == 0:
            print(f"    加载进度: {i + 1}/{len(normal_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        normal_train_data.append(audio)

    # 自动分出20%正常样本作为测试
    from sklearn.model_selection import train_test_split
    normal_train, normal_test = train_test_split(
        normal_train_data, test_size=0.2, random_state=42
    )

    print(f"  正常样本分割: 训练 {len(normal_train)}, 测试 {len(normal_test)}")

    # 加载异常测试样本
    anomaly_files = glob(os.path.join(args.anomaly_test_dir, "*.wav"))
    print(f"  异常测试样本: {len(anomaly_files)} 个")

    anomaly_test_data = []
    for i, file_path in enumerate(anomaly_files):
        if (i + 1) % 20 == 0:
            print(f"    加载进度: {i + 1}/{len(anomaly_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        anomaly_test_data.append(audio)

    # 2. 提取特征
    print("\n[步骤 2/6] 提取特征...")

    print("  正常训练样本...")
    train_features = []
    for i, audio in enumerate(normal_train):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(normal_train)}")
        features = extract_industrial_features(audio, args.sr, denoise=args.denoise)
        train_features.append(features)
    train_features = np.array(train_features)

    print(f"  训练集特征: {train_features.shape}")

    # 测试集特征
    test_features = []
    test_labels = []

    print("  正常测试样本...")
    for audio in normal_test:
        features = extract_industrial_features(audio, args.sr, denoise=args.denoise)
        test_features.append(features)
        test_labels.append(0)

    print("  异常测试样本...")
    for i, audio in enumerate(anomaly_test_data):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_test_data)}")
        features = extract_industrial_features(audio, args.sr, denoise=args.denoise)
        test_features.append(features)
        test_labels.append(1)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(f"  测试集特征: {test_features.shape} (正常: {np.sum(test_labels==0)}, 异常: {np.sum(test_labels==1)})")

    # 3. 特征标准化
    print("\n[步骤 3/6] 特征标准化...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)

    # 4. 计算正常样本的统计特征
    print(f"\n[步骤 4/6] 在正常样本上建立评分模型 (方法: {args.score_method})...")

    if args.score_method == 'mahalanobis':
        # 计算均值和协方差矩阵
        mean = np.mean(train_features_scaled, axis=0)
        cov = np.cov(train_features_scaled.T)

        # 添加正则化避免奇异矩阵
        cov += np.eye(cov.shape[0]) * 1e-6
        cov_inv = np.linalg.inv(cov)

        # 计算训练集的分数（用于确定正常区间）
        train_scores = calculate_mahalanobis_score(train_features_scaled, mean, cov_inv)

        score_model = {
            'method': 'mahalanobis',
            'mean': mean,
            'cov_inv': cov_inv
        }

    elif args.score_method == 'euclidean':
        # 计算均值
        mean = np.mean(train_features_scaled, axis=0)

        # 计算训练集的分数
        train_scores = calculate_euclidean_score(train_features_scaled, mean)

        score_model = {
            'method': 'euclidean',
            'mean': mean
        }

    elif args.score_method == 'pca_reconstruction':
        # PCA降维
        pca = PCA(n_components=min(50, train_features_scaled.shape[1] // 2))
        pca.fit(train_features_scaled)

        mean = np.mean(train_features_scaled, axis=0)

        # 计算训练集的分数
        train_scores = calculate_pca_reconstruction_score(train_features_scaled, pca, mean)

        score_model = {
            'method': 'pca_reconstruction',
            'mean': mean,
            'pca': pca
        }

    # 5. 确定正常区间
    print("\n[步骤 5/6] 确定正常得分区间...")

    score_mean = np.mean(train_scores)
    score_std = np.std(train_scores)

    # 正常区间：均值 ± sigma_threshold * 标准差
    normal_min = score_mean - args.sigma_threshold * score_std
    normal_max = score_mean + args.sigma_threshold * score_std

    print(f"  训练集得分统计:")
    print(f"    均值: {score_mean:.4f}")
    print(f"    标准差: {score_std:.4f}")
    print(f"    最小值: {np.min(train_scores):.4f}")
    print(f"    最大值: {np.max(train_scores):.4f}")
    print(f"\n  正常区间: [{normal_min:.4f}, {normal_max:.4f}]")
    print(f"    (均值 ± {args.sigma_threshold} * 标准差)")

    threshold = normal_max

    # 6. 在测试集上评估
    print("\n[步骤 6/6] 在测试集上评估...")

    # 计算测试集的分数
    if args.score_method == 'mahalanobis':
        test_scores = calculate_mahalanobis_score(
            test_features_scaled, score_model['mean'], score_model['cov_inv']
        )
    elif args.score_method == 'euclidean':
        test_scores = calculate_euclidean_score(
            test_features_scaled, score_model['mean']
        )
    elif args.score_method == 'pca_reconstruction':
        test_scores = calculate_pca_reconstruction_score(
            test_features_scaled, score_model['pca'], score_model['mean']
        )

    # 预测：得分超出正常区间 = 异常
    predictions = (test_scores > threshold).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    try:
        auc = roc_auc_score(test_labels, test_scores)
    except:
        auc = None

    print("\n" + "=" * 70)
    print("评估结果:")
    print("=" * 70)
    print(f"  准确率:   {accuracy:.4f}")
    print(f"  精确率:   {precision:.4f}")
    print(f"  召回率:   {recall:.4f}")
    print(f"  F1分数:   {f1:.4f}")
    if auc is not None:
        print(f"  AUC:      {auc:.4f}")
    print(f"\n混淆矩阵:")
    print(f"  {cm}")

    # 分析得分分布
    normal_test_scores = test_scores[test_labels == 0]
    anomaly_test_scores = test_scores[test_labels == 1]

    print(f"\n得分分布:")
    print(f"  正常测试样本:")
    print(f"    均值: {np.mean(normal_test_scores):.4f}")
    print(f"    标准差: {np.std(normal_test_scores):.4f}")
    print(f"    范围: [{np.min(normal_test_scores):.4f}, {np.max(normal_test_scores):.4f}]")
    print(f"  异常测试样本:")
    print(f"    均值: {np.mean(anomaly_test_scores):.4f}")
    print(f"    标准差: {np.std(anomaly_test_scores):.4f}")
    print(f"    范围: [{np.min(anomaly_test_scores):.4f}, {np.max(anomaly_test_scores):.4f}]")

    # 分离度
    separation = (np.mean(anomaly_test_scores) - np.mean(normal_test_scores)) / score_std
    print(f"\n  分离度: {separation:.4f} (异常和正常的均值差距 / 训练集标准差)")
    print(f"    > 2.0: 优秀")
    print(f"    1.0-2.0: 良好")
    print(f"    < 1.0: 需要改进")

    # 7. 保存模型
    print("\n保存模型...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'score_based_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'score_model': score_model,
            'scaler': scaler,
            'threshold': threshold,
            'normal_range': (normal_min, normal_max),
            'denoise': args.denoise,
            'sigma_threshold': args.sigma_threshold
        }, f)
    print(f"  模型已保存: {model_path}")

    # 保存训练信息
    info = {
        'model_type': 'score_based_anomaly',
        'score_method': args.score_method,
        'denoise': args.denoise,
        'sigma_threshold': args.sigma_threshold,
        'normal_range': [float(normal_min), float(normal_max)],
        'threshold': float(threshold),
        'sample_rate': args.sr,
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc else None
        },
        'score_statistics': {
            'train': {
                'mean': float(score_mean),
                'std': float(score_std)
            },
            'test_normal': {
                'mean': float(np.mean(normal_test_scores)),
                'std': float(np.std(normal_test_scores))
            },
            'test_anomaly': {
                'mean': float(np.mean(anomaly_test_scores)),
                'std': float(np.std(anomaly_test_scores))
            },
            'separation': float(separation)
        }
    }

    info_path = os.path.join(args.output_dir, 'score_based_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  训练信息已保存: {info_path}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 性能评估
    if f1 > 0.6 and separation > 1.0:
        print("\n[成功] 模型效果良好！")
        print(f"  F1={f1:.4f}, 分离度={separation:.4f}")
    elif f1 > 0.4:
        print("\n[提示] 有一定效果，可以尝试优化:")
        print(f"  1. 调整 --sigma_threshold (当前: {args.sigma_threshold})")
        print(f"  2. 尝试其他评分方法 (--score_method)")
        print(f"  3. 调整降噪设置")
    else:
        print("\n[需要改进] 效果不佳，建议:")
        print("  1. 检查数据标签是否正确")
        print("  2. 人工听样本确认正常和异常的差异")
        print("  3. 尝试不同的评分方法")


if __name__ == "__main__":
    main()
