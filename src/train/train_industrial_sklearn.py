#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工业设备异常检测训练 - 使用sklearn（无需PyTorch）
"""

import os
import sys
import argparse
import numpy as np
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import load_anomaly_detection_data
from features.industrial_features import extract_industrial_features
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
import librosa
import pickle


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='工业设备异常检测训练（sklearn版本）')
    parser.add_argument('--normal_train_dir', type=str, required=True,
                        help='正常样本目录（用于训练）')
    parser.add_argument('--normal_test_dir', type=str, default=None,
                        help='正常测试样本目录（可选）')
    parser.add_argument('--anomaly_test_dir', type=str, default=None,
                        help='异常测试样本目录（可选）')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--denoise', action='store_true', default=True,
                        help='是否进行背景噪声抑制')
    parser.add_argument('--no_denoise', action='store_false', dest='denoise',
                        help='不进行背景噪声抑制')
    parser.add_argument('--algorithm', type=str, default='isolation_forest',
                        choices=['isolation_forest', 'one_class_svm', 'elliptic_envelope'],
                        help='异常检测算法')
    parser.add_argument('--contamination', type=float, default=0.05,
                        help='预期异常样本比例（0-0.5）')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')
    return parser.parse_args()


def extract_features_from_data(data, sr, denoise, desc="数据"):
    """从已加载的音频数据中提取工业特征"""
    print(f"正在从{desc}提取工业特征...")
    print(f"  降噪: {'是' if denoise else '否'}")

    features_list = []
    valid_count = 0

    for i, (audio, label) in enumerate(data):
        if (i + 1) % 10 == 0 or i == len(data) - 1:
            print(f"  进度: {i + 1}/{len(data)}")

        try:
            # 提取工业特征
            features = extract_industrial_features(audio, sr, denoise=denoise)
            features_list.append(features)
            valid_count += 1

        except Exception as e:
            print(f"  警告: 处理样本失败: {e}")
            continue

    if valid_count == 0:
        raise ValueError("没有成功提取任何特征")

    features_array = np.array(features_list)
    print(f"{desc}特征提取完成: {features_array.shape} ({valid_count}/{len(data)} 成功)")

    return features_array


def create_model(algorithm, contamination, n_features):
    """创建异常检测模型"""
    if algorithm == 'isolation_forest':
        print(f"  算法: Isolation Forest")
        print(f"  预期异常比例: {contamination}")
        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1
        )
    elif algorithm == 'one_class_svm':
        print(f"  算法: One-Class SVM")
        model = OneClassSVM(
            nu=contamination,
            kernel='rbf',
            gamma='auto'
        )
    elif algorithm == 'elliptic_envelope':
        print(f"  算法: Elliptic Envelope")
        model = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
    else:
        raise ValueError(f"未知算法: {algorithm}")

    return model


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 70)
    print("工业设备异常检测训练（sklearn版本）")
    print("=" * 70)
    print(f"训练模式: 异常检测（只使用正常样本）")
    print(f"背景噪声抑制: {'启用' if args.denoise else '禁用'}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")

    # 加载正常训练样本
    from glob import glob
    normal_files = glob(os.path.join(args.normal_train_dir, "*.wav"))
    print(f"找到 {len(normal_files)} 个正常训练样本")

    normal_data = []
    for file_path in normal_files:
        audio, _ = librosa.load(file_path, sr=args.sr)
        normal_data.append((audio, 0))

    # 分割训练集和验证集
    from sklearn.model_selection import train_test_split

    # 如果没有提供正常测试样本，从正常样本中额外分出一部分作为测试
    if args.anomaly_test_dir and not args.normal_test_dir:
        print("  提示：未提供正常测试样本，将从正常样本中分出20%作为测试集")
        train_val_data, normal_test_data = train_test_split(
            normal_data, test_size=0.2, random_state=42
        )
        train_data, val_data = train_test_split(
            train_val_data, test_size=args.val_ratio, random_state=42
        )
    else:
        normal_test_data = []
        train_data, val_data = train_test_split(
            normal_data, test_size=args.val_ratio, random_state=42
        )

    print(f"  训练集: {len(train_data)} 个正常样本")
    print(f"  验证集: {len(val_data)} 个正常样本")

    # 加载测试数据
    test_data = []
    test_labels = []

    # 添加自动分出的正常测试样本
    if len(normal_test_data) > 0:
        for audio, label in normal_test_data:
            test_data.append((audio, label))
            test_labels.append(label)
        print(f"  自动分出的正常测试样本: {len(normal_test_data)} 个")

    if args.normal_test_dir:
        normal_test_files = glob(os.path.join(args.normal_test_dir, "*.wav"))
        print(f"  加载正常测试样本: {len(normal_test_files)} 个")
        for file_path in normal_test_files:
            audio, _ = librosa.load(file_path, sr=args.sr)
            test_data.append((audio, 0))
            test_labels.append(0)

    if args.anomaly_test_dir:
        anomaly_test_files = glob(os.path.join(args.anomaly_test_dir, "*.wav"))
        print(f"  加载异常测试样本: {len(anomaly_test_files)} 个")
        for file_path in anomaly_test_files:
            audio, _ = librosa.load(file_path, sr=args.sr)
            test_data.append((audio, 1))
            test_labels.append(1)

    if len(test_data) > 0:
        test_labels = np.array(test_labels)
        print(f"  测试集: {len(test_data)} 个样本（正常: {np.sum(test_labels==0)}, 异常: {np.sum(test_labels==1)}）")
    else:
        print("  未提供测试集")
        test_data = None
        test_labels = None

    # 2. 提取特征
    print("\n[2/5] 提取工业特征...")
    start_time = time.time()

    # 合并训练集和验证集用于训练
    all_train_data = train_data + val_data

    train_features = extract_features_from_data(
        all_train_data, args.sr, args.denoise, "训练集"
    )

    if test_data:
        test_features = extract_features_from_data(
            test_data, args.sr, args.denoise, "测试集"
        )
    else:
        test_features = None

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 特征标准化
    print("\n[3/5] 特征标准化...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)

    if test_features is not None:
        test_features_scaled = scaler.transform(test_features)
    else:
        test_features_scaled = None

    # 4. 训练异常检测模型
    print("\n[4/5] 训练异常检测模型...")
    print(f"  特征维度: {train_features_scaled.shape[1]}")

    model = create_model(args.algorithm, args.contamination, train_features_scaled.shape[1])

    # 训练
    start_time = time.time()
    model.fit(train_features_scaled)
    training_time = time.time() - start_time

    print(f"模型训练完成（耗时: {training_time:.2f} 秒）")

    # 5. 评估模型
    if test_features_scaled is not None and test_labels is not None:
        print("\n[5/5] 在测试集上评估...")

        # 预测：-1表示异常，1表示正常
        predictions_raw = model.predict(test_features_scaled)

        # 转换为0（正常）和1（异常）
        predictions = (predictions_raw == -1).astype(int)

        # 获取异常分数
        if hasattr(model, 'decision_function'):
            anomaly_scores = -model.decision_function(test_features_scaled)  # 分数越高越异常
        else:
            anomaly_scores = -model.score_samples(test_features_scaled)

        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, roc_auc_score

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        cm = confusion_matrix(test_labels, predictions)

        # AUC
        try:
            auc = roc_auc_score(test_labels, anomaly_scores)
        except:
            auc = None

        print("\n评估结果:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        if auc is not None:
            print(f"  AUC: {auc:.4f}")
        print(f"混淆矩阵:")
        print(cm)

        # 分析异常分数分布
        normal_scores = anomaly_scores[test_labels == 0]
        anomaly_scores_filtered = anomaly_scores[test_labels == 1]

        print(f"\n异常分数分析:")
        if len(normal_scores) > 0:
            print(f"  正常样本 - 均值: {np.mean(normal_scores):.4f}, 标准差: {np.std(normal_scores):.4f}")
        if len(anomaly_scores_filtered) > 0:
            print(f"  异常样本 - 均值: {np.mean(anomaly_scores_filtered):.4f}, 标准差: {np.std(anomaly_scores_filtered):.4f}")

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None
        }
    else:
        print("\n[5/5] 跳过测试集评估（未提供测试集）")
        metrics_dict = None

    # 6. 保存模型
    print("\n保存模型和预处理器...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存模型和预处理器
    model_path = os.path.join(args.output_dir, 'industrial_sklearn_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler': scaler,
            'algorithm': args.algorithm,
            'denoise': args.denoise
        }, f)
    print(f"模型已保存: {model_path}")

    # 保存训练信息
    info = {
        'model_type': 'industrial_anomaly_sklearn',
        'algorithm': args.algorithm,
        'feature_type': 'industrial_features',
        'denoise': args.denoise,
        'contamination': args.contamination,
        'sample_rate': args.sr,
        'n_features': train_features_scaled.shape[1],
        'train_samples': len(all_train_data),
        'test_samples': len(test_data) if test_data else 0,
        'training_time': training_time,
        'metrics': metrics_dict
    }

    info_path = os.path.join(args.output_dir, 'industrial_sklearn_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 性能评估
    if metrics_dict:
        print(f"\n结果对比 - 之前监督学习: F1=0.52, AUC=0.53")
        auc_str = f"{metrics_dict['auc']:.4f}" if metrics_dict['auc'] and not np.isnan(metrics_dict['auc']) else "N/A"
        print(f"         现在异常检测: F1={metrics_dict['f1']:.4f}, AUC={auc_str}")

        if metrics_dict['auc'] and metrics_dict['auc'] > 0.65:
            print("\n[成功] 性能提升明显！异常检测模式适合你的数据")
        elif metrics_dict['f1'] > 0.55:
            print("\n[提示] 有一定提升，建议调整参数:")
            print("  1. 尝试不同算法 (--algorithm one_class_svm)")
            print("  2. 调整contamination (--contamination 0.1)")
            print("  3. 禁用降噪试试 (--no_denoise)")
        else:
            print("\n[需要改进] 效果仍需改进，可能的原因:")
            print("  1. 训练集中可能混入了异常样本")
            print("  2. 正常样本之间差异太大")
            print("  3. 异常与正常确实非常相似")


if __name__ == "__main__":
    main()
