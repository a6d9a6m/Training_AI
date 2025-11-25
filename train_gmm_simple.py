#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于GMM的异常检测 - 简单直观版本

思路：
1. 使用librosa提取音频特征
2. 在正常样本上训练GMM（学习正常声音的概率分布）
3. GMM对每个样本计算得分（对数似然）
   - 正常样本：得分高（符合正常分布）
   - 异常样本：得分低（不符合正常分布）
4. 在正常样本上确定得分阈值
5. 低于阈值判定为异常
"""

import os
import sys
import argparse
import numpy as np
import json
import pickle
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import librosa


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='基于GMM的异常检测')
    parser.add_argument('--normal_train_dir', type=str, required=True,
                        help='正常样本目录（用于训练）')
    parser.add_argument('--anomaly_test_dir', type=str, required=True,
                        help='异常测试样本目录')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--n_components', type=int, default=3,
                        help='GMM组件数量')
    parser.add_argument('--percentile', type=float, default=5,
                        help='阈值百分位数（0-100），越小越严格')
    return parser.parse_args()


def extract_features(audio, sr):
    """
    提取音频特征（使用librosa）

    特征包括：
    - MFCC及其差分
    - 梅尔频谱统计
    - 频谱质心
    - 频谱带宽
    - 频谱对比度
    - 过零率
    - RMS能量
    """
    features = []

    # 1. MFCC特征（最重要）
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # MFCC一阶差分
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    # 2. 梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec)
    features.extend([
        np.mean(mel_spec_db),
        np.std(mel_spec_db),
        np.max(mel_spec_db),
        np.min(mel_spec_db)
    ])

    # 3. 频谱质心（声音的"重心"）
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.extend([
        np.mean(centroid),
        np.std(centroid)
    ])

    # 4. 频谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.extend([
        np.mean(bandwidth),
        np.std(bandwidth)
    ])

    # 5. 频谱对比度
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    features.extend(np.mean(contrast, axis=1))

    # 6. 过零率
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([
        np.mean(zcr),
        np.std(zcr)
    ])

    # 7. RMS能量
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([
        np.mean(rms),
        np.std(rms)
    ])

    return np.array(features)


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 70)
    print("基于GMM的异常检测")
    print("=" * 70)
    print(f"GMM组件数: {args.n_components}")
    print(f"阈值百分位: {args.percentile}% (越小越严格)")
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
        features = extract_features(audio, args.sr)
        train_features.append(features)
    train_features = np.array(train_features)

    print(f"  训练集特征: {train_features.shape}")

    # 测试集特征
    test_features = []
    test_labels = []

    print("  正常测试样本...")
    for audio in normal_test:
        features = extract_features(audio, args.sr)
        test_features.append(features)
        test_labels.append(0)

    print("  异常测试样本...")
    for i, audio in enumerate(anomaly_test_data):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_test_data)}")
        features = extract_features(audio, args.sr)
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

    # 4. 训练GMM模型
    print(f"\n[步骤 4/6] 训练GMM模型 (组件数: {args.n_components})...")

    gmm = GaussianMixture(
        n_components=args.n_components,
        covariance_type='full',
        max_iter=200,
        random_state=42
    )

    gmm.fit(train_features_scaled)
    print("  GMM训练完成")

    # 5. 计算得分并确定阈值
    print("\n[步骤 5/6] 计算得分并确定阈值...")

    # 计算训练集的得分（对数似然）
    train_scores = gmm.score_samples(train_features_scaled)

    print(f"  训练集得分统计:")
    print(f"    均值: {np.mean(train_scores):.4f}")
    print(f"    标准差: {np.std(train_scores):.4f}")
    print(f"    最小值: {np.min(train_scores):.4f}")
    print(f"    最大值: {np.max(train_scores):.4f}")

    # 确定阈值：使用百分位数
    # 例如：5%百分位意味着5%的正常样本会被误判为异常
    threshold = np.percentile(train_scores, args.percentile)

    print(f"\n  阈值（第{args.percentile}百分位）: {threshold:.4f}")
    print(f"    含义：{args.percentile}%的正常训练样本得分低于此值")
    print(f"    判定规则：得分 < {threshold:.4f} → 异常")

    # 6. 在测试集上评估
    print("\n[步骤 6/6] 在测试集上评估...")

    # 计算测试集的得分
    test_scores = gmm.score_samples(test_features_scaled)

    # 预测：得分低于阈值 = 异常
    predictions = (test_scores < threshold).astype(int)

    # 计算评估指标
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    try:
        # 注意：对于得分，越高越正常，所以需要取负
        auc = roc_auc_score(test_labels, -test_scores)
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
    print(f"            预测正常  预测异常")
    print(f"  实际正常     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"  实际异常     {cm[1,0]:3d}      {cm[1,1]:3d}")

    # 分析得分分布
    normal_test_scores = test_scores[test_labels == 0]
    anomaly_test_scores = test_scores[test_labels == 1]

    print(f"\n得分分布:")
    print(f"  正常测试样本:")
    print(f"    均值: {np.mean(normal_test_scores):.4f}")
    print(f"    标准差: {np.std(normal_test_scores):.4f}")
    print(f"    范围: [{np.min(normal_test_scores):.4f}, {np.max(normal_test_scores):.4f}]")
    print(f"    低于阈值: {np.sum(normal_test_scores < threshold)}/{len(normal_test_scores)}")

    print(f"  异常测试样本:")
    print(f"    均值: {np.mean(anomaly_test_scores):.4f}")
    print(f"    标准差: {np.std(anomaly_test_scores):.4f}")
    print(f"    范围: [{np.min(anomaly_test_scores):.4f}, {np.max(anomaly_test_scores):.4f}]")
    print(f"    低于阈值: {np.sum(anomaly_test_scores < threshold)}/{len(anomaly_test_scores)}")

    # 分离度
    score_diff = np.mean(normal_test_scores) - np.mean(anomaly_test_scores)
    separation = score_diff / np.std(train_scores)

    print(f"\n  分离度: {separation:.4f}")
    print(f"    (正常和异常的均值差距 / 训练集标准差)")
    print(f"    > 1.0: 良好分离")
    print(f"    0.5-1.0: 有一定区分")
    print(f"    < 0.5: 区分困难")

    # 7. 保存模型
    print("\n保存模型...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'gmm_anomaly_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'gmm': gmm,
            'scaler': scaler,
            'threshold': threshold,
            'n_components': args.n_components,
            'sample_rate': args.sr
        }, f)
    print(f"  模型已保存: {model_path}")

    # 保存训练信息
    info = {
        'model_type': 'gmm_anomaly_detection',
        'n_components': args.n_components,
        'threshold': float(threshold),
        'threshold_percentile': args.percentile,
        'sample_rate': args.sr,
        'n_features': train_features_scaled.shape[1],
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc else None
        },
        'score_statistics': {
            'train': {
                'mean': float(np.mean(train_scores)),
                'std': float(np.std(train_scores))
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

    info_path = os.path.join(args.output_dir, 'gmm_anomaly_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  训练信息已保存: {info_path}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 性能评估和建议
    print(f"\n与之前对比:")
    print(f"  监督学习GMM: F1=0.52, AUC=0.53")
    print(f"  异常检测GMM: F1={f1:.4f}, AUC={auc:.4f if auc is not None else 0:.4f}")

    if f1 > 0.65 and separation > 0.8:
        print("\n[成功] 模型效果良好！")
    elif f1 > 0.5:
        print("\n[提示] 有一定效果，可以尝试:")
        print(f"  1. 调整GMM组件数 (--n_components 5)")
        print(f"  2. 调整阈值百分位 (--percentile 10) 使其更宽松")
        print(f"  3. 当前阈值可能过于严格，尝试 --percentile 10 或 15")
    else:
        print("\n[需要改进] 效果不佳")
        print("  可能原因：正常和异常样本在特征空间中高度重叠")
        print("  建议检查数据质量和标签正确性")


if __name__ == "__main__":
    main()
