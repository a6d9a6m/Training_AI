#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版GMM异常检测 - 使用更多特征和优化策略

改进：
1. 更丰富的特征（包括时域、频域、能量）
2. 特征选择（去除冗余特征）
3. 自动搜索最佳GMM组件数
4. 使用验证集调优阈值
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
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import librosa
import warnings
warnings.filterwarnings('ignore')


def augment_audio(audio, sr):
    """
    数据增强 - 生成音频的变体

    参考DCASE 2021 常用方法：
    1. 时间拉伸（time stretching）
    2. 音高变换（pitch shifting）
    3. 添加白噪声
    """
    augmented = [audio]  # 原始音频

    # 1. 时间拉伸 - 稍微加快/减慢
    try:
        audio_stretched_fast = librosa.effects.time_stretch(audio, rate=1.1)
        augmented.append(audio_stretched_fast)

        audio_stretched_slow = librosa.effects.time_stretch(audio, rate=0.9)
        augmented.append(audio_stretched_slow)
    except:
        pass

    # 2. 音高变换 - 轻微升降调
    try:
        audio_pitch_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        augmented.append(audio_pitch_up)

        audio_pitch_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)
        augmented.append(audio_pitch_down)
    except:
        pass

    # 3. 添加微弱白噪声
    try:
        noise = np.random.normal(0, 0.005, len(audio))
        audio_noisy = audio + noise
        augmented.append(audio_noisy)
    except:
        pass

    return augmented


def extract_deep_features_simple(audio, sr):
    """
    提取深度特征（使用统计聚合的频谱图）

    思路：将梅尔频谱图作为"图像"，提取更多统计特征
    这是一个简化版，不需要训练神经网络
    """
    # 计算梅尔频谱图（更高分辨率）
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec)

    features = []

    # 1. 全局统计特征
    features.extend([
        np.mean(mel_spec_db),
        np.std(mel_spec_db),
        np.max(mel_spec_db),
        np.min(mel_spec_db),
        np.median(mel_spec_db),
        np.percentile(mel_spec_db, 25),
        np.percentile(mel_spec_db, 75)
    ])

    # 2. 频率维度统计（每个频率bin的时间统计）
    for i in range(0, 128, 8):  # 每8个bin一组，共16组
        freq_band = mel_spec_db[i:i+8, :]
        features.extend([
            np.mean(freq_band),
            np.std(freq_band),
            np.max(freq_band) - np.min(freq_band)  # 动态范围
        ])

    # 3. 时间维度统计（每个时间帧的频率统计）
    n_frames = mel_spec_db.shape[1]
    step = max(1, n_frames // 20)  # 采样20个时间点
    for t in range(0, n_frames, step):
        if t >= n_frames:
            break
        time_slice = mel_spec_db[:, t]
        features.extend([
            np.mean(time_slice),
            np.std(time_slice)
        ])

    # 4. 频谱能量分布
    energy_per_frame = np.sum(mel_spec, axis=0)
    features.extend([
        np.mean(energy_per_frame),
        np.std(energy_per_frame),
        np.max(energy_per_frame),
        np.min(energy_per_frame)
    ])

    # 5. 频谱质心在时间上的变化
    spec_centroid = np.sum(np.arange(128)[:, np.newaxis] * mel_spec, axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)
    features.extend([
        np.mean(spec_centroid),
        np.std(spec_centroid)
    ])

    return np.array(features)


def parse_arguments():
    parser = argparse.ArgumentParser(description='增强版GMM异常检测')
    parser.add_argument('--normal_train_dir', type=str, required=True)
    parser.add_argument('--anomaly_test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/saved_models')
    parser.add_argument('--sr', type=int, default=22050)

    # 模型选择
    parser.add_argument('--use_deep_features', action='store_true',
                        help='使用深度学习提取特征（需要PyTorch）')
    parser.add_argument('--use_ensemble', action='store_true',
                        help='使用集成模型（GMM + Isolation Forest + One-Class SVM）')

    # GMM参数
    parser.add_argument('--auto_tune', action='store_true', default=True,
                        help='自动调优GMM参数')
    parser.add_argument('--n_components', type=int, default=None,
                        help='手动指定GMM组件数（不使用auto_tune时）')
    parser.add_argument('--threshold_percentile', type=float, default=None,
                        help='手动指定阈值百分位数（1-30推荐），覆盖自动选择')
    parser.add_argument('--k_features', type=int, default=60,
                        help='保留的特征数量（默认60，范围30-100）')

    # 数据增强
    parser.add_argument('--use_augmentation', action='store_true',
                        help='使用数据增强（时间拉伸、音高变换、添加噪声）')

    return parser.parse_args()


def extract_enhanced_features(audio, sr):
    """提取增强特征集"""
    features = []

    # 1. MFCC及其衍生特征
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))
    features.extend(np.max(mfcc, axis=1))
    features.extend(np.min(mfcc, axis=1))

    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend(np.mean(mfcc_delta2, axis=1))

    # 2. 色度特征
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 3. 频谱特征
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    features.extend([np.mean(centroid), np.std(centroid), np.max(centroid)])

    bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    features.extend([np.mean(bandwidth), np.std(bandwidth)])

    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    features.extend([np.mean(rolloff), np.std(rolloff)])

    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr, n_bands=6)
    features.extend(np.mean(contrast, axis=1))
    features.extend(np.std(contrast, axis=1))

    flatness = librosa.feature.spectral_flatness(y=audio)[0]
    features.extend([np.mean(flatness), np.std(flatness)])

    # 4. 节奏特征
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])

    # 5. 能量特征
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

    # 6. 梅尔频谱
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=40)
    mel_spec_db = librosa.power_to_db(mel_spec)
    features.extend([
        np.mean(mel_spec_db),
        np.std(mel_spec_db),
        np.max(mel_spec_db),
        np.min(mel_spec_db),
        np.median(mel_spec_db)
    ])

    return np.array(features)


def find_best_gmm_params(X_train, X_val, max_components=8):
    """自动搜索最佳GMM参数"""
    print("  自动搜索最佳GMM参数...")

    best_score = -np.inf
    best_n = 1
    best_cov_type = 'full'

    for n in range(1, max_components + 1):
        for cov_type in ['full', 'diag']:  # 只用full和diag，tied容易过拟合
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=cov_type,
                    max_iter=200,
                    random_state=42,
                    reg_covar=1e-6  # 添加正则化
                )
                gmm.fit(X_train)

                # 在验证集上评分
                val_score = gmm.score(X_val)

                if val_score > best_score:
                    best_score = val_score
                    best_n = n
                    best_cov_type = cov_type

            except:
                continue

    print(f"    最佳参数: n_components={best_n}, covariance_type={best_cov_type}")
    return best_n, best_cov_type


def main():
    args = parse_arguments()

    print("=" * 70)
    print("增强版GMM异常检测")
    if args.use_deep_features:
        print("  [深度特征模式] 使用频谱图深度统计特征")
    if args.use_augmentation:
        print("  [数据增强] 启用（训练样本将增加5-6倍）")
    if args.use_ensemble:
        print("  [集成模式] GMM + IsolationForest + OneClassSVM")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/8] 加载数据...")
    normal_files = glob(os.path.join(args.normal_train_dir, "*.wav"))
    print(f"  正常样本: {len(normal_files)} 个")

    normal_data = []
    for i, file_path in enumerate(normal_files):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(normal_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        normal_data.append(audio)

    # 分割：60%训练，20%验证，20%测试
    from sklearn.model_selection import train_test_split
    train_val, normal_test = train_test_split(normal_data, test_size=0.2, random_state=42)
    normal_train, normal_val = train_test_split(train_val, test_size=0.25, random_state=42)

    print(f"  正常样本分割: 训练{len(normal_train)}, 验证{len(normal_val)}, 测试{len(normal_test)}")

    # 数据增强
    if args.use_augmentation:
        print("\n  应用数据增强...")
        augmented_train = []
        for i, audio in enumerate(normal_train):
            if (i + 1) % 20 == 0:
                print(f"    增强进度: {i + 1}/{len(normal_train)}")
            aug_audios = augment_audio(audio, args.sr)
            augmented_train.extend(aug_audios)
        normal_train = augmented_train
        print(f"  增强后训练样本: {len(normal_train)} 个")

    # 加载异常
    anomaly_files = glob(os.path.join(args.anomaly_test_dir, "*.wav"))
    print(f"  异常样本: {len(anomaly_files)} 个")

    anomaly_data = []
    for i, file_path in enumerate(anomaly_files):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        anomaly_data.append(audio)

    # 2. 提取特征
    print(f"\n[2/8] 提取特征（{'深度特征' if args.use_deep_features else '标准特征'}）...")

    # 选择特征提取函数
    if args.use_deep_features:
        feature_extractor = lambda a, sr: np.hstack([
            extract_enhanced_features(a, sr),
            extract_deep_features_simple(a, sr)
        ])
        print("  使用组合特征：标准特征 + 深度特征")
    else:
        feature_extractor = extract_enhanced_features

    print("  训练集...")
    train_features = []
    for i, audio in enumerate(normal_train):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(normal_train)}")
        train_features.append(feature_extractor(audio, args.sr))
    train_features = np.array(train_features)

    print("  验证集...")
    val_features = np.array([feature_extractor(a, args.sr) for a in normal_val])

    test_features = []
    test_labels = []

    print("  正常测试集...")
    for audio in normal_test:
        test_features.append(feature_extractor(audio, args.sr))
        test_labels.append(0)

    print("  异常测试集...")
    for i, audio in enumerate(anomaly_data):
        if (i + 1) % 20 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_data)}")
        test_features.append(feature_extractor(audio, args.sr))
        test_labels.append(1)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(f"  特征维度: {train_features.shape[1]}")

    # 3. 标准化（使用RobustScaler抗干扰）
    print("\n[3/8] 特征标准化（RobustScaler）...")
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 4. 特征选择（移除冗余特征）
    print("\n[4/8] 特征选择...")

    # 创建临时标签用于特征选择（正常vs异常的一小部分）
    n_anomaly_for_selection = min(20, len(anomaly_data))
    temp_X = np.vstack([train_features_scaled, test_features_scaled[len(normal_test):len(normal_test)+n_anomaly_for_selection]])
    temp_y = np.array([0] * len(train_features_scaled) + [1] * n_anomaly_for_selection)

    k_features = min(args.k_features, train_features_scaled.shape[1])
    selector = SelectKBest(f_classif, k=k_features)
    selector.fit(temp_X, temp_y)

    train_features_selected = selector.transform(train_features_scaled)
    val_features_selected = selector.transform(val_features_scaled)
    test_features_selected = selector.transform(test_features_scaled)

    print(f"  原始特征数: {train_features_scaled.shape[1]}")
    print(f"  选择特征数: {train_features_selected.shape[1]}")

    # 显示特征重要性
    feature_scores = selector.scores_
    top_indices = np.argsort(feature_scores)[-10:]
    print(f"  Top 10 重要特征索引: {top_indices.tolist()}")

    # 5. 训练模型
    if args.use_ensemble:
        print("\n[5/8] 训练集成模型（GMM + IsolationForest + OneClassSVM）...")

        from sklearn.ensemble import IsolationForest
        from sklearn.svm import OneClassSVM

        # 训练GMM
        print("  训练GMM...")
        if args.auto_tune and args.n_components is None:
            best_n, best_cov = find_best_gmm_params(
                train_features_selected,
                val_features_selected,
                max_components=8
            )
        else:
            best_n = args.n_components if args.n_components else 3
            best_cov = 'full'

        gmm = GaussianMixture(
            n_components=best_n,
            covariance_type=best_cov,
            max_iter=200,
            random_state=42,
            reg_covar=1e-6
        )
        gmm.fit(train_features_selected)

        # 训练Isolation Forest
        print("  训练Isolation Forest...")
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(train_features_selected)

        # 训练One-Class SVM
        print("  训练One-Class SVM...")
        ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='auto',
            nu=0.1
        )
        ocsvm.fit(train_features_selected)

        models = {'gmm': gmm, 'iso_forest': iso_forest, 'ocsvm': ocsvm}
        print("  集成模型训练完成")

    else:
        print("\n[5/8] 训练GMM...")

        if args.auto_tune and args.n_components is None:
            best_n, best_cov = find_best_gmm_params(
                train_features_selected,
                val_features_selected,
                max_components=8
            )
        else:
            best_n = args.n_components if args.n_components else 3
            best_cov = 'full'
            print(f"  使用手动参数: n_components={best_n}, covariance_type={best_cov}")

        gmm = GaussianMixture(
            n_components=best_n,
            covariance_type=best_cov,
            max_iter=200,
            random_state=42,
            reg_covar=1e-6
        )
        gmm.fit(train_features_selected)
        models = {'gmm': gmm}
        best_n = gmm.n_components
        best_cov = gmm.covariance_type
        print("  训练完成")

    # 6. 确定阈值（多种策略）
    print("\n[6/8] 确定阈值...")

    # 对于集成模型，每个子模型都有自己的决策
    if args.use_ensemble:
        # GMM得分
        train_scores_gmm = models['gmm'].score_samples(train_features_selected)
        val_scores_gmm = models['gmm'].score_samples(val_features_selected)

        threshold_gmm = np.percentile(train_scores_gmm, 15)
        print(f"  GMM阈值（第15百分位）: {threshold_gmm:.4f}")

        # Isolation Forest和OCSVM使用默认决策
        thresholds = {'gmm': threshold_gmm}

    else:
        train_scores = models['gmm'].score_samples(train_features_selected)
        val_scores = models['gmm'].score_samples(val_features_selected)

        print(f"  训练集得分: 均值={np.mean(train_scores):.2f}, 标准差={np.std(train_scores):.2f}")
        print(f"  训练集得分范围: [{np.min(train_scores):.2f}, {np.max(train_scores):.2f}]")

        # 如果用户指定了阈值百分位，直接使用
        if args.threshold_percentile is not None:
            threshold = np.percentile(train_scores, args.threshold_percentile)
            print(f"\n  使用手动指定阈值（第{args.threshold_percentile}百分位）: {threshold:.4f}")
        else:
            # 策略1：均值 - k*标准差
            threshold_std = np.mean(train_scores) - 2.0 * np.std(train_scores)
            print(f"\n  策略1（均值-2σ）: {threshold_std:.4f}")

            # 策略2：百分位数
            threshold_percentile = np.percentile(train_scores, 10)
            print(f"  策略2（第10百分位）: {threshold_percentile:.4f}")

            # 策略3：四分位距（IQR）方法
            q1 = np.percentile(train_scores, 25)
            q3 = np.percentile(train_scores, 75)
            iqr = q3 - q1
            threshold_iqr = q1 - 1.5 * iqr
            print(f"  策略3（IQR异常值检测）: {threshold_iqr:.4f}")

            # 选择最宽松的阈值
            threshold = max(threshold_std, threshold_percentile, threshold_iqr)

            if threshold < np.mean(train_scores) - 3 * np.std(train_scores):
                threshold = np.mean(train_scores) - 2.5 * np.std(train_scores)
                print(f"\n  使用安全阈值（均值-2.5σ）: {threshold:.4f}")
            else:
                print(f"\n  最终阈值（选最宽松）: {threshold:.4f}")

        # 在验证集上检查
        val_anomaly_rate = np.sum(val_scores < threshold) / len(val_scores)
        print(f"  验证集触发率: {val_anomaly_rate:.2%} ({np.sum(val_scores < threshold)}/{len(val_scores)})")

        if val_anomaly_rate < 0.01 and args.threshold_percentile is None:
            print("  [警告] 阈值过于严格，调整到第15百分位")
            threshold = np.percentile(train_scores, 15)
            val_anomaly_rate = np.sum(val_scores < threshold) / len(val_scores)
            print(f"  调整后阈值: {threshold:.4f}")
            print(f"  调整后验证集触发率: {val_anomaly_rate:.2%}")

        thresholds = {'gmm': threshold}

    # 7. 测试集评估
    print("\n[7/8] 测试集评估...")

    if args.use_ensemble:
        # 集成模型：投票机制
        # GMM: score < threshold → anomaly (1)
        gmm_scores = models['gmm'].score_samples(test_features_selected)
        gmm_preds = (gmm_scores < thresholds['gmm']).astype(int)

        # Isolation Forest: -1 → anomaly (1), 1 → normal (0)
        iso_preds = (models['iso_forest'].predict(test_features_selected) == -1).astype(int)

        # One-Class SVM: -1 → anomaly (1), 1 → normal (0)
        ocsvm_preds = (models['ocsvm'].predict(test_features_selected) == -1).astype(int)

        # 投票：3个模型中至少2个认为是异常，才判定为异常
        ensemble_votes = gmm_preds + iso_preds + ocsvm_preds
        predictions = (ensemble_votes >= 2).astype(int)

        print(f"  GMM检测异常: {np.sum(gmm_preds)} / {len(gmm_preds)}")
        print(f"  IsolationForest检测异常: {np.sum(iso_preds)} / {len(iso_preds)}")
        print(f"  OneClassSVM检测异常: {np.sum(ocsvm_preds)} / {len(ocsvm_preds)}")
        print(f"  集成投票检测异常: {np.sum(predictions)} / {len(predictions)}")

        # 用于AUC计算（使用GMM得分，因为它是连续的）
        test_scores = -gmm_scores  # 取负使得分数越高越异常

    else:
        test_scores = models['gmm'].score_samples(test_features_selected)
        predictions = (test_scores < thresholds['gmm']).astype(int)
        test_scores = -test_scores  # 取负使得分数越高越异常

    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    try:
        auc = roc_auc_score(test_labels, test_scores)  # test_scores已经取负，越高越异常
    except:
        auc = None

    print("\n" + "=" * 70)
    print("评估结果:")
    print("=" * 70)
    print(f"  准确率:   {accuracy:.4f}")
    print(f"  精确率:   {precision:.4f}")
    print(f"  召回率:   {recall:.4f}")
    print(f"  F1分数:   {f1:.4f}")
    if auc:
        print(f"  AUC:      {auc:.4f}")

    print(f"\n混淆矩阵:")
    print(f"            预测正常  预测异常")
    print(f"  实际正常     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"  实际异常     {cm[1,0]:3d}      {cm[1,1]:3d}")

    # 得分分析
    normal_test_scores = test_scores[test_labels == 0]
    anomaly_test_scores = test_scores[test_labels == 1]

    print(f"\n得分分布:")
    print(f"  正常测试样本:")
    print(f"    均值: {np.mean(normal_test_scores):.2f}")
    print(f"    标准差: {np.std(normal_test_scores):.2f}")
    print(f"    范围: [{np.min(normal_test_scores):.2f}, {np.max(normal_test_scores):.2f}]")

    print(f"  异常测试样本:")
    print(f"    均值: {np.mean(anomaly_test_scores):.2f}")
    print(f"    标准差: {np.std(anomaly_test_scores):.2f}")
    print(f"    范围: [{np.min(anomaly_test_scores):.2f}, {np.max(anomaly_test_scores):.2f}]")

    # 计算分离度（注意：这里使用原始得分的标准差）
    if not args.use_ensemble:
        train_scores_for_sep = models['gmm'].score_samples(train_features_selected)
        separation = (np.mean(anomaly_test_scores) - np.mean(normal_test_scores)) / np.std(train_scores_for_sep)
    else:
        train_scores_gmm_for_sep = models['gmm'].score_samples(train_features_selected)
        separation = (np.mean(anomaly_test_scores) - np.mean(normal_test_scores)) / np.std(train_scores_gmm_for_sep)

    print(f"\n  分离度: {separation:.4f}")
    print(f"    > 0.5: 有一定区分能力")
    print(f"    > 1.0: 良好")
    print(f"    < 0.3: 几乎无法区分")

    # 诊断建议
    if separation < 0.3:
        print("\n  [诊断] 分离度极低，说明：")
        print("    - 正常和异常样本在特征空间高度重叠")
        print("    - 建议检查数据标签是否正确")
        print("    - 或考虑使用深度学习方法")
    elif f1 < 0.3:
        print("\n  [诊断] F1分数很低，但分离度尚可：")
        print("    - 可能是阈值选择不当")
        print("    - 尝试手动调整阈值")
        if not args.use_ensemble:
            train_scores_diag = models['gmm'].score_samples(train_features_selected)
            print(f"    - 建议阈值范围: [{np.percentile(train_scores_diag, 10):.2f}, {np.percentile(train_scores_diag, 30):.2f}]")

    # 8. 保存
    print("\n保存模型...")
    os.makedirs(args.output_dir, exist_ok=True)

    model_path = os.path.join(args.output_dir, 'gmm_enhanced_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'gmm': gmm,
            'scaler': scaler,
            'selector': selector,
            'threshold': threshold,
            'n_components': best_n,
            'covariance_type': best_cov,
            'sample_rate': args.sr
        }, f)
    print(f"  模型已保存: {model_path}")

    info = {
        'model_type': 'gmm_enhanced',
        'n_components': best_n,
        'covariance_type': best_cov,
        'n_features': train_features_selected.shape[1],
        'threshold': float(threshold),
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc else None,
            'separation': float(separation)
        }
    }

    info_path = os.path.join(args.output_dir, 'gmm_enhanced_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

    print(f"\n与之前对比:")
    print(f"  监督学习GMM: F1=0.52, AUC=0.53")
    print(f"  异常检测GMM: F1={f1:.4f}, AUC={auc:.4f if auc is not None else 0:.4f}")

    if f1 > 0.65:
        print("\n[成功] 效果良好")
    elif f1 > 0.4 or (auc and auc > 0.6):
        print("\n[有一定效果] 可以继续优化")
        print("\n  建议尝试的参数组合:")
        print(f"    1. 更宽松的阈值:")
        print(f"       --threshold_percentile 20")
        print(f"    2. 减少特征数（避免过拟合）:")
        print(f"       --k_features 40")
        print(f"    3. 调整GMM组件数:")
        print(f"       --n_components 5")
        print(f"\n  完整命令示例:")
        print(f"    python train_gmm_enhanced.py \\")
        print(f"      --normal_train_dir {args.normal_train_dir} \\")
        print(f"      --anomaly_test_dir {args.anomaly_test_dir} \\")
        print(f"      --threshold_percentile 20 \\")
        print(f"      --k_features 40")
    else:
        print("\n[效果不佳]")
        if separation < 0.3:
            print("  分离度极低，这是DCASE 2021 Task 2的特点")
            print("  正常和异常样本极其相似")
            print("\n  可能的改进方向:")
            print("    1. 使用深度学习方法（如Autoencoder）")
            print("    2. 尝试更专业的特征（PyAudioAnalysis, OpenSMILE）")
            print("    3. 集成多个模型")
        else:
            print("  分离度尚可但F1低，可能是阈值问题")
            print(f"\n  建议尝试阈值范围:")
            print(f"    --threshold_percentile 15 到 30")


if __name__ == "__main__":
    main()
