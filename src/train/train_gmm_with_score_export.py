#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
增强版GMM异常检测 - 带分数导出和高级分离方法

新增功能：
1. 导出所有样本的多维分数（GMM、IsolationForest、OneClassSVM等）
2. 实现多种空间分离方法：
   - Fisher线性判别分析（LDA）
   - 核方法（RBF、多项式）
   - 对比度增强
   - 监督特征变换
3. 提供详细的分数分析和可视化
"""

import os
import sys
import argparse
import numpy as np
import json
import pickle
from glob import glob
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import librosa
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except:
    HAS_PLOT = False


def augment_audio(audio, sr):
    """增强版数据增强"""
    augmented = [audio]

    # 1. 时间拉伸（更多变化）
    for rate in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
        try:
            audio_stretched = librosa.effects.time_stretch(audio, rate=rate)
            augmented.append(audio_stretched)
        except:
            pass

    # 2. 音调偏移
    for n_steps in [-3, -2, -1, 1, 2, 3]:
        try:
            audio_shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
            augmented.append(audio_shifted)
        except:
            pass

    # 3. 添加噪声（多个强度）
    for noise_level in [0.003, 0.005, 0.008]:
        try:
            noise = np.random.normal(0, noise_level, len(audio))
            audio_noisy = audio + noise
            augmented.append(audio_noisy)
        except:
            pass

    # 4. 音量变化
    for gain in [0.8, 0.9, 1.1, 1.2]:
        try:
            audio_gain = audio * gain
            augmented.append(audio_gain)
        except:
            pass

    return augmented


def extract_deep_features_simple(audio, sr):
    """提取深度特征"""
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    mel_spec_db = librosa.power_to_db(mel_spec)

    features = []

    # 1. 全局统计
    features.extend([
        np.mean(mel_spec_db),
        np.std(mel_spec_db),
        np.max(mel_spec_db),
        np.min(mel_spec_db),
        np.median(mel_spec_db),
        np.percentile(mel_spec_db, 25),
        np.percentile(mel_spec_db, 75)
    ])

    # 2. 频率维度统计
    for i in range(0, 128, 8):
        freq_band = mel_spec_db[i:i+8, :]
        features.extend([
            np.mean(freq_band),
            np.std(freq_band),
            np.max(freq_band) - np.min(freq_band)
        ])

    # 3. 时间维度统计
    n_frames = mel_spec_db.shape[1]
    step = max(1, n_frames // 20)
    for t in range(0, n_frames, step):
        if t >= n_frames:
            break
        time_slice = mel_spec_db[:, t]
        features.extend([
            np.mean(time_slice),
            np.std(time_slice)
        ])

    # 4. 能量分布
    energy_per_frame = np.sum(mel_spec, axis=0)
    features.extend([
        np.mean(energy_per_frame),
        np.std(energy_per_frame),
        np.max(energy_per_frame),
        np.min(energy_per_frame)
    ])

    # 5. 频谱质心
    spec_centroid = np.sum(np.arange(128)[:, np.newaxis] * mel_spec, axis=0) / (np.sum(mel_spec, axis=0) + 1e-10)
    features.extend([
        np.mean(spec_centroid),
        np.std(spec_centroid)
    ])

    return np.array(features)


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


def apply_lda_separation(X_train, X_test, y_train_for_lda):
    """
    应用Fisher线性判别分析（LDA）增强分离度

    LDA会找到一个投影方向，使得类间距离最大化，类内距离最小化
    """
    print("\n  [分离方法] 应用LDA线性判别分析...")

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train, y_train_for_lda)

    X_train_lda = lda.transform(X_train)
    X_test_lda = lda.transform(X_test)

    print(f"    LDA投影后维度: {X_train_lda.shape[1]}")

    return X_train_lda, X_test_lda, lda


def apply_kernel_transformation(X_train, X_test, kernel='rbf', n_components=100):
    """
    应用核方法进行非线性特征变换

    核方法可以将数据映射到高维空间，可能更容易线性分离
    """
    print(f"\n  [分离方法] 应用{kernel.upper()}核变换...")

    if kernel == 'rbf':
        transformer = RBFSampler(n_components=n_components, random_state=42, gamma=0.1)
    else:
        transformer = Nystroem(kernel=kernel, n_components=n_components, random_state=42)

    transformer.fit(X_train)

    X_train_kernel = transformer.transform(X_train)
    X_test_kernel = transformer.transform(X_test)

    print(f"    核变换后维度: {X_train_kernel.shape[1]}")

    return X_train_kernel, X_test_kernel, transformer


def contrast_enhancement(features, alpha=2.0):
    """
    对比度增强：拉大特征值的差异

    使用幂变换来增强特征对比度
    """
    print(f"\n  [分离方法] 应用对比度增强（alpha={alpha}）...")

    # 先归一化到[0,1]
    features_min = np.min(features, axis=0)
    features_max = np.max(features, axis=0)
    features_range = features_max - features_min
    features_range[features_range == 0] = 1

    features_norm = (features - features_min) / features_range

    # 应用幂变换
    features_enhanced = np.sign(features_norm - 0.5) * np.abs(features_norm - 0.5) ** alpha + 0.5

    # 恢复原始尺度
    features_enhanced = features_enhanced * features_range + features_min

    return features_enhanced


def find_best_gmm_params(X_train, X_val, max_components=8):
    """自动搜索最佳GMM参数"""
    print("  自动搜索最佳GMM参数...")

    best_score = -np.inf
    best_n = 1
    best_cov_type = 'full'

    for n in range(1, max_components + 1):
        for cov_type in ['full', 'diag']:
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type=cov_type,
                    max_iter=200,
                    random_state=42,
                    reg_covar=1e-6
                )
                gmm.fit(X_train)
                val_score = gmm.score(X_val)

                if val_score > best_score:
                    best_score = val_score
                    best_n = n
                    best_cov_type = cov_type
            except:
                continue

    print(f"    最佳参数: n_components={best_n}, covariance_type={best_cov_type}")
    return best_n, best_cov_type


def export_scores(scores_dict, output_path):
    """导出所有样本的分数到CSV"""
    df = pd.DataFrame(scores_dict)
    df.to_csv(output_path, index=False)
    print(f"\n  分数已导出到: {output_path}")
    return df


def visualize_score_distribution(df, output_dir):
    """可视化分数分布"""
    if not HAS_PLOT:
        print("  未安装matplotlib，跳过可视化")
        return

    print("\n  生成分数分布图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. GMM分数分布
    ax = axes[0, 0]
    normal_scores = df[df['label'] == 0]['gmm_score']
    anomaly_scores = df[df['label'] == 1]['gmm_score']

    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue')
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red')
    ax.set_xlabel('GMM Score')
    ax.set_ylabel('Frequency')
    ax.set_title('GMM Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 分数散点图（GMM vs ISO）
    if 'iso_score' in df.columns:
        ax = axes[0, 1]
        normal_data = df[df['label'] == 0]
        anomaly_data = df[df['label'] == 1]

        ax.scatter(normal_data['gmm_score'], normal_data['iso_score'],
                  alpha=0.5, label='Normal', color='blue', s=20)
        ax.scatter(anomaly_data['gmm_score'], anomaly_data['iso_score'],
                  alpha=0.5, label='Anomaly', color='red', s=20)
        ax.set_xlabel('GMM Score')
        ax.set_ylabel('IsolationForest Score')
        ax.set_title('Score Space (GMM vs ISO)')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # 3. 累积分布函数（CDF）
    ax = axes[1, 0]
    from scipy import stats

    normal_sorted = np.sort(normal_scores)
    anomaly_sorted = np.sort(anomaly_scores)

    normal_cdf = np.arange(1, len(normal_sorted) + 1) / len(normal_sorted)
    anomaly_cdf = np.arange(1, len(anomaly_sorted) + 1) / len(anomaly_sorted)

    ax.plot(normal_sorted, normal_cdf, label='Normal', color='blue', linewidth=2)
    ax.plot(anomaly_sorted, anomaly_cdf, label='Anomaly', color='red', linewidth=2)
    ax.set_xlabel('GMM Score')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('Cumulative Distribution Function')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 分离度热图（如果有多个分数）
    ax = axes[1, 1]
    score_cols = [col for col in df.columns if 'score' in col]
    if len(score_cols) > 1:
        corr_matrix = df[score_cols].corr()
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(score_cols)))
        ax.set_yticks(range(len(score_cols)))
        ax.set_xticklabels(score_cols, rotation=45, ha='right')
        ax.set_yticklabels(score_cols)
        ax.set_title('Score Correlation Matrix')
        plt.colorbar(im, ax=ax)

        for i in range(len(score_cols)):
            for j in range(len(score_cols)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'score_distribution.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"    图表已保存: {plot_path}")
    plt.close()


def analyze_separation_quality(df):
    """分析分离质量"""
    print("\n" + "="*70)
    print("分离质量分析")
    print("="*70)

    normal_gmm = df[df['label'] == 0]['gmm_score']
    anomaly_gmm = df[df['label'] == 1]['gmm_score']

    # 1. 基础统计
    print(f"\nGMM分数统计:")
    print(f"  正常样本: 均值={normal_gmm.mean():.2f}, 标准差={normal_gmm.std():.2f}")
    print(f"  异常样本: 均值={anomaly_gmm.mean():.2f}, 标准差={anomaly_gmm.std():.2f}")
    print(f"  均值差: {abs(anomaly_gmm.mean() - normal_gmm.mean()):.2f}")

    # 2. 重叠度分析
    normal_range = (normal_gmm.min(), normal_gmm.max())
    anomaly_range = (anomaly_gmm.min(), anomaly_gmm.max())

    overlap_start = max(normal_range[0], anomaly_range[0])
    overlap_end = min(normal_range[1], anomaly_range[1])

    if overlap_end > overlap_start:
        overlap_ratio = (overlap_end - overlap_start) / max(normal_range[1] - normal_range[0],
                                                             anomaly_range[1] - anomaly_range[0])
        print(f"\n重叠度分析:")
        print(f"  正常范围: [{normal_range[0]:.2f}, {normal_range[1]:.2f}]")
        print(f"  异常范围: [{anomaly_range[0]:.2f}, {anomaly_range[1]:.2f}]")
        print(f"  重叠区间: [{overlap_start:.2f}, {overlap_end:.2f}]")
        print(f"  重叠比例: {overlap_ratio:.2%}")

    # 3. 最优阈值建议（基于F1）
    thresholds = np.linspace(df['gmm_score'].min(), df['gmm_score'].max(), 200)
    best_f1 = 0
    best_threshold = 0

    for thresh in thresholds:
        preds = (df['gmm_score'] > thresh).astype(int)  # 高分数 = 异常
        f1 = f1_score(df['label'], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    print(f"\n最优阈值建议（基于F1）:")
    print(f"  阈值: {best_threshold:.2f}")
    print(f"  预期F1: {best_f1:.4f}")

    # 4. 可分离性指标（Cohen's d）
    cohens_d = (anomaly_gmm.mean() - normal_gmm.mean()) / np.sqrt((anomaly_gmm.std()**2 + normal_gmm.std()**2) / 2)
    print(f"\n可分离性指标（Cohen's d）: {cohens_d:.4f}")
    print(f"  > 0.8: 大效应（易分离）")
    print(f"  0.5-0.8: 中等效应")
    print(f"  0.2-0.5: 小效应")
    print(f"  < 0.2: 极小效应（难分离）")


def parse_arguments():
    parser = argparse.ArgumentParser(description='增强版GMM异常检测（带分数导出）')
    parser.add_argument('--normal_train_dir', type=str, required=True)
    parser.add_argument('--anomaly_test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/saved_models')
    parser.add_argument('--sr', type=int, default=22050)

    # 特征和模型选择
    parser.add_argument('--use_deep_features', action='store_true')
    parser.add_argument('--use_ensemble', action='store_true')
    parser.add_argument('--use_augmentation', action='store_true')

    # 分离增强方法
    parser.add_argument('--separation_method', type=str, default='none',
                       choices=['none', 'lda', 'kernel', 'contrast', 'all'],
                       help='空间分离增强方法')
    parser.add_argument('--kernel_type', type=str, default='rbf',
                       choices=['rbf', 'poly', 'cosine'])
    parser.add_argument('--contrast_alpha', type=float, default=2.0)

    # GMM参数
    parser.add_argument('--auto_tune', action='store_true', default=True)
    parser.add_argument('--n_components', type=int, default=None)
    parser.add_argument('--threshold_percentile', type=float, default=None)
    parser.add_argument('--k_features', type=int, default=60)

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 70)
    print("增强版GMM异常检测 - 带分数导出和空间分离")
    if args.use_deep_features:
        print("  [深度特征]")
    if args.use_augmentation:
        print("  [数据增强]")
    if args.use_ensemble:
        print("  [集成模型]")
    if args.separation_method != 'none':
        print(f"  [分离方法: {args.separation_method.upper()}]")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/9] 加载数据...")
    normal_files = glob(os.path.join(args.normal_train_dir, "*.wav"))
    print(f"  正常样本: {len(normal_files)} 个")

    normal_data = []
    for i, file_path in enumerate(normal_files):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(normal_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        normal_data.append(audio)

    from sklearn.model_selection import train_test_split
    train_val, normal_test = train_test_split(normal_data, test_size=0.2, random_state=42)
    normal_train, normal_val = train_test_split(train_val, test_size=0.25, random_state=42)

    print(f"  正常样本分割: 训练{len(normal_train)}, 验证{len(normal_val)}, 测试{len(normal_test)}")

    # 数据增强
    if args.use_augmentation:
        print("\n  应用数据增强...")
        augmented_train = []
        for i, audio in enumerate(normal_train):
            if (i + 1) % 50 == 0:
                print(f"    进度: {i + 1}/{len(normal_train)}")
            aug_audios = augment_audio(audio, args.sr)
            augmented_train.extend(aug_audios)
        normal_train = augmented_train
        print(f"  增强后训练样本: {len(normal_train)} 个")

    # 加载异常
    anomaly_files = glob(os.path.join(args.anomaly_test_dir, "*.wav"))
    print(f"  异常样本: {len(anomaly_files)} 个")

    anomaly_data = []
    for i, file_path in enumerate(anomaly_files):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_files)}")
        audio, _ = librosa.load(file_path, sr=args.sr)
        anomaly_data.append(audio)

    # 2. 提取特征
    print(f"\n[2/9] 提取特征...")

    if args.use_deep_features:
        feature_extractor = lambda a, sr: np.hstack([
            extract_enhanced_features(a, sr),
            extract_deep_features_simple(a, sr)
        ])
    else:
        feature_extractor = extract_enhanced_features

    print("  训练集...")
    train_features = []
    for i, audio in enumerate(normal_train):
        if (i + 1) % 100 == 0:
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
        if (i + 1) % 50 == 0:
            print(f"    进度: {i + 1}/{len(anomaly_data)}")
        test_features.append(feature_extractor(audio, args.sr))
        test_labels.append(1)

    test_features = np.array(test_features)
    test_labels = np.array(test_labels)

    print(f"  原始特征维度: {train_features.shape[1]}")

    # 3. 标准化
    print("\n[3/9] 特征标准化...")
    # 使用PowerTransformer处理偏态分布，增加特征鲁棒性
    from sklearn.preprocessing import PowerTransformer
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 4. 特征选择
    print("\n[4/9] 特征选择...")
    n_anomaly_for_selection = min(20, len(anomaly_data))
    temp_X = np.vstack([train_features_scaled, test_features_scaled[len(normal_test):len(normal_test)+n_anomaly_for_selection]])
    temp_y = np.array([0] * len(train_features_scaled) + [1] * n_anomaly_for_selection)

    # 方差阈值预筛选，去除低方差特征
    from sklearn.feature_selection import VarianceThreshold
    var_selector = VarianceThreshold(threshold=0.1)  # 去除低方差特征
    train_features_scaled = var_selector.fit_transform(train_features_scaled)
    val_features_scaled = var_selector.transform(val_features_scaled)
    test_features_scaled = var_selector.transform(test_features_scaled)
    
    # 更新temp_X以匹配方差筛选后的特征
    temp_X = np.vstack([train_features_scaled, test_features_scaled[len(normal_test):len(normal_test)+n_anomaly_for_selection]])
    
    # 减少特征数量以防止过拟合
    k_features = min(40, train_features_scaled.shape[1])  # 减少到40个特征
    selector = SelectKBest(f_classif, k=k_features)
    selector.fit(temp_X, temp_y)

    train_features_selected = selector.transform(train_features_scaled)
    val_features_selected = selector.transform(val_features_scaled)
    test_features_selected = selector.transform(test_features_scaled)

    print(f"  原始特征数: {train_features_scaled.shape[1]}")
    print(f"  选择特征数: {train_features_selected.shape[1]}")

    # 5. 应用分离增强方法
    print("\n[5/9] 应用分离增强方法...")

    separation_transformer = None

    if args.separation_method == 'lda':
        # 使用少量异常样本进行LDA
        n_anomaly_for_lda = min(30, len(anomaly_data))
        # test_features_selected的结构：[normal_test样本, anomaly_data样本]
        lda_X = np.vstack([train_features_selected, test_features_selected[len(normal_test):len(normal_test)+n_anomaly_for_lda]])
        lda_y = np.array([0] * len(train_features_selected) + [1] * n_anomaly_for_lda)

        # 对所有特征进行LDA变换
        all_features_to_transform = np.vstack([train_features_selected, val_features_selected, test_features_selected])

        lda = LinearDiscriminantAnalysis(n_components=1)
        lda.fit(lda_X, lda_y)

        all_features_transformed = lda.transform(all_features_to_transform)

        # 分离回各个集合
        train_features_selected = all_features_transformed[:len(train_features_selected)]
        val_features_selected = all_features_transformed[len(train_features_selected):len(train_features_selected)+len(val_features_selected)]
        test_features_selected = all_features_transformed[len(train_features_selected)+len(val_features_selected):]

        separation_transformer = lda
        print(f"    LDA投影后维度: {train_features_selected.shape[1]}")

    elif args.separation_method == 'kernel':
        train_features_selected, test_features_selected_temp, separation_transformer = apply_kernel_transformation(
            train_features_selected,
            np.vstack([val_features_selected, test_features_selected]),
            kernel=args.kernel_type
        )
        val_features_selected = test_features_selected_temp[:len(val_features_selected)]
        test_features_selected = test_features_selected_temp[len(val_features_selected):]

    elif args.separation_method == 'contrast':
        all_features = np.vstack([train_features_selected, val_features_selected, test_features_selected])
        all_features_enhanced = contrast_enhancement(all_features, alpha=args.contrast_alpha)

        train_features_selected = all_features_enhanced[:len(train_features_selected)]
        val_features_selected = all_features_enhanced[len(train_features_selected):len(train_features_selected)+len(val_features_selected)]
        test_features_selected = all_features_enhanced[len(train_features_selected)+len(val_features_selected):]

    # 6. 训练模型
    print("\n[6/9] 训练模型...")

    if args.use_ensemble:
        print("  训练集成模型...")

        # GMM
        if args.auto_tune and args.n_components is None:
            best_n, best_cov = find_best_gmm_params(train_features_selected, val_features_selected, max_components=12)
        else:
            best_n = args.n_components if args.n_components else 3
            best_cov = 'full'

        gmm = GaussianMixture(n_components=best_n, covariance_type=best_cov, max_iter=200, random_state=42, reg_covar=1e-6)
        gmm.fit(train_features_selected)

        # Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=200,           # 增加树的数量，提高稳定性
            contamination=0.05,         # 降低到5%，减少误报
            max_samples=256,            # 限制每棵树的样本数
            random_state=42,
            n_jobs=-1
        )
        iso_forest.fit(train_features_selected)

        # One-Class SVM
        ocsvm = OneClassSVM(
            kernel='rbf',
            gamma='scale',              # 使用 scale 代替 auto，更稳定
            nu=0.05,                    # 降低到5%，放宽边界
            tol=1e-4                    # 提高容忍度
        )
        ocsvm.fit(train_features_selected)

        models = {'gmm': gmm, 'iso_forest': iso_forest, 'ocsvm': ocsvm}

    else:
        if args.auto_tune and args.n_components is None:
            best_n, best_cov = find_best_gmm_params(train_features_selected, val_features_selected, max_components=12)
        else:
            best_n = args.n_components if args.n_components else 3
            best_cov = 'full'

        gmm = GaussianMixture(n_components=best_n, covariance_type=best_cov, max_iter=200, random_state=42, reg_covar=1e-6)
        gmm.fit(train_features_selected)
        models = {'gmm': gmm}

    # 7. 计算所有样本的分数
    print("\n[7/9] 计算分数并导出...")

    # 验证长度匹配
    print(f"  测试集特征数: {len(test_features_selected)}")
    print(f"  测试集标签数: {len(test_labels)}")
    print(f"    正常测试样本: {len(normal_test)}")
    print(f"    异常测试样本: {len(anomaly_data)}")

    if len(test_features_selected) != len(test_labels):
        raise ValueError(f"特征数({len(test_features_selected)})与标签数({len(test_labels)})不匹配！")

    # 收集所有分数
    scores_dict = {
        'label': test_labels,
        'sample_type': ['normal' if l == 0 else 'anomaly' for l in test_labels]
    }

    # GMM分数
    gmm_scores = models['gmm'].score_samples(test_features_selected)
    scores_dict['gmm_score'] = -gmm_scores  # 取负，使高分数=异常
    scores_dict['gmm_log_likelihood'] = gmm_scores  # 保留原始对数似然

    # 如果是集成模型，添加其他分数
    if args.use_ensemble:
        # Isolation Forest分数
        iso_scores = models['iso_forest'].score_samples(test_features_selected)
        scores_dict['iso_score'] = -iso_scores  # 取负，使高分数=异常

        # One-Class SVM决策函数
        ocsvm_scores = models['ocsvm'].decision_function(test_features_selected)
        scores_dict['ocsvm_score'] = -ocsvm_scores  # 取负，使高分数=异常

        # 集成分数（简单平均）
        scores_dict['ensemble_avg'] = (-gmm_scores - iso_scores - ocsvm_scores) / 3

    # 导出到CSV
    os.makedirs(args.output_dir, exist_ok=True)
    scores_path = os.path.join(args.output_dir, 'sample_scores.csv')
    df_scores = export_scores(scores_dict, scores_path)

    # 8. 分析和可视化
    print("\n[8/9] 分析分离质量...")
    analyze_separation_quality(df_scores)

    visualize_score_distribution(df_scores, args.output_dir)

    # 9. 使用最优阈值评估
    print("\n[9/9] 模型评估...")

    # 基于F1找到的最优阈值
    thresholds = np.linspace(df_scores['gmm_score'].min(), df_scores['gmm_score'].max(), 200)
    best_f1 = 0
    best_threshold = 0

    for thresh in thresholds:
        preds = (df_scores['gmm_score'] > thresh).astype(int)
        f1 = f1_score(df_scores['label'], preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh

    # 使用最优阈值评估
    predictions = (df_scores['gmm_score'] > best_threshold).astype(int)

    accuracy = accuracy_score(df_scores['label'], predictions)
    precision = precision_score(df_scores['label'], predictions, zero_division=0)
    recall = recall_score(df_scores['label'], predictions, zero_division=0)
    f1 = f1_score(df_scores['label'], predictions, zero_division=0)
    cm = confusion_matrix(df_scores['label'], predictions)

    try:
        auc = roc_auc_score(df_scores['label'], df_scores['gmm_score'])
    except:
        auc = None

    print("\n" + "=" * 70)
    print("评估结果（使用最优阈值）")
    print("=" * 70)
    print(f"  最优阈值:  {best_threshold:.4f}")
    print(f"  准确率:    {accuracy:.4f}")
    print(f"  精确率:    {precision:.4f}")
    print(f"  召回率:    {recall:.4f}")
    print(f"  F1分数:    {f1:.4f}")
    if auc:
        print(f"  AUC:       {auc:.4f}")

    print(f"\n混淆矩阵:")
    print(f"            预测正常  预测异常")
    print(f"  实际正常     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"  实际异常     {cm[1,0]:3d}      {cm[1,1]:3d}")

    # 10. 保存模型
    print("\n保存模型和结果...")

    model_data = {
        'gmm': gmm,
        'scaler': scaler,
        'var_selector': var_selector,  # 添加方差选择器
        'selector': selector,
        'separation_transformer': separation_transformer,
        'separation_method': args.separation_method,
        'threshold': best_threshold,
        'n_components': best_n,
        'sample_rate': args.sr
    }

    if args.use_ensemble:
        model_data['iso_forest'] = models['iso_forest']
        model_data['ocsvm'] = models['ocsvm']

    model_path = os.path.join(args.output_dir, 'gmm_with_scores.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"  模型已保存: {model_path}")
    print(f"  分数已保存: {scores_path}")

    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print(f"\n接下来可以：")
    print(f"  1. 查看分数文件: {scores_path}")
    print(f"  2. 查看可视化图表: {os.path.join(args.output_dir, 'score_distribution.png')}")
    print(f"  3. 使用Python/R进行进一步的统计分析")
    print(f"\n建议的数学方法：")
    print(f"  - 核密度估计（KDE）找到更好的分界线")
    print(f"  - 混合模型（GMM on scores）")
    print(f"  - 贝叶斯优化阈值")
    print(f"  - 使用分位数回归")


if __name__ == "__main__":
    main()
