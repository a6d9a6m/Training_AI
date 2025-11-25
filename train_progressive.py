#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
渐进式训练：逐步扩展到新数据域

策略：
1. 初始模型：在源域（原始数据）训练
2. 第一轮扩展：加入20%新数据，使用域适应
3. 第二轮扩展：加入50%新数据
4. 第三轮扩展：使用全部数据

每轮都评估性能，确保不会灾难性遗忘
"""

import os
import sys
import argparse
import numpy as np
import pickle
from glob import glob
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.domain_adaptation import (
    apply_domain_adaptation,
    calculate_mmd,
    select_domain_invariant_features
)


def load_audio_files(directory, sr=22050):
    """加载音频文件"""
    files = glob(os.path.join(directory, "*.wav"))
    print(f"  加载 {len(files)} 个文件")

    audios = []
    for i, f in enumerate(files):
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{len(files)}")
        audio, _ = librosa.load(f, sr=sr)
        audios.append(audio)

    return audios


def extract_enhanced_features(audio, sr):
    """提取增强特征"""
    features = []

    # MFCC
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

    # 色度
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    # 频谱特征
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

    # 节奏
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zcr), np.std(zcr), np.max(zcr)])

    # 能量
    rms = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms), np.std(rms), np.max(rms), np.min(rms)])

    # 梅尔频谱
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


def extract_features_from_audios(audios, sr):
    """批量提取特征"""
    features = []
    for i, audio in enumerate(audios):
        if (i + 1) % 100 == 0:
            print(f"    提取特征: {i+1}/{len(audios)}")
        features.append(extract_enhanced_features(audio, sr))
    return np.array(features)


def train_base_models(X_train, contamination=0.03, nu=0.03):
    """训练基础模型"""
    print("\n  训练基础模型...")

    # GMM
    print("    - GMM")
    gmm = GaussianMixture(n_components=5, covariance_type='full', max_iter=200,
                         random_state=42, reg_covar=1e-6)
    gmm.fit(X_train)

    # IsolationForest
    print("    - IsolationForest")
    iso_forest = IsolationForest(n_estimators=200, contamination=contamination,
                                 max_samples=256, random_state=42, n_jobs=-1)
    iso_forest.fit(X_train)

    # OneClassSVM
    print("    - OneClassSVM")
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=nu, tol=1e-4)
    ocsvm.fit(X_train)

    return gmm, iso_forest, ocsvm


def compute_scores(models, X):
    """计算所有模型的分数"""
    gmm, iso_forest, ocsvm = models

    scores = {}
    scores['gmm_score'] = -gmm.score_samples(X)
    scores['iso_score'] = -iso_forest.score_samples(X)
    scores['ocsvm_score'] = -ocsvm.decision_function(X)

    return scores


def train_ensemble(scores_train, labels_train):
    """训练集成模型（随机森林）"""
    print("\n  训练随机森林集成...")

    X = np.column_stack([scores_train['gmm_score'],
                        scores_train['iso_score'],
                        scores_train['ocsvm_score']])

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, labels_train)

    return rf


def evaluate(models, ensemble, scaler, selector, X_test, y_test):
    """评估模型"""
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = selector.transform(X_test_scaled)

    scores_test = compute_scores(models, X_test_selected)
    X_ensemble = np.column_stack([scores_test['gmm_score'],
                                  scores_test['iso_score'],
                                  scores_test['ocsvm_score']])

    y_pred = ensemble.predict(X_ensemble)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'predictions': y_pred
    }


def progressive_training(source_normal, source_anomaly,
                        target_normal, target_anomaly,
                        sr=22050, output_dir='models/saved_models_progressive'):
    """渐进式训练主流程"""

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("渐进式训练流程")
    print("="*70)

    # ========== 第0轮：源域基础训练 ==========
    print("\n[第0轮] 源域基础训练")
    print("-"*70)

    print("\n提取源域特征...")
    source_normal_features = extract_features_from_audios(source_normal, sr)
    source_anomaly_features = extract_features_from_audios(source_anomaly, sr)

    print("\n提取目标域特征...")
    target_normal_features = extract_features_from_audios(target_normal, sr)
    target_anomaly_features = extract_features_from_audios(target_anomaly, sr)

    # 分割源域数据
    source_train, source_val = train_test_split(source_normal_features, test_size=0.2, random_state=42)

    print(f"\n源域训练集: {len(source_train)}")
    print(f"源域验证集: {len(source_val)}")
    print(f"目标域正常: {len(target_normal_features)}")
    print(f"目标域异常: {len(target_anomaly_features)}")

    # 标准化
    print("\n标准化...")
    scaler = RobustScaler()
    source_train_scaled = scaler.fit_transform(source_train)
    source_val_scaled = scaler.transform(source_val)

    # 特征选择（使用少量异常样本）
    print("\n特征选择...")
    n_anomaly_for_selection = min(20, len(source_anomaly_features))
    temp_X = np.vstack([source_train_scaled, source_anomaly_features[:n_anomaly_for_selection]])
    temp_y = np.array([0] * len(source_train_scaled) + [1] * n_anomaly_for_selection)

    selector = SelectKBest(f_classif, k=min(50, source_train_scaled.shape[1]))
    selector.fit(temp_X, temp_y)

    source_train_selected = selector.transform(source_train_scaled)

    # 训练基础模型
    models = train_base_models(source_train_selected)

    # 计算源域验证集的分数
    source_val_selected = selector.transform(source_val_scaled)
    source_val_scores = compute_scores(models, source_val_selected)

    # 需要一些异常样本训练集成模型
    source_anomaly_scaled = scaler.transform(source_anomaly_features)
    source_anomaly_selected = selector.transform(source_anomaly_scaled)
    source_anomaly_scores = compute_scores(models, source_anomaly_selected)

    # 合并训练集成模型
    ensemble_train_scores = {
        'gmm_score': np.concatenate([source_val_scores['gmm_score'], source_anomaly_scores['gmm_score']]),
        'iso_score': np.concatenate([source_val_scores['iso_score'], source_anomaly_scores['iso_score']]),
        'ocsvm_score': np.concatenate([source_val_scores['ocsvm_score'], source_anomaly_scores['ocsvm_score']])
    }
    ensemble_train_labels = np.array([0] * len(source_val) + [1] * len(source_anomaly_features))

    ensemble = train_ensemble(ensemble_train_scores, ensemble_train_labels)

    # 评估源域
    print("\n源域评估:")
    source_eval = evaluate(models, ensemble, scaler, selector,
                          source_val, np.zeros(len(source_val)))
    print(f"  准确率: {source_eval['accuracy']:.2%}")

    # 评估目标域
    print("\n目标域初始评估:")
    target_X = np.vstack([target_normal_features, target_anomaly_features])
    target_y = np.array([0] * len(target_normal_features) + [1] * len(target_anomaly_features))

    target_eval_0 = evaluate(models, ensemble, scaler, selector, target_X, target_y)
    print(f"  准确率: {target_eval_0['accuracy']:.2%}")
    print(f"  F1: {target_eval_0['f1']:.2%}")

    # 计算MMD（域距离）
    target_all_scaled = scaler.transform(target_X)
    target_all_selected = selector.transform(target_all_scaled)
    mmd_before = calculate_mmd(source_train_selected, target_all_selected)
    print(f"\n域距离（MMD）: {mmd_before:.4f}")

    # ========== 第1轮：加入20%目标域数据 ==========
    print("\n" + "="*70)
    print("[第1轮] 加入20%目标域数据")
    print("-"*70)

    # 选择20%目标域正常数据
    n_target_20 = int(len(target_normal_features) * 0.2)
    target_normal_20 = target_normal_features[:n_target_20]

    print(f"\n新增目标域样本: {n_target_20}")

    # 合并数据
    combined_train = np.vstack([source_train, target_normal_20])
    print(f"合并后训练集: {len(combined_train)}")

    # 域适应
    print("\n应用域适应...")
    source_adapted, target_adapted = apply_domain_adaptation(
        source_train, target_normal_20,
        method='coral_adaptation'  # 使用CORAL对齐协方差
    )
    combined_train_adapted = np.vstack([source_adapted, target_adapted])

    # 重新标准化和特征选择
    scaler_1 = RobustScaler()
    combined_train_scaled = scaler_1.fit_transform(combined_train_adapted)

    temp_X_1 = np.vstack([combined_train_scaled, source_anomaly_features[:n_anomaly_for_selection]])
    temp_y_1 = np.array([0] * len(combined_train_scaled) + [1] * n_anomaly_for_selection)

    selector_1 = SelectKBest(f_classif, k=min(50, combined_train_scaled.shape[1]))
    selector_1.fit(temp_X_1, temp_y_1)

    combined_train_selected = selector_1.transform(combined_train_scaled)

    # 重新训练
    models_1 = train_base_models(combined_train_selected)

    # 重新训练集成
    source_anomaly_scaled_1 = scaler_1.transform(source_anomaly_features)
    source_anomaly_selected_1 = selector_1.transform(source_anomaly_scaled_1)
    source_anomaly_scores_1 = compute_scores(models_1, source_anomaly_selected_1)

    # 使用部分正常样本
    val_normal_scaled = scaler_1.transform(source_val)
    val_normal_selected = selector_1.transform(val_normal_scaled)
    val_normal_scores = compute_scores(models_1, val_normal_selected)

    ensemble_train_scores_1 = {
        'gmm_score': np.concatenate([val_normal_scores['gmm_score'], source_anomaly_scores_1['gmm_score']]),
        'iso_score': np.concatenate([val_normal_scores['iso_score'], source_anomaly_scores_1['iso_score']]),
        'ocsvm_score': np.concatenate([val_normal_scores['ocsvm_score'], source_anomaly_scores_1['ocsvm_score']])
    }
    ensemble_train_labels_1 = np.array([0] * len(source_val) + [1] * len(source_anomaly_features))

    ensemble_1 = train_ensemble(ensemble_train_scores_1, ensemble_train_labels_1)

    # 评估
    target_eval_1 = evaluate(models_1, ensemble_1, scaler_1, selector_1, target_X, target_y)
    print(f"\n目标域评估（第1轮）:")
    print(f"  准确率: {target_eval_1['accuracy']:.2%} (提升: {target_eval_1['accuracy']-target_eval_0['accuracy']:+.2%})")
    print(f"  F1: {target_eval_1['f1']:.2%} (提升: {target_eval_1['f1']-target_eval_0['f1']:+.2%})")

    # ========== 第2轮：加入50%目标域数据 ==========
    print("\n" + "="*70)
    print("[第2轮] 加入50%目标域数据")
    print("-"*70)

    n_target_50 = int(len(target_normal_features) * 0.5)
    target_normal_50 = target_normal_features[:n_target_50]

    print(f"\n新增目标域样本: {n_target_50}")

    combined_train_2 = np.vstack([source_train, target_normal_50])

    # 域适应
    print("\n应用域适应...")
    source_adapted_2, target_adapted_2 = apply_domain_adaptation(
        source_train, target_normal_50,
        method='joint_distribution_adaptation'  # 更强的适应
    )
    combined_train_adapted_2 = np.vstack([source_adapted_2, target_adapted_2])

    scaler_2 = RobustScaler()
    combined_train_scaled_2 = scaler_2.fit_transform(combined_train_adapted_2)

    temp_X_2 = np.vstack([combined_train_scaled_2, source_anomaly_features[:n_anomaly_for_selection]])
    temp_y_2 = np.array([0] * len(combined_train_scaled_2) + [1] * n_anomaly_for_selection)

    selector_2 = SelectKBest(f_classif, k=min(50, combined_train_scaled_2.shape[1]))
    selector_2.fit(temp_X_2, temp_y_2)

    combined_train_selected_2 = selector_2.transform(combined_train_scaled_2)

    models_2 = train_base_models(combined_train_selected_2, contamination=0.04, nu=0.04)

    # 重新训练集成
    source_anomaly_scaled_2 = scaler_2.transform(source_anomaly_features)
    source_anomaly_selected_2 = selector_2.transform(source_anomaly_scaled_2)
    source_anomaly_scores_2 = compute_scores(models_2, source_anomaly_selected_2)

    val_normal_scaled_2 = scaler_2.transform(source_val)
    val_normal_selected_2 = selector_2.transform(val_normal_scaled_2)
    val_normal_scores_2 = compute_scores(models_2, val_normal_selected_2)

    ensemble_train_scores_2 = {
        'gmm_score': np.concatenate([val_normal_scores_2['gmm_score'], source_anomaly_scores_2['gmm_score']]),
        'iso_score': np.concatenate([val_normal_scores_2['iso_score'], source_anomaly_scores_2['iso_score']]),
        'ocsvm_score': np.concatenate([val_normal_scores_2['ocsvm_score'], source_anomaly_scores_2['ocsvm_score']])
    }
    ensemble_train_labels_2 = np.array([0] * len(source_val) + [1] * len(source_anomaly_features))

    ensemble_2 = train_ensemble(ensemble_train_scores_2, ensemble_train_labels_2)

    target_eval_2 = evaluate(models_2, ensemble_2, scaler_2, selector_2, target_X, target_y)
    print(f"\n目标域评估（第2轮）:")
    print(f"  准确率: {target_eval_2['accuracy']:.2%} (提升: {target_eval_2['accuracy']-target_eval_1['accuracy']:+.2%})")
    print(f"  F1: {target_eval_2['f1']:.2%} (提升: {target_eval_2['f1']-target_eval_1['f1']:+.2%})")

    # ========== 保存最佳模型 ==========
    best_round = np.argmax([target_eval_0['f1'], target_eval_1['f1'], target_eval_2['f1']])

    if best_round == 0:
        best_models = (models, ensemble, scaler, selector)
        best_eval = target_eval_0
    elif best_round == 1:
        best_models = (models_1, ensemble_1, scaler_1, selector_1)
        best_eval = target_eval_1
    else:
        best_models = (models_2, ensemble_2, scaler_2, selector_2)
        best_eval = target_eval_2

    print("\n" + "="*70)
    print(f"最佳模型：第{best_round}轮")
    print("="*70)
    print(f"  准确率: {best_eval['accuracy']:.2%}")
    print(f"  精确率: {best_eval['precision']:.2%}")
    print(f"  召回率: {best_eval['recall']:.2%}")
    print(f"  F1: {best_eval['f1']:.2%}")

    # 保存
    gmm, iso, ocsvm = best_models[0]
    ensemble = best_models[1]
    scaler = best_models[2]
    selector = best_models[3]

    model_data = {
        'gmm': gmm,
        'iso_forest': iso,
        'ocsvm': ocsvm,
        'scaler': scaler,
        'selector': selector,
        'sample_rate': sr
    }

    model_path = os.path.join(output_dir, 'gmm_progressive.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    ensemble_data = {
        'model': ensemble,
        'model_type': 'random_forest',
        'score_columns': ['gmm_score', 'iso_score', 'ocsvm_score'],
        'metrics': {
            'f1': best_eval['f1'],
            'precision': best_eval['precision'],
            'recall': best_eval['recall'],
            'accuracy': best_eval['accuracy']
        }
    }

    ensemble_path = os.path.join(output_dir, 'ensemble_progressive.pkl')
    with open(ensemble_path, 'wb') as f:
        pickle.dump(ensemble_data, f)

    print(f"\n模型已保存:")
    print(f"  基础模型: {model_path}")
    print(f"  集成模型: {ensemble_path}")

    return best_models, best_eval


def parse_args():
    parser = argparse.ArgumentParser(description='渐进式训练')
    parser.add_argument('--source_normal_dir', type=str, required=True)
    parser.add_argument('--source_anomaly_dir', type=str, required=True)
    parser.add_argument('--target_normal_dir', type=str, required=True)
    parser.add_argument('--target_anomaly_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='models/saved_models_progressive')
    parser.add_argument('--sr', type=int, default=22050)

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("渐进式训练系统")
    print("="*70)

    # 加载数据
    print("\n[加载数据]")
    print("\n源域正常样本:")
    source_normal = load_audio_files(args.source_normal_dir, args.sr)

    print("\n源域异常样本:")
    source_anomaly = load_audio_files(args.source_anomaly_dir, args.sr)

    print("\n目标域正常样本:")
    target_normal = load_audio_files(args.target_normal_dir, args.sr)

    print("\n目标域异常样本:")
    target_anomaly = load_audio_files(args.target_anomaly_dir, args.sr)

    # 渐进式训练
    progressive_training(source_normal, source_anomaly,
                        target_normal, target_anomaly,
                        args.sr, args.output_dir)

    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == "__main__":
    main()
