#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试时增强（TTA）：通过对测试样本进行多次增强预测，取平均结果

原理：
- 对单个音频样本进行微小变换（时间拉伸、音高偏移、噪声）
- 对所有变换后的版本分别预测
- 取平均概率作为最终预测

效果：大幅提升模型对新数据的泛化能力
"""

import os
import sys
import numpy as np
import librosa
import pickle
import argparse
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def augment_audio_for_tta(audio, sr, n_augmentations=5):
    """
    生成测试时增强版本

    策略：轻微变换，不改变音频本质特征
    """
    augmented = [audio]  # 原始版本

    # 1. 时间拉伸（轻微）
    try:
        audio_fast = librosa.effects.time_stretch(audio, rate=1.05)
        augmented.append(audio_fast)

        audio_slow = librosa.effects.time_stretch(audio, rate=0.95)
        augmented.append(audio_slow)
    except:
        pass

    # 2. 音高偏移（轻微）
    try:
        audio_pitch_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=1)
        augmented.append(audio_pitch_up)

        audio_pitch_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-1)
        augmented.append(audio_pitch_down)
    except:
        pass

    # 3. 轻微噪声
    if len(augmented) < n_augmentations:
        try:
            noise = np.random.normal(0, 0.003, len(audio))
            audio_noisy = audio + noise
            augmented.append(audio_noisy)
        except:
            pass

    return augmented[:n_augmentations]


def extract_enhanced_features(audio, sr):
    """提取特征（与训练时相同）"""
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

    # 频谱
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

    # 梅尔
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


def compute_scores(base_model_data, features):
    """计算基础模型分数"""
    # 预处理
    features = features.reshape(1, -1)

    if 'scaler' in base_model_data:
        features = base_model_data['scaler'].transform(features)

    if 'selector' in base_model_data:
        features = base_model_data['selector'].transform(features)

    scores = {}

    # GMM
    if 'gmm' in base_model_data:
        gmm_score = base_model_data['gmm'].score_samples(features)[0]
        scores['gmm_score'] = -gmm_score

    # IsolationForest
    if 'iso_forest' in base_model_data:
        iso_score = base_model_data['iso_forest'].score_samples(features)[0]
        scores['iso_score'] = -iso_score

    # OneClassSVM
    if 'ocsvm' in base_model_data:
        ocsvm_score = base_model_data['ocsvm'].decision_function(features)[0]
        scores['ocsvm_score'] = -ocsvm_score

    return scores


def predict_with_tta(audio_path, base_model_data, ensemble_data, sr=22050, n_augmentations=5):
    """
    使用TTA进行预测

    返回：
    - prediction: 0/1
    - proba: [prob_normal, prob_anomaly]
    - confidence: 预测置信度
    """
    # 加载音频
    audio, _ = librosa.load(audio_path, sr=sr)

    # 生成增强版本
    augmented_audios = augment_audio_for_tta(audio, sr, n_augmentations)

    all_predictions = []
    all_probas = []

    # 对每个增强版本预测
    for aug_audio in augmented_audios:
        # 提取特征
        features = extract_enhanced_features(aug_audio, sr)

        # 计算分数
        scores = compute_scores(base_model_data, features)

        # 集成预测
        score_cols = ensemble_data['score_columns']
        X = np.array([scores[col] for col in score_cols]).reshape(1, -1)

        model = ensemble_data['model']
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]

        all_predictions.append(pred)
        all_probas.append(proba)

    # 平均概率
    avg_proba = np.mean(all_probas, axis=0)

    # 最终预测
    final_prediction = 1 if avg_proba[1] > 0.5 else 0

    # 置信度：预测类别的平均概率
    confidence = avg_proba[final_prediction]

    # 一致性：所有增强版本的预测一致性
    consistency = np.mean(np.array(all_predictions) == final_prediction)

    return {
        'prediction': final_prediction,
        'proba': avg_proba,
        'confidence': confidence,
        'consistency': consistency,
        'n_augmentations': len(augmented_audios)
    }


def batch_predict_with_tta(audio_dir, base_model_path, ensemble_model_path, sr=22050, n_augmentations=5):
    """批量预测（带TTA）"""
    # 加载模型
    print(f"加载基础模型: {base_model_path}")
    with open(base_model_path, 'rb') as f:
        base_model_data = pickle.load(f)

    print(f"加载集成模型: {ensemble_model_path}")
    with open(ensemble_model_path, 'rb') as f:
        ensemble_data = pickle.load(f)

    # 获取音频文件
    audio_files = glob(os.path.join(audio_dir, "*.wav"))
    print(f"\n找到 {len(audio_files)} 个音频文件")
    print(f"使用TTA（每个样本{n_augmentations}次增强）\n")

    results = []

    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"进度: {i+1}/{len(audio_files)}")

        try:
            result = predict_with_tta(audio_path, base_model_data, ensemble_data, sr, n_augmentations)

            # 提取真实标签
            filename = os.path.basename(audio_path)
            if '_normal_' in filename.lower():
                true_label = 0
            elif '_anomaly_' in filename.lower():
                true_label = 1
            else:
                true_label = -1

            result['file'] = filename
            result['true_label'] = true_label
            result['correct'] = (result['prediction'] == true_label) if true_label != -1 else None

            results.append(result)

            # 显示
            correct_marker = ""
            if true_label != -1:
                correct_marker = "✓" if result['correct'] else "✗ ERROR"

            pred_label = "异常" if result['prediction'] == 1 else "正常"
            true_label_str = "正常" if true_label == 0 else "异常" if true_label == 1 else "未知"

            print(f"  [{i+1:3d}] 真实:{true_label_str:4s} | 预测:{pred_label:4s} {correct_marker:7s} "
                  f"(置信:{result['confidence']:.1%}, 一致:{result['consistency']:.1%})")

        except Exception as e:
            print(f"  [{i+1:3d}] ERROR - {filename}: {str(e)[:50]}")

    # 统计
    print("\n" + "="*70)
    print("TTA预测统计")
    print("="*70)

    labeled = [r for r in results if r['true_label'] != -1]

    if labeled:
        y_true = [r['true_label'] for r in labeled]
        y_pred = [r['prediction'] for r in labeled]

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        print(f"\n准确率: {acc:.2%}")
        print(f"精确率: {prec:.2%}")
        print(f"召回率: {rec:.2%}")
        print(f"F1分数: {f1:.2%}")

        print(f"\n混淆矩阵:")
        print(f"                 预测:正常   预测:异常")
        print(f"  真实:正常      {cm[0,0]:6d}      {cm[0,1]:6d}")
        print(f"  真实:异常      {cm[1,0]:6d}      {cm[1,1]:6d}")

        # 置信度分析
        correct_results = [r for r in labeled if r['correct']]
        wrong_results = [r for r in labeled if not r['correct']]

        if correct_results:
            avg_conf_correct = np.mean([r['confidence'] for r in correct_results])
            avg_cons_correct = np.mean([r['consistency'] for r in correct_results])
            print(f"\n正确预测的平均置信度: {avg_conf_correct:.2%}")
            print(f"正确预测的平均一致性: {avg_cons_correct:.2%}")

        if wrong_results:
            avg_conf_wrong = np.mean([r['confidence'] for r in wrong_results])
            avg_cons_wrong = np.mean([r['consistency'] for r in wrong_results])
            print(f"\n错误预测的平均置信度: {avg_conf_wrong:.2%}")
            print(f"错误预测的平均一致性: {avg_cons_wrong:.2%}")

    return results


def parse_args():
    parser = argparse.ArgumentParser(description='TTA预测')
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--ensemble_model', type=str, required=True)
    parser.add_argument('--audio_dir', type=str, required=True)
    parser.add_argument('--sr', type=int, default=22050)
    parser.add_argument('--n_augmentations', type=int, default=5,
                       help='每个样本的增强次数（越多越稳定但越慢）')

    return parser.parse_args()


def main():
    args = parse_args()

    print("="*70)
    print("TTA（测试时增强）预测系统")
    print("="*70)

    batch_predict_with_tta(
        args.audio_dir,
        args.base_model,
        args.ensemble_model,
        args.sr,
        args.n_augmentations
    )

    print("\n" + "="*70)
    print("完成！")
    print("="*70)


if __name__ == "__main__":
    main()
