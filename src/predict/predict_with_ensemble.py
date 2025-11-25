#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用保存的集成模型进行预测

加载完整的流程：
1. 加载基础模型（GMM, IsolationForest, OneClassSVM）
2. 对新音频提取特征
3. 计算多个分数
4. 使用集成模型融合分数得到最终判断
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import librosa
from glob import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_base_models(model_path):
    """加载基础模型（GMM等）"""
    print(f"加载基础模型: {model_path}")

    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print("  包含组件:")
    for key in model_data.keys():
        if key not in ['threshold', 'n_components', 'sample_rate', 'separation_method']:
            print(f"    - {key}")

    return model_data


def load_ensemble_model(ensemble_path):
    """加载集成模型"""
    print(f"\n加载集成模型: {ensemble_path}")

    with open(ensemble_path, 'rb') as f:
        ensemble_data = pickle.load(f)

    print(f"  模型类型: {ensemble_data['model_type']}")
    print(f"  需要的分数: {ensemble_data['score_columns']}")
    print(f"  训练集性能: F1={ensemble_data['metrics']['f1']:.4f}")

    return ensemble_data


def extract_enhanced_features(audio, sr):
    """标准特征提取（与训练时保持完全一致）"""
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


def extract_deep_features_simple(audio, sr):
    """深度特征提取"""
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


def extract_features_from_audio(audio_path, base_model_data, use_deep_features=True):
    """从音频提取特征"""
    # 加载音频
    sr = base_model_data.get('sample_rate', 22050)
    audio, _ = librosa.load(audio_path, sr=sr)

    # 标准特征
    features_standard = extract_enhanced_features(audio, sr)

    # 深度特征
    if use_deep_features:
        features_deep = extract_deep_features_simple(audio, sr)
        features = np.hstack([features_standard, features_deep])
    else:
        features = features_standard

    return features


def compute_all_scores(features, base_model_data):
    """计算所有模型的分数"""
    # 预处理特征
    features = features.reshape(1, -1)

    # Scaler
    if 'scaler' in base_model_data:
        features = base_model_data['scaler'].transform(features)

    # Variance Selector (新增)
    if 'var_selector' in base_model_data:
        features = base_model_data['var_selector'].transform(features)

    # Selector
    if 'selector' in base_model_data:
        features = base_model_data['selector'].transform(features)

    # Separation transformer
    if 'separation_transformer' in base_model_data and base_model_data['separation_transformer'] is not None:
        features = base_model_data['separation_transformer'].transform(features)

    scores = {}

    # GMM
    if 'gmm' in base_model_data:
        gmm_score = base_model_data['gmm'].score_samples(features)[0]
        scores['gmm_score'] = -gmm_score  # 取负

    # Isolation Forest
    if 'iso_forest' in base_model_data:
        iso_score = base_model_data['iso_forest'].score_samples(features)[0]
        scores['iso_score'] = -iso_score

    # One-Class SVM
    if 'ocsvm' in base_model_data:
        ocsvm_score = base_model_data['ocsvm'].decision_function(features)[0]
        scores['ocsvm_score'] = -ocsvm_score

    return scores


def predict_with_ensemble(scores_dict, ensemble_data):
    """使用集成模型预测"""
    # 构建特征向量
    score_cols = ensemble_data['score_columns']
    
    # 检查是否是加权平均模型
    if ensemble_data['model_type'] == 'weighted_average':
        # 加权平均预测
        weights = ensemble_data['weights']
        threshold = ensemble_data['threshold']
        
        # 计算加权分数
        weighted_score = 0.0
        for col in score_cols:
            if col in scores_dict and col in weights:
                weighted_score += scores_dict[col] * weights[col]
        
        # 基于阈值进行预测
        prediction = 1 if weighted_score > threshold else 0
        
        # 计算置信度（基于距离阈值的远近）
        distance = abs(weighted_score - threshold)
        max_distance = max(abs(threshold), abs(1.0 - threshold))
        confidence = min(distance / max_distance, 1.0)
        
        # 构建概率数组
        if prediction == 1:
            proba = np.array([1 - confidence, confidence])
        else:
            proba = np.array([confidence, 1 - confidence])
            
        return prediction, proba
    else:
        # 传统机器学习模型预测
        X = np.array([scores_dict[col] for col in score_cols]).reshape(1, -1)
        model = ensemble_data['model']
        prediction = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        return prediction, proba


def predict_audio_file(audio_path, base_model_data, ensemble_data, use_deep_features=True):
    """预测单个音频文件"""
    print(f"\n预测: {os.path.basename(audio_path)}")

    # 1. 提取特征
    features = extract_features_from_audio(audio_path, base_model_data, use_deep_features)
    print(f"  特征维度: {len(features)}")

    # 2. 计算分数
    scores = compute_all_scores(features, base_model_data)
    print(f"  分数:")
    for score_name, score_value in scores.items():
        print(f"    {score_name}: {score_value:.4f}")

    # 3. 集成预测
    prediction, proba = predict_with_ensemble(scores, ensemble_data)

    print(f"\n  集成模型判断: {'异常' if prediction == 1 else '正常'}")
    print(f"  置信度: 正常={proba[0]:.2%}, 异常={proba[1]:.2%}")

    return prediction, proba, scores


def extract_true_label_from_filename(filename):
    """从文件名中提取真实标签

    文件名格式: section_00_source_train_normal_0000_strength_1_ambient.wav
    或: section_00_source_train_anomaly_0000_strength_1_ambient.wav
    """
    filename_lower = filename.lower()

    if '_normal_' in filename_lower:
        return 0  # normal
    elif '_anomaly_' in filename_lower:
        return 1  # anomaly
    else:
        return None  # 无法识别


def batch_predict(audio_dir, base_model_data, ensemble_data, use_deep_features=True):
    """批量预测目录下的音频"""
    audio_files = glob(os.path.join(audio_dir, "*.wav"))

    if not audio_files:
        print(f"错误: 未找到音频文件 ({audio_dir})")
        return []

    print(f"\n批量预测: {len(audio_files)} 个文件")
    print("="*70)

    results = []
    errors = []

    for i, audio_path in enumerate(audio_files):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"\n进度: {i+1}/{len(audio_files)}")

        try:
            # 提取特征
            features = extract_features_from_audio(audio_path, base_model_data, use_deep_features)

            # 计算分数
            scores = compute_all_scores(features, base_model_data)

            # 集成预测
            prediction, proba = predict_with_ensemble(scores, ensemble_data)

            # 提取真实标签
            true_label = extract_true_label_from_filename(os.path.basename(audio_path))
            true_label_str = 'normal' if true_label == 0 else 'anomaly' if true_label == 1 else 'unknown'

            result = {
                'file': os.path.basename(audio_path),
                'true_label': int(true_label) if true_label is not None else -1,
                'true_label_str': true_label_str,
                'prediction': int(prediction),
                'prediction_label': 'anomaly' if prediction == 1 else 'normal',
                'correct': (prediction == true_label) if true_label is not None else None,
                'confidence_normal': float(proba[0]),
                'confidence_anomaly': float(proba[1]),
                **{k: float(v) for k, v in scores.items()},
                'status': 'success'
            }

            results.append(result)

            # 简洁输出，显示对错
            correct_marker = ""
            if true_label is not None:
                if prediction == true_label:
                    correct_marker = "✓"
                else:
                    correct_marker = "✗ ERROR"

            print(f"  [{i+1:3d}] 真实:{true_label_str:7s} | 预测:{result['prediction_label']:7s} {correct_marker:7s} (置信度: {proba[prediction]:.1%})")

        except Exception as e:
            error_msg = str(e)
            print(f"  [{i+1:3d}] ERROR    - {os.path.basename(audio_path)}: {error_msg[:50]}")

            errors.append({
                'file': os.path.basename(audio_path),
                'error': error_msg
            })

            # 仍然记录失败的结果
            results.append({
                'file': os.path.basename(audio_path),
                'true_label': -1,
                'true_label_str': 'unknown',
                'prediction': -1,
                'prediction_label': 'error',
                'correct': None,
                'confidence_normal': 0.0,
                'confidence_anomaly': 0.0,
                'status': f'error: {error_msg}'
            })

    # 统计
    print("\n" + "="*70)
    print("批量预测统计")
    print("="*70)

    successful = [r for r in results if r['status'] == 'success']

    print(f"\n总数: {len(results)}")
    print(f"  成功: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"  失败: {len(errors)} ({100*len(errors)/len(results):.1f}%)")

    if successful:
        n_normal = sum(1 for r in successful if r['prediction'] == 0)
        n_anomaly = sum(1 for r in successful if r['prediction'] == 1)

        print(f"\n预测结果分布:")
        print(f"  正常: {n_normal} ({100*n_normal/len(successful):.1f}%)")
        print(f"  异常: {n_anomaly} ({100*n_anomaly/len(successful):.1f}%)")

        # 平均置信度
        if n_normal > 0:
            avg_conf_normal = np.mean([r['confidence_normal'] for r in successful if r['prediction'] == 0])
            print(f"\n平均置信度 (正常样本): {avg_conf_normal:.2%}")

        if n_anomaly > 0:
            avg_conf_anomaly = np.mean([r['confidence_anomaly'] for r in successful if r['prediction'] == 1])
            print(f"平均置信度 (异常样本): {avg_conf_anomaly:.2%}")

        # ===== 新增：评估指标 =====
        # 筛选有真实标签的样本
        labeled_results = [r for r in successful if r['true_label'] != -1]

        if labeled_results:
            print("\n" + "="*70)
            print("模型评估指标 (与真实标签对比)")
            print("="*70)

            # 计算准确率
            correct_count = sum(1 for r in labeled_results if r['correct'])
            accuracy = correct_count / len(labeled_results)

            print(f"\n准确率 (Accuracy): {accuracy:.2%} ({correct_count}/{len(labeled_results)})")

            # 混淆矩阵
            y_true = [r['true_label'] for r in labeled_results]
            y_pred = [r['prediction'] for r in labeled_results]

            from sklearn.metrics import confusion_matrix, classification_report

            # 混淆矩阵
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            print(f"\n混淆矩阵:")
            print(f"                 预测:正常   预测:异常")
            print(f"  真实:正常      {tn:6d}      {fp:6d}")
            print(f"  真实:异常      {fn:6d}      {tp:6d}")

            # 计算各项指标
            if tp + fp > 0:
                precision = tp / (tp + fp)
                print(f"\n精确率 (Precision): {precision:.2%}")
            else:
                print(f"\n精确率 (Precision): N/A (无异常预测)")

            if tp + fn > 0:
                recall = tp / (tp + fn)
                print(f"召回率 (Recall):    {recall:.2%}")
            else:
                print(f"召回率 (Recall):    N/A (无真实异常)")

            if tp + fp > 0 and tp + fn > 0:
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                print(f"F1 分数:            {f1:.2%}")

            # 详细分类报告
            print(f"\n详细分类报告:")
            print(classification_report(y_true, y_pred, target_names=['正常', '异常'], digits=4))

            # 错误样本分析
            wrong_results = [r for r in labeled_results if not r['correct']]
            if wrong_results:
                print(f"\n错误分类样本 ({len(wrong_results)} 个):")
                print("-" * 70)
                for r in wrong_results[:10]:  # 只显示前10个
                    print(f"  {r['file'][:50]}")
                    print(f"    真实: {r['true_label_str']:7s} | 预测: {r['prediction_label']:7s} | 置信度: {r['confidence_anomaly' if r['prediction'] == 1 else 'confidence_normal']:.1%}")
                if len(wrong_results) > 10:
                    print(f"  ... 还有 {len(wrong_results) - 10} 个错误样本")

    if errors:
        print(f"\n失败的文件:")
        for err in errors[:5]:  # 只显示前5个
            print(f"  - {err['file']}: {err['error'][:80]}")
        if len(errors) > 5:
            print(f"  ... 还有 {len(errors)-5} 个错误")

    # 保存结果
    df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(audio_dir), 'prediction_results.csv')
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n完整结果已保存: {output_path}")

    return results


def parse_arguments():
    parser = argparse.ArgumentParser(description='使用集成模型预测')
    parser.add_argument('--base_model', type=str, required=True,
                       help='基础模型路径（gmm_with_scores.pkl）')
    parser.add_argument('--ensemble_model', type=str, required=True,
                       help='集成模型路径（ensemble_model.pkl）')
    parser.add_argument('--audio_file', type=str,
                       help='单个音频文件路径')
    parser.add_argument('--audio_dir', type=str,
                       help='音频目录（批量预测）')
    parser.add_argument('--use_deep_features', action='store_true', default=True,
                       help='使用深度特征（默认True）')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("="*70)
    print("集成模型预测器")
    print("="*70)

    # 加载模型
    base_model_data = load_base_models(args.base_model)
    ensemble_data = load_ensemble_model(args.ensemble_model)

    # 单文件预测
    if args.audio_file:
        predict_audio_file(args.audio_file, base_model_data, ensemble_data, args.use_deep_features)

    # 批量预测
    elif args.audio_dir:
        batch_predict(args.audio_dir, base_model_data, ensemble_data, args.use_deep_features)

    else:
        print("\n请指定 --audio_file 或 --audio_dir")


if __name__ == "__main__":
    main()
