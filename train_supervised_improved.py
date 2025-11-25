#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
改进的监督学习训练脚本 - 提升性能版本

改进点：
1. 更好的特征归一化
2. 自动寻找最佳GMM组件数
3. 特征选择
4. 多种评估指标
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, f1_score


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='改进的监督学习训练')
    parser.add_argument('--normal_dir', type=str, required=True,
                        help='正常样本目录')
    parser.add_argument('--anomaly_dir', type=str, required=True,
                        help='异常样本目录')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--n_components', type=int, default=None,
                        help='GMM组件数量（None表示自动搜索）')
    parser.add_argument('--feature_selection', type=float, default=0.8,
                        help='特征选择比例（0-1）')
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


def find_best_n_components(train_features, train_labels, min_comp=2, max_comp=15):
    """寻找最佳GMM组件数"""
    print(f"\n正在寻找最佳GMM组件数 (范围: {min_comp}-{max_comp})...")

    best_score = -np.inf
    best_n = min_comp

    for n in range(min_comp, max_comp + 1):
        try:
            model = GMMModel(n_components=n, covariance_type='diag', random_state=42)
            model.fit(train_features, train_labels)

            # 使用训练集的对数似然作为评分
            score = 0
            for label in [0, 1]:
                class_samples = train_features[train_labels == label]
                if len(class_samples) > 0:
                    likelihood = model.models[label].score(class_samples)
                    score += likelihood

            print(f"  n_components={n}: 分数={score:.2f}")

            if score > best_score:
                best_score = score
                best_n = n

        except Exception as e:
            print(f"  n_components={n}: 失败 ({e})")
            continue

    print(f"\n最佳组件数: {best_n} (分数: {best_score:.2f})")
    return best_n


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 60)
    print("改进的监督学习训练 - GMM分类模型")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_data, val_data, test_data = load_supervised_data(
        normal_dir=args.normal_dir,
        anomaly_dir=args.anomaly_dir,
        sr=args.sr,
        test_size=args.test_size,
        val_size=args.val_size
    )

    # 2. 提取特征
    print("\n[2/6] 提取特征...")
    start_time = time.time()

    train_features, train_labels = extract_features_from_data(train_data, args.sr, "训练集")
    val_features, val_labels = extract_features_from_data(val_data, args.sr, "验证集")
    test_features, test_labels = extract_features_from_data(test_data, args.sr, "测试集")

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 特征预处理
    print("\n[3/6] 特征预处理...")

    # 3.1 使用RobustScaler（对异常值更稳健）
    print("  使用RobustScaler进行特征归一化...")
    scaler = RobustScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)
    test_features_scaled = scaler.transform(test_features)

    # 3.2 特征选择（移除低区分度特征）
    if args.feature_selection < 1.0:
        n_features_to_select = int(train_features_scaled.shape[1] * args.feature_selection)
        print(f"  选择最佳特征: {n_features_to_select}/{train_features_scaled.shape[1]}")

        selector = SelectKBest(f_classif, k=n_features_to_select)
        train_features_selected = selector.fit_transform(train_features_scaled, train_labels)
        val_features_selected = selector.transform(val_features_scaled)
        test_features_selected = selector.transform(test_features_scaled)

        print(f"  特征选择后维度: {train_features_selected.shape[1]}")
    else:
        print("  跳过特征选择")
        train_features_selected = train_features_scaled
        val_features_selected = val_features_scaled
        test_features_selected = test_features_scaled
        selector = None

    # 4. 确定最佳GMM组件数
    print("\n[4/6] 训练GMM模型...")

    if args.n_components is None:
        n_components = find_best_n_components(train_features_selected, train_labels,
                                               min_comp=2, max_comp=15)
    else:
        n_components = args.n_components
        print(f"使用指定的组件数量: {n_components}")

    # 训练最终模型
    print(f"\n训练最终模型 (n_components={n_components})...")
    model = GMMModel(n_components=n_components, covariance_type='diag', random_state=42)
    model.fit(train_features_selected, train_labels)

    print("模型训练完成")

    # 5. 确定最佳阈值
    print("\n[5/6] 在验证集上确定最佳阈值...")
    optimal_threshold = find_optimal_threshold(
        model, val_features_selected, val_labels,
        method='f1'
    )
    print(f"最佳阈值: {optimal_threshold:.4f}")

    # 6. 在测试集上评估
    print("\n[6/6] 在测试集上评估模型...")
    evaluator, metrics = evaluate_model(
        model, test_features_selected, test_labels,
        threshold=optimal_threshold,
        target_names=['正常', '异常']
    )

    print("\n评估结果:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  精确率: {metrics['precision']:.4f}")
    print(f"  召回率: {metrics['recall']:.4f}")
    print(f"  F1分数: {metrics['f1']:.4f}")
    if 'roc_auc' in metrics and metrics['roc_auc'] is not None:
        print(f"  AUC: {metrics['roc_auc']:.4f}")
    print(f"混淆矩阵:")
    print(metrics['confusion_matrix'])

    # 7. 保存模型
    print("\n保存模型和预处理器...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(args.output_dir, 'supervised_gmm_model_improved.pkl')
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存预处理器
    import pickle
    preprocessor_path = os.path.join(args.output_dir, 'preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'selector': selector
        }, f)
    print(f"预处理器已保存: {preprocessor_path}")

    # 保存训练信息
    info = {
        'model_type': 'supervised_gmm_improved',
        'n_components': n_components,
        'threshold': optimal_threshold,
        'sample_rate': args.sr,
        'feature_selection_ratio': args.feature_selection,
        'n_features_selected': train_features_selected.shape[1],
        'n_features_original': train_features.shape[1],
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['roc_auc']) if 'roc_auc' in metrics and metrics['roc_auc'] is not None else None
        },
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data)
    }

    info_path = os.path.join(args.output_dir, 'supervised_gmm_model_improved_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)

    # 性能评估
    if metrics['f1'] >= 0.8:
        print("\n性能评估: 优秀 (F1 >= 0.8)")
    elif metrics['f1'] >= 0.7:
        print("\n性能评估: 良好 (F1 >= 0.7)")
    elif metrics['f1'] >= 0.6:
        print("\n性能评估: 一般 (F1 >= 0.6)")
    else:
        print("\n性能评估: 需要改进 (F1 < 0.6)")
        print("\n改进建议:")
        print("  1. 检查数据标签是否正确")
        print("  2. 确认正常和异常样本在听感上是否有明显区别")
        print("  3. 增加更多训练样本")
        print("  4. 尝试异常检测模式: python train_anomaly_detection.py")


if __name__ == "__main__":
    main()
