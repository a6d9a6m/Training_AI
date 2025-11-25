#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工业设备异常检测训练 - 专门针对工业场景优化
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
from models.autoencoder import AutoencoderModel
from sklearn.preprocessing import StandardScaler
import librosa


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='工业设备异常检测训练')
    parser.add_argument('--normal_train_dir', type=str, required=True,
                        help='正常样本目录（用于训练）')
    parser.add_argument('--mixed_test_dir', type=str, default=None,
                        help='混合测试集目录（包含正常和异常样本）')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                        help='模型保存目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--denoise', action='store_true', default=True,
                        help='是否进行背景噪声抑制')
    parser.add_argument('--no_denoise', action='store_false', dest='denoise',
                        help='不进行背景噪声抑制')
    parser.add_argument('--encoding_dim', type=int, default=32,
                        help='自动编码器编码维度')
    parser.add_argument('--epochs', type=int, default=150,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--threshold_percentile', type=float, default=95,
                        help='异常阈值百分位数')
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


def main():
    """主函数"""
    args = parse_arguments()

    print("=" * 70)
    print("工业设备异常检测训练")
    print("=" * 70)
    print(f"训练模式: 异常检测（只使用正常样本）")
    print(f"背景噪声抑制: {'启用' if args.denoise else '禁用'}")
    print("=" * 70)

    # 1. 加载数据
    print("\n[1/6] 加载数据...")
    train_data, val_data, test_data = load_anomaly_detection_data(
        normal_train_dir=args.normal_train_dir,
        mixed_test_dir=args.mixed_test_dir,
        sr=args.sr
    )

    print(f"  训练集: {len(train_data)} 个正常样本")
    print(f"  验证集: {len(val_data)} 个正常样本")

    if test_data:
        print(f"  测试集: {len(test_data)} 个样本")

    # 2. 提取特征
    print("\n[2/6] 提取工业特征...")
    start_time = time.time()

    train_features = extract_features_from_data(
        train_data, args.sr, args.denoise, "训练集"
    )
    val_features = extract_features_from_data(
        val_data, args.sr, args.denoise, "验证集"
    )

    test_features = None
    test_labels = None
    if test_data:
        test_labels = np.array([label for _, label in test_data])
        test_features = extract_features_from_data(
            test_data, args.sr, args.denoise, "测试集"
        )

    print(f"特征提取耗时: {time.time() - start_time:.2f} 秒")

    # 3. 特征标准化
    print("\n[3/6] 特征标准化...")
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    val_features_scaled = scaler.transform(val_features)

    if test_features is not None:
        test_features_scaled = scaler.transform(test_features)
    else:
        test_features_scaled = None

    # 4. 训练自动编码器
    print("\n[4/6] 训练自动编码器...")
    print(f"  输入维度: {train_features_scaled.shape[1]}")
    print(f"  编码维度: {args.encoding_dim}")
    print(f"  训练轮数: {args.epochs}")

    model = AutoencoderModel(
        input_dim=train_features_scaled.shape[1],
        encoding_dim=args.encoding_dim
    )

    model.fit(
        train_features_scaled,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=val_features_scaled
    )

    print("模型训练完成")

    # 5. 确定异常阈值
    print("\n[5/6] 确定异常阈值...")

    # 在验证集上计算重构误差
    val_reconstructed = model.predict(val_features_scaled)
    val_reconstruction_errors = np.mean(np.square(val_features_scaled - val_reconstructed), axis=1)

    # 使用百分位数作为阈值
    threshold = np.percentile(val_reconstruction_errors, args.threshold_percentile)
    print(f"  阈值（{args.threshold_percentile}th百分位数）: {threshold:.6f}")

    val_mean_error = np.mean(val_reconstruction_errors)
    val_std_error = np.std(val_reconstruction_errors)
    print(f"  验证集重构误差 - 均值: {val_mean_error:.6f}, 标准差: {val_std_error:.6f}")

    # 6. 评估模型
    if test_features_scaled is not None and test_labels is not None:
        print("\n[6/6] 在测试集上评估...")

        # 计算测试集的重构误差
        test_reconstructed = model.predict(test_features_scaled)
        test_reconstruction_errors = np.mean(
            np.square(test_features_scaled - test_reconstructed), axis=1
        )

        # 预测：重构误差大于阈值则为异常
        predictions = (test_reconstruction_errors > threshold).astype(int)

        # 计算评估指标
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import confusion_matrix, roc_auc_score

        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions, zero_division=0)
        recall = recall_score(test_labels, predictions, zero_division=0)
        f1 = f1_score(test_labels, predictions, zero_division=0)
        cm = confusion_matrix(test_labels, predictions)

        # AUC（使用重构误差作为连续得分）
        try:
            auc = roc_auc_score(test_labels, test_reconstruction_errors)
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

        # 分析重构误差分布
        normal_errors = test_reconstruction_errors[test_labels == 0]
        anomaly_errors = test_reconstruction_errors[test_labels == 1]

        print(f"\n重构误差分析:")
        if len(normal_errors) > 0:
            print(f"  正常样本 - 均值: {np.mean(normal_errors):.6f}, 标准差: {np.std(normal_errors):.6f}")
        if len(anomaly_errors) > 0:
            print(f"  异常样本 - 均值: {np.mean(anomaly_errors):.6f}, 标准差: {np.std(anomaly_errors):.6f}")

        metrics_dict = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auc': float(auc) if auc is not None else None
        }
    else:
        print("\n[6/6] 跳过测试集评估（未提供测试集）")
        metrics_dict = None

    # 7. 保存模型
    print("\n保存模型和预处理器...")
    os.makedirs(args.output_dir, exist_ok=True)

    # 保存模型
    model_path = os.path.join(args.output_dir, 'industrial_anomaly_model.pth')
    model.save(model_path)
    print(f"模型已保存: {model_path}")

    # 保存预处理器
    import pickle
    preprocessor_path = os.path.join(args.output_dir, 'industrial_preprocessor.pkl')
    with open(preprocessor_path, 'wb') as f:
        pickle.dump({
            'scaler': scaler,
            'denoise': args.denoise
        }, f)
    print(f"预处理器已保存: {preprocessor_path}")

    # 保存训练信息
    info = {
        'model_type': 'industrial_anomaly_detection',
        'feature_type': 'industrial_features',
        'denoise': args.denoise,
        'encoding_dim': args.encoding_dim,
        'threshold': float(threshold),
        'threshold_percentile': args.threshold_percentile,
        'sample_rate': args.sr,
        'n_features': train_features_scaled.shape[1],
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'test_samples': len(test_data) if test_data else 0,
        'epochs': args.epochs,
        'metrics': metrics_dict
    }

    info_path = os.path.join(args.output_dir, 'industrial_anomaly_model_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"训练信息已保存: {info_path}")

    print("\n" + "=" * 70)
    print("训练完成！")
    print("=" * 70)

    # 性能评估
    if metrics_dict:
        if metrics_dict['f1'] >= 0.7:
            print("\n✓ 性能评估: 良好 (F1 >= 0.7)")
            print("模型已学会区分正常和异常模式")
        elif metrics_dict['f1'] >= 0.5:
            print("\n⚠ 性能评估: 一般 (0.5 <= F1 < 0.7)")
            print("建议:")
            print("  1. 增加训练轮数 (--epochs 200)")
            print("  2. 调整编码维度 (--encoding_dim 64)")
            print("  3. 确保训练集只包含正常样本")
        else:
            print("\n❌ 性能评估: 需要改进 (F1 < 0.5)")
            print("可能的原因:")
            print("  1. 正常和异常样本差异过小")
            print("  2. 训练集中可能混入了异常样本")
            print("  3. 需要人工检查样本标签")


if __name__ == "__main__":
    main()
