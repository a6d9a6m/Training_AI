#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
训练诊断脚本 - 帮助诊断训练数据和模型问题
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.simple_data_loader import load_supervised_data, get_audio_and_labels
from features.extract_features import extract_all_features


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='训练数据诊断')
    parser.add_argument('--normal_dir', type=str, required=True,
                        help='正常样本目录')
    parser.add_argument('--anomaly_dir', type=str, required=True,
                        help='异常样本目录')
    parser.add_argument('--sr', type=int, default=22050,
                        help='采样率')
    parser.add_argument('--n_samples', type=int, default=10,
                        help='每类检查的样本数')
    return parser.parse_args()


def extract_features_from_data(data, sr, max_samples=None):
    """从数据中提取特征"""
    audios, labels = get_audio_and_labels(data)

    if max_samples:
        audios = audios[:max_samples]
        labels = labels[:max_samples]

    features_list = []
    for audio in audios:
        features, _ = extract_all_features(audio, sr)
        features_list.append(features)

    features_array = np.array(features_list)
    labels_array = np.array(labels)

    return features_array, labels_array


def diagnose_data(normal_dir, anomaly_dir, sr, n_samples):
    """诊断数据质量"""
    print("=" * 60)
    print("数据诊断")
    print("=" * 60)

    # 1. 加载数据
    print("\n[1/5] 加载数据...")
    train_data, val_data, test_data = load_supervised_data(
        normal_dir=normal_dir,
        anomaly_dir=anomaly_dir,
        sr=sr
    )

    # 2. 检查数据量
    print(f"\n[2/5] 数据量检查")
    print(f"  训练集: {len(train_data)} 个样本")
    print(f"  验证集: {len(val_data)} 个样本")
    print(f"  测试集: {len(test_data)} 个样本")

    train_audios, train_labels = get_audio_and_labels(train_data)
    print(f"\n  训练集分布:")
    print(f"    正常: {sum(1 for l in train_labels if l == 0)} 个")
    print(f"    异常: {sum(1 for l in train_labels if l == 1)} 个")

    # 判断数据量是否足够
    if len(train_data) < 50:
        print("  ⚠️ 警告: 训练集样本数少于50，建议增加样本数")

    # 3. 提取特征进行分析
    print(f"\n[3/5] 提取特征（每类{n_samples}个样本用于分析）...")

    # 分别提取正常和异常样本
    normal_samples = [d for d in train_data if d[1] == 0][:n_samples]
    anomaly_samples = [d for d in train_data if d[1] == 1][:n_samples]

    normal_features, _ = extract_features_from_data(normal_samples, sr)
    anomaly_features, _ = extract_features_from_data(anomaly_samples, sr)

    print(f"  特征维度: {normal_features.shape[1]}")

    # 4. 特征统计分析
    print(f"\n[4/5] 特征统计分析")

    # 计算特征均值和标准差
    normal_mean = np.mean(normal_features, axis=0)
    normal_std = np.std(normal_features, axis=0)
    anomaly_mean = np.mean(anomaly_features, axis=0)
    anomaly_std = np.std(anomaly_features, axis=0)

    # 计算特征差异
    feature_diff = np.abs(normal_mean - anomaly_mean)
    feature_diff_normalized = feature_diff / (normal_std + anomaly_std + 1e-10)

    print(f"  特征差异度 (归一化):")
    print(f"    平均: {np.mean(feature_diff_normalized):.4f}")
    print(f"    最大: {np.max(feature_diff_normalized):.4f}")
    print(f"    最小: {np.min(feature_diff_normalized):.4f}")

    if np.mean(feature_diff_normalized) < 0.1:
        print("  ❌ 问题: 正常和异常样本的特征差异非常小")
        print("     可能原因:")
        print("     1. 正常和异常样本在音频上没有明显差异")
        print("     2. 数据标签可能有误")
        print("     3. 需要更好的特征提取方法")
    elif np.mean(feature_diff_normalized) < 0.5:
        print("  ⚠️ 警告: 特征差异较小，可能影响分类效果")
    else:
        print("  ✓ 特征差异度良好")

    # 检查特征中的NaN或Inf
    if np.any(np.isnan(normal_features)) or np.any(np.isnan(anomaly_features)):
        print("  ❌ 问题: 特征中存在NaN值")
    if np.any(np.isinf(normal_features)) or np.any(np.isinf(anomaly_features)):
        print("  ❌ 问题: 特征中存在Inf值")

    # 5. 可视化特征分布
    print(f"\n[5/5] 生成特征可视化...")

    # 合并特征
    all_features = np.vstack([normal_features, anomaly_features])
    all_labels = np.hstack([np.zeros(len(normal_features)), np.ones(len(anomaly_features))])

    # PCA降维
    try:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(all_features)

        plt.figure(figsize=(12, 5))

        # PCA可视化
        plt.subplot(1, 2, 1)
        plt.scatter(features_2d[all_labels==0, 0], features_2d[all_labels==0, 1],
                   c='blue', label='正常', alpha=0.6, s=50)
        plt.scatter(features_2d[all_labels==1, 0], features_2d[all_labels==1, 1],
                   c='red', label='异常', alpha=0.6, s=50)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA特征分布')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 特征差异热图
        plt.subplot(1, 2, 2)
        top_features = np.argsort(feature_diff_normalized)[-20:]  # 选择差异最大的20个特征

        data_to_plot = []
        data_to_plot.append(normal_mean[top_features])
        data_to_plot.append(anomaly_mean[top_features])

        plt.imshow(data_to_plot, aspect='auto', cmap='RdYlBu_r')
        plt.colorbar(label='特征值')
        plt.yticks([0, 1], ['正常', '异常'])
        plt.xlabel('特征维度 (差异最大的20个)')
        plt.title('特征值对比')

        plt.tight_layout()
        plt.savefig('data_diagnosis.png', dpi=150, bbox_inches='tight')
        print(f"  可视化已保存: data_diagnosis.png")

        # 检查PCA后的分离度
        normal_2d = features_2d[all_labels==0]
        anomaly_2d = features_2d[all_labels==1]

        normal_center = np.mean(normal_2d, axis=0)
        anomaly_center = np.mean(anomaly_2d, axis=0)
        center_distance = np.linalg.norm(normal_center - anomaly_center)

        normal_spread = np.mean(np.linalg.norm(normal_2d - normal_center, axis=1))
        anomaly_spread = np.mean(np.linalg.norm(anomaly_2d - anomaly_center, axis=1))
        avg_spread = (normal_spread + anomaly_spread) / 2

        separation_ratio = center_distance / (avg_spread + 1e-10)

        print(f"\n  PCA空间分离度分析:")
        print(f"    类间距离: {center_distance:.4f}")
        print(f"    类内平均散度: {avg_spread:.4f}")
        print(f"    分离比率: {separation_ratio:.4f}")

        if separation_ratio < 0.5:
            print("    ❌ 问题: 两类样本在PCA空间高度重叠")
            print("       建议: 检查数据标签是否正确，或样本是否真的有差异")
        elif separation_ratio < 1.0:
            print("    ⚠️ 警告: 两类样本有一定重叠，可能影响分类效果")
        else:
            print("    ✓ 两类样本分离度良好")

    except Exception as e:
        print(f"  可视化失败: {e}")

    # 6. 总结和建议
    print("\n" + "=" * 60)
    print("诊断总结")
    print("=" * 60)

    issues = []
    suggestions = []

    if len(train_data) < 50:
        issues.append("训练数据量不足")
        suggestions.append("增加训练样本至每类至少50个")

    if np.mean(feature_diff_normalized) < 0.1:
        issues.append("正常和异常样本特征差异极小")
        suggestions.append("检查数据标签是否正确")
        suggestions.append("确认正常和异常样本在听感上是否有明显区别")
        suggestions.append("考虑使用其他特征提取方法")

    if 'separation_ratio' in locals() and separation_ratio < 0.5:
        issues.append("样本在特征空间高度重叠")
        suggestions.append("尝试使用深度学习模型（自动编码器）")
        suggestions.append("增加更多有区分度的样本")

    if len(issues) == 0:
        print("✓ 未发现明显问题，数据质量良好")
        print("\n如果训练效果仍然不好，建议:")
        print("  1. 增加训练样本数量")
        print("  2. 调整GMM组件数量 (--n_components)")
        print("  3. 尝试异常检测模式 (train_anomaly_detection.py)")
    else:
        print("发现以下问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print("\n改进建议:")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"  {i}. {suggestion}")


def main():
    """主函数"""
    args = parse_arguments()

    diagnose_data(
        normal_dir=args.normal_dir,
        anomaly_dir=args.anomaly_dir,
        sr=args.sr,
        n_samples=args.n_samples
    )


if __name__ == "__main__":
    main()
