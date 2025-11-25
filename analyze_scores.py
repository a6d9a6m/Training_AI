#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分数分析工具 - 对导出的样本分数进行深入数学分析

功能：
1. 核密度估计（KDE）- 找到更精确的分界线
2. 混合高斯模型 - 对分数建模
3. 贝叶斯优化阈值
4. 分位数分析
5. 非线性决策边界（多项式、RBF）
6. 集成分数融合优化
"""

import numpy as np
import pandas as pd
import argparse
import os
import json
from scipy import stats
from scipy.optimize import minimize_scalar
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT = True
except:
    HAS_PLOT = False


def load_scores(csv_path):
    """加载分数CSV"""
    df = pd.read_csv(csv_path)
    print(f"加载了 {len(df)} 个样本的分数")
    print(f"  正常样本: {sum(df['label'] == 0)}")
    print(f"  异常样本: {sum(df['label'] == 1)}")
    print(f"  可用分数列: {[col for col in df.columns if 'score' in col]}")
    return df


def kde_based_threshold(df, score_col='gmm_score', bandwidth='scott'):
    """
    基于核密度估计（KDE）找到最优阈值

    原理：估计正常和异常样本的概率密度，找到交叉点作为阈值
    """
    print(f"\n{'='*70}")
    print(f"方法1: 核密度估计（KDE）")
    print(f"{'='*70}")

    normal_scores = df[df['label'] == 0][score_col].values
    anomaly_scores = df[df['label'] == 1][score_col].values

    # 拟合KDE
    kde_normal = stats.gaussian_kde(normal_scores, bw_method=bandwidth)
    kde_anomaly = stats.gaussian_kde(anomaly_scores, bw_method=bandwidth)

    # 在分数范围内搜索交叉点
    score_range = np.linspace(df[score_col].min(), df[score_col].max(), 1000)
    pdf_normal = kde_normal(score_range)
    pdf_anomaly = kde_anomaly(score_range)

    # 找到交叉点（正常概率 = 异常概率）
    diff = np.abs(pdf_normal - pdf_anomaly)
    crossover_idx = np.argmin(diff)
    threshold_kde = score_range[crossover_idx]

    # 评估
    predictions = (df[score_col] > threshold_kde).astype(int)
    f1 = f1_score(df['label'], predictions)
    precision = precision_score(df['label'], predictions, zero_division=0)
    recall = recall_score(df['label'], predictions)

    print(f"\nKDE阈值: {threshold_kde:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    if HAS_PLOT:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(score_range, pdf_normal, label='Normal PDF', color='blue', linewidth=2)
        ax.plot(score_range, pdf_anomaly, label='Anomaly PDF', color='red', linewidth=2)
        ax.axvline(threshold_kde, color='green', linestyle='--', linewidth=2, label=f'KDE Threshold={threshold_kde:.2f}')
        ax.fill_between(score_range, 0, pdf_normal, where=(score_range <= threshold_kde), alpha=0.3, color='blue')
        ax.fill_between(score_range, 0, pdf_anomaly, where=(score_range > threshold_kde), alpha=0.3, color='red')
        ax.set_xlabel('Score')
        ax.set_ylabel('Probability Density')
        ax.set_title('KDE-based Threshold Selection')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    return threshold_kde, f1, (score_range, pdf_normal, pdf_anomaly)


def gmm_on_scores(df, score_col='gmm_score', n_components=2):
    """
    在分数上应用GMM建模

    原理：将分数本身作为1维数据，用GMM拟合，找到两个高斯分量的分界
    """
    print(f"\n{'='*70}")
    print(f"方法2: 分数GMM建模")
    print(f"{'='*70}")

    scores = df[score_col].values.reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(scores)

    # 预测每个样本属于哪个高斯分量
    components = gmm.predict(scores)

    # 判断哪个分量对应异常（均值更高的）
    means = gmm.means_.flatten()
    anomaly_component = np.argmax(means)

    predictions = (components == anomaly_component).astype(int)

    f1 = f1_score(df['label'], predictions)
    precision = precision_score(df['label'], predictions, zero_division=0)
    recall = recall_score(df['label'], predictions)

    # 阈值可以定义为两个高斯分量均值的中点
    threshold_gmm = np.mean(means)

    print(f"\nGMM组件均值: {means}")
    print(f"GMM阈值（中点）: {threshold_gmm:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    return threshold_gmm, f1, gmm


def bayesian_threshold_optimization(df, score_col='gmm_score', prior_normal=0.5):
    """
    贝叶斯优化阈值

    原理：考虑先验概率，最小化贝叶斯风险
    假设误分类代价相同，寻找后验概率相等的点
    """
    print(f"\n{'='*70}")
    print(f"方法3: 贝叶斯阈值优化")
    print(f"{'='*70}")

    normal_scores = df[df['label'] == 0][score_col].values
    anomaly_scores = df[df['label'] == 1][score_col].values

    # 拟合分布
    kde_normal = stats.gaussian_kde(normal_scores)
    kde_anomaly = stats.gaussian_kde(anomaly_scores)

    prior_anomaly = 1 - prior_normal

    # 贝叶斯决策：找到 P(normal|x) = P(anomaly|x) 的点
    # P(class|x) ∝ P(x|class) * P(class)

    def posterior_diff(threshold):
        """后验概率差"""
        likelihood_normal = kde_normal(threshold)[0]
        likelihood_anomaly = kde_anomaly(threshold)[0]

        posterior_normal = likelihood_normal * prior_normal
        posterior_anomaly = likelihood_anomaly * prior_anomaly

        return abs(posterior_normal - posterior_anomaly)

    # 寻找使后验概率相等的阈值
    result = minimize_scalar(
        posterior_diff,
        bounds=(df[score_col].min(), df[score_col].max()),
        method='bounded'
    )

    threshold_bayes = result.x

    predictions = (df[score_col] > threshold_bayes).astype(int)
    f1 = f1_score(df['label'], predictions)
    precision = precision_score(df['label'], predictions, zero_division=0)
    recall = recall_score(df['label'], predictions)

    print(f"\n贝叶斯阈值: {threshold_bayes:.4f}")
    print(f"  先验: P(normal)={prior_normal}, P(anomaly)={prior_anomaly}")
    print(f"  F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

    return threshold_bayes, f1


def quantile_based_analysis(df, score_col='gmm_score'):
    """
    基于分位数的分析

    找到正常样本的高分位数作为阈值
    """
    print(f"\n{'='*70}")
    print(f"方法4: 分位数分析")
    print(f"{'='*70}")

    normal_scores = df[df['label'] == 0][score_col].values

    results = {}

    for percentile in [90, 95, 97.5, 99]:
        threshold = np.percentile(normal_scores, percentile)
        predictions = (df[score_col] > threshold).astype(int)

        f1 = f1_score(df['label'], predictions)
        precision = precision_score(df['label'], predictions, zero_division=0)
        recall = recall_score(df['label'], predictions)

        results[percentile] = {
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

        print(f"\n第{percentile}百分位: {threshold:.4f}")
        print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 找到最佳
    best_percentile = max(results, key=lambda k: results[k]['f1'])
    best_result = results[best_percentile]

    print(f"\n最佳分位数: 第{best_percentile}百分位")
    print(f"  阈值: {best_result['threshold']:.4f}")
    print(f"  F1: {best_result['f1']:.4f}")

    return best_result['threshold'], best_result['f1'], results


def ensemble_score_optimization(df):
    """
    如果有多个分数，优化它们的组合

    使用逻辑回归找到最优权重
    """
    print(f"\n{'='*70}")
    print(f"方法5: 集成分数优化")
    print(f"{'='*70}")

    score_cols = [col for col in df.columns if 'score' in col and col != 'gmm_log_likelihood']

    if len(score_cols) < 2:
        print("  只有一个分数列，跳过集成优化")
        return None, None

    print(f"  使用 {len(score_cols)} 个分数: {score_cols}")

    X = df[score_cols].values
    y = df['label'].values

    # 1. 逻辑回归（线性组合）
    print("\n  [a] 逻辑回归权重学习...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    predictions_lr = lr.predict(X)
    f1_lr = f1_score(y, predictions_lr)
    print(f"    F1: {f1_lr:.4f}")
    print(f"    权重: {dict(zip(score_cols, lr.coef_[0]))}")

    # 2. 随机森林（非线性组合）
    print("\n  [b] 随机森林学习...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
    rf.fit(X, y)

    predictions_rf = rf.predict(X)
    f1_rf = f1_score(y, predictions_rf)
    print(f"    F1: {f1_rf:.4f}")
    print(f"    特征重要性: {dict(zip(score_cols, rf.feature_importances_))}")

    # 3. SVM（非线性边界）
    print("\n  [c] SVM学习...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X, y)

    predictions_svm = svm.predict(X)
    f1_svm = f1_score(y, predictions_svm)
    print(f"    F1: {f1_svm:.4f}")

    # 选择最佳
    best_f1 = max(f1_lr, f1_rf, f1_svm)
    if best_f1 == f1_lr:
        best_model = lr
        best_name = "逻辑回归"
    elif best_f1 == f1_rf:
        best_model = rf
        best_name = "随机森林"
    else:
        best_model = svm
        best_name = "SVM"

    print(f"\n  最佳集成方法: {best_name}, F1={best_f1:.4f}")

    return best_model, best_f1


def plot_all_methods_comparison(df, thresholds_dict, output_dir):
    """比较所有方法的ROC曲线"""
    if not HAS_PLOT:
        return

    print(f"\n生成方法对比图...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 1. ROC曲线对比
    ax = axes[0]
    score_col = 'gmm_score'

    # 原始ROC
    fpr, tpr, _ = roc_curve(df['label'], df[score_col])
    auc = roc_auc_score(df['label'], df[score_col])
    ax.plot(fpr, tpr, label=f'ROC Curve (AUC={auc:.3f})', linewidth=2)

    # 标记各方法的工作点
    colors = ['red', 'green', 'blue', 'orange', 'purple']
    for i, (method, threshold) in enumerate(thresholds_dict.items()):
        preds = (df[score_col] > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(df['label'], preds).ravel()
        fpr_point = fp / (fp + tn)
        tpr_point = tp / (tp + fn)

        ax.scatter(fpr_point, tpr_point, s=200, color=colors[i % len(colors)],
                  marker='o', edgecolor='black', linewidth=2, zorder=5,
                  label=f'{method} (FPR={fpr_point:.3f}, TPR={tpr_point:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Method Comparison')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2. F1分数对比柱状图
    ax = axes[1]
    methods = list(thresholds_dict.keys())
    f1_scores = []

    for method, threshold in thresholds_dict.items():
        preds = (df[score_col] > threshold).astype(int)
        f1 = f1_score(df['label'], preds)
        f1_scores.append(f1)

    bars = ax.bar(range(len(methods)), f1_scores, color=colors[:len(methods)], edgecolor='black', linewidth=1.5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=15, ha='right')
    ax.set_ylabel('F1 Score')
    ax.set_title('F1 Score Comparison')
    ax.set_ylim([0, 1])
    ax.grid(True, axis='y', alpha=0.3)

    # 在柱状图上标注数值
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{f1:.3f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'methods_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  对比图已保存: {plot_path}")
    plt.close()


def save_analysis_results(results, output_path):
    """保存分析结果"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n分析结果已保存: {output_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='分数分析工具')
    parser.add_argument('--scores_csv', type=str, required=True,
                       help='分数CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='results/score_analysis',
                       help='输出目录')
    parser.add_argument('--score_col', type=str, default='gmm_score',
                       help='要分析的分数列名')
    parser.add_argument('--methods', type=str, nargs='+',
                       default=['kde', 'gmm', 'bayes', 'quantile', 'ensemble'],
                       help='要使用的方法')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("="*70)
    print("分数分析工具 - 数学方法优化")
    print("="*70)

    # 加载数据
    df = load_scores(args.scores_csv)

    os.makedirs(args.output_dir, exist_ok=True)

    results = {}
    thresholds_dict = {}

    # 方法1: KDE
    if 'kde' in args.methods:
        threshold_kde, f1_kde, kde_data = kde_based_threshold(df, args.score_col)
        results['kde'] = {'threshold': float(threshold_kde), 'f1': float(f1_kde)}
        thresholds_dict['KDE'] = threshold_kde

        if HAS_PLOT:
            plt.savefig(os.path.join(args.output_dir, 'kde_threshold.png'), dpi=150, bbox_inches='tight')
            plt.close()

    # 方法2: GMM on scores
    if 'gmm' in args.methods:
        threshold_gmm, f1_gmm, gmm_model = gmm_on_scores(df, args.score_col)
        results['gmm_on_scores'] = {'threshold': float(threshold_gmm), 'f1': float(f1_gmm)}
        thresholds_dict['GMM-Scores'] = threshold_gmm

    # 方法3: Bayesian
    if 'bayes' in args.methods:
        threshold_bayes, f1_bayes = bayesian_threshold_optimization(df, args.score_col)
        results['bayesian'] = {'threshold': float(threshold_bayes), 'f1': float(f1_bayes)}
        thresholds_dict['Bayesian'] = threshold_bayes

    # 方法4: Quantile
    if 'quantile' in args.methods:
        threshold_quantile, f1_quantile, quantile_results = quantile_based_analysis(df, args.score_col)
        results['quantile'] = {
            'best_threshold': float(threshold_quantile),
            'best_f1': float(f1_quantile),
            'all_results': {str(k): {kk: float(vv) for kk, vv in v.items()}
                           for k, v in quantile_results.items()}
        }
        thresholds_dict['Quantile'] = threshold_quantile

    # 方法5: Ensemble
    if 'ensemble' in args.methods:
        best_model, best_f1_ensemble = ensemble_score_optimization(df)
        if best_model is not None:
            results['ensemble'] = {'best_f1': float(best_f1_ensemble)}

    # 生成对比图
    if thresholds_dict:
        plot_all_methods_comparison(df, thresholds_dict, args.output_dir)

    # 保存结果
    results_path = os.path.join(args.output_dir, 'analysis_results.json')
    save_analysis_results(results, results_path)

    # 总结
    print("\n" + "="*70)
    print("分析总结")
    print("="*70)

    if thresholds_dict:
        # 找到最佳方法
        best_method = max(results, key=lambda k: results[k].get('f1', results[k].get('best_f1', 0)))
        best_f1 = results[best_method].get('f1', results[best_method].get('best_f1', 0))

        print(f"\n推荐方法: {best_method.upper()}")
        print(f"  F1分数: {best_f1:.4f}")

        if 'threshold' in results[best_method]:
            print(f"  推荐阈值: {results[best_method]['threshold']:.4f}")

    print("\n所有方法对比:")
    for method, data in results.items():
        if 'f1' in data:
            print(f"  {method}: F1={data['f1']:.4f}, 阈值={data.get('threshold', 'N/A')}")
        elif 'best_f1' in data:
            print(f"  {method}: F1={data['best_f1']:.4f}")

    print(f"\n结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
