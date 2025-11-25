#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从已导出的分数训练集成模型

读取sample_scores.csv，训练随机森林/逻辑回归/SVM来融合多个分数
保存最佳集成模型供后续使用
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, classification_report


def load_scores(csv_path):
    """加载分数CSV"""
    df = pd.read_csv(csv_path)
    print(f"加载了 {len(df)} 个样本")
    print(f"  正常: {sum(df['label']==0)}, 异常: {sum(df['label']==1)}")

    score_cols = [col for col in df.columns if 'score' in col and col != 'gmm_log_likelihood']
    print(f"  可用分数: {score_cols}")

    return df, score_cols


def train_ensemble_models(df, score_cols):
    """训练多种集成模型"""
    X = df[score_cols].values
    y = df['label'].values

    print("\n" + "="*70)
    print("训练集成模型")
    print("="*70)

    models = {}
    results = {}

    # 1. 随机森林
    print("\n[1] 随机森林...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)

    preds_rf = rf.predict(X)
    proba_rf = rf.predict_proba(X)[:, 1]

    results['random_forest'] = {
        'f1': f1_score(y, preds_rf),
        'precision': precision_score(y, preds_rf),
        'recall': recall_score(y, preds_rf),
        'accuracy': accuracy_score(y, preds_rf),
        'auc': roc_auc_score(y, proba_rf)
    }

    print(f"  F1: {results['random_forest']['f1']:.4f}")
    print(f"  Precision: {results['random_forest']['precision']:.4f}")
    print(f"  Recall: {results['random_forest']['recall']:.4f}")
    print(f"  AUC: {results['random_forest']['auc']:.4f}")
    print(f"  特征重要性: {dict(zip(score_cols, rf.feature_importances_))}")

    models['random_forest'] = rf

    # 2. 逻辑回归
    print("\n[2] 逻辑回归...")
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X, y)

    preds_lr = lr.predict(X)
    proba_lr = lr.predict_proba(X)[:, 1]

    results['logistic_regression'] = {
        'f1': f1_score(y, preds_lr),
        'precision': precision_score(y, preds_lr),
        'recall': recall_score(y, preds_lr),
        'accuracy': accuracy_score(y, preds_lr),
        'auc': roc_auc_score(y, proba_lr)
    }

    print(f"  F1: {results['logistic_regression']['f1']:.4f}")
    print(f"  Precision: {results['logistic_regression']['precision']:.4f}")
    print(f"  Recall: {results['logistic_regression']['recall']:.4f}")
    print(f"  权重: {dict(zip(score_cols, lr.coef_[0]))}")

    models['logistic_regression'] = lr

    # 3. SVM
    print("\n[3] SVM (RBF核)...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X, y)

    preds_svm = svm.predict(X)
    proba_svm = svm.predict_proba(X)[:, 1]

    results['svm'] = {
        'f1': f1_score(y, preds_svm),
        'precision': precision_score(y, preds_svm),
        'recall': recall_score(y, preds_svm),
        'accuracy': accuracy_score(y, preds_svm),
        'auc': roc_auc_score(y, proba_svm)
    }

    print(f"  F1: {results['svm']['f1']:.4f}")
    print(f"  Precision: {results['svm']['precision']:.4f}")
    print(f"  Recall: {results['svm']['recall']:.4f}")

    models['svm'] = svm

    # 4. 简单平均（作为基线）
    print("\n[4] 简单平均（基线）...")
    if 'ensemble_avg' in df.columns:
        avg_scores = df['ensemble_avg'].values
    else:
        avg_scores = X.mean(axis=1)

    # 找最优阈值
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y, avg_scores)
    f1_scores = []
    for thresh in thresholds:
        preds = (avg_scores > thresh).astype(int)
        f1_scores.append(f1_score(y, preds))

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    preds_avg = (avg_scores > best_thresh).astype(int)

    results['simple_average'] = {
        'f1': f1_score(y, preds_avg),
        'precision': precision_score(y, preds_avg),
        'recall': recall_score(y, preds_avg),
        'accuracy': accuracy_score(y, preds_avg),
        'auc': roc_auc_score(y, avg_scores),
        'threshold': float(best_thresh)
    }

    print(f"  最优阈值: {best_thresh:.4f}")
    print(f"  F1: {results['simple_average']['f1']:.4f}")
    print(f"  Precision: {results['simple_average']['precision']:.4f}")
    print(f"  Recall: {results['simple_average']['recall']:.4f}")

    return models, results


def print_detailed_report(model, X, y, model_name):
    """打印详细报告"""
    print("\n" + "="*70)
    print(f"{model_name} - 详细评估")
    print("="*70)

    preds = model.predict(X)

    print("\n分类报告:")
    print(classification_report(y, preds, target_names=['Normal', 'Anomaly']))

    print("\n混淆矩阵:")
    cm = confusion_matrix(y, preds)
    print(f"            预测正常  预测异常")
    print(f"  实际正常     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"  实际异常     {cm[1,0]:3d}      {cm[1,1]:3d}")

    # 误分类分析
    wrong_indices = np.where(preds != y)[0]
    print(f"\n误分类样本: {len(wrong_indices)} / {len(y)} ({100*len(wrong_indices)/len(y):.1f}%)")

    false_positives = np.where((preds == 1) & (y == 0))[0]
    false_negatives = np.where((preds == 0) & (y == 1))[0]

    print(f"  假阳性（正常误判为异常）: {len(false_positives)}")
    print(f"  假阴性（异常误判为正常）: {len(false_negatives)}")


def save_best_model(models, results, score_cols, output_dir):
    """保存最佳模型"""
    # 找到最佳模型
    best_model_name = max(results, key=lambda k: results[k]['f1'])
    best_model = models.get(best_model_name)
    best_result = results[best_model_name]

    print("\n" + "="*70)
    print("保存最佳模型")
    print("="*70)
    print(f"\n最佳模型: {best_model_name}")
    print(f"  F1: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  AUC: {best_result['auc']:.4f}")

    # 保存模型
    os.makedirs(output_dir, exist_ok=True)

    model_data = {
        'model': best_model,
        'model_type': best_model_name,
        'score_columns': score_cols,
        'metrics': best_result
    }

    model_path = os.path.join(output_dir, 'ensemble_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    print(f"\n模型已保存: {model_path}")

    # 保存结果JSON
    results_path = os.path.join(output_dir, 'ensemble_results.json')
    results_json = {k: {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                        for kk, vv in v.items()}
                   for k, v in results.items()}

    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    print(f"结果已保存: {results_path}")

    return best_model_name, best_model


def parse_arguments():
    parser = argparse.ArgumentParser(description='从分数训练集成模型')
    parser.add_argument('--scores_csv', type=str, required=True,
                       help='分数CSV文件路径')
    parser.add_argument('--output_dir', type=str, default='models/saved_models',
                       help='输出目录')

    return parser.parse_args()


def main():
    args = parse_arguments()

    print("="*70)
    print("集成模型训练器")
    print("="*70)

    # 加载数据
    df, score_cols = load_scores(args.scores_csv)

    if len(score_cols) < 2:
        print("\n[警告] 只有一个分数列，无法进行集成学习")
        return

    # 训练模型
    models, results = train_ensemble_models(df, score_cols)

    # 保存最佳模型
    best_model_name, best_model = save_best_model(models, results, score_cols, args.output_dir)

    # 详细报告
    X = df[score_cols].values
    y = df['label'].values
    print_detailed_report(best_model, X, y, best_model_name.upper())

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    print(f"\n使用方法:")
    print(f"  1. 模型已保存到: {os.path.join(args.output_dir, 'ensemble_model.pkl')}")
    print(f"  2. 需要的输入分数: {score_cols}")
    print(f"  3. 预测时提供这些分数，模型会输出异常/正常判断")

    print(f"\n所有模型对比:")
    for model_name, result in sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True):
        print(f"  {model_name:20s}: F1={result['f1']:.4f}, Precision={result['precision']:.4f}, Recall={result['recall']:.4f}")


if __name__ == "__main__":
    main()
