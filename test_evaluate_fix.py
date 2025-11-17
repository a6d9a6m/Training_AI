#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试评估函数修复
"""

import os
import sys
import numpy as np
from utils.config_manager import load_config, create_default_config
from utils.evaluator import ModelEvaluator, evaluate_model

# 模拟测试数据
X_test = np.random.rand(100, 10)  # 100个样本，10个特征
y_test = np.random.randint(0, 2, 100)  # 二分类标签

# 模拟模型类
class MockModel:
    def predict(self, X):
        return np.random.randint(0, 2, X.shape[0])
    
    def predict_proba(self, X):
        probas = np.random.rand(X.shape[0], 2)
        return probas / probas.sum(axis=1, keepdims=True)

# 测试评估函数
def test_evaluate_model():
    print("测试 evaluate_model 函数...")
    model = MockModel()
    
    # 调用evaluate_model函数
    try:
        evaluator, metrics = evaluate_model(
            model, X_test, y_test,
            threshold=0.5,
            target_names=['normal', 'anomaly'],
            output_dir=None
        )
        print("✓ evaluate_model 函数调用成功")
        print("返回的metrics字典键:", metrics.keys())
        return True
    except Exception as e:
        print(f"✗ evaluate_model 函数调用失败: {e}")
        return False

if __name__ == "__main__":
    print("开始测试评估函数修复...")
    success = test_evaluate_model()
    
    if success:
        print("\n测试通过！评估函数修复成功。")
        sys.exit(0)
    else:
        print("\n测试失败，请检查修复。")
        sys.exit(1)