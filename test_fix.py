import numpy as np
import matplotlib.pyplot as plt
from models.gmm_model import train_gmm_model, find_optimal_components
from features.extract_features import extract_all_features
import librosa
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_extract_all_features():
    """测试extract_all_features函数的参数和返回值"""
    print("测试extract_all_features函数...")
    
    # 创建一个简单的音频信号（1kHz正弦波）
    sr = 22050
    duration = 2.0  # 2秒
    t = np.linspace(0, duration, int(sr * duration))
    y = 0.5 * np.sin(2 * np.pi * 1000 * t)  # 1kHz正弦波
    
    # 测试默认参数
    try:
        combined_features, feature_dict = extract_all_features(y, sr)
        print(f"✓ extract_all_features调用成功")
        print(f"  - 组合特征形状: {combined_features.shape}")
        print(f"  - 提取的特征类型: {list(feature_dict.keys())}")
        return True
    except Exception as e:
        print(f"✗ extract_all_features调用失败: {e}")
        return False

def test_find_optimal_components():
    """测试find_optimal_components函数的参数和返回值"""
    print("\n测试find_optimal_components函数...")
    
    # 创建简单的测试数据
    X = np.random.rand(50, 10)  # 50个样本，每个样本10个特征
    y = np.array([0] * 25 + [1] * 25)  # 二分类问题
    
    try:
        best_n, best_score, scores = find_optimal_components(
            X, y, min_components=2, max_components=4, cv=3
        )
        print(f"✓ find_optimal_components调用成功")
        print(f"  - 最佳组件数量: {best_n}")
        print(f"  - 最佳得分: {best_score:.4f}")
        print(f"  - 各组件得分: {scores}")
        return True
    except Exception as e:
        print(f"✗ find_optimal_components调用失败: {e}")
        return False

def test_train_gmm_model():
    """测试train_gmm_model函数的参数和返回值"""
    print("\n测试train_gmm_model函数...")
    
    # 创建简单的测试数据
    X = np.random.rand(50, 10)  # 50个样本，每个样本10个特征
    y = np.array([0] * 25 + [1] * 25)  # 二分类问题
    
    try:
        model = train_gmm_model(
            X, y, n_components=3, covariance_type='diag', random_state=42
        )
        print(f"✓ train_gmm_model调用成功")
        print(f"  - 模型类别: {model.classes}")
        print(f"  - 模型组件数: {model.n_components}")
        
        # 测试预测功能
        y_pred = model.predict(X[:5])
        print(f"  - 预测结果示例: {y_pred}")
        return True
    except Exception as e:
        print(f"✗ train_gmm_model调用失败: {e}")
        return False

def main():
    """运行所有测试"""
    print("开始测试修复后的函数调用...\n")
    
    results = {
        'extract_all_features': test_extract_all_features(),
        'find_optimal_components': test_find_optimal_components(),
        'train_gmm_model': test_train_gmm_model()
    }
    
    print("\n=== 测试结果汇总 ===")
    all_passed = True
    for func_name, passed in results.items():
        status = "通过" if passed else "失败"
        print(f"{func_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ 所有测试通过！函数调用参数匹配正确。")
    else:
        print("\n❌ 部分测试失败，请检查代码修复。")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)