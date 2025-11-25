#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试数据加载功能
用于验证修改后的代码是否能正确加载fan文件夹的数据
"""

import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_manager import load_config
from utils.data_loader import load_dataset


def test_fan_data_loading():
    """
    测试加载fan文件夹的数据
    """
    print("=== 测试fan文件夹数据加载 ===")
    
    # 加载配置
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    
    # 确保使用设备类型加载
    device_type = config.get('data.device_type', 'fan')
    base_data_dir = config.get('paths.base_data_dir', '../dev_data')
    sample_rate = config.get('data.sample_rate', 22050)
    
    print(f"设备类型: {device_type}")
    print(f"基础数据目录: {base_data_dir}")
    print(f"采样率: {sample_rate}")
    
    try:
        # 直接调用load_dataset函数测试
        train_data, val_data, test_data = load_dataset(
            device_type=device_type,
            base_data_dir=base_data_dir,
            sr=sample_rate,
            test_size=0.2,
            val_size=0.2,
            random_state=42
        )
        
        # 打印数据统计信息
        print("\n数据加载成功！")
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        # 统计正常和异常样本
        if test_data:
            normal_count = sum(1 for _, label in test_data if label == 0)
            anomaly_count = sum(1 for _, label in test_data if label == 1)
            print(f"\n测试集中正常样本: {normal_count}")
            print(f"测试集中异常样本: {anomaly_count}")
        
        # 测试通过标志
        print("\n✅ 测试通过：fan文件夹数据加载正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        return False


def test_prepare_data_integration():
    """
    测试集成到main.py中的prepare_data函数
    """
    print("\n=== 测试prepare_data函数集成 ===")
    
    try:
        # 导入prepare_data函数
        from main import prepare_data
        
        # 加载配置
        config_path = 'config/config.yaml'
        config = load_config(config_path)
        
        # 确保配置使用设备类型
        config.config['data']['use_device_type'] = True
        config.config['data']['device_type'] = 'fan'
        
        # 调用prepare_data函数
        train_data, val_data, test_data = prepare_data(config)
        
        # 验证结果
        print("\nprepare_data函数调用成功！")
        print(f"训练集大小: {len(train_data)}")
        print(f"验证集大小: {len(val_data)}")
        print(f"测试集大小: {len(test_data)}")
        
        print("\n✅ 测试通过：prepare_data函数集成正常")
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败：{str(e)}")
        return False


def test_multi_device_preparation():
    """
    测试为后续添加的六个设备文件夹做准备
    """
    print("\n=== 测试多设备准备 ===")
    
    # 加载配置
    config_path = 'config/config.yaml'
    config = load_config(config_path)
    
    # 获取支持的设备列表
    supported_devices = config.get('data.supported_devices', [])
    
    print(f"配置中支持的设备列表: {supported_devices}")
    print(f"当前使用的设备类型: {config.get('data.device_type', '未设置')}")
    
    # 验证是否包含七个设备类型（fan + 六个其他设备）
    if len(supported_devices) >= 7:
        print("\n✅ 测试通过：配置已准备好支持多个设备类型")
        print("后续添加新设备时，只需将设备文件夹放在dev_data目录下，")
        print("并将config.yaml中的device_type设置为对应设备名称即可")
        return True
    else:
        print(f"\n⚠️  警告：配置中的设备列表数量不足")
        return False


if __name__ == "__main__":
    print("开始测试数据加载功能...")
    print("=" * 50)
    
    # 运行所有测试
    test1_passed = test_fan_data_loading()
    test2_passed = test_prepare_data_integration()
    test3_passed = test_multi_device_preparation()
    
    print("\n" + "=" * 50)
    print("测试总结:")
    print(f"1. fan文件夹数据加载测试: {'通过' if test1_passed else '失败'}")
    print(f"2. prepare_data函数集成测试: {'通过' if test2_passed else '失败'}")
    print(f"3. 多设备准备测试: {'通过' if test3_passed else '失败'}")
    
    if all([test1_passed, test2_passed, test3_passed]):
        print("\n🎉 所有测试通过！系统已成功对接fan文件夹，并为后续添加其他设备做好准备。")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息并进行修复。")