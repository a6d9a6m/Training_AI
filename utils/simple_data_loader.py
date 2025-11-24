"""
简化的数据加载器
支持两种模式：监督学习和异常检测
"""

import os
import numpy as np
import librosa
from glob import glob
from sklearn.model_selection import train_test_split


def load_audio_file(file_path, sr=22050, duration=None):
    """
    加载单个音频文件

    参数:
    file_path: 音频文件路径
    sr: 采样率
    duration: 音频时长（秒），None表示加载完整文件

    返回:
    audio: 音频数据
    """
    try:
        audio, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        return audio
    except Exception as e:
        print(f"加载音频文件失败 {file_path}: {e}")
        return None


def load_supervised_data(normal_dir, anomaly_dir, sr=22050, test_size=0.2, val_size=0.2, random_state=42):
    """
    加载监督学习数据

    参数:
    normal_dir: 正常样本目录
    anomaly_dir: 异常样本目录
    sr: 采样率
    test_size: 测试集比例
    val_size: 验证集比例（从训练集中划分）
    random_state: 随机种子

    返回:
    train_data: [(audio, label), ...]
    val_data: [(audio, label), ...]
    test_data: [(audio, label), ...]
    """
    print("加载监督学习数据...")

    # 检查目录
    if not os.path.exists(normal_dir):
        raise FileNotFoundError(f"正常样本目录不存在: {normal_dir}")
    if not os.path.exists(anomaly_dir):
        raise FileNotFoundError(f"异常样本目录不存在: {anomaly_dir}")

    # 加载正常样本
    normal_files = glob(os.path.join(normal_dir, "*.wav"))
    print(f"找到 {len(normal_files)} 个正常样本")

    normal_data = []
    for file_path in normal_files:
        audio = load_audio_file(file_path, sr=sr)
        if audio is not None:
            normal_data.append((audio, 0))  # 标签0表示正常

    # 加载异常样本
    anomaly_files = glob(os.path.join(anomaly_dir, "*.wav"))
    print(f"找到 {len(anomaly_files)} 个异常样本")

    anomaly_data = []
    for file_path in anomaly_files:
        audio = load_audio_file(file_path, sr=sr)
        if audio is not None:
            anomaly_data.append((audio, 1))  # 标签1表示异常

    # 合并数据
    all_data = normal_data + anomaly_data
    print(f"总样本数: {len(all_data)} (正常: {len(normal_data)}, 异常: {len(anomaly_data)})")

    # 分割数据集
    # 先分出测试集
    train_val_data, test_data = train_test_split(
        all_data, test_size=test_size, random_state=random_state,
        stratify=[label for _, label in all_data]
    )

    # 再从训练集分出验证集
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_size, random_state=random_state,
        stratify=[label for _, label in train_val_data]
    )

    print(f"数据分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}, 测试集 {len(test_data)}")

    return train_data, val_data, test_data


def load_anomaly_detection_data(normal_train_dir, mixed_test_dir=None, sr=22050, val_ratio=0.2, random_state=42):
    """
    加载异常检测数据

    参数:
    normal_train_dir: 纯净正常样本目录（用于训练）
    mixed_test_dir: 混合样本目录（大部分正常，少量异常，用于测试）
    sr: 采样率
    val_ratio: 从训练集划分验证集的比例
    random_state: 随机种子

    返回:
    train_data: [(audio, 0), ...] - 全部正常样本
    val_data: [(audio, 0), ...] - 验证用的正常样本
    test_data: [(audio, label), ...] - 混合样本（如果提供）
    """
    print("加载异常检测数据...")

    # 检查目录
    if not os.path.exists(normal_train_dir):
        raise FileNotFoundError(f"正常训练样本目录不存在: {normal_train_dir}")

    # 加载正常训练样本
    normal_files = glob(os.path.join(normal_train_dir, "*.wav"))
    print(f"找到 {len(normal_files)} 个正常训练样本")

    normal_data = []
    for file_path in normal_files:
        audio = load_audio_file(file_path, sr=sr)
        if audio is not None:
            normal_data.append((audio, 0))  # 全部标记为正常

    # 分割训练集和验证集（都是正常样本）
    train_data, val_data = train_test_split(
        normal_data, test_size=val_ratio, random_state=random_state
    )

    print(f"正常样本分割: 训练集 {len(train_data)}, 验证集 {len(val_data)}")

    # 加载混合测试集（可选）
    test_data = []
    if mixed_test_dir and os.path.exists(mixed_test_dir):
        print(f"加载混合测试集: {mixed_test_dir}")

        # 支持两种组织方式
        # 方式1: 有 normal/ 和 anomaly/ 子目录
        normal_test_dir = os.path.join(mixed_test_dir, "normal")
        anomaly_test_dir = os.path.join(mixed_test_dir, "anomaly")

        if os.path.exists(normal_test_dir) and os.path.exists(anomaly_test_dir):
            # 加载正常测试样本
            normal_test_files = glob(os.path.join(normal_test_dir, "*.wav"))
            print(f"找到 {len(normal_test_files)} 个正常测试样本")
            for file_path in normal_test_files:
                audio = load_audio_file(file_path, sr=sr)
                if audio is not None:
                    test_data.append((audio, 0))

            # 加载异常测试样本
            anomaly_test_files = glob(os.path.join(anomaly_test_dir, "*.wav"))
            print(f"找到 {len(anomaly_test_files)} 个异常测试样本")
            for file_path in anomaly_test_files:
                audio = load_audio_file(file_path, sr=sr)
                if audio is not None:
                    test_data.append((audio, 1))
        else:
            # 方式2: 所有文件在同一目录，假设大部分正常
            mixed_files = glob(os.path.join(mixed_test_dir, "*.wav"))
            print(f"找到 {len(mixed_files)} 个混合测试样本")
            print("警告: 无法区分正常和异常，所有样本标记为正常(0)")
            for file_path in mixed_files:
                audio = load_audio_file(file_path, sr=sr)
                if audio is not None:
                    test_data.append((audio, 0))

        print(f"测试集样本数: {len(test_data)}")
    else:
        print("未提供测试集，将使用验证集进行评估")

    return train_data, val_data, test_data


def get_audio_and_labels(data):
    """
    从数据列表中分离音频和标签

    参数:
    data: [(audio, label), ...]

    返回:
    audios: [audio, ...]
    labels: [label, ...]
    """
    audios = [audio for audio, _ in data]
    labels = [label for _, label in data]
    return audios, labels
