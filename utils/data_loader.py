import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split


def load_audio(file_path, sr=22050, duration=None, offset=0.0):
    """
    加载音频文件
    
    参数:
    file_path: 音频文件路径
    sr: 采样率
    duration: 加载音频的时长（秒），None表示加载整个文件
    offset: 从文件开始的偏移量（秒）
    
    返回:
    y: 音频时间序列
    sr: 采样率
    """
    try:
        y, sr = librosa.load(file_path, sr=sr, duration=duration, offset=offset)
        return y, sr
    except Exception as e:
        print(f"加载音频文件失败 {file_path}: {e}")
        return None, sr


def preprocess_audio(y, sr, normalize=True, resample_rate=None):
    """
    预处理音频信号
    
    参数:
    y: 音频时间序列
    sr: 原始采样率
    normalize: 是否归一化音频幅度
    resample_rate: 重采样率，如果为None则不重采样
    
    返回:
    y_processed: 处理后的音频时间序列
    sr_new: 处理后的采样率
    """
    y_processed = y.copy()
    sr_new = sr
    
    # 重采样
    if resample_rate is not None and resample_rate != sr:
        y_processed = librosa.resample(y_processed, orig_sr=sr, target_sr=resample_rate)
        sr_new = resample_rate
    
    # 归一化
    if normalize:
        y_max = np.max(np.abs(y_processed))
        if y_max > 0:
            y_processed = y_processed / y_max
    
    return y_processed, sr_new


def load_dataset(normal_dir=None, anomaly_dir=None, sr=22050, test_size=0.2, val_size=0.2, random_state=42, device_type=None, base_data_dir=None):
    """
    加载数据集并分割为训练集、验证集和测试集
    
    参数:
    normal_dir: 正常数据目录
    anomaly_dir: 异常数据目录
    sr: 采样率
    test_size: 测试集比例
    val_size: 验证集比例（相对于训练集）
    random_state: 随机种子
    device_type: 设备类型（如'fan'），如果提供，将从base_data_dir/device_type加载数据
    base_data_dir: 基础数据目录，当提供device_type时使用
    
    返回:
    train_data: 训练集数据列表 [(audio_data, label), ...]
    val_data: 验证集数据列表 [(audio_data, label), ...]
    test_data: 测试集数据列表 [(audio_data, label), ...]
    """
    audio_files = []
    audio_labels = []
    
    # 如果指定了设备类型和基础数据目录，使用这种方式加载
    if device_type and base_data_dir:
        print(f"正在加载设备类型 '{device_type}' 的数据...")
        device_dir = os.path.join(base_data_dir, device_type)
        
        # 检查设备目录是否存在
        if not os.path.exists(device_dir):
            raise FileNotFoundError(f"设备目录不存在: {device_dir}")
        
        # 加载训练、测试数据
        # 训练数据通常在train目录
        train_dir = os.path.join(device_dir, 'train')
        if os.path.exists(train_dir):
            for file_name in os.listdir(train_dir):
                if file_name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    file_path = os.path.join(train_dir, file_name)
                    audio_files.append(file_path)
                    audio_labels.append(0)  # 训练集通常只包含正常样本
        
        # 测试数据在source_test和target_test目录
        test_dirs = ['source_test', 'target_test']
        for test_dir_name in test_dirs:
            test_dir = os.path.join(device_dir, test_dir_name)
            if os.path.exists(test_dir):
                for file_name in os.listdir(test_dir):
                    if file_name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        file_path = os.path.join(test_dir, file_name)
                        audio_files.append(file_path)
                        # 根据文件名判断是否为异常样本
                        if 'anomaly' in file_name.lower():
                            audio_labels.append(1)
                        else:
                            audio_labels.append(0)
    
    # 传统方式：分别指定正常和异常数据目录
    elif normal_dir and anomaly_dir:
        # 加载正常样本
        print(f"正在加载正常数据: {normal_dir}")
        if os.path.exists(normal_dir):
            for root, _, files in os.walk(normal_dir):
                for file_name in files:
                    if file_name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        file_path = os.path.join(root, file_name)
                        audio_files.append(file_path)
                        audio_labels.append(0)
        else:
            print(f"警告: 正常数据目录不存在 {normal_dir}")
        
        # 加载异常样本
        print(f"正在加载异常数据: {anomaly_dir}")
        if os.path.exists(anomaly_dir):
            for root, _, files in os.walk(anomaly_dir):
                for file_name in files:
                    if file_name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        file_path = os.path.join(root, file_name)
                        audio_files.append(file_path)
                        audio_labels.append(1)
        else:
            print(f"警告: 异常数据目录不存在 {anomaly_dir}")
    else:
        raise ValueError("必须提供(normal_dir, anomaly_dir)或(device_type, base_data_dir)")
    
    if len(audio_files) == 0:
        raise ValueError(f"没有找到音频文件")
    
    print(f"总共找到 {len(audio_files)} 个音频文件")
    print(f"正常样本: {audio_labels.count(0)}")
    print(f"异常样本: {audio_labels.count(1)}")
    
    # 分割数据集
    X_files_train_val, X_files_test, y_train_val, y_test = train_test_split(
        audio_files, audio_labels, test_size=test_size, random_state=random_state, stratify=audio_labels
    )
    
    X_files_train, X_files_val, y_train, y_val = train_test_split(
        X_files_train_val, y_train_val, test_size=val_size, random_state=random_state, stratify=y_train_val
    )
    
    # 加载音频数据
    print("正在加载音频数据...")
    train_data = [(load_audio(file_path, sr=sr)[0], label) for file_path, label in zip(X_files_train, y_train) if load_audio(file_path, sr=sr)[0] is not None]
    val_data = [(load_audio(file_path, sr=sr)[0], label) for file_path, label in zip(X_files_val, y_train_val) if load_audio(file_path, sr=sr)[0] is not None]
    test_data = [(load_audio(file_path, sr=sr)[0], label) for file_path, label in zip(X_files_test, y_test) if load_audio(file_path, sr=sr)[0] is not None]
    
    print(f"数据集分割完成:")
    print(f"- 训练集: {len(train_data)} 个有效样本")
    print(f"- 验证集: {len(val_data)} 个有效样本")
    print(f"- 测试集: {len(test_data)} 个有效样本")
    
    return train_data, val_data, test_data


def load_audio_batch(file_paths, sr=22050, normalize=True, resample_rate=None):
    """
    批量加载音频文件
    
    参数:
    file_paths: 音频文件路径列表
    sr: 采样率
    normalize: 是否归一化
    resample_rate: 重采样率
    
    返回:
    audio_data: 音频数据列表
    valid_indices: 有效样本的索引
    """
    audio_data = []
    valid_indices = []
    
    for i, file_path in enumerate(file_paths):
        y, _ = load_audio(file_path, sr=sr)
        if y is not None:
            y_processed, _ = preprocess_audio(y, sr, normalize=normalize, resample_rate=resample_rate)
            audio_data.append(y_processed)
            valid_indices.append(i)
    
    return audio_data, valid_indices