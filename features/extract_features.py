import numpy as np
import librosa
import librosa.feature


def extract_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, fmin=20, fmax=None):
    """
    提取梅尔频率倒谱系数 (MFCC) 特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_mfcc: MFCC特征数量
    n_fft: FFT窗口大小
    hop_length: 帧移
    fmin: 最低频率
    fmax: 最高频率，None表示sr/2
    
    返回:
    mfcc_features: MFCC特征数组
    mfcc_mean: 均值特征向量
    mfcc_std: 标准差特征向量
    """
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    
    # 计算均值和标准差作为特征向量
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    
    # 水平堆叠均值和标准差
    mfcc_features = np.hstack((mfcc_mean, mfcc_std))
    
    return mfcc, mfcc_features


def extract_melspectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512, fmin=20, fmax=None):
    """
    提取梅尔频谱特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_mels: 梅尔滤波器数量
    n_fft: FFT窗口大小
    hop_length: 帧移
    fmin: 最低频率
    fmax: 最高频率，None表示sr/2
    
    返回:
    mel_spectrogram: 梅尔频谱
    mel_features: 梅尔频谱特征向量
    """
    # 提取梅尔频谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, 
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    
    # 转换为对数刻度
    mel_spectrogram_log = librosa.power_to_db(mel_spectrogram)
    
    # 计算均值和标准差作为特征向量
    mel_mean = np.mean(mel_spectrogram_log, axis=1)
    mel_std = np.std(mel_spectrogram_log, axis=1)
    
    # 水平堆叠均值和标准差
    mel_features = np.hstack((mel_mean, mel_std))
    
    return mel_spectrogram_log, mel_features


def extract_chroma(y, sr, n_fft=2048, hop_length=512):
    """
    提取色度特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    
    返回:
    chroma: 色度特征
    chroma_features: 色度特征向量
    """
    # 计算色度特征
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 计算均值和标准差作为特征向量
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)
    
    # 水平堆叠均值和标准差
    chroma_features = np.hstack((chroma_mean, chroma_std))
    
    return chroma, chroma_features


def extract_spectral_contrast(y, sr, n_fft=2048, hop_length=512):
    """
    提取频谱对比度特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    
    返回:
    spectral_contrast: 频谱对比度
    contrast_features: 频谱对比度特征向量
    """
    # 提取频谱对比度
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # 计算均值和标准差作为特征向量
    contrast_mean = np.mean(spectral_contrast, axis=1)
    contrast_std = np.std(spectral_contrast, axis=1)
    
    # 水平堆叠均值和标准差
    contrast_features = np.hstack((contrast_mean, contrast_std))
    
    return spectral_contrast, contrast_features


def extract_zero_crossing_rate(y, hop_length=512):
    """
    提取过零率特征
    
    参数:
    y: 音频时间序列
    hop_length: 帧移
    
    返回:
    zcr: 过零率序列
    zcr_features: 过零率特征向量
    """
    # 提取过零率
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)
    
    # 计算统计特征
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    zcr_max = np.max(zcr)
    zcr_min = np.min(zcr)
    
    # 构建特征向量
    zcr_features = np.array([zcr_mean, zcr_std, zcr_max, zcr_min])
    
    return zcr, zcr_features


def extract_rms_energy(y, frame_length=2048, hop_length=512):
    """
    提取RMS能量特征
    
    参数:
    y: 音频时间序列
    frame_length: 帧长度
    hop_length: 帧移
    
    返回:
    rms: RMS能量序列
    rms_features: RMS能量特征向量
    """
    # 提取RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)
    
    # 计算统计特征
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_max = np.max(rms)
    rms_min = np.min(rms)
    
    # 构建特征向量
    rms_features = np.array([rms_mean, rms_std, rms_max, rms_min])
    
    return rms, rms_features


def extract_all_features(y, sr, feature_config=None):
    """
    提取所有特征并组合成一个特征向量
    
    参数:
    y: 音频时间序列
    sr: 采样率
    feature_config: 特征配置字典
    
    返回:
    combined_features: 组合后的特征向量
    feature_dict: 包含各特征的字典
    """
    # 默认配置
    if feature_config is None:
        feature_config = {
            'mfcc': {'n_mfcc': 13},
            'melspectrogram': {'n_mels': 64},
            'chroma': {},
            'spectral_contrast': {},
            'zero_crossing_rate': {},
            'rms_energy': {}
        }
    
    feature_dict = {}
    combined_features = []
    
    # 提取MFCC特征
    if 'mfcc' in feature_config:
        _, mfcc_features = extract_mfcc(y, sr, **feature_config['mfcc'])
        combined_features.append(mfcc_features)
        feature_dict['mfcc'] = mfcc_features
    
    # 提取梅尔频谱特征
    if 'melspectrogram' in feature_config:
        _, mel_features = extract_melspectrogram(y, sr, **feature_config['melspectrogram'])
        combined_features.append(mel_features)
        feature_dict['melspectrogram'] = mel_features
    
    # 提取色度特征
    if 'chroma' in feature_config:
        _, chroma_features = extract_chroma(y, sr, **feature_config['chroma'])
        combined_features.append(chroma_features)
        feature_dict['chroma'] = chroma_features
    
    # 提取频谱对比度特征
    if 'spectral_contrast' in feature_config:
        _, contrast_features = extract_spectral_contrast(y, sr, **feature_config['spectral_contrast'])
        combined_features.append(contrast_features)
        feature_dict['spectral_contrast'] = contrast_features
    
    # 提取过零率特征
    if 'zero_crossing_rate' in feature_config:
        _, zcr_features = extract_zero_crossing_rate(y, **feature_config['zero_crossing_rate'])
        combined_features.append(zcr_features)
        feature_dict['zero_crossing_rate'] = zcr_features
    
    # 提取RMS能量特征
    if 'rms_energy' in feature_config:
        _, rms_features = extract_rms_energy(y, **feature_config['rms_energy'])
        combined_features.append(rms_features)
        feature_dict['rms_energy'] = rms_features
    
    # 组合所有特征
    if combined_features:
        combined_features = np.hstack(combined_features)
    else:
        raise ValueError("至少需要选择一种特征类型")
    
    return combined_features, feature_dict


def extract_features_from_files(file_paths, sr=22050, feature_config=None):
    """
    从多个音频文件中提取特征
    
    参数:
    file_paths: 音频文件路径列表
    sr: 采样率
    feature_config: 特征配置字典
    
    返回:
    features: 特征矩阵，每行是一个样本的特征
    valid_indices: 有效样本的索引
    """
    features = []
    valid_indices = []
    
    for i, file_path in enumerate(file_paths):
        try:
            # 加载音频
            y, _ = librosa.load(file_path, sr=sr)
            
            # 提取特征
            sample_features, _ = extract_all_features(y, sr, feature_config)
            
            features.append(sample_features)
            valid_indices.append(i)
        except Exception as e:
            print(f"处理文件失败 {file_path}: {e}")
    
    if features:
        features = np.vstack(features)
    else:
        features = np.array([])
    
    return features, valid_indices