import numpy as np
import librosa
import librosa.feature
from scipy import stats  # 用于更高级的统计特征计算
import warnings  # 用于抑制警告

# 小波变换（可选依赖）
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("警告: PyWavelets未安装，小波特征将被跳过。安装: pip install PyWavelets")


def extract_mfcc(y, sr, n_mfcc=13, n_fft=2048, hop_length=512, fmin=20, fmax=None, apply_delta=True):
    """
    提取梅尔频率倒谱系数 (MFCC) 特征，增加更稳健的统计特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_mfcc: MFCC特征数量
    n_fft: FFT窗口大小
    hop_length: 帧移
    fmin: 最低频率
    fmax: 最高频率，None表示sr/2
    apply_delta: 是否计算MFCC的一阶和二阶差分
    
    返回:
    mfcc: MFCC特征数组
    mfcc_features: MFCC特征向量
    """
    # 提取MFCC特征
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, 
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    
    # 应用倒谱均值减法(CMC)，提高领域不变性
    mfcc_cmvn = librosa.feature.stack_memory(mfcc)
    
    # 计算一阶和二阶差分
    features = [mfcc]
    if apply_delta:
        delta1 = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features.extend([delta1, delta2])
    
    # 为每个特征类型计算更丰富的统计特征
    all_stats = []
    for feat in features:
        # 基础统计特征
        feat_mean = np.mean(feat, axis=1)
        feat_std = np.std(feat, axis=1)
        feat_min = np.min(feat, axis=1)
        feat_max = np.max(feat, axis=1)
        feat_median = np.median(feat, axis=1)
        
        # 更高级的统计特征
        feat_skew = stats.skew(feat, axis=1)
        feat_kurtosis = stats.kurtosis(feat, axis=1)
        feat_percentile25 = np.percentile(feat, 25, axis=1)
        feat_percentile75 = np.percentile(feat, 75, axis=1)
        
        # 合并统计特征
        stats_features = np.hstack([
            feat_mean, feat_std, feat_min, feat_max, feat_median,
            feat_skew, feat_kurtosis, feat_percentile25, feat_percentile75
        ])
        all_stats.append(stats_features)
    
    # 水平堆叠所有统计特征
    mfcc_features = np.hstack(all_stats)
    
    return mfcc, mfcc_features

def extract_mfcc_delta_stats(y, sr, n_mfcc=13):
    """
    提取MFCC及其差分的统计特征，特别适合领域不变表示
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_mfcc: MFCC特征数量
    
    返回:
    mfcc_delta_features: 包含MFCC及其差分统计特征的向量
    """
    # 提取MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    
    # 计算一阶和二阶差分
    delta1 = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    
    # 合并MFCC和差分
    mfcc_delta = np.vstack([mfcc, delta1, delta2])
    
    # 计算稳健统计量
    features = []
    for i in range(mfcc_delta.shape[0]):
        features.extend([
            np.mean(mfcc_delta[i]),
            np.median(mfcc_delta[i]),
            np.std(mfcc_delta[i]),
            np.percentile(mfcc_delta[i], 25),
            np.percentile(mfcc_delta[i], 75),
            stats.skew(mfcc_delta[i]),
            stats.kurtosis(mfcc_delta[i])
        ])
    
    return np.array(features)


def extract_melspectrogram(y, sr, n_mels=128, n_fft=2048, hop_length=512, fmin=20, fmax=None):
    """
    提取梅尔频谱特征，增加更多统计特征以提高稳健性
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_mels: 梅尔滤波器数量
    n_fft: FFT窗口大小
    hop_length: 帧移
    fmin: 最低频率
    fmax: 最高频率，None表示sr/2
    
    返回:
    mel_spectrogram_log: 对数梅尔频谱
    mel_features: 梅尔频谱特征向量
    """
    # 提取梅尔频谱
    mel_spectrogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, 
        hop_length=hop_length, fmin=fmin, fmax=fmax
    )
    
    # 转换为对数刻度
    mel_spectrogram_log = librosa.power_to_db(mel_spectrogram)
    
    # 计算更丰富的统计特征
    mel_mean = np.mean(mel_spectrogram_log, axis=1)
    mel_std = np.std(mel_spectrogram_log, axis=1)
    mel_min = np.min(mel_spectrogram_log, axis=1)
    mel_max = np.max(mel_spectrogram_log, axis=1)
    mel_median = np.median(mel_spectrogram_log, axis=1)
    
    # 添加数值稳定性处理，避免偏度和峰度计算时的精度损失警告
    # 1. 对每个频谱帧添加微小扰动以增加数值多样性
    mel_spectrogram_stable = mel_spectrogram_log.copy()
    
    # 只对标准差很小的数据添加扰动
    small_std_mask = mel_std < 1e-5
    if np.any(small_std_mask):
        # 添加微小的高斯噪声作为扰动
        epsilon = 1e-6
        noise = np.random.normal(0, epsilon, mel_spectrogram_log.shape)
        mel_spectrogram_stable = mel_spectrogram_log + noise
    
    # 2. 在计算偏度和峰度时抑制警告
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mel_skew = stats.skew(mel_spectrogram_stable, axis=1)
        mel_kurtosis = stats.kurtosis(mel_spectrogram_stable, axis=1)
    
    # 水平堆叠所有统计特征
    mel_features = np.hstack([
        mel_mean, mel_std, mel_min, mel_max, mel_median,
        mel_skew, mel_kurtosis
    ])
    
    return mel_spectrogram_log, mel_features

def extract_spectral_bandwidth(y, sr, n_fft=2048, hop_length=512):
    """
    提取频谱带宽特征，对声音的频率分布范围敏感
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    
    返回:
    bandwidth: 频谱带宽
    bandwidth_features: 频谱带宽特征向量
    """
    # 提取频谱带宽
    bandwidth = librosa.feature.spectral_bandwidth(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length
    )
    
    # 计算统计特征
    features = [
        np.mean(bandwidth),
        np.std(bandwidth),
        np.min(bandwidth),
        np.max(bandwidth),
        np.median(bandwidth),
        stats.skew(bandwidth)[0],
        stats.kurtosis(bandwidth)[0]
    ]
    
    return bandwidth, np.array(features)


def extract_chroma(y, sr, n_fft=2048, hop_length=512, use_cqt=True):
    """
    提取色度特征，支持使用色度恒常性变换提高领域不变性
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    use_cqt: 是否使用常数Q变换计算色度特征（更稳定）
    
    返回:
    chroma: 色度特征
    chroma_features: 色度特征向量
    """
    # 使用常数Q变换计算色度特征，对音高变化更鲁棒
    if use_cqt:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    else:
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 计算色度能量归一化，提高领域不变性
    chroma_norm = librosa.util.normalize(chroma, axis=0)
    
    # 计算丰富的统计特征
    chroma_mean = np.mean(chroma_norm, axis=1)
    chroma_std = np.std(chroma_norm, axis=1)
    chroma_min = np.min(chroma_norm, axis=1)
    chroma_max = np.max(chroma_norm, axis=1)
    chroma_median = np.median(chroma_norm, axis=1)
    
    # 计算色度相关统计量
    chroma_contrast = np.max(chroma_norm, axis=1) - np.min(chroma_norm, axis=1)
    
    # 水平堆叠所有统计特征
    chroma_features = np.hstack([
        chroma_mean, chroma_std, chroma_min, chroma_max, chroma_median,
        chroma_contrast
    ])
    
    return chroma, chroma_features

def extract_chroma_constant(y, sr):
    """
    提取基于常数Q变换的色度特征，对音高变化和噪声具有不变性
    
    参数:
    y: 音频时间序列
    sr: 采样率
    
    返回:
    chroma_features: 色度恒常特征向量
    """
    # 使用不同的参数计算色度特征以提高不变性
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, bins_per_octave=36, n_octaves=7)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
    
    # 计算两种色度特征的统计量
    features = []
    for chroma in [chroma_cqt, chroma_cens]:
        features.extend([
            np.mean(chroma, axis=1),
            np.median(chroma, axis=1),
            np.std(chroma, axis=1),
            np.max(chroma, axis=1) - np.min(chroma, axis=1),
            np.mean(np.diff(chroma, axis=1), axis=1)  # 时间变化特征
        ])
    
    # 扁平化特征列表
    return np.concatenate(features)


def extract_spectral_contrast(y, sr, n_fft=2048, hop_length=512, n_bands=6):
    """
    提取频谱对比度特征，增加更多统计特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    n_bands: 频谱带数量
    
    返回:
    spectral_contrast: 频谱对比度
    contrast_features: 频谱对比度特征向量
    """
    # 提取频谱对比度
    spectral_contrast = librosa.feature.spectral_contrast(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_bands=n_bands
    )
    
    # 计算更丰富的统计特征
    contrast_mean = np.mean(spectral_contrast, axis=1)
    contrast_std = np.std(spectral_contrast, axis=1)
    contrast_min = np.min(spectral_contrast, axis=1)
    contrast_max = np.max(spectral_contrast, axis=1)
    contrast_median = np.median(spectral_contrast, axis=1)
    
    # 水平堆叠所有统计特征
    contrast_features = np.hstack([
        contrast_mean, contrast_std, contrast_min, contrast_max, contrast_median
    ])
    
    return spectral_contrast, contrast_features

def extract_spectral_flatness(y, sr, n_fft=2048, hop_length=512):
    """
    提取频谱平坦度特征，对音色特性敏感
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    
    返回:
    flatness: 频谱平坦度
    flatness_features: 频谱平坦度特征向量
    """
    # 提取频谱平坦度
    flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)
    
    # 计算统计特征
    features = [
        np.mean(flatness),
        np.std(flatness),
        np.min(flatness),
        np.max(flatness),
        np.median(flatness),
        stats.skew(flatness)[0],
        stats.kurtosis(flatness)[0]
    ]
    
    return flatness, np.array(features)

def extract_spectral_centroid(y, sr, n_fft=2048, hop_length=512):
    """
    提取频谱质心特征，对声音的明亮度敏感
    
    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    
    返回:
    centroid: 频谱质心
    centroid_features: 频谱质心特征向量
    """
    # 提取频谱质心
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    
    # 计算统计特征
    features = [
        np.mean(centroid),
        np.std(centroid),
        np.min(centroid),
        np.max(centroid),
        np.median(centroid),
        stats.skew(centroid)[0],
        stats.kurtosis(centroid)[0]
    ]
    
    return centroid, np.array(features)


def extract_zero_crossing_rate(y, hop_length=512, frame_length=2048):
    """
    提取过零率特征，增加更多统计特征
    
    参数:
    y: 音频时间序列
    hop_length: 帧移
    frame_length: 帧长度
    
    返回:
    zcr: 过零率序列
    zcr_features: 过零率特征向量
    """
    # 提取过零率
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length, frame_length=frame_length)
    
    # 计算更丰富的统计特征
    zcr_mean = np.mean(zcr)
    zcr_std = np.std(zcr)
    zcr_min = np.min(zcr)
    zcr_max = np.max(zcr)
    zcr_median = np.median(zcr)
    zcr_skew = stats.skew(zcr)[0]
    zcr_kurtosis = stats.kurtosis(zcr)[0]
    
    # 构建特征向量
    zcr_features = np.array([
        zcr_mean, zcr_std, zcr_min, zcr_max, zcr_median,
        zcr_skew, zcr_kurtosis
    ])
    
    return zcr, zcr_features

def extract_wavelet_features(y, wavelet='db4', level=4):
    """
    提取小波特征，提供多尺度时间-频率表示，对领域变化更稳健

    参数:
    y: 音频时间序列
    wavelet: 小波基函数名称
    level: 分解级别

    返回:
    wavelet_features: 小波特征向量
    """
    if not PYWT_AVAILABLE:
        # 如果pywt未安装，返回空数组
        return np.array([])

    try:
        # 执行小波分解
        coeffs = pywt.wavedec(y, wavelet, level=level)
        
        features = []
        
        # 对每个系数级别提取统计特征
        for i, coeff in enumerate(coeffs):
            # 基础统计量
            features.extend([
                np.mean(coeff),
                np.std(coeff),
                np.min(coeff),
                np.max(coeff),
                np.median(coeff),
                stats.skew(coeff),
                stats.kurtosis(coeff),
                np.percentile(coeff, 25),
                np.percentile(coeff, 75)
            ])
            
            # 能量特征
            energy = np.sum(coeff ** 2)
            features.append(energy)
        
        return np.array(features)
    except Exception as e:
        print(f"小波特征提取失败: {e}")
        # 如果小波特征提取失败，返回空数组
        return np.array([])


def extract_rms_energy(y, frame_length=2048, hop_length=512):
    """
    提取RMS能量特征，增加更多统计特征
    
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
    
    # 计算更丰富的统计特征
    rms_mean = np.mean(rms)
    rms_std = np.std(rms)
    rms_min = np.min(rms)
    rms_max = np.max(rms)
    rms_median = np.median(rms)
    rms_skew = stats.skew(rms)[0]
    rms_kurtosis = stats.kurtosis(rms)[0]
    
    # 计算能量变化特征
    rms_diff = np.diff(rms, axis=1)
    rms_diff_mean = np.mean(np.abs(rms_diff))
    rms_diff_std = np.std(rms_diff)
    
    # 构建特征向量
    rms_features = np.array([
        rms_mean, rms_std, rms_min, rms_max, rms_median,
        rms_skew, rms_kurtosis, rms_diff_mean, rms_diff_std
    ])
    
    return rms, rms_features

def extract_temporal_features(y):
    """
    提取时域特征，对声音的时间特性敏感
    
    参数:
    y: 音频时间序列
    
    返回:
    temporal_features: 时域特征向量
    """
    # 基础时域统计特征
    mean_amplitude = np.mean(np.abs(y))
    std_amplitude = np.std(y)
    max_amplitude = np.max(np.abs(y))
    min_amplitude = np.min(np.abs(y))
    rms_amplitude = np.sqrt(np.mean(y**2))
    
    # 能量相关特征
    energy = np.sum(y**2)
    energy_entropy = -np.sum((y**2/energy) * np.log2(y**2/energy + 1e-10))
    
    # 过零率相关特征
    zero_crossings = np.where(np.diff(np.signbit(y)))[0]
    zero_crossing_rate = len(zero_crossings) / len(y)
    
    # 构建特征向量
    features = np.array([
        mean_amplitude, std_amplitude, max_amplitude, min_amplitude, rms_amplitude,
        energy, energy_entropy, zero_crossing_rate
    ])
    
    return features


def extract_all_features(y, sr, feature_config=None):
    """
    提取所有特征并组合成一个特征向量，增加了更稳健的领域不变特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    feature_config: 特征配置字典
    
    返回:
    combined_features: 组合后的特征向量
    feature_dict: 包含各特征的字典
    """
    # 默认配置，添加了新的稳健特征
    if feature_config is None:
        feature_config = {
            'mfcc': {'n_mfcc': 13, 'apply_delta': True},
            'melspectrogram': {'n_mels': 64},
            'chroma': {'use_cqt': True},
            'spectral_contrast': {'n_bands': 6},
            'zero_crossing_rate': {},
            'rms_energy': {},
            # 新增特征
            'spectral_bandwidth': {},
            'spectral_flatness': {},
            'spectral_centroid': {},
            'chroma_constant': {},
            'wavelet': {'wavelet': 'db4', 'level': 4},
            'temporal': {}
        }
    
    feature_dict = {}
    combined_features = []
    
    # 提取MFCC特征（带差分）
    if 'mfcc' in feature_config:
        _, mfcc_features = extract_mfcc(y, sr, **feature_config['mfcc'])
        combined_features.append(mfcc_features)
        feature_dict['mfcc'] = mfcc_features
    
    # 提取梅尔频谱特征
    if 'melspectrogram' in feature_config:
        _, mel_features = extract_melspectrogram(y, sr, **feature_config['melspectrogram'])
        combined_features.append(mel_features)
        feature_dict['melspectrogram'] = mel_features
    
    # 提取色度特征（使用CQT）
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
    
    # 新增特征：频谱带宽
    if 'spectral_bandwidth' in feature_config:
        _, bandwidth_features = extract_spectral_bandwidth(y, sr, **feature_config['spectral_bandwidth'])
        combined_features.append(bandwidth_features)
        feature_dict['spectral_bandwidth'] = bandwidth_features
    
    # 新增特征：频谱平坦度
    if 'spectral_flatness' in feature_config:
        _, flatness_features = extract_spectral_flatness(y, sr, **feature_config['spectral_flatness'])
        combined_features.append(flatness_features)
        feature_dict['spectral_flatness'] = flatness_features
    
    # 新增特征：频谱质心
    if 'spectral_centroid' in feature_config:
        _, centroid_features = extract_spectral_centroid(y, sr, **feature_config['spectral_centroid'])
        combined_features.append(centroid_features)
        feature_dict['spectral_centroid'] = centroid_features
    
    # 新增特征：色度恒常特征
    if 'chroma_constant' in feature_config:
        chroma_const_features = extract_chroma_constant(y, sr)
        if len(chroma_const_features) > 0:
            combined_features.append(chroma_const_features)
            feature_dict['chroma_constant'] = chroma_const_features
    
    # 新增特征：小波特征
    if 'wavelet' in feature_config:
        wavelet_features = extract_wavelet_features(y, **feature_config['wavelet'])
        if len(wavelet_features) > 0:
            combined_features.append(wavelet_features)
            feature_dict['wavelet'] = wavelet_features
    
    # 新增特征：时域特征
    if 'temporal' in feature_config:
        temporal_features = extract_temporal_features(y)
        combined_features.append(temporal_features)
        feature_dict['temporal'] = temporal_features
    
    # 组合所有特征
    if combined_features:
        combined_features = np.hstack(combined_features)
        # 处理可能的NaN或无穷值
        combined_features = np.nan_to_num(combined_features, nan=0.0, posinf=1e10, neginf=-1e10)
    else:
        raise ValueError("至少需要选择一种特征类型")
    
    return combined_features, feature_dict

def extract_domain_invariant_features(y, sr):
    """
    专门提取领域不变特征的函数，组合了对领域变化最不敏感的特征
    
    参数:
    y: 音频时间序列
    sr: 采样率
    
    返回:
    invariant_features: 领域不变特征向量
    """
    try:
        # 使用色度恒常特征
        chroma_const = extract_chroma_constant(y, sr)
        
        # 使用MFCC差分统计特征
        mfcc_delta = extract_mfcc_delta_stats(y, sr)
        
        # 使用小波特征
        wavelet = extract_wavelet_features(y)
        
        # 组合领域不变特征
        features_list = [chroma_const, mfcc_delta]
        if len(wavelet) > 0:
            features_list.append(wavelet)
        
        invariant_features = np.hstack(features_list)
        
        # 处理可能的NaN或无穷值
        invariant_features = np.nan_to_num(invariant_features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return invariant_features
    except Exception as e:
        print(f"领域不变特征提取失败: {e}")
        # 回退到标准MFCC特征
        _, mfcc_features = extract_mfcc(y, sr)
        return mfcc_features


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