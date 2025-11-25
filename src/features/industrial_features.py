#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工业设备声音预处理 - 背景噪声抑制和信号增强
"""

import numpy as np
import librosa
from scipy.signal import wiener


def reduce_background_noise_stationary(y, sr, noise_duration=0.5):
    """
    抑制平稳背景噪声 - 基于噪声估计的频谱减法

    参数:
    y: 音频时间序列
    sr: 采样率
    noise_duration: 用于噪声估计的持续时间（秒）

    返回:
    y_denoised: 降噪后的音频
    """
    # 从开头估计噪声特征（假设前0.5秒是纯噪声或背景音）
    noise_samples = int(noise_duration * sr)
    noise_clip = y[:noise_samples]

    # 计算噪声的频谱
    noise_stft = librosa.stft(noise_clip)
    noise_power = np.mean(np.abs(noise_stft) ** 2, axis=1, keepdims=True)

    # 计算信号的频谱
    signal_stft = librosa.stft(y)
    signal_power = np.abs(signal_stft) ** 2

    # 频谱减法
    # 减去噪声功率，但保持相位不变
    alpha = 2.0  # 过减因子
    signal_power_cleaned = np.maximum(signal_power - alpha * noise_power, 0.1 * signal_power)

    # 重构
    magnitude = np.sqrt(signal_power_cleaned)
    phase = np.angle(signal_stft)
    signal_stft_cleaned = magnitude * np.exp(1j * phase)

    # ISTFT
    y_denoised = librosa.istft(signal_stft_cleaned)

    return y_denoised


def enhance_signal_contrast(y, sr):
    """
    增强信号对比度 - 突出异常部分

    参数:
    y: 音频时间序列
    sr: 采样率

    返回:
    y_enhanced: 增强后的音频
    """
    # 计算短时能量
    frame_length = 2048
    hop_length = 512

    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 扩展到原始长度
    rms_full = np.interp(
        np.arange(len(y)),
        np.arange(len(rms)) * hop_length,
        rms
    )

    # 动态范围压缩 - 突出微弱信号
    rms_normalized = rms_full / (np.max(rms_full) + 1e-10)
    gain = np.power(rms_normalized, -0.3)  # 压缩系数
    gain = np.clip(gain, 0.5, 2.0)  # 限制增益范围

    y_enhanced = y * gain

    # 归一化
    y_enhanced = y_enhanced / (np.max(np.abs(y_enhanced)) + 1e-10)

    return y_enhanced


def extract_differential_features(y, sr):
    """
    提取差分特征 - 关注变化而非绝对值

    适用于有持续背景音的场景，异常通常表现为与正常模式的"偏离"

    参数:
    y: 音频时间序列
    sr: 采样率

    返回:
    features: 差分特征向量
    """
    features = []

    # 1. MFCC的时间差分
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # 关注变化的统计特征
    for feat in [mfcc_delta, mfcc_delta2]:
        features.extend([
            np.mean(np.abs(feat), axis=1),  # 平均变化幅度
            np.std(feat, axis=1),           # 变化的波动
            np.max(np.abs(feat), axis=1),   # 最大变化
            np.percentile(np.abs(feat), 90, axis=1)  # 90分位数
        ])

    # 2. 频谱质心的变化
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_diff = np.diff(centroid)

    features.extend([
        np.mean(np.abs(centroid_diff)),
        np.std(centroid_diff),
        np.max(np.abs(centroid_diff))
    ])

    # 3. RMS能量的变化
    rms = librosa.feature.rms(y=y)[0]
    rms_diff = np.diff(rms)

    features.extend([
        np.mean(np.abs(rms_diff)),
        np.std(rms_diff),
        np.max(np.abs(rms_diff)),
        # 能量突变检测
        np.sum(np.abs(rms_diff) > 2 * np.std(rms_diff))
    ])

    # 4. 过零率的变化
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    zcr_diff = np.diff(zcr)

    features.extend([
        np.mean(np.abs(zcr_diff)),
        np.std(zcr_diff)
    ])

    # 扁平化
    features = [f.flatten() if hasattr(f, 'flatten') else [f] for f in features]
    features = np.concatenate([np.atleast_1d(f) for f in features])

    return features


def extract_periodic_features(y, sr):
    """
    提取周期性特征 - 工业设备通常有周期性运转

    异常可能表现为周期性的破坏

    参数:
    y: 音频时间序列
    sr: 采样率

    返回:
    features: 周期性特征向量
    """
    features = []

    # 1. 自相关函数 - 检测周期性
    autocorr = librosa.autocorrelate(y, max_size=sr)  # 最多1秒的延迟

    features.extend([
        np.max(autocorr[1:]),  # 最大自相关（排除0延迟）
        np.argmax(autocorr[1:]) / sr,  # 主周期（秒）
        np.mean(autocorr),
        np.std(autocorr)
    ])

    # 2. 频谱的周期性
    # 计算频谱图每一帧的自相关
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)

    frame_autocorr = []
    for i in range(mel_spec.shape[1]):
        frame = mel_spec[:, i]
        # 简单的自相关
        ac = np.correlate(frame, frame, mode='same')
        frame_autocorr.append(np.max(ac))

    features.extend([
        np.mean(frame_autocorr),
        np.std(frame_autocorr),
        np.min(frame_autocorr),
        np.max(frame_autocorr)
    ])

    # 3. Tempogram - 节奏特征
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr)

        features.extend([
            np.mean(tempogram),
            np.std(tempogram),
            np.max(tempogram)
        ])
    except:
        # 如果tempogram计算失败，用零填充
        features.extend([0, 0, 0])

    return np.array(features)


def extract_industrial_features(y, sr, denoise=True):
    """
    提取适用于工业设备的综合特征

    包含：
    1. 背景噪声抑制
    2. 差分特征（关注变化）
    3. 周期性特征（设备运转模式）

    参数:
    y: 音频时间序列
    sr: 采样率
    denoise: 是否进行降噪预处理

    返回:
    features: 工业特征向量
    """
    # 1. 预处理
    if denoise:
        y = reduce_background_noise_stationary(y, sr)
        y = enhance_signal_contrast(y, sr)

    # 2. 提取差分特征
    diff_features = extract_differential_features(y, sr)

    # 3. 提取周期性特征
    periodic_features = extract_periodic_features(y, sr)

    # 4. 基础频谱特征（简化版）
    # MFCC - 只保留关键统计量
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats = np.hstack([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    # 频谱对比度
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_stats = np.hstack([
        np.mean(contrast, axis=1),
        np.std(contrast, axis=1)
    ])

    # 5. 组合所有特征
    all_features = np.hstack([
        diff_features,
        periodic_features,
        mfcc_stats,
        contrast_stats
    ])

    # 处理NaN和Inf
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=1e10, neginf=-1e10)

    return all_features


def preprocess_industrial_audio(file_path, sr=22050, denoise=True):
    """
    加载并预处理工业音频文件

    参数:
    file_path: 音频文件路径
    sr: 采样率
    denoise: 是否降噪

    返回:
    y_processed: 预处理后的音频
    features: 提取的特征
    """
    # 加载音频
    y, _ = librosa.load(file_path, sr=sr)

    # 预处理
    if denoise:
        y_processed = reduce_background_noise_stationary(y, sr)
        y_processed = enhance_signal_contrast(y_processed, sr)
    else:
        y_processed = y

    # 提取特征
    features = extract_industrial_features(y_processed, sr, denoise=False)  # 已经预处理过了

    return y_processed, features
