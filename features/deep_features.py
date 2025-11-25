#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
深度学习特征提取 - 使用预训练模型
"""

import numpy as np
import torch
import torch.nn as nn
import librosa

class SimpleAudioEncoder(nn.Module):
    """
    简单的音频编码器 - 自动学习特征表示
    """
    def __init__(self, input_dim, encoding_dim=64):
        super(SimpleAudioEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),

            nn.Linear(128, encoding_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


def extract_raw_spectrogram_features(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """
    提取原始频谱特征用于深度学习

    参数:
    y: 音频时间序列
    sr: 采样率
    n_fft: FFT窗口大小
    hop_length: 帧移
    n_mels: 梅尔滤波器数量

    返回:
    features: 扁平化的频谱特征
    """
    # 提取梅尔频谱
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft,
        hop_length=hop_length, n_mels=n_mels
    )

    # 转换为对数刻度
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # 统计池化 - 对时间维度进行汇总
    features = []

    # 每个频率带的统计特征
    for i in range(n_mels):
        freq_band = mel_spec_db[i, :]
        features.extend([
            np.mean(freq_band),
            np.std(freq_band),
            np.max(freq_band),
            np.min(freq_band),
            np.median(freq_band)
        ])

    # 全局统计特征
    features.extend([
        np.mean(mel_spec_db),
        np.std(mel_spec_db),
        np.max(mel_spec_db),
        np.min(mel_spec_db)
    ])

    return np.array(features)


def extract_multi_resolution_features(y, sr):
    """
    提取多分辨率特征 - 不同时间尺度的特征

    参数:
    y: 音频时间序列
    sr: 采样率

    返回:
    features: 多分辨率特征向量
    """
    features = []

    # 不同的窗口大小捕捉不同时间尺度的信息
    window_sizes = [512, 1024, 2048, 4096]

    for n_fft in window_sizes:
        hop_length = n_fft // 4

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        features.extend([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1)
        ])

        # 频谱质心
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
        features.extend([
            np.mean(centroid),
            np.std(centroid)
        ])

    # 扁平化
    features = [f.flatten() if hasattr(f, 'flatten') else [f] for f in features]
    features = np.concatenate([np.atleast_1d(f) for f in features])

    return features


def extract_temporal_context_features(y, sr, context_frames=5):
    """
    提取时间上下文特征 - 考虑相邻帧的关系

    参数:
    y: 音频时间序列
    sr: 采样率
    context_frames: 上下文帧数

    返回:
    features: 时间上下文特征向量
    """
    # 提取MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    features = []

    # 基础统计
    features.extend([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1)
    ])

    # 时间变化特征
    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend([
        np.mean(mfcc_delta, axis=1),
        np.std(mfcc_delta, axis=1)
    ])

    # 加速度特征
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    features.extend([
        np.mean(mfcc_delta2, axis=1),
        np.std(mfcc_delta2, axis=1)
    ])

    # 扁平化
    features = np.concatenate([np.atleast_1d(f) for f in features])

    return features
