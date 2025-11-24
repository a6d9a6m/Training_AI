"""
特征提取器包装类
用于将项目中的特征提取函数适配到实时分类器接口
"""

import numpy as np
from features.extract_features import extract_all_features


class FeatureExtractorWrapper:
    """
    特征提取器包装类，适配实时分类器接口
    """

    def __init__(self, config):
        """
        初始化特征提取器

        参数:
        config: 配置管理器实例
        """
        self.config = config
        self.sample_rate = config.get('data.sample_rate', 22050)

        # 构建特征配置
        self.feature_config = {
            'mfcc': {'n_mfcc': config.get('features.n_mfcc', 13)},
            'melspectrogram': {'n_mels': config.get('features.n_mels', 128)},
            'chroma': {},
            'spectral_contrast': {},
            'zero_crossing_rate': {},
            'rms_energy': {}
        }

    def extract_features(self, audio_data, sample_rate=None):
        """
        从音频数据中提取特征

        参数:
        audio_data: 音频数据数组
        sample_rate: 采样率（可选，如果为None则使用配置中的采样率）

        返回:
        features: 提取的特征向量
        """
        if sample_rate is None:
            sample_rate = self.sample_rate

        # 确保音频数据是一维数组
        if len(audio_data.shape) > 1:
            audio_data = audio_data.flatten()

        # 使用项目的特征提取函数
        combined_features, _ = extract_all_features(
            audio_data,
            sample_rate,
            self.feature_config
        )

        return combined_features
