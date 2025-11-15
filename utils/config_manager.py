import os
import yaml
import logging
from typing import Dict, Any, List


class ConfigManager:
    """
    配置管理器，用于加载、保存和管理项目配置
    
    支持YAML格式的配置文件，提供参数的获取、设置和验证功能
    """
    
    def __init__(self, config_path=None):
        """
        初始化配置管理器
        
        参数:
        config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = {}
        self.logger = self._setup_logger()
        
        if config_path:
            self.load_config(config_path)
    
    def _setup_logger(self):
        """
        设置日志记录器
        
        返回:
        logger: 日志记录器实例
        """
        logger = logging.getLogger('ConfigManager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_config(self, config_path):
        """
        从YAML文件加载配置
        
        参数:
        config_path: 配置文件路径
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.config_path = config_path
            self.logger.info(f"成功加载配置文件: {config_path}")
        except yaml.YAMLError as e:
            self.logger.error(f"解析配置文件失败: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise
        
        return self
    
    def save_config(self, config_path=None):
        """
        保存配置到YAML文件
        
        参数:
        config_path: 配置文件路径，如果为None则使用当前配置路径
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        path = config_path or self.config_path
        if not path:
            raise ValueError("未指定配置文件路径")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
            self.logger.info(f"成功保存配置文件: {path}")
        except Exception as e:
            self.logger.error(f"保存配置文件失败: {e}")
            raise
        
        return self
    
    def get(self, key, default=None, required=False):
        """
        获取配置项
        
        支持点表示法访问嵌套配置，例如: "data.batch_size"
        
        参数:
        key: 配置键
        default: 默认值，如果配置项不存在则返回
        required: 如果为True且配置项不存在则抛出异常
        
        返回:
        value: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
        except (KeyError, TypeError):
            if required:
                raise KeyError(f"必需的配置项不存在: {key}")
            return default
        
        return value
    
    def set(self, key, value):
        """
        设置配置项
        
        支持点表示法设置嵌套配置，例如: "data.batch_size"
        
        参数:
        key: 配置键
        value: 配置值
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        keys = key.split('.')
        config = self.config
        
        # 遍历除最后一个键外的所有键，确保嵌套结构存在
        for k in keys[:-1]:
            if k not in config or not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        # 设置最后一个键的值
        config[keys[-1]] = value
        
        self.logger.debug(f"设置配置项: {key} = {value}")
        return self
    
    def update(self, updates, prefix=None):
        """
        批量更新配置
        
        参数:
        updates: 要更新的配置字典
        prefix: 配置前缀
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        for key, value in updates.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                self.update(value, full_key)
            else:
                self.set(full_key, value)
        
        return self
    
    def validate(self, schema):
        """
        验证配置是否符合给定的模式
        
        参数:
        schema: 配置模式，定义必需的配置项及其类型
        
        返回:
        bool: 如果配置有效则返回True
        
        异常:
        ValueError: 如果配置无效
        """
        for key, expected_type in schema.items():
            try:
                value = self.get(key, required=True)
                
                # 检查类型
                if not isinstance(value, expected_type):
                    raise ValueError(
                        f"配置项 '{key}' 类型错误，期望 {expected_type.__name__}，得到 {type(value).__name__}")
                
                # 对于数值类型，可能需要额外的验证
                if isinstance(value, (int, float)):
                    # 可以在这里添加范围检查等
                    pass
                    
            except KeyError:
                raise ValueError(f"缺失必需的配置项: {key}")
        
        self.logger.info("配置验证通过")
        return True
    
    def get_all(self):
        """
        获取所有配置
        
        返回:
        config: 完整的配置字典
        """
        return self.config
    
    def merge(self, other_config):
        """
        合并另一个配置到当前配置
        
        参数:
        other_config: 另一个配置字典或ConfigManager实例
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        if isinstance(other_config, ConfigManager):
            other_config = other_config.get_all()
        
        def _merge_recursive(dict1, dict2):
            for key, value in dict2.items():
                if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                    _merge_recursive(dict1[key], value)
                else:
                    dict1[key] = value
            return dict1
        
        self.config = _merge_recursive(self.config, other_config)
        self.logger.info("配置合并完成")
        return self
    
    def get_data_config(self):
        """
        获取数据相关配置
        
        返回:
        data_config: 数据配置字典
        """
        return self.get('data', {})
    
    def get_feature_config(self):
        """
        获取特征提取相关配置
        
        返回:
        feature_config: 特征提取配置字典
        """
        return self.get('features', {})
    
    def get_model_config(self):
        """
        获取模型相关配置
        
        返回:
        model_config: 模型配置字典
        """
        return self.get('model', {})
    
    def get_training_config(self):
        """
        获取训练相关配置
        
        返回:
        training_config: 训练配置字典
        """
        return self.get('training', {})
    
    def get_evaluation_config(self):
        """
        获取评估相关配置
        
        返回:
        evaluation_config: 评估配置字典
        """
        return self.get('evaluation', {})
    
    def get_paths_config(self):
        """
        获取路径相关配置
        
        返回:
        paths_config: 路径配置字典
        """
        return self.get('paths', {})
    
    def resolve_paths(self, base_dir=None):
        """
        解析相对路径为绝对路径
        
        参数:
        base_dir: 基础目录，如果为None则使用配置文件所在目录
        
        返回:
        self: 配置管理器实例，支持链式调用
        """
        if base_dir is None and self.config_path:
            base_dir = os.path.dirname(self.config_path)
        
        if not base_dir:
            self.logger.warning("无法解析相对路径，没有基础目录")
            return self
        
        def _resolve_paths_recursive(config):
            if isinstance(config, dict):
                for key, value in config.items():
                    if isinstance(value, str) and key.endswith('_path'):
                        if not os.path.isabs(value):
                            config[key] = os.path.normpath(os.path.join(base_dir, value))
                    elif isinstance(value, (dict, list)):
                        _resolve_paths_recursive(value)
            elif isinstance(config, list):
                for i, item in enumerate(config):
                    if isinstance(item, (dict, list)):
                        _resolve_paths_recursive(item)
        
        _resolve_paths_recursive(self.config)
        self.logger.info("路径解析完成")
        return self
    
    def __str__(self):
        """
        返回配置的字符串表示
        
        返回:
        str: 配置的字符串表示
        """
        return yaml.dump(self.config, default_flow_style=False, allow_unicode=True)
    
    def __repr__(self):
        """
        返回配置管理器的字符串表示
        
        返回:
        str: 配置管理器的字符串表示
        """
        return f"ConfigManager(config_path={self.config_path})"


def load_config(config_path):
    """
    加载配置文件的便捷函数
    
    参数:
    config_path: 配置文件路径
    
    返回:
    ConfigManager: 配置管理器实例
    """
    return ConfigManager(config_path)


def create_default_config(config_path):
    """
    创建默认配置文件
    
    参数:
    config_path: 配置文件保存路径
    
    返回:
    ConfigManager: 配置管理器实例
    """
    # 默认配置模板
    default_config = {
        'paths': {
            'data_dir': './dev_data',
            'normal_data_path': './dev_data/normal',
            'anomaly_data_path': './dev_data/anomaly',
            'model_save_path': './models/saved_models',
            'features_save_path': './features/saved_features',
            'results_save_path': './results',
            'logs_save_path': './logs'
        },
        'data': {
            'sample_rate': 22050,
            'duration': 2.0,
            'mono': True,
            'normalize': True,
            'train_val_split': 0.8,
            'batch_size': 32
        },
        'features': {
            'extract_features': ['mfcc', 'melspectrogram', 'chroma', 'spectral_contrast', 'zcr', 'rms'],
            'n_mfcc': 13,
            'n_fft': 2048,
            'hop_length': 512,
            'n_mels': 128,
            'fmin': 0,
            'fmax': 8000,
            'win_length': 2048
        },
        'model': {
            'type': 'gmm',
            'n_components': 8,
            'covariance_type': 'full',
            'max_iter': 100,
            'n_init': 3,
            'tol': 0.001,
            'reg_covar': 1e-06
        },
        'training': {
            'cross_validation': True,
            'cv_folds': 5,
            'optimize_components': True,
            'component_range': [2, 4, 8, 16, 32],
            'random_state': 42
        },
        'evaluation': {
            'threshold_method': 'f1',
            'n_thresholds': 100,
            'target_metric': 'f1',
            'plot_metrics': True,
            'save_plots': True
        },
        'real_time': {
            'buffer_size': 2048,
            'hop_length': 512,
            'window_length': 2,
            'detection_threshold': 0.5,
            'consecutive_frames': 3
        }
    }
    
    # 创建配置管理器
    config_manager = ConfigManager()
    config_manager.config = default_config
    
    # 保存配置文件
    config_manager.save_config(config_path)
    
    return config_manager


def get_config_schema():
    """
    获取配置验证模式
    
    返回:
    schema: 配置验证模式字典
    """
    schema = {
        'paths.data_dir': str,
        'paths.normal_data_path': str,
        'paths.anomaly_data_path': str,
        'paths.model_save_path': str,
        'paths.features_save_path': str,
        'paths.results_save_path': str,
        'data.sample_rate': int,
        'data.duration': (int, float),
        'data.mono': bool,
        'data.normalize': bool,
        'data.train_val_split': (int, float),
        'features.extract_features': list,
        'features.n_mfcc': int,
        'features.n_fft': int,
        'features.hop_length': int,
        'features.n_mels': int,
        'model.type': str,
        'model.n_components': int,
        'model.covariance_type': str,
        'evaluation.threshold_method': str
    }
    
    return schema