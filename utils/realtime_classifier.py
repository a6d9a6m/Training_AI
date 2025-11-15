import numpy as np
import queue
import threading
import time
from typing import Optional, Callable, List, Dict, Any


class AudioBuffer:
    """
    音频缓冲区，用于存储和管理音频数据流
    """
    
    def __init__(self, buffer_size: int = 44100):
        """
        初始化音频缓冲区
        
        参数:
        buffer_size: 缓冲区大小（样本数）
        """
        self.buffer_size = buffer_size
        self.buffer = np.zeros(buffer_size, dtype=np.float32)
        self.lock = threading.Lock()
        self.buffer_full = False
    
    def add_samples(self, samples: np.ndarray):
        """
        添加音频样本到缓冲区
        
        参数:
        samples: 音频样本数组
        """
        with self.lock:
            # 如果缓冲区未满，直接添加
            if not self.buffer_full:
                available_space = self.buffer_size - len(self.buffer[self.buffer != 0])
                if len(samples) <= available_space:
                    self.buffer[len(self.buffer[self.buffer != 0]):len(self.buffer[self.buffer != 0]) + len(samples)] = samples
                    if len(self.buffer[self.buffer != 0]) >= self.buffer_size:
                        self.buffer_full = True
                else:
                    # 缓冲区空间不足，覆盖旧数据
                    self.buffer = np.roll(self.buffer, -available_space)
                    self.buffer[-len(samples):] = samples[-available_space:]
                    self.buffer_full = True
            else:
                # 缓冲区已满，滚动缓冲区并添加新数据
                self.buffer = np.roll(self.buffer, -len(samples))
                self.buffer[-len(samples):] = samples
    
    def get_buffer(self) -> np.ndarray:
        """
        获取完整的缓冲区数据
        
        返回:
        buffer: 缓冲区数据
        """
        with self.lock:
            return self.buffer.copy()
    
    def clear(self):
        """
        清空缓冲区
        """
        with self.lock:
            self.buffer.fill(0)
            self.buffer_full = False


class FrameProcessor:
    """
    帧处理器，用于从音频流中提取帧并进行处理
    """
    
    def __init__(self, frame_size: int, hop_length: int):
        """
        初始化帧处理器
        
        参数:
        frame_size: 帧大小
        hop_length: 帧移
        """
        self.frame_size = frame_size
        self.hop_length = hop_length
        self.current_position = 0
        self.overlap_buffer = None
    
    def process_audio(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """
        处理音频数据，提取帧
        
        参数:
        audio_data: 音频数据
        
        返回:
        frames: 帧列表
        """
        frames = []
        
        # 如果有重叠缓冲区，先与新数据合并
        if self.overlap_buffer is not None:
            audio_data = np.concatenate([self.overlap_buffer, audio_data])
            self.overlap_buffer = None
        
        # 处理每个帧
        while self.current_position + self.frame_size <= len(audio_data):
            frame = audio_data[self.current_position:self.current_position + self.frame_size]
            frames.append(frame)
            self.current_position += self.hop_length
        
        # 保存重叠部分
        if self.current_position < len(audio_data):
            self.overlap_buffer = audio_data[self.current_position:]
            self.current_position = 0
        
        return frames
    
    def reset(self):
        """
        重置帧处理器
        """
        self.current_position = 0
        self.overlap_buffer = None


class RealTimeClassifier:
    """
    实时分类器接口，用于实时音频分类
    
    注：这是一个预留接口，实际的实时音频输入需要额外的音频库支持
    如PyAudio、sounddevice等
    """
    
    def __init__(self, 
                 model: Any,
                 feature_extractor: Any,
                 threshold_detector: Any = None,
                 sample_rate: int = 22050,
                 frame_size: int = 2048,
                 hop_length: int = 512,
                 buffer_size: Optional[int] = None,
                 threshold: Optional[float] = None):
        """
        初始化实时分类器
        
        参数:
        model: 分类模型，需要支持预测功能
        feature_extractor: 特征提取器，需要支持特征提取功能
        threshold_detector: 阈值检测器（可选）
        sample_rate: 采样率
        frame_size: 帧大小
        hop_length: 帧移
        buffer_size: 缓冲区大小（如果为None，则使用sample_rate * 2，即2秒）
        threshold: 分类阈值
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold_detector = threshold_detector
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_length = hop_length
        
        if buffer_size is None:
            buffer_size = sample_rate * 2  # 默认2秒的缓冲区
        
        self.buffer_size = buffer_size
        self.threshold = threshold
        
        # 初始化缓冲区和帧处理器
        self.audio_buffer = AudioBuffer(buffer_size)
        self.frame_processor = FrameProcessor(frame_size, hop_length)
        
        # 初始化分类结果队列
        self.result_queue = queue.Queue()
        
        # 初始化状态
        self.running = False
        self.processing_thread = None
        self.stop_event = threading.Event()
        
        # 分类计数，用于连续检测
        self.anomaly_count = 0
        self.normal_count = 0
    
    def start(self):
        """
        启动实时分类器
        
        注：实际使用时，需要单独的线程或回调来获取音频数据并调用process_audio_data方法
        """
        if self.running:
            print("分类器已经在运行中")
            return
        
        self.running = True
        self.stop_event.clear()
        self.anomaly_count = 0
        self.normal_count = 0
        
        # 启动处理线程
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print(f"实时分类器已启动 - 采样率: {self.sample_rate}, 帧大小: {self.frame_size}, 帧移: {self.hop_length}")
    
    def stop(self):
        """
        停止实时分类器
        """
        if not self.running:
            print("分类器未运行")
            return
        
        self.running = False
        self.stop_event.set()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        self.audio_buffer.clear()
        self.frame_processor.reset()
        
        print("实时分类器已停止")
    
    def process_audio_data(self, audio_data: np.ndarray):
        """
        处理音频数据（应由音频输入回调调用）
        
        参数:
        audio_data: 音频数据（归一化到[-1, 1]范围）
        """
        if not self.running:
            return
        
        # 添加到缓冲区
        self.audio_buffer.add_samples(audio_data)
    
    def _processing_loop(self):
        """
        处理循环，定期从缓冲区提取特征并进行分类
        """
        while self.running and not self.stop_event.is_set():
            try:
                # 从缓冲区获取数据
                buffer_data = self.audio_buffer.get_buffer()
                
                # 处理帧
                frames = self.frame_processor.process_audio(buffer_data)
                
                for frame in frames:
                    # 提取特征
                    features = self._extract_features(frame)
                    
                    if features is not None:
                        # 进行分类
                        prediction, score = self._classify(features)
                        
                        # 发布结果
                        self._publish_result(prediction, score)
                
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.01)
                
            except Exception as e:
                print(f"处理循环错误: {e}")
                # 短暂休眠后继续
                time.sleep(0.1)
    
    def _extract_features(self, audio_data: np.ndarray) -> Optional[np.ndarray]:
        """
        从音频数据中提取特征
        
        参数:
        audio_data: 音频数据
        
        返回:
        features: 提取的特征
        """
        try:
            # 这里需要根据实际的特征提取器进行调整
            # 假设feature_extractor有extract_features方法
            features = self.feature_extractor.extract_features(audio_data, self.sample_rate)
            return features
        except Exception as e:
            print(f"特征提取错误: {e}")
            return None
    
    def _classify(self, features: np.ndarray) -> tuple:
        """
        使用模型进行分类
        
        参数:
        features: 音频特征
        
        返回:
        prediction: 预测结果（0: 正常, 1: 异常）
        score: 分类分数
        """
        try:
            if self.threshold_detector is not None:
                # 使用阈值检测器进行分类
                prediction, score = self.threshold_detector.apply_threshold(
                    np.array([features]), self.threshold
                )
                return prediction[0], score[0]
            else:
                # 直接使用模型进行分类
                prediction = self.model.predict(np.array([features]))[0]
                
                # 尝试获取分数
                if hasattr(self.model, 'get_class_likelihood'):
                    normal_likelihood = self.model.get_class_likelihood(
                        np.array([features]), 0
                    )[0]
                    anomaly_likelihood = self.model.get_class_likelihood(
                        np.array([features]), 1
                    )[0]
                    score = anomaly_likelihood - normal_likelihood
                else:
                    score = 0.0
                
                return prediction, score
        except Exception as e:
            print(f"分类错误: {e}")
            return 0, 0.0
    
    def _publish_result(self, prediction: int, score: float):
        """
        发布分类结果
        
        参数:
        prediction: 预测结果
        score: 分类分数
        """
        # 更新计数
        if prediction == 1:  # 异常
            self.anomaly_count += 1
            self.normal_count = 0
        else:  # 正常
            self.normal_count += 1
            self.anomaly_count = 0
        
        # 创建结果对象
        result = {
            'timestamp': time.time(),
            'prediction': prediction,
            'score': score,
            'anomaly_count': self.anomaly_count,
            'normal_count': self.normal_count,
            'is_anomaly': prediction == 1,
            'confidence': abs(score)
        }
        
        # 将结果放入队列
        try:
            self.result_queue.put(result, block=False)
        except queue.Full:
            # 如果队列已满，移除最旧的结果
            try:
                self.result_queue.get_nowait()
                self.result_queue.put(result, block=False)
            except:
                pass
    
    def get_latest_result(self, timeout: float = 0.1) -> Optional[Dict[str, Any]]:
        """
        获取最新的分类结果
        
        参数:
        timeout: 超时时间（秒）
        
        返回:
        result: 最新的分类结果，如果没有则返回None
        """
        try:
            # 获取队列中的所有结果，返回最新的
            results = []
            while not self.result_queue.empty():
                try:
                    results.append(self.result_queue.get_nowait())
                except queue.Empty:
                    break
            
            # 将结果放回队列（可选）
            # for result in results:
            #     try:
            #         self.result_queue.put(result, block=False)
            #     except queue.Full:
            #         break
            
            return results[-1] if results else None
            
        except Exception as e:
            print(f"获取结果错误: {e}")
            return None
    
    def set_threshold(self, threshold: float):
        """
        设置分类阈值
        
        参数:
        threshold: 新的阈值
        """
        self.threshold = threshold
        print(f"阈值已更新为: {threshold}")


class AudioStreamHandler:
    """
    音频流处理器接口（预留）
    
    实际使用时需要实现具体的音频输入输出功能
    """
    
    def __init__(self, 
                 classifier: RealTimeClassifier,
                 sample_rate: int = 22050,
                 channels: int = 1,
                 chunk_size: int = 1024):
        """
        初始化音频流处理器
        
        参数:
        classifier: 实时分类器
        sample_rate: 采样率
        channels: 通道数
        chunk_size: 块大小
        """
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.stream = None
        self.running = False
    
    def start_stream(self):
        """
        启动音频流
        
        注：需要实现具体的音频库调用
        """
        print("启动音频流（预留接口，需要实现具体的音频库调用）")
        print("建议使用PyAudio、sounddevice等库来实现实际的音频输入")
        
        # 实际实现示例（使用PyAudio）:
        # import pyaudio
        # self.pa = pyaudio.PyAudio()
        # self.stream = self.pa.open(
        #     format=pyaudio.paFloat32,
        #     channels=self.channels,
        #     rate=self.sample_rate,
        #     input=True,
        #     frames_per_buffer=self.chunk_size,
        #     stream_callback=self._audio_callback
        # )
        # self.stream.start_stream()
        # self.running = True
    
    def stop_stream(self):
        """
        停止音频流
        """
        print("停止音频流（预留接口）")
        
        # 实际实现示例:
        # if self.stream is not None:
        #     self.stream.stop_stream()
        #     self.stream.close()
        # if hasattr(self, 'pa'):
        #     self.pa.terminate()
        # self.running = False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """
        音频回调函数（预留）
        
        注：这是PyAudio的回调函数格式，其他库可能有所不同
        """
        # 将字节数据转换为numpy数组
        # audio_data = np.frombuffer(in_data, dtype=np.float32)
        # 
        # # 处理音频数据
        # self.classifier.process_audio_data(audio_data)
        # 
        # # 继续流
        # return (in_data, pyaudio.paContinue)
        return None


def create_realtime_classifier(model, feature_extractor, threshold_detector=None, config=None):
    """
    创建实时分类器的便捷函数
    
    参数:
    model: 分类模型
    feature_extractor: 特征提取器
    threshold_detector: 阈值检测器
    config: 配置字典或ConfigManager实例
    
    返回:
    classifier: RealTimeClassifier实例
    """
    # 默认配置
    default_config = {
        'sample_rate': 22050,
        'frame_size': 2048,
        'hop_length': 512,
        'buffer_size': 44100,
        'threshold': None
    }
    
    # 如果提供了配置，更新默认配置
    if config is not None:
        if hasattr(config, 'get'):
            # ConfigManager实例
            rt_config = config.get('real_time', {})
            default_config.update({
                'sample_rate': config.get('data.sample_rate', 22050),
                'frame_size': rt_config.get('buffer_size', 2048),
                'hop_length': rt_config.get('hop_length', 512),
                'threshold': rt_config.get('detection_threshold', None)
            })
        else:
            # 字典
            rt_config = config.get('real_time', {})
            default_config.update(rt_config)
    
    # 创建并返回分类器
    classifier = RealTimeClassifier(
        model=model,
        feature_extractor=feature_extractor,
        threshold_detector=threshold_detector,
        **default_config
    )
    
    return classifier