"""
音频流处理器实现
使用 sounddevice 库实现实时音频输入和异常检测
"""

import numpy as np
import sounddevice as sd
import queue
import threading
import time
from typing import Optional, Callable
from utils.realtime_classifier import RealTimeClassifier


class SoundDeviceStreamHandler:
    """
    基于 sounddevice 的音频流处理器
    实现实时音频输入、异常检测和警告功能
    """

    def __init__(self,
                 classifier: RealTimeClassifier,
                 sample_rate: int = 22050,
                 channels: int = 1,
                 chunk_size: int = 1024,
                 device: Optional[int] = None,
                 alert_callback: Optional[Callable] = None,
                 consecutive_anomaly_threshold: int = 3):
        """
        初始化音频流处理器

        参数:
        classifier: 实时分类器实例
        sample_rate: 采样率
        channels: 通道数（1=单声道，2=立体声）
        chunk_size: 每次读取的样本数
        device: 音频输入设备ID（None表示使用默认设备）
        alert_callback: 警告回调函数，当检测到异常时调用
        consecutive_anomaly_threshold: 连续异常帧数阈值，超过此值触发警告
        """
        self.classifier = classifier
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.device = device
        self.alert_callback = alert_callback
        self.consecutive_anomaly_threshold = consecutive_anomaly_threshold

        self.stream = None
        self.running = False
        self.audio_queue = queue.Queue()

        # 统计信息
        self.total_frames = 0
        self.anomaly_frames = 0
        self.consecutive_anomalies = 0

        # 监听线程
        self.monitor_thread = None
        self.stop_event = threading.Event()

    def list_devices(self):
        """
        列出所有可用的音频设备
        """
        print("可用的音频设备:")
        print(sd.query_devices())

    def start_stream(self):
        """
        启动音频流
        """
        if self.running:
            print("音频流已在运行中")
            return

        try:
            # 启动分类器
            self.classifier.start()

            # 创建音频流
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='float32',
                blocksize=self.chunk_size,
                device=self.device,
                callback=self._audio_callback
            )

            # 启动流
            self.stream.start()
            self.running = True
            self.stop_event.clear()

            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            print(f"音频流已启动 - 采样率: {self.sample_rate} Hz, 通道数: {self.channels}")
            print(f"使用设备: {sd.query_devices(self.device, 'input')['name']}")

        except Exception as e:
            print(f"启动音频流失败: {e}")
            self.stop_stream()

    def stop_stream(self):
        """
        停止音频流
        """
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        # 停止音频流
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # 等待监控线程结束
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)

        # 停止分类器
        self.classifier.stop()

        # 清空队列
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        print("音频流已停止")
        self._print_statistics()

    def _audio_callback(self, indata, frames, time_info, status):
        """
        音频输入回调函数

        参数:
        indata: 输入音频数据
        frames: 帧数
        time_info: 时间信息
        status: 状态标志
        """
        if status:
            print(f"音频回调状态: {status}")

        # 将音频数据复制到numpy数组
        audio_data = indata.copy()

        # 如果是立体声，转换为单声道
        if self.channels > 1:
            audio_data = np.mean(audio_data, axis=1)
        else:
            audio_data = audio_data.flatten()

        # 将音频数据传递给分类器
        self.classifier.process_audio_data(audio_data)

        # 更新统计
        self.total_frames += 1

    def _monitor_loop(self):
        """
        监控循环，检查分类结果并触发警告
        """
        last_alert_time = 0
        alert_cooldown = 2.0  # 警告冷却时间（秒）

        while self.running and not self.stop_event.is_set():
            try:
                # 获取最新的分类结果
                result = self.classifier.get_latest_result(timeout=0.1)

                if result is not None:
                    is_anomaly = result['is_anomaly']
                    score = result['score']
                    confidence = result['confidence']
                    anomaly_count = result['anomaly_count']

                    if is_anomaly:
                        self.anomaly_frames += 1
                        self.consecutive_anomalies = anomaly_count

                        # 检查是否应该触发警告
                        current_time = time.time()
                        if (anomaly_count >= self.consecutive_anomaly_threshold and
                            current_time - last_alert_time > alert_cooldown):

                            # 触发警告
                            self._trigger_alert(result)
                            last_alert_time = current_time
                    else:
                        self.consecutive_anomalies = 0

                # 短暂休眠
                time.sleep(0.05)

            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(0.1)

    def _trigger_alert(self, result):
        """
        触发异常警告

        参数:
        result: 分类结果字典
        """
        # 打印警告信息
        print("\n" + "="*50)
        print("⚠️  检测到声音异常！")
        print(f"时间: {time.strftime('%H:%M:%S', time.localtime(result['timestamp']))}")
        print(f"异常分数: {result['score']:.4f}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"连续异常帧数: {result['anomaly_count']}")
        print("="*50 + "\n")

        # 如果提供了回调函数，调用它
        if self.alert_callback is not None:
            try:
                self.alert_callback(result)
            except Exception as e:
                print(f"警告回调函数执行失败: {e}")

    def _print_statistics(self):
        """
        打印统计信息
        """
        print("\n统计信息:")
        print(f"总处理帧数: {self.total_frames}")
        print(f"异常帧数: {self.anomaly_frames}")
        if self.total_frames > 0:
            anomaly_rate = (self.anomaly_frames / self.total_frames) * 100
            print(f"异常率: {anomaly_rate:.2f}%")

    def get_statistics(self):
        """
        获取统计信息

        返回:
        stats: 统计信息字典
        """
        return {
            'total_frames': self.total_frames,
            'anomaly_frames': self.anomaly_frames,
            'consecutive_anomalies': self.consecutive_anomalies,
            'anomaly_rate': (self.anomaly_frames / self.total_frames * 100) if self.total_frames > 0 else 0.0
        }


class FileStreamSimulator:
    """
    文件流模拟器
    用于从音频文件模拟实时音频流，方便测试
    """

    def __init__(self,
                 classifier: RealTimeClassifier,
                 audio_file: str,
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 alert_callback: Optional[Callable] = None,
                 consecutive_anomaly_threshold: int = 3,
                 playback_speed: float = 1.0):
        """
        初始化文件流模拟器

        参数:
        classifier: 实时分类器实例
        audio_file: 音频文件路径
        sample_rate: 目标采样率
        chunk_size: 每次处理的样本数
        alert_callback: 警告回调函数
        consecutive_anomaly_threshold: 连续异常帧数阈值
        playback_speed: 播放速度（1.0=正常速度）
        """
        self.classifier = classifier
        self.audio_file = audio_file
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.alert_callback = alert_callback
        self.consecutive_anomaly_threshold = consecutive_anomaly_threshold
        self.playback_speed = playback_speed

        self.running = False
        self.audio_data = None
        self.current_position = 0

        # 统计信息
        self.total_frames = 0
        self.anomaly_frames = 0
        self.consecutive_anomalies = 0

        # 线程
        self.playback_thread = None
        self.monitor_thread = None
        self.stop_event = threading.Event()

    def load_audio(self):
        """
        加载音频文件
        """
        try:
            import librosa
            # 加载音频文件
            audio, sr = librosa.load(self.audio_file, sr=self.sample_rate, mono=True)
            self.audio_data = audio
            print(f"音频文件已加载: {self.audio_file}")
            print(f"时长: {len(audio) / self.sample_rate:.2f} 秒")
            return True
        except Exception as e:
            print(f"加载音频文件失败: {e}")
            return False

    def start_stream(self):
        """
        开始模拟流
        """
        if self.running:
            print("模拟流已在运行中")
            return

        if self.audio_data is None:
            if not self.load_audio():
                return

        try:
            # 启动分类器
            self.classifier.start()

            self.running = True
            self.stop_event.clear()
            self.current_position = 0

            # 启动播放线程
            self.playback_thread = threading.Thread(target=self._playback_loop)
            self.playback_thread.daemon = True
            self.playback_thread.start()

            # 启动监控线程
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            print(f"开始模拟音频流 - 文件: {self.audio_file}")
            print(f"播放速度: {self.playback_speed}x")

        except Exception as e:
            print(f"启动模拟流失败: {e}")
            self.stop_stream()

    def stop_stream(self):
        """
        停止模拟流
        """
        if not self.running:
            return

        self.running = False
        self.stop_event.set()

        # 等待线程结束
        if self.playback_thread is not None:
            self.playback_thread.join(timeout=2.0)
        if self.monitor_thread is not None:
            self.monitor_thread.join(timeout=2.0)

        # 停止分类器
        self.classifier.stop()

        print("模拟流已停止")
        self._print_statistics()

    def _playback_loop(self):
        """
        播放循环
        """
        while self.running and not self.stop_event.is_set():
            # 检查是否到达文件末尾
            if self.current_position >= len(self.audio_data):
                print("音频文件播放完毕")
                break

            # 获取当前块
            end_position = min(self.current_position + self.chunk_size, len(self.audio_data))
            chunk = self.audio_data[self.current_position:end_position]

            # 将数据传递给分类器
            self.classifier.process_audio_data(chunk)

            # 更新位置
            self.current_position = end_position
            self.total_frames += 1

            # 模拟实时播放（根据播放速度调整）
            sleep_time = (self.chunk_size / self.sample_rate) / self.playback_speed
            time.sleep(sleep_time)

        # 播放结束后自动停止
        if self.running:
            self.stop_stream()

    def _monitor_loop(self):
        """
        监控循环
        """
        last_alert_time = 0
        alert_cooldown = 1.0

        while self.running and not self.stop_event.is_set():
            try:
                result = self.classifier.get_latest_result(timeout=0.1)

                if result is not None:
                    is_anomaly = result['is_anomaly']
                    anomaly_count = result['anomaly_count']

                    if is_anomaly:
                        self.anomaly_frames += 1
                        self.consecutive_anomalies = anomaly_count

                        current_time = time.time()
                        if (anomaly_count >= self.consecutive_anomaly_threshold and
                            current_time - last_alert_time > alert_cooldown):

                            self._trigger_alert(result)
                            last_alert_time = current_time
                    else:
                        self.consecutive_anomalies = 0

                time.sleep(0.05)

            except Exception as e:
                print(f"监控循环错误: {e}")
                time.sleep(0.1)

    def _trigger_alert(self, result):
        """
        触发警告
        """
        # 计算当前播放时间
        current_time_in_audio = self.current_position / self.sample_rate

        print("\n" + "="*50)
        print("⚠️  检测到声音异常！")
        print(f"播放时间: {current_time_in_audio:.2f} 秒")
        print(f"异常分数: {result['score']:.4f}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"连续异常帧数: {result['anomaly_count']}")
        print("="*50 + "\n")

        if self.alert_callback is not None:
            try:
                result['playback_time'] = current_time_in_audio
                self.alert_callback(result)
            except Exception as e:
                print(f"警告回调函数执行失败: {e}")

    def _print_statistics(self):
        """
        打印统计信息
        """
        print("\n统计信息:")
        print(f"总处理帧数: {self.total_frames}")
        print(f"异常帧数: {self.anomaly_frames}")
        if self.total_frames > 0:
            anomaly_rate = (self.anomaly_frames / self.total_frames) * 100
            print(f"异常率: {anomaly_rate:.2f}%")
