#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时音频异常检测 - 可视化版本

此脚本提供带图形界面的实时音频流异常检测：
1. 实时波形显示
2. 异常检测状态可视化
3. 统计图表
"""

import os
import sys
import argparse
import time
import threading
import numpy as np
from collections import deque

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle

from utils.config_manager import load_config
from utils.feature_extractor_wrapper import FeatureExtractorWrapper
from utils.realtime_classifier import RealTimeClassifier
from utils.audio_stream_handler import SoundDeviceStreamHandler, FileStreamSimulator
from models.gmm_model import GMMModel
from models.autoencoder import AudioAutoencoder
from models.threshold_detector import ThresholdDetector


class VisualizationWindow:
    """
    可视化窗口类
    """

    def __init__(self, classifier, max_history=500):
        """
        初始化可视化窗口

        参数:
        classifier: 实时分类器
        max_history: 历史数据最大长度
        """
        self.classifier = classifier
        self.max_history = max_history

        # 数据缓冲区
        self.time_points = deque(maxlen=max_history)
        self.scores = deque(maxlen=max_history)
        self.predictions = deque(maxlen=max_history)
        self.audio_buffer = deque(maxlen=2048)

        self.start_time = time.time()
        self.anomaly_count = 0
        self.normal_count = 0
        self.current_status = "正常"

        # 创建图形界面
        self.setup_figure()

    def setup_figure(self):
        """
        设置图形界面
        """
        plt.style.use('dark_background')
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.suptitle('实时音频异常检测监控', fontsize=16, fontweight='bold')

        # 创建子图
        gs = self.fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. 音频波形图
        self.ax_waveform = self.fig.add_subplot(gs[0, :])
        self.ax_waveform.set_title('实时音频波形')
        self.ax_waveform.set_xlabel('样本')
        self.ax_waveform.set_ylabel('幅度')
        self.ax_waveform.set_ylim(-1, 1)
        self.line_waveform, = self.ax_waveform.plot([], [], 'cyan', linewidth=0.5)

        # 2. 异常分数曲线
        self.ax_score = self.fig.add_subplot(gs[1, :])
        self.ax_score.set_title('异常分数时间曲线')
        self.ax_score.set_xlabel('时间 (秒)')
        self.ax_score.set_ylabel('异常分数')
        self.line_score, = self.ax_score.plot([], [], 'lime', linewidth=2, label='异常分数')
        self.ax_score.axhline(y=0, color='white', linestyle='--', alpha=0.3)
        self.ax_score.legend()

        # 3. 状态指示器
        self.ax_status = self.fig.add_subplot(gs[2, 0])
        self.ax_status.set_title('当前状态')
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.axis('off')
        self.status_rect = Rectangle((0.1, 0.3), 0.8, 0.4,
                                      facecolor='green', edgecolor='white', linewidth=2)
        self.ax_status.add_patch(self.status_rect)
        self.status_text = self.ax_status.text(0.5, 0.5, '正常',
                                               ha='center', va='center',
                                               fontsize=20, fontweight='bold', color='white')

        # 4. 统计信息
        self.ax_stats = self.fig.add_subplot(gs[2, 1])
        self.ax_stats.set_title('统计信息')
        self.ax_stats.set_xlim(0, 1)
        self.ax_stats.set_ylim(0, 1)
        self.ax_stats.axis('off')
        self.stats_text = self.ax_stats.text(0.1, 0.5, '',
                                             fontsize=10, va='center',
                                             family='monospace')

    def update_data(self):
        """
        更新数据
        """
        # 获取最新的分类结果
        result = self.classifier.get_latest_result(timeout=0.01)

        if result is not None:
            current_time = time.time() - self.start_time
            self.time_points.append(current_time)
            self.scores.append(result['score'])
            self.predictions.append(result['prediction'])

            # 更新计数
            if result['is_anomaly']:
                self.anomaly_count += 1
                self.current_status = "异常"
            else:
                self.normal_count += 1
                self.current_status = "正常"

        # 获取音频缓冲区（模拟）
        buffer = self.classifier.audio_buffer.get_buffer()
        if len(buffer) > 0:
            # 取最后2048个样本
            self.audio_buffer = deque(buffer[-2048:], maxlen=2048)

    def update_plot(self, frame):
        """
        更新图表
        """
        self.update_data()

        # 更新波形
        if len(self.audio_buffer) > 0:
            self.line_waveform.set_data(range(len(self.audio_buffer)),
                                        list(self.audio_buffer))
            self.ax_waveform.set_xlim(0, len(self.audio_buffer))

        # 更新分数曲线
        if len(self.time_points) > 0:
            self.line_score.set_data(list(self.time_points), list(self.scores))
            self.ax_score.set_xlim(max(0, list(self.time_points)[-1] - 30),
                                    list(self.time_points)[-1] + 1)

            # 动态调整Y轴范围
            if len(self.scores) > 0:
                score_min = min(self.scores)
                score_max = max(self.scores)
                margin = (score_max - score_min) * 0.1 + 0.1
                self.ax_score.set_ylim(score_min - margin, score_max + margin)

        # 更新状态指示器
        if self.current_status == "异常":
            self.status_rect.set_facecolor('red')
            self.status_text.set_text('⚠️ 异常')
        else:
            self.status_rect.set_facecolor('green')
            self.status_text.set_text('✓ 正常')

        # 更新统计信息
        total = self.anomaly_count + self.normal_count
        anomaly_rate = (self.anomaly_count / total * 100) if total > 0 else 0
        stats_str = f"""
运行时间: {time.time() - self.start_time:.1f} 秒

总帧数: {total}
正常帧数: {self.normal_count}
异常帧数: {self.anomaly_count}
异常率: {anomaly_rate:.2f}%
        """.strip()
        self.stats_text.set_text(stats_str)

        return [self.line_waveform, self.line_score, self.status_rect,
                self.status_text, self.stats_text]

    def show(self):
        """
        显示窗口
        """
        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig, self.update_plot,
            interval=50,  # 20 FPS
            blit=True,
            cache_frame_data=False
        )

        plt.show()


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='实时音频异常检测系统（可视化版本）')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--threshold', type=float,
                        help='分类阈值')
    parser.add_argument('--mode', type=str, default='microphone',
                        choices=['microphone', 'file'],
                        help='输入模式')
    parser.add_argument('--audio_file', type=str,
                        help='音频文件路径（file模式必需）')
    parser.add_argument('--device', type=int,
                        help='音频输入设备ID')
    parser.add_argument('--consecutive_threshold', type=int, default=3,
                        help='连续异常帧数阈值')
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help='音频块大小')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                        help='文件播放速度')
    return parser.parse_args()


def load_model(model_path, config):
    """
    加载模型
    """
    print(f"正在加载模型: {model_path}")

    try:
        if 'autoencoder' in model_path.lower() or config.get('model.type') == 'autoencoder':
            model = AudioAutoencoder.load(model_path)
            print("已加载自动编码器模型")
            threshold = model.threshold if hasattr(model, 'threshold') else None
        else:
            model = GMMModel.load(model_path)
            print("已加载GMM模型")
            threshold = None

        return model, threshold

    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)


def main():
    """
    主函数
    """
    args = parse_arguments()

    print("=" * 60)
    print("实时音频异常检测系统 - 可视化版本")
    print("=" * 60)

    # 检查参数
    if args.mode == 'file' and not args.audio_file:
        print("错误: file 模式需要指定 --audio_file 参数")
        sys.exit(1)

    # 加载配置
    try:
        config = load_config(args.config)
        print(f"已加载配置文件: {args.config}")
    except FileNotFoundError:
        print(f"错误: 配置文件不存在: {args.config}")
        sys.exit(1)

    # 加载模型
    model, model_threshold = load_model(args.model, config)

    # 确定阈值
    threshold = args.threshold if args.threshold is not None else (model_threshold or 0.0)
    print(f"使用阈值: {threshold}")

    # 创建特征提取器
    print("初始化特征提取器...")
    feature_extractor = FeatureExtractorWrapper(config)

    # 创建阈值检测器
    threshold_detector = None
    if not hasattr(model, 'calculate_reconstruction_error'):
        threshold_detector = ThresholdDetector(model)

    # 创建实时分类器
    print("初始化实时分类器...")
    sample_rate = config.get('data.sample_rate', 22050)
    classifier = RealTimeClassifier(
        model=model,
        feature_extractor=feature_extractor,
        threshold_detector=threshold_detector,
        sample_rate=sample_rate,
        frame_size=args.chunk_size,
        hop_length=args.chunk_size // 2,
        threshold=threshold
    )

    # 创建音频流处理器
    print(f"初始化音频流处理器 (模式: {args.mode})...")

    if args.mode == 'microphone':
        stream_handler = SoundDeviceStreamHandler(
            classifier=classifier,
            sample_rate=sample_rate,
            channels=1,
            chunk_size=args.chunk_size,
            device=args.device,
            consecutive_anomaly_threshold=args.consecutive_threshold
        )
    else:
        stream_handler = FileStreamSimulator(
            classifier=classifier,
            audio_file=args.audio_file,
            sample_rate=sample_rate,
            chunk_size=args.chunk_size,
            consecutive_anomaly_threshold=args.consecutive_threshold,
            playback_speed=args.playback_speed
        )

    # 启动音频流
    print("\n开始实时检测...")
    print("关闭窗口以停止检测\n")

    try:
        # 启动流
        stream_handler.start_stream()

        # 等待一小段时间让数据开始流入
        time.sleep(1.0)

        # 创建并显示可视化窗口
        viz = VisualizationWindow(classifier)
        viz.show()

    except KeyboardInterrupt:
        print("\n收到中断信号...")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 停止流
        print("正在停止音频流...")
        stream_handler.stop_stream()

        # 显示最终统计
        if hasattr(stream_handler, 'get_statistics'):
            stats = stream_handler.get_statistics()
            print("\n" + "=" * 60)
            print("检测会话统计:")
            print(f"总处理帧数: {stats['total_frames']}")
            print(f"异常帧数: {stats['anomaly_frames']}")
            print(f"异常率: {stats['anomaly_rate']:.2f}%")
            print("=" * 60)

    print("\n实时检测已结束")


if __name__ == "__main__":
    main()
