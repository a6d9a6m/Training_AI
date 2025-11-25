#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时音频异常检测主程序

此脚本提供实时音频流异常检测功能：
1. 支持麦克风实时输入
2. 支持音频文件模拟流
3. 实时检测并警告异常
"""

import os
import sys
import argparse
import time
import threading

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config_manager import load_config
from utils.feature_extractor_wrapper import FeatureExtractorWrapper
from utils.realtime_classifier import RealTimeClassifier
from utils.audio_stream_handler import SoundDeviceStreamHandler, FileStreamSimulator
from models.gmm_model import GMMModel
from models.autoencoder import AudioAutoencoder
from models.threshold_detector import ThresholdDetector


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='实时音频异常检测系统')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--model', type=str, required=True,
                        help='训练好的模型路径')
    parser.add_argument('--threshold', type=float,
                        help='分类阈值（可选，如果模型已保存阈值则使用模型的）')
    parser.add_argument('--mode', type=str, default='microphone',
                        choices=['microphone', 'file'],
                        help='输入模式：microphone（麦克风）或 file（文件模拟）')
    parser.add_argument('--audio_file', type=str,
                        help='音频文件路径（file模式必需）')
    parser.add_argument('--device', type=int,
                        help='音频输入设备ID（可选）')
    parser.add_argument('--list_devices', action='store_true',
                        help='列出所有可用的音频设备')
    parser.add_argument('--duration', type=int, default=0,
                        help='运行时长（秒），0表示手动停止')
    parser.add_argument('--consecutive_threshold', type=int, default=3,
                        help='触发警告所需的连续异常帧数')
    parser.add_argument('--chunk_size', type=int, default=2048,
                        help='音频块大小')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                        help='文件播放速度（仅file模式，1.0=正常速度）')
    return parser.parse_args()


def load_model(model_path, config):
    """
    加载训练好的模型

    参数:
    model_path: 模型文件路径
    config: 配置管理器

    返回:
    model: 加载的模型
    threshold: 模型的阈值（如果有）
    """
    print(f"正在加载模型: {model_path}")

    try:
        # 检测模型类型
        if 'autoencoder' in model_path.lower() or config.get('model.type') == 'autoencoder':
            # 加载自动编码器模型
            model = AudioAutoencoder.load(model_path)
            print("已加载自动编码器模型")

            # 获取阈值
            threshold = model.threshold if hasattr(model, 'threshold') else None
        else:
            # 加载GMM模型
            model = GMMModel.load(model_path)
            print("已加载GMM模型")
            threshold = None  # GMM模型需要单独指定阈值

        return model, threshold

    except Exception as e:
        print(f"加载模型失败: {e}")
        sys.exit(1)


def custom_alert_callback(result):
    """
    自定义警告回调函数
    可以在这里添加自定义的警告处理逻辑，例如：
    - 发送邮件通知
    - 记录到日志
    - 触发其他系统动作
    - 播放警告声音
    """
    # 示例：保存异常日志
    log_file = "anomaly_log.txt"
    with open(log_file, 'a', encoding='utf-8') as f:
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(result['timestamp']))
        f.write(f"{timestamp} - 异常分数: {result['score']:.4f}, 置信度: {result['confidence']:.4f}\n")


def main():
    """
    主函数
    """
    args = parse_arguments()

    print("=" * 60)
    print("实时音频异常检测系统")
    print("=" * 60)

    # 如果只是列出设备，直接显示并退出
    if args.list_devices:
        import sounddevice as sd
        print("\n可用的音频设备:")
        print(sd.query_devices())
        return

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
    if args.threshold is not None:
        threshold = args.threshold
        print(f"使用命令行指定的阈值: {threshold}")
    elif model_threshold is not None:
        threshold = model_threshold
        print(f"使用模型保存的阈值: {threshold}")
    else:
        threshold = 0.0
        print(f"使用默认阈值: {threshold}")

    # 创建特征提取器
    print("初始化特征提取器...")
    feature_extractor = FeatureExtractorWrapper(config)

    # 创建阈值检测器（如果是GMM模型）
    threshold_detector = None
    if not hasattr(model, 'calculate_reconstruction_error'):
        threshold_detector = ThresholdDetector(model)
        print("已创建阈值检测器（GMM模型）")

    # 创建实时分类器
    print("初始化实时分类器...")
    sample_rate = config.get('data.sample_rate', 22050)
    classifier = RealTimeClassifier(
        model=model,
        feature_extractor=feature_extractor,
        threshold_detector=threshold_detector,
        sample_rate=sample_rate,
        frame_size=args.chunk_size,
        hop_length=args.chunk_size // 2,  # 50% 重叠
        threshold=threshold
    )

    # 创建音频流处理器
    print(f"初始化音频流处理器 (模式: {args.mode})...")
    stream_handler = None

    if args.mode == 'microphone':
        # 麦克风模式
        stream_handler = SoundDeviceStreamHandler(
            classifier=classifier,
            sample_rate=sample_rate,
            channels=1,
            chunk_size=args.chunk_size,
            device=args.device,
            alert_callback=custom_alert_callback,
            consecutive_anomaly_threshold=args.consecutive_threshold
        )
    else:
        # 文件模拟模式
        stream_handler = FileStreamSimulator(
            classifier=classifier,
            audio_file=args.audio_file,
            sample_rate=sample_rate,
            chunk_size=args.chunk_size,
            alert_callback=custom_alert_callback,
            consecutive_anomaly_threshold=args.consecutive_threshold,
            playback_speed=args.playback_speed
        )

    # 启动音频流
    print("\n" + "=" * 60)
    print("开始实时检测...")
    print("按 Ctrl+C 停止检测")
    print("=" * 60 + "\n")

    try:
        stream_handler.start_stream()

        # 如果指定了运行时长
        if args.duration > 0:
            print(f"将运行 {args.duration} 秒...")
            time.sleep(args.duration)
        else:
            # 等待用户中断
            print("正在监听音频流...")
            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n收到中断信号，正在停止...")
    except Exception as e:
        print(f"\n发生错误: {e}")
    finally:
        # 停止流
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
