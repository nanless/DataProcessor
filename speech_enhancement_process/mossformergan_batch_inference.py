#!/usr/bin/env python3
"""
MossFormerGAN批量语音增强处理器

该脚本提供了一个基于ClearVoice MossFormerGAN的批量语音增强解决方案，
支持多GPU并行处理和优化的批处理功能。

主要功能:
- 批量处理多种音频格式 (wav, mp3, flac, m4a, mp4)
- 多GPU并行处理
- 内存优化和异步I/O
- 支持隔离GPU处理模式
- 详细的性能统计和诊断
- 多通道音频自动转换为单声道（取第一个通道）
- 自动转换到模型采样率并保存（16kHz，使用自定义保存方法）
- 智能文件名处理（支持长文件名和特殊字符清理，空格替换为下划线）
- 正确处理ClearVoice返回的2D音频数组格式

依赖要求:
- 处理MP4和M4A文件需要安装: pip install pydub
- 输出采样率: 16kHz (使用AudioProcessor自定义保存，不依赖ClearVoice的write方法)
- 自动处理长文件名和特殊字符，避免保存错误
- 修复ClearVoice返回2D数组(1, N)导致的"Format not recognised"错误
- ClearVoice使用文件进文件出模式，通过临时文件传递音频数据

作者: AI Assistant
版本: 2.5
"""

import os
import multiprocessing as mp
import time
import tempfile
import logging
import random
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# 设置multiprocessing启动方法为spawn（CUDA多进程必需）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，会抛出RuntimeError，可以忽略
    pass

# 设置pytorch线程数
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 尝试导入ClearVoice
try:
    from clearvoice import ClearVoice
    CLEARVOICE_AVAILABLE = True
except ImportError as e:
    logger.error(f"ClearVoice未安装成功: {e}")
    logger.error("请执行: pip install clearvoice")
    CLEARVOICE_AVAILABLE = False


@dataclass
class MossFormerGANConfig:
    """MossFormerGAN配置类"""
    
    # 路径配置
    input_dir: str = "/root/group-shared/voiceprint/data/speech/speaker_verification/aidatatang_200zh"
    output_dir: str = "/root/group-shared/voiceprint/data/speech/speaker_verification/aidatatang_200zh_mossformergan_enhanced"
    
    # 模型配置
    model_name: str = "MossFormerGAN_SE_16K"
    task: str = 'speech_enhancement'
    
    # 硬件配置
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    
    # 音频配置
    target_sr: int = 16000  # 模型的采样率（输出音频将以此采样率保存）
    skip_existing: bool = True  # 跳过已存在的文件
    
    # FFT下采样配置
    use_fft_downsample: bool = True  # 是否使用FFT下采样
    fft_downsample_quality: str = "high"  # FFT下采样质量: "high", "medium", "low"
    anti_alias_filter: bool = True  # 是否使用抗混叠滤波器
    
    # 性能优化参数
    batch_size: int = 32  # 批处理大小
    num_workers: int = 8  # 并行工作线程数
    use_parallel: bool = False  # 使用并行处理
    use_multi_gpu: bool = True  # 是否使用多GPU
    use_multiprocess: bool = True  # 是否使用多进程
    batch_processing_mode: str = 'isolated_gpu'  # 批处理模式
    
    # 大文件处理和错误恢复参数
    max_file_size_mb: float = 200.0  # 单文件最大大小限制(MB)
    max_audio_duration_minutes: float = 120.0  # 单文件最大时长限制(分钟)
    enable_process_recovery: bool = True  # 启用进程崩溃恢复
    max_retry_attempts: int = 2  # 最大重试次数
    process_timeout_minutes: int = 600  # 单个进程超时时间(分钟)
    skip_large_files: bool = True  # 跳过超大文件
    
    # 异步处理管道配置
    enable_async_pipeline: bool = True  # 启用异步处理管道
    pipeline_queue_size: int = 8  # 管道队列大小（预处理缓冲区）
    max_workers_per_gpu: int = 3  # 每个GPU的工作线程数
    gpu_timeout_seconds: int = 1800  # GPU处理超时时间（秒）
    
    # 动态负载均衡参数
    enable_dynamic_balancing: bool = False  # 禁用动态负载均衡（大规模文件时复杂度计算开销过大）
    initial_batch_ratio: float = 0.3  # 初始批次比例（30%文件先分配）
    fast_gpu_bonus_ratio: float = 1.5  # 快速GPU奖励比例
    
    # 性能阈值配置
    preprocess_warning_seconds: int = 120  # 预处理时间警告阈值（秒）
    large_file_warning_mb: float = 200.0  # 大文件警告阈值（MB）
    memory_warning_duration_minutes: float = 60.0  # 内存警告时长阈值（分钟）
    thread_join_timeout_seconds: int = 30  # 线程等待超时时间（秒）
    gpu_memory_usage_limit: float = 0.8  # GPU内存使用限制（80%）
    
    def __post_init__(self):
        """初始化后的验证"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        if not CLEARVOICE_AVAILABLE:
            raise RuntimeError("ClearVoice未安装")
        
        # 验证GPU ID
        if torch.cuda.is_available():
            available_gpus = list(range(torch.cuda.device_count()))
            invalid_gpus = [gpu_id for gpu_id in self.gpu_ids if gpu_id not in available_gpus]
            if invalid_gpus:
                logger.warning(f"无效的GPU ID: {invalid_gpus}, 可用GPU: {available_gpus}")
                self.gpu_ids = [gpu_id for gpu_id in self.gpu_ids if gpu_id in available_gpus]
        else:
            logger.warning("CUDA不可用，将使用CPU模式")
            self.device = "cpu"
            self.use_multi_gpu = False


class AudioProcessor:
    """音频处理工具类"""
    
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.m4a', '.mp4')
    
    @staticmethod
    def load_and_validate_audio(file_path: str, config: Optional['MossFormerGANConfig'] = None) -> Tuple[np.ndarray, int]:
        """
        加载并验证音频文件
        多通道音频将被转换为单声道（取第一个通道）
        支持FFT下采样到目标采样率
        
        Args:
            file_path: 音频文件路径
            config: 配置对象，包含FFT下采样参数
            
        Returns:
            音频数据和采样率的元组
        """
        try:
            if file_path.lower().endswith(('.mp4', '.m4a')):
                # 处理mp4和m4a文件
                try:
                    from pydub import AudioSegment
                except ImportError:
                    raise ImportError("处理MP4/M4A文件需要安装pydub: pip install pydub")
                
                audio = AudioSegment.from_file(file_path)
                
                # 转换为numpy数组，如果是多声道，直接取第一个通道
                audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
                if audio.sample_width == 2:  # 16位音频
                    audio_data = audio_data / (1 << 15)
                elif audio.sample_width == 3:  # 24位音频
                    audio_data = audio_data / (1 << 23)
                elif audio.sample_width == 4:  # 32位音频
                    audio_data = audio_data / (1 << 31)
                else:
                    audio_data = audio_data / np.max(np.abs(audio_data))
                
                # 处理多声道：如果是立体声或多声道，取第一个通道
                if audio.channels > 1:
                    # 对于多声道，数据是交错存储的，我们需要取每隔channels个样本
                    audio_data = audio_data[::audio.channels]
                
                sr = audio.frame_rate
            else:
                # 读取其他格式的音频文件
                audio_data, sr = sf.read(file_path)
            
            # 转换为单声道 - 只取第一个通道
            if audio_data.ndim > 1:
                audio_data = audio_data[:, 0] if audio_data.shape[1] > 0 else audio_data[0]
            
            # 基本验证
            if len(audio_data) == 0:
                raise ValueError("音频文件为空")
            
            # 应用FFT下采样（如果配置启用且需要）
            if config and config.use_fft_downsample and sr != config.target_sr:
                logger.info(f"对文件 {os.path.basename(file_path)} 执行FFT下采样: {sr}Hz -> {config.target_sr}Hz")
                audio_data = AudioProcessor.fft_downsample(
                    audio_data, sr, config.target_sr, 
                    quality=config.fft_downsample_quality,
                    anti_alias=config.anti_alias_filter
                )
                sr = config.target_sr
            
            return audio_data, sr
            
        except Exception as e:
            raise RuntimeError(f"加载音频文件失败 {file_path}: {e}")
    
    @staticmethod
    def save_audio(audio_data: np.ndarray, file_path: str, sample_rate: int):
        """
        保存音频文件
        使用自定义保存方法确保采样率控制，不依赖ClearVoice的write方法
        支持长文件名和特殊字符处理
        
        Args:
            audio_data: 音频数据
            file_path: 保存路径
            sample_rate: 采样率
        """
        try:
            # 检查文件路径是否需要清理（避免重复处理）
            if AudioProcessor._needs_sanitization(file_path):
                file_path = AudioProcessor._sanitize_file_path(file_path)
            
            # 确保输出目录存在
            output_dir = os.path.dirname(file_path)
            os.makedirs(output_dir, exist_ok=True)
            
            # 验证音频数据
            if audio_data is None or len(audio_data) == 0:
                raise ValueError("音频数据为空")
            
            # 确保音频数据类型正确
            if not isinstance(audio_data, np.ndarray):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            # 处理数据类型
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # 确保音频数据在合理范围内
            if np.abs(audio_data).max() > 1.0:
                audio_data = audio_data / np.abs(audio_data).max()
            
            # 确保音频数据是1D的（ClearVoice返回(1, N)形状的数组）
            if audio_data.ndim == 2:
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.squeeze(0)  # (1, N) -> (N,)
                elif audio_data.shape[1] == 1:
                    audio_data = audio_data.squeeze(1)  # (N, 1) -> (N,)
                else:
                    # 多通道，取第一个通道
                    audio_data = audio_data[0]
                print(f"调试信息 - 2D数组转换为1D，新形状: {audio_data.shape}")
            elif audio_data.ndim > 2:
                # 更高维度的数组，尽力转换为1D
                audio_data = audio_data.flatten()
                print(f"调试信息 - 高维数组展平为1D，新形状: {audio_data.shape}")
            
            print(f"调试信息 - 最终保存的音频形状: {audio_data.shape}, 数据类型: {audio_data.dtype}")
            
            # 保存音频
            sf.write(file_path, audio_data, sample_rate, format='WAV', subtype='PCM_16')
            
            # 验证文件是否成功保存
            if not os.path.exists(file_path):
                raise RuntimeError("文件保存后不存在")
                
        except Exception as e:
            raise RuntimeError(f"保存音频文件失败 {file_path}: {e}")
    
    @staticmethod
    def _needs_sanitization(file_path: str) -> bool:
        """
        检查文件路径是否需要清理
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否需要清理
        """
        import re
        
        filename = os.path.basename(file_path)
        name = os.path.splitext(filename)[0]
        
        # 检查是否包含需要清理的字符（包括空格）
        has_special_chars = bool(re.search(r'[^\w\-\.\(\)（）【】\u4e00-\u9fff]', name))
        
        # 检查文件名是否过长
        is_too_long = len(name) > 200
        
        return has_special_chars or is_too_long
    
    @staticmethod
    def _sanitize_file_path(file_path: str) -> str:
        """
        清理和缩短文件路径，处理特殊字符和长文件名
        
        Args:
            file_path: 原始文件路径
            
        Returns:
            处理后的文件路径
        """
        import re
        import hashlib
        
        # 分离目录和文件名
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        # 分离文件名和扩展名
        name, ext = os.path.splitext(filename)
        
        # 清理文件名中的特殊字符
        # 保留中文、英文、数字、下划线、连字符、点号和括号，但不保留空格
        clean_name = re.sub(r'[^\w\-\.\(\)（）【】\u4e00-\u9fff]', '_', name)
        
        # 处理连续的下划线
        clean_name = re.sub(r'_+', '_', clean_name)
        clean_name = clean_name.strip('_')
        
        # 如果文件名过长，进行截断和哈希处理
        max_filename_length = 200  # 大多数文件系统支持的安全长度
        
        if len(clean_name) > max_filename_length:
            # 保留前半部分和后半部分，中间用哈希值连接
            hash_obj = hashlib.md5(clean_name.encode('utf-8'))
            hash_str = hash_obj.hexdigest()[:8]
            
            # 计算前后部分的长度
            remaining_length = max_filename_length - len(hash_str) - 2  # 2 for '__'
            front_length = remaining_length // 2
            back_length = remaining_length - front_length
            
            if back_length > 0:
                clean_name = f"{clean_name[:front_length]}__{hash_str}__{clean_name[-back_length:]}"
            else:
                clean_name = f"{clean_name[:front_length]}__{hash_str}"
        
        # 重新组合文件路径
        new_filename = clean_name + ext
        new_file_path = os.path.join(directory, new_filename)
        
        # 如果路径被修改，记录日志
        if new_file_path != file_path:
            print(f"文件名清理: {os.path.basename(file_path)} -> {new_filename}")
        
        return new_file_path
    
    @staticmethod
    def validate_audio_file(file_path: str, config: Optional['MossFormerGANConfig'] = None) -> bool:
        """
        验证音频文件的有效性
        
        Args:
            file_path: 文件路径
            config: 配置对象
            
        Returns:
            是否有效
        """
        try:
            audio, sr = AudioProcessor.load_and_validate_audio(file_path, config)
            return True
        except Exception:
            return False

    @staticmethod
    def fft_downsample(audio_data: np.ndarray, orig_sr: int, target_sr: int, 
                      quality: str = "high", anti_alias: bool = True) -> np.ndarray:
        """
        使用librosa的FFT方法进行高质量音频下采样
        
        Args:
            audio_data: 输入音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            quality: 下采样质量 ("high", "medium", "low")
            anti_alias: 是否使用抗混叠滤波器
            
        Returns:
            下采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio_data
            
        try:
            import librosa
        except ImportError:
            logger.error("librosa未安装，回退到线性下采样")
            return AudioProcessor._linear_downsample(audio_data, orig_sr, target_sr)
        
        try:
            # 确保输入是1D数组
            if audio_data.ndim > 1:
                if audio_data.shape[0] == 1:
                    audio_data = audio_data.squeeze(0)
                elif audio_data.shape[1] == 1:
                    audio_data = audio_data.squeeze(1)
                else:
                    audio_data = audio_data[:, 0]  # 取第一个通道
            
            # 根据质量设置参数
            if quality == "high":
                res_type = 'fft'  # 使用FFT方法
            elif quality == "medium":
                res_type = 'polyphase'
            else:  # low
                res_type = 'linear'
            
            # 使用librosa的resample函数进行FFT下采样
            if orig_sr > target_sr:
                # 下采样
                downsampled_audio = librosa.resample(
                    audio_data, 
                    orig_sr=orig_sr, 
                    target_sr=target_sr,
                    res_type=res_type,
                    fix=True,  # 修复长度不匹配问题
                    scale=False  # 不进行幅度缩放
                )
            else:
                # 上采样
                downsampled_audio = librosa.resample(
                    audio_data, 
                    orig_sr=orig_sr, 
                    target_sr=target_sr,
                    res_type=res_type,
                    fix=True,
                    scale=False
                )
            
            # 确保输出数据类型与输入一致
            downsampled_audio = downsampled_audio.astype(audio_data.dtype)
            
            logger.info(f"Librosa FFT重采样完成: {orig_sr}Hz -> {target_sr}Hz, 质量: {quality}")
            return downsampled_audio
            
        except Exception as e:
            logger.error(f"Librosa FFT重采样失败: {e}，回退到线性下采样")
            return AudioProcessor._linear_downsample(audio_data, orig_sr, target_sr)
    
    @staticmethod
    def _linear_downsample(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """线性下采样（回退方案）"""
        downsample_factor = orig_sr / target_sr
        target_length = int(len(audio_data) / downsample_factor)
        indices = np.linspace(0, len(audio_data) - 1, target_length)
        return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(audio_data.dtype)





def process_gpu_batch_isolated(args) -> List[Tuple[bool, str]]:
    """
    完全隔离的GPU处理函数 - 使用异步处理管道优化
    
    Args:
        args: 包含文件列表、GPU ID和其他参数的元组
        
    Returns:
        处理结果列表
    """
    file_list, gpu_id, model_name, task, target_sr, input_dir, output_dir, position, config = args
    
    try:
        # CUDA_VISIBLE_DEVICES 已在子进程启动前设置，模块导入时 CUDA 只看到 1 张卡
        import torch
        import numpy as np
        import soundfile as sf
        from tqdm import tqdm
        import threading
        import queue
        import tempfile
        import time
        
        time.sleep(1)
        torch.set_num_threads(8)
        
        if not torch.cuda.is_available():
            print(f"GPU {gpu_id} 进程: CUDA不可用")
            return [(False, "CUDA不可用") for _ in file_list]
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print(f"GPU {gpu_id} 进程: 没有检测到GPU设备")
            return [(False, "没有检测到GPU设备") for _ in file_list]
        
        # 在 CUDA_VISIBLE_DEVICES 隔离后, cuda:0 即为指定物理 GPU
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        try:
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(0)
            device_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU {gpu_id} 进程启动成功:")
            print(f"  - 当前设备: {current_device}")
            print(f"  - 设备名称: {device_name}")
            print(f"  - 设备内存: {device_memory:.1f} GB")
            print(f"  - 可见设备数: {device_count}")
        except Exception as e:
            print(f"GPU {gpu_id} 进程: 获取设备信息失败: {e}")
            return [(False, f"获取设备信息失败: {e}") for _ in file_list]

        
        print(f"GPU {gpu_id} 启用异步处理管道")
        
        return process_with_async_pipeline(
            file_list, gpu_id, model_name, task, target_sr, position, config
        )
            
    except Exception as e:
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return [(False, f"进程初始化失败: {e}") for _ in file_list]


def process_with_async_pipeline(file_list, gpu_id, model_name, task, target_sr, position, config=None):
    """
    使用异步处理管道处理文件列表
    
    Args:
        file_list: 待处理的文件列表
        gpu_id: GPU ID
        model_name: 模型名称
        task: 任务类型
        target_sr: 目标采样率
        position: 进度条位置
        config: MossFormerGANConfig 配置对象
    """
    import torch
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
    import threading
    import queue
    import tempfile
    import time
    from types import SimpleNamespace
    
    if config is None:
        config = SimpleNamespace()
        config.pipeline_queue_size = 1
        config.max_workers_per_gpu = 2
        config.gpu_timeout_seconds = 600
        config.use_fft_downsample = True
        config.fft_downsample_quality = "high"
        config.anti_alias_filter = True
    
    # 创建管道队列
    queue_size = getattr(config, 'pipeline_queue_size', 1)
    preprocessing_queue = queue.Queue(maxsize=queue_size)  # 预处理完成的文件队列
    results_queue = queue.Queue()  # 结果队列
    
    results = []
    
    # 在进程中导入ClearVoice并初始化模型
    try:
        from clearvoice import ClearVoice
        
        torch.cuda.set_device(0)
        
        # ClearVoice 通过 nvidia-smi 探测空闲 GPU，不认 CUDA_VISIBLE_DEVICES
        # 需 monkey-patch 使其始终使用 cuda:0
        import clearvoice.networks as _cvn
        _cvn.SpeechModel.get_free_gpu = lambda self: 0 if torch.cuda.is_available() else None
        
        clearvoice_instance = ClearVoice(task=task, model_names=[model_name])
        
        print(f"GPU {gpu_id} 模型加载成功，开始异步处理 {len(file_list)} 个文件")
        
    except Exception as e:
        print(f"GPU {gpu_id} 模型初始化失败: {e}")
        return [(False, f"模型初始化失败: {e}") for _ in file_list]
    
    # 预处理线程函数：负责音频加载和FFT下采样
    def preprocessing_worker():
        """预处理工作线程：音频加载和FFT下采样"""
        # 使用传入的配置参数
        gpu_config = config if hasattr(config, 'use_fft_downsample') else SimpleNamespace()
        if not hasattr(gpu_config, 'use_fft_downsample'):
            gpu_config.use_fft_downsample = True
        if not hasattr(gpu_config, 'fft_downsample_quality'):
            gpu_config.fft_downsample_quality = "high"
        if not hasattr(gpu_config, 'anti_alias_filter'):
            gpu_config.anti_alias_filter = True
        gpu_config.target_sr = target_sr
        
        for i, (input_file, output_file) in enumerate(file_list):
            try:
                # 确保输出目录存在（强制覆盖已存在的文件）
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # 如果文件已存在，先删除以确保重新处理
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                        print(f"🔄 GPU {gpu_id} 删除已存在文件: {os.path.basename(output_file)}")
                    except Exception as e:
                        print(f"⚠️ GPU {gpu_id} 删除文件失败，将覆盖: {e}")
                
                # 音频加载和FFT下采样
                preprocess_start = time.time()
                
                # 获取文件详细信息和大文件预检查
                try:
                    file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
                except:
                    file_size_mb = 0
                
                print(f"⏳ GPU {gpu_id} 开始预处理 {i+1}/{len(file_list)}:")
                print(f"   📁 输入文件: {input_file}")
                print(f"   📤 输出文件: {output_file}")  
                print(f"   📊 文件大小: {file_size_mb:.2f} MB")
                print(f"   🎵 文件格式: {os.path.splitext(input_file)[1].upper()}")
                print(f"   📂 相对路径: {os.path.relpath(input_file)}")
                try:
                    import time as time_module
                    mtime = os.path.getmtime(input_file)
                    mtime_str = time_module.strftime('%Y-%m-%d %H:%M:%S', time_module.localtime(mtime))
                    print(f"   🕒 修改时间: {mtime_str}")
                except:
                    pass
                
                # 大文件预检查 - 防止内存爆炸
                max_file_size = getattr(config, 'max_file_size_mb', 100.0)
                if file_size_mb > max_file_size:
                    error_msg = f"文件过大({file_size_mb:.1f}MB > {max_file_size}MB)，跳过以防止内存溢出"
                    print(f"⚠️ GPU {gpu_id}: {error_msg}")
                    preprocessing_queue.put(('error', i, input_file, output_file, error_msg))
                    continue
                
                try:
                    audio, sr = AudioProcessor.load_and_validate_audio(input_file, gpu_config)
                except MemoryError as mem_err:
                    error_msg = f"内存不足，无法加载音频文件: {mem_err}"
                    print(f"🚨 GPU {gpu_id}: {error_msg}")
                    preprocessing_queue.put(('error', i, input_file, output_file, error_msg))
                    continue
                except Exception as load_err:
                    error_msg = f"音频加载失败: {load_err}"
                    print(f"❌ GPU {gpu_id}: {error_msg}")
                    preprocessing_queue.put(('error', i, input_file, output_file, error_msg))
                    continue
                
                if len(audio) == 0:
                    preprocessing_queue.put(('error', i, input_file, output_file, "音频文件为空"))
                    continue
                
                # 计算音频时长
                audio_duration_minutes = len(audio) / sr / 60
                
                # 创建临时文件
                temp_dir = tempfile.gettempdir()
                temp_filename = f"temp_audio_gpu{gpu_id}_{int(time.time())}_{os.getpid()}_{i}.wav"
                temp_input_path = os.path.join(temp_dir, temp_filename)
                
                # 保存预处理后的音频到临时文件
                sf.write(temp_input_path, audio, sr, format='WAV', subtype='PCM_16')
                
                preprocess_time = time.time() - preprocess_start
                
                # 将预处理完成的数据放入队列
                preprocessing_queue.put((
                    'ready', i, input_file, output_file, temp_input_path, 
                    audio_duration_minutes, preprocess_time
                ))
                
                # 显示队列状态和性能警告
                queue_size = preprocessing_queue.qsize()
                
                # 性能警告
                warning_threshold = getattr(config, 'preprocess_warning_seconds', 120)
                if preprocess_time > warning_threshold:
                    print(f"🔄 GPU {gpu_id} 预处理完成 {i+1}/{len(file_list)}: {os.path.basename(input_file)} (时长 {audio_duration_minutes:.1f}min, 预处理 {preprocess_time:.2f}s, 队列 {queue_size}) ⚠️ 预处理时间较长")
                else:
                    print(f"🔄 GPU {gpu_id} 预处理完成 {i+1}/{len(file_list)}: {os.path.basename(input_file)} (时长 {audio_duration_minutes:.1f}min, 预处理 {preprocess_time:.2f}s, 队列 {queue_size})")
                
            except Exception as e:
                preprocessing_queue.put(('error', i, input_file, output_file, str(e)))
        
        # 预处理完成，为每个GPU处理线程发送结束信号
        max_workers = getattr(config, 'max_workers_per_gpu', 2)
        for _ in range(max_workers):
            preprocessing_queue.put(('done', None, None, None, None))
    
    # GPU处理线程函数：负责AI模型推理
    def gpu_processing_worker():
        """GPU处理工作线程：AI模型推理"""
        processed_count = 0
        
        while True:
            try:
                # 从队列获取预处理完成的数据（增加超时时间以适应长音频文件）
                item = preprocessing_queue.get()  # 无超时，适应长音频预处理
                
                if item[0] == 'done':
                    break
                elif item[0] == 'error':
                    _, i, input_file, output_file, error_msg = item
                    results_queue.put((False, f"预处理失败 {input_file}: {error_msg}"))
                    processed_count += 1
                elif item[0] == 'ready':
                    _, i, input_file, output_file, temp_input_path, audio_duration_minutes, preprocess_time = item
                    
                    try:
                        # GPU内存检查 - AI处理前（cuda:0 即为当前 GPU）
                        if torch.cuda.is_available():
                            gpu_mem_before = torch.cuda.memory_allocated(0) / (1024**3)
                            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                            gpu_mem_free = gpu_mem_total - gpu_mem_before
                            
                            # 如果可用内存不足，先清理缓存
                            if gpu_mem_free < 2.0:  # 少于2GB可用内存
                                print(f"⚠️ GPU {gpu_id} 内存不足 ({gpu_mem_free:.1f}GB可用)，清理缓存...")
                                torch.cuda.empty_cache()
                                import gc
                                gc.collect()
                                
                                # 重新检查内存
                                gpu_mem_after_cleanup = torch.cuda.memory_allocated(0) / (1024**3)
                                gpu_mem_free_after = gpu_mem_total - gpu_mem_after_cleanup
                                print(f"   清理后可用内存: {gpu_mem_free_after:.1f}GB")
                                
                                # 如果清理后仍然内存不足，跳过该文件
                                if gpu_mem_free_after < 1.0:  # 少于1GB可用内存
                                    error_msg = f"GPU内存严重不足({gpu_mem_free_after:.1f}GB可用)，跳过大文件"
                                    print(f"🚨 GPU {gpu_id}: {error_msg}")
                                    results_queue.put((False, f"{error_msg}: {input_file}"))
                                    continue
                        
                        # GPU AI处理
                        print(f"🎯 GPU {gpu_id} AI处理开始: {os.path.basename(input_file)} (时长 {audio_duration_minutes:.1f}min)")
                        gpu_start = time.time()
                        
                        try:
                            enhanced_audio = clearvoice_instance(temp_input_path, online_write=False)
                        except RuntimeError as cuda_err:
                            # 处理CUDA内存溢出错误
                            if "out of memory" in str(cuda_err).lower():
                                error_msg = f"GPU显存不足，无法处理大文件 ({audio_duration_minutes:.1f}min)"
                                print(f"🚨 GPU {gpu_id}: {error_msg}")
                                
                                # 强制清理GPU内存
                                try:
                                    torch.cuda.empty_cache()
                                    import gc
                                    gc.collect()
                                    torch.cuda.synchronize()
                                except:
                                    pass
                                
                                results_queue.put((False, f"{error_msg}: {input_file}"))
                                continue
                            else:
                                raise cuda_err
                        except MemoryError as mem_err:
                            error_msg = f"系统内存不足，无法处理大文件 ({audio_duration_minutes:.1f}min)"
                            print(f"🚨 GPU {gpu_id}: {error_msg}")
                            results_queue.put((False, f"{error_msg}: {input_file}"))
                            continue
                        
                        gpu_time = time.time() - gpu_start
                        print(f"⚡ GPU {gpu_id} AI处理完成: {os.path.basename(input_file)} (GPU耗时 {gpu_time:.1f}s)")
                        
                        # 保存结果
                        if enhanced_audio is not None:
                            try:
                                # 处理增强音频数据
                                audio_data = None
                                if isinstance(enhanced_audio, np.ndarray):
                                    audio_data = enhanced_audio.astype(np.float32)
                                elif hasattr(enhanced_audio, 'detach'):  # torch.Tensor
                                    audio_data = enhanced_audio.detach().cpu().numpy().astype(np.float32)
                                else:
                                    audio_data = np.array(enhanced_audio, dtype=np.float32)
                                
                                # 处理多维数组
                                if audio_data.ndim > 1:
                                    if audio_data.shape[0] == 1:
                                        audio_data = audio_data.squeeze(0)
                                    elif audio_data.shape[1] == 1:
                                        audio_data = audio_data.squeeze(1)
                                    else:
                                        audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                                
                                # 保存音频
                                model_sr = 16000
                                AudioProcessor.save_audio(audio_data, output_file, model_sr)
                                
                                total_time = preprocess_time + gpu_time
                                speed_ratio = audio_duration_minutes * 60 / total_time if total_time > 0 else 0
                                
                                results_queue.put((True, f"成功处理: {input_file}"))
                                
                                # 显示处理统计和队列状态
                                queue_size = preprocessing_queue.qsize()
                                print(f"✅ GPU {gpu_id} 完成 {processed_count+1}/{len(file_list)}: {os.path.basename(input_file)} "
                                      f"(音频 {audio_duration_minutes:.1f}min, 预处理 {preprocess_time:.1f}s, "
                                      f"GPU {gpu_time:.1f}s, 速度 {speed_ratio:.1f}x, 队列 {queue_size})")
                                
                            except Exception as save_error:
                                results_queue.put((False, f"保存失败 {input_file}: {save_error}"))
                        else:
                            results_queue.put((False, f"处理失败，无音频数据: {input_file}"))
                            
                    except Exception as gpu_error:
                        results_queue.put((False, f"GPU处理失败 {input_file}: {gpu_error}"))
                    
                    finally:
                        # 清理临时文件
                        if 'temp_input_path' in locals() and os.path.exists(temp_input_path):
                            try:
                                os.remove(temp_input_path)
                            except:
                                pass
                    
                    processed_count += 1
                    
            except queue.Empty:
                queue_size = preprocessing_queue.qsize()
                print(f"⚠️ GPU {gpu_id}: 处理队列超时 (队列中还有 {queue_size} 个文件等待处理)")
                if queue_size == 0:
                    print(f"   预处理线程可能已经完成或出现错误")
                    break
                else:
                    print(f"   继续等待预处理完成...")
                    continue
            except Exception as e:
                print(f"🚨 GPU {gpu_id}: GPU处理线程错误: {e}")
                break
    
    # 启动异步线程
    print(f"🚀 GPU {gpu_id} 启动异步处理管道:")
    print(f"  📥 预处理线程: 音频加载 + FFT下采样")
    print(f"  🎯 GPU处理线程: AI模型推理 + 结果保存")
    print(f"  📊 管道队列大小: {queue_size} (缓冲区)")
    print(f"  🧵 GPU工作线程数: {getattr(config, 'max_workers_per_gpu', 2)}")
    print(f"  ⏱️ GPU处理超时: {getattr(config, 'gpu_timeout_seconds', 1200)}秒")
    
    # 创建预处理线程
    preprocess_thread = threading.Thread(target=preprocessing_worker, daemon=True, name=f"GPU{gpu_id}-Preprocess")
    
    # 根据配置创建GPU处理线程
    max_workers = getattr(config, 'max_workers_per_gpu', 2)
    gpu_threads = []
    for worker_id in range(max_workers):
        if max_workers > 1:
            thread_name = f"GPU{gpu_id}-Processing-{worker_id}"
        else:
            thread_name = f"GPU{gpu_id}-Processing"
        gpu_thread = threading.Thread(target=gpu_processing_worker, daemon=True, name=thread_name)
        gpu_threads.append(gpu_thread)
    
    preprocess_thread.start()
    for gpu_thread in gpu_threads:
        gpu_thread.start()
    
    print(f"✅ GPU {gpu_id} 异步管道启动完成，开始处理文件...")
    
    # 创建进度条
    pbar = tqdm(total=len(file_list), desc=f"GPU {gpu_id}", position=position, leave=True)
    
        # 收集结果
    completed = 0
    while completed < len(file_list):
        try:
            # 使用配置的超时时间
            timeout_seconds = getattr(config, 'gpu_timeout_seconds', 1200)
            result = results_queue.get(timeout=timeout_seconds)
            results.append(result)
            completed += 1
            pbar.update(1)
            
            # 更新进度条状态
            if result[0]:
                pbar.set_postfix({"状态": "成功"})
            else:
                pbar.set_postfix({"状态": "失败"})
                
        except queue.Empty:
            timeout_minutes = getattr(config, 'gpu_timeout_seconds', 1200) // 60
            print(f"⚠️ GPU {gpu_id}: 结果收集超时 ({timeout_minutes}分钟)")
            break
    
    # 等待线程完成
    thread_timeout = getattr(config, 'thread_join_timeout_seconds', 10)
    preprocess_thread.join(timeout=thread_timeout)
    for gpu_thread in gpu_threads:
        gpu_thread.join(timeout=thread_timeout)
    
    pbar.close()
    
    # 统计结果和性能
    success_count = sum(1 for success, _ in results if success)
    failed_count = len(results) - success_count
    
    print(f"\n📊 GPU {gpu_id} 异步管道处理完成:")
    print(f"  ✅ 成功处理: {success_count} 个文件")
    print(f"  ❌ 处理失败: {failed_count} 个文件")
    print(f"  📈 成功率: {(success_count/len(file_list)*100):.1f}%")
    print(f"  🔄 异步管道: 预处理线程 + GPU处理线程 并行工作")
    
    return results


def process_batch_on_gpu_worker(args) -> List[Tuple[bool, str, Optional[np.ndarray]]]:
    """
    多进程工作函数
    
    Args:
        args: 包含批次文件、GPU ID和其他参数的元组
        
    Returns:
        处理结果列表
    """
    batch_files, gpu_id, model_name, task, target_sr, config = args
    
    # 强制设置GPU设备
    if gpu_id is not None and torch.cuda.is_available():
        # 设置CUDA_VISIBLE_DEVICES，让进程只看到指定GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 现在GPU 0对应于实际的指定GPU
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # 验证GPU设置
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(0)
        print(f"进程工作GPU: 指定={gpu_id}, 当前={current_device}, 设备名={device_name}")
    
    # 在工作进程中初始化ClearVoice
    try:
        from clearvoice import ClearVoice
        
        # 强制使用设备0（在CUDA_VISIBLE_DEVICES隔离环境中）
        torch.cuda.set_device(0)
        
        clearvoice_instance = ClearVoice(task=task, model_names=[model_name])
        
        logger.info(f"进程GPU {gpu_id} 开始处理 {len(batch_files)} 个文件")
        
        results = []
        for i, audio_file in enumerate(batch_files):
            try:
                # 验证音频文件
                audio, sr = AudioProcessor.load_and_validate_audio(audio_file, config)
                
                # 处理音频 - 使用ClearVoice进行语音增强（文件进文件出）
                # 创建临时文件保存下采样后的音频
                
                # 创建临时文件
                temp_dir = tempfile.gettempdir()
                temp_filename = f"temp_audio_gpu{gpu_id}_{int(time.time())}_{os.getpid()}_{i}.wav"
                temp_input_path = os.path.join(temp_dir, temp_filename)
                
                try:
                    # 保存音频到临时文件
                    sf.write(temp_input_path, audio, sr, format='WAV', subtype='PCM_16')
                    
                    # 使用ClearVoice处理临时文件（文件进文件出）
                    enhanced_audio = clearvoice_instance(temp_input_path, online_write=False)
                    
                finally:
                    # 清理临时文件
                    if os.path.exists(temp_input_path):
                        try:
                            os.remove(temp_input_path)
                        except:
                            pass  # 忽略删除失败
                
                # 处理结果
                if enhanced_audio is not None:
                    results.append((True, f"成功处理: {audio_file} (GPU {gpu_id})", enhanced_audio))
                else:
                    results.append((False, f"处理失败: {audio_file} (GPU {gpu_id})", None))
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} 处理失败 {audio_file}: {e}")
                results.append((False, f"处理失败 {audio_file}: {e}", None))
        
        logger.info(f"进程GPU {gpu_id} 完成处理")
        return results
        
    except Exception as e:
        logger.error(f"进程GPU {gpu_id} 初始化失败: {e}")
        return [(False, f"进程初始化失败: {e}", None) for _ in batch_files]


class ProcessingStats:
    """处理统计信息类"""
    
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = 0
        self.total_duration = 0
        self.processing_time = 0.0
        
        # 错误分类统计
        self.error_categories = {
            "cuda_memory_errors": 0,      # CUDA显存不足错误
            "system_memory_errors": 0,    # 系统内存不足错误
            "load_errors": 0,             # 音频加载错误
            "process_errors": 0,          # 音频处理错误
            "save_errors": 0,             # 音频保存错误
            "unknown_errors": 0           # 其他未知错误
        }
    
    def update_from_results(self, results: List[Tuple[bool, str]]):
        """从结果更新统计信息（不包含跳过文件的重复计算）"""
        for success, message in results:
            if success:
                # 跳过文件数量已经在处理前统计，这里不再重复计算
                if "跳过已存在文件" not in message:
                    self.processed_files += 1
            else:
                self.failed_files += 1
                
                # 分类错误类型
                message_lower = message.lower()
                if "显存不足" in message_lower or "cuda内存不足" in message_lower or "out of memory" in message_lower:
                    self.error_categories["cuda_memory_errors"] += 1
                elif "系统内存不足" in message_lower or "内存不足" in message_lower:
                    self.error_categories["system_memory_errors"] += 1
                elif "加载失败" in message_lower:
                    self.error_categories["load_errors"] += 1
                elif "处理失败" in message_lower:
                    self.error_categories["process_errors"] += 1
                elif "保存失败" in message_lower:
                    self.error_categories["save_errors"] += 1
                else:
                    self.error_categories["unknown_errors"] += 1
    
    def print_summary(self, config: MossFormerGANConfig):
        """打印统计摘要"""
        print("\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        print(f"总文件数:     {self.total_files}")
        print(f"成功处理:     {self.processed_files}")
        print(f"跳过文件:     {self.skipped_files}")
        print(f"失败文件:     {self.failed_files}")
        
        if self.total_files > 0:
            success_rate = (self.processed_files / self.total_files) * 100
            print(f"成功率:       {success_rate:.1f}%")
        
        print(f"处理时间:     {self.processing_time:.2f}秒")
        
        # 显示错误分类统计
        if self.failed_files > 0:
            print("\n错误分类统计:")
            print("-" * 30)
            
            for error_type, count in self.error_categories.items():
                if count > 0:
                    error_name = {
                        "cuda_memory_errors": "CUDA显存不足",
                        "system_memory_errors": "系统内存不足", 
                        "load_errors": "音频加载错误",
                        "process_errors": "音频处理错误",
                        "save_errors": "音频保存错误",
                        "unknown_errors": "其他未知错误"
                    }.get(error_type, error_type)
                    
                    percentage = (count / self.failed_files) * 100
                    print(f"{error_name}:    {count} ({percentage:.1f}%)")
            
            # 提供针对性建议
            print("\n💡 错误处理建议:")
            if self.error_categories["cuda_memory_errors"] > 0:
                print("  - CUDA显存不足：尝试处理较小的音频文件，或增加GPU显存")
            if self.error_categories["system_memory_errors"] > 0:
                print("  - 系统内存不足：关闭其他程序释放内存，或处理较小的文件")
            if self.error_categories["load_errors"] > 0:
                print("  - 音频加载错误：检查音频文件格式和完整性")
            if self.error_categories["process_errors"] > 0:
                print("  - 音频处理错误：检查ClearVoice模型和CUDA环境")
            if self.error_categories["save_errors"] > 0:
                print("  - 音频保存错误：检查输出目录权限和磁盘空间")
        
        # 显示GPU使用情况
        if config.use_multi_gpu:
            print(f"\n处理模式:     {'隔离GPU模式' if config.batch_processing_mode == 'isolated_gpu' else '多GPU模式'}")
            print(f"使用GPU数:    {len(config.gpu_ids)}")
            print(f"GPU列表:      {config.gpu_ids}")
        
        print("=" * 60)


class MossFormerGANBatchProcessor:
    """MossFormerGAN批量处理器"""
    
    def __init__(self, config: MossFormerGANConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.protection_strategy = LargeFileProtectionStrategy(config)  # 大文件保护策略
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.available_gpus = [i for i in config.gpu_ids if i < torch.cuda.device_count()]
            if not self.available_gpus:
                logger.warning("指定的GPU不可用，使用CPU")
                self.config.device = "cpu"
                self.config.use_multi_gpu = False
            else:
                logger.info(f"可用GPU: {self.available_gpus}")
        else:
            logger.warning("CUDA不可用，使用CPU")
            self.config.device = "cpu"
            self.config.use_multi_gpu = False
        
        # 根据批处理模式决定是否初始化模型
        if self.config.batch_processing_mode == 'isolated_gpu':
            # 隔离GPU模式不在主进程中初始化模型
            self.clearvoice_instances = []
            logger.info("隔离GPU模式：将在子进程中初始化模型")
        else:
            # 初始化模型
            self.clearvoice_instances = []
            self.init_model()
    
    def init_model(self):
        """初始化MossFormerGAN模型"""
        if not CLEARVOICE_AVAILABLE:
            raise ImportError("ClearVoice未安装，无法使用MossFormerGAN")
        
        try:
            if self.config.use_multi_gpu and len(self.available_gpus) > 1:
                # 多GPU模式：为每个GPU创建一个ClearVoice实例
                logger.info(f"初始化多GPU模式，使用GPU: {self.available_gpus}")
                for gpu_id in self.available_gpus:
                    # 设置当前GPU设备
                    torch.cuda.set_device(gpu_id)
                    
                    # 创建ClearVoice实例
                    clearvoice_instance = ClearVoice(
                        task=self.config.task, 
                        model_names=[self.config.model_name]
                    )
                    
                    # 存储实例和对应的GPU ID
                    self.clearvoice_instances.append((clearvoice_instance, gpu_id))
                    logger.info(f"✓ 在GPU {gpu_id}上加载模型成功")
            else:
                # 单GPU或CPU模式
                if self.available_gpus:
                    torch.cuda.set_device(self.available_gpus[0])
                
                clearvoice_instance = ClearVoice(
                    task=self.config.task, 
                    model_names=[self.config.model_name]
                )
                self.clearvoice_instances.append(
                    (clearvoice_instance, self.available_gpus[0] if self.available_gpus else None)
                )
                logger.info(f"✓ 在设备 {self.config.device} 上加载模型成功")
            
            logger.info(f"✓ 成功加载MossFormerGAN模型: {self.config.model_name}")
            logger.info(f"✓ 总共加载了 {len(self.clearvoice_instances)} 个模型实例")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def enhance_single_file(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """增强单个文件"""
        # 获取文件大小用于错误报告
        try:
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        except:
            file_size_mb = 0
            
        try:
            # 检查是否跳过已存在的文件
            if self.config.skip_existing and os.path.exists(output_file):
                return True, f"跳过已存在文件: {input_file}"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 使用第一个ClearVoice实例处理
            clearvoice_instance, gpu_id = self.clearvoice_instances[0]
            
            # 确保设置正确的GPU
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
            
            # 加载音频数据
            try:
                audio, sr = AudioProcessor.load_and_validate_audio(input_file, self.config)
                
                # 检查音频长度，如果太长给出警告
                audio_duration_minutes = len(audio) / sr / 60
                warning_duration = getattr(self.config, 'memory_warning_duration_minutes', 30.0)
                if audio_duration_minutes > warning_duration:
                    print(f"⚠️ 单文件处理: 文件 {os.path.basename(input_file)} 时长 {audio_duration_minutes:.1f}分钟，可能消耗大量显存")
                    
            except Exception as load_error:
                return False, f"音频加载失败 {input_file} ({file_size_mb:.1f}MB): {load_error}"
            
            # 在指定GPU上处理 - 使用ClearVoice进行语音增强（文件进文件出）
            # 创建临时文件保存下采样后的音频
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_audio_gpu{gpu_id}_{int(time.time())}_{os.getpid()}_single.wav"
            temp_input_path = os.path.join(temp_dir, temp_filename)
            
            enhanced_audio = None
            try:
                # 保存音频到临时文件
                sf.write(temp_input_path, audio, sr, format='WAV', subtype='PCM_16')
                
                # 清理音频变量释放内存
                del audio
                
                # 使用ClearVoice处理临时文件（文件进文件出）
                with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                    enhanced_audio = clearvoice_instance(temp_input_path, online_write=False)
                    
            except RuntimeError as cuda_error:
                # 处理CUDA相关错误（如显存不足）
                error_msg = str(cuda_error).lower()
                if "out of memory" in error_msg or "cuda" in error_msg:
                    print(f"🚨 单文件处理: CUDA内存不足，无法处理大文件 {os.path.basename(input_file)} ({file_size_mb:.1f}MB)")
                    
                    # 强制清理GPU内存
                    try:
                        import gc
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        # 尝试创建一个小tensor来测试GPU是否恢复
                        test_tensor = torch.tensor([1.0]).cuda()
                        del test_tensor
                        torch.cuda.empty_cache()
                        print(f"✓ 单文件处理: 内存清理完成")
                    except Exception as cleanup_error:
                        print(f"⚠️ 单文件处理: 内存清理失败: {cleanup_error}")
                    
                    return False, f"显存不足，无法处理大文件 {input_file} ({file_size_mb:.1f}MB): {cuda_error}"
                else:
                    return False, f"CUDA处理失败 {input_file} ({file_size_mb:.1f}MB): {cuda_error}"
                    
            except MemoryError as mem_error:
                # 处理内存不足错误
                print(f"🚨 单文件处理: 系统内存不足，无法处理大文件 {os.path.basename(input_file)} ({file_size_mb:.1f}MB)")
                
                # 清理内存
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
                    
                return False, f"系统内存不足，无法处理大文件 {input_file} ({file_size_mb:.1f}MB): {mem_error}"
                
            except Exception as process_error:
                return False, f"音频处理失败 {input_file} ({file_size_mb:.1f}MB): {process_error}"
                
            finally:
                # 清理临时文件
                if os.path.exists(temp_input_path):
                    try:
                        os.remove(temp_input_path)
                    except:
                        pass  # 忽略删除失败
            
            # 保存结果 - 不使用ClearVoice的write方法，使用自定义保存确保采样率控制 (16kHz)
            if enhanced_audio is not None:
                try:
                    # 使用模型的采样率 (16kHz)
                    model_sr = 16000
                    
                    # 处理不同格式的增强音频数据
                    import numpy as np
                    import torch
                    
                    audio_data = None
                    if isinstance(enhanced_audio, np.ndarray):
                        audio_data = enhanced_audio.astype(np.float32)
                    elif isinstance(enhanced_audio, torch.Tensor):
                        audio_data = enhanced_audio.detach().cpu().numpy().astype(np.float32)
                    elif isinstance(enhanced_audio, list):
                        audio_data = np.array(enhanced_audio, dtype=np.float32)
                    else:
                        audio_data = np.array(enhanced_audio, dtype=np.float32)
                    
                    # 处理多维数组 - 确保是1D
                    if audio_data.ndim > 1:
                        if audio_data.shape[0] == 1:
                            audio_data = audio_data.squeeze(0)
                        elif audio_data.shape[1] == 1:
                            audio_data = audio_data.squeeze(1)
                        else:
                            audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                    
                    # 使用AudioProcessor的save_audio方法保存
                    AudioProcessor.save_audio(audio_data, output_file, model_sr)
                    
                    # 检查结果
                    if os.path.exists(output_file):
                        return True, f"成功处理: {input_file}"
                    else:
                        return False, f"处理失败，输出文件未生成: {input_file}"
                        
                except Exception as e:
                    return False, f"保存失败 {input_file}: {e}"
            else:
                return False, f"处理失败，无音频数据: {input_file}"
                
        except Exception as e:
            # 捕获所有其他未处理的异常
            error_msg = str(e).lower()
            
            if "out of memory" in error_msg or "cuda" in error_msg:
                print(f"🚨 单文件处理: 意外的CUDA错误，无法处理文件 {os.path.basename(input_file)} ({file_size_mb:.1f}MB): {e}")
                
                # 尝试清理GPU内存
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                    
                return False, f"意外CUDA错误，无法处理文件 {input_file} ({file_size_mb:.1f}MB): {e}"
            else:
                print(f"🚨 单文件处理: 未知错误，无法处理文件 {os.path.basename(input_file)} ({file_size_mb:.1f}MB): {e}")
                return False, f"未知错误，无法处理文件 {input_file} ({file_size_mb:.1f}MB): {e}"
    
    def get_audio_files(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
        """
        获取所有音频文件路径并分类
        
        Returns:
            元组包含三个列表: (要处理的文件, 要跳过的文件, 所有文件)
        """
        all_files = []
        files_to_process = []
        files_to_skip = []
        
        for root, dirs, files in os.walk(self.config.input_dir):
            for file in files:
                if file.lower().endswith(AudioProcessor.SUPPORTED_FORMATS):
                    input_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(input_path, self.config.input_dir)
                    
                    # 构造输出路径，确保为wav格式
                    raw_output_path = os.path.join(
                        self.config.output_dir, 
                        os.path.splitext(rel_path)[0] + '.wav'
                    )
                    
                    # 立即清理输出路径，避免后续保存时出错
                    output_path = AudioProcessor._sanitize_file_path(raw_output_path)
                    
                    file_pair = (input_path, output_path)
                    all_files.append(file_pair)
                    
                    # 检查输出文件是否已存在，决定是否跳过
                    if self.config.skip_existing and os.path.exists(output_path):
                        files_to_skip.append(file_pair)
                    else:
                        files_to_process.append(file_pair)
        
        return files_to_process, files_to_skip, all_files
    
    def get_file_complexity_score(self, file_path: str) -> float:
        """
        计算文件复杂度分数（基于文件大小和时长）
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            复杂度分数（越大越复杂）
        """
        try:
            # 获取文件大小（MB）
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            # 尝试获取音频时长
            try:
                import soundfile as sf
                with sf.SoundFile(file_path) as f:
                    duration_minutes = len(f) / f.samplerate / 60
            except:
                # 如果无法读取，基于文件大小估算（假设1MB约1分钟）
                duration_minutes = file_size_mb
            
            # 复杂度分数 = 时长权重 * 时长 + 文件大小权重 * 文件大小
            complexity_score = 0.7 * duration_minutes + 0.3 * file_size_mb
            return complexity_score
            
        except Exception:
            return 1.0  # 默认复杂度
    
    def distribute_files_smartly(self, file_pairs: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """
        智能分配文件给各个GPU，考虑文件复杂度和负载均衡
        
        Args:
            file_pairs: 文件路径对列表
            
        Returns:
            每个GPU分配的文件列表
        """
        if not self.available_gpus:
            return []
        
        num_gpus = len(self.available_gpus)
        
        # 如果禁用动态负载均衡，使用原来的简单分配
        if not self.config.enable_dynamic_balancing:
            logger.info("使用简单均等分配")
            files_per_gpu = len(file_pairs) // num_gpus
            remainder = len(file_pairs) % num_gpus
            
            gpu_file_batches = []
            start_idx = 0
            
            for i in range(num_gpus):
                current_batch_size = files_per_gpu + (1 if i < remainder else 0)
                end_idx = start_idx + current_batch_size
                gpu_files = file_pairs[start_idx:end_idx]
                gpu_file_batches.append(gpu_files)
                start_idx = end_idx
            
            return gpu_file_batches
        
        # 启用智能分配
        logger.info("使用智能负载均衡分配")
        
        # 计算每个文件的复杂度
        file_complexity = []
        for input_file, output_file in file_pairs:
            complexity = self.get_file_complexity_score(input_file)
            file_complexity.append((input_file, output_file, complexity))
        
        # 按复杂度排序（复杂的文件先分配）
        file_complexity.sort(key=lambda x: x[2], reverse=True)
        
        # 初始化每个GPU的工作负载
        gpu_workloads = [0.0] * num_gpus
        gpu_file_lists = [[] for _ in range(num_gpus)]
        
        # 使用贪心算法分配文件（总是分配给当前负载最轻的GPU）
        for input_file, output_file, complexity in file_complexity:
            # 找到负载最轻的GPU
            min_workload_gpu = gpu_workloads.index(min(gpu_workloads))
            
            # 分配文件
            gpu_file_lists[min_workload_gpu].append((input_file, output_file))
            gpu_workloads[min_workload_gpu] += complexity
        
        # 打印分配统计
        for i, (gpu_id, workload, file_count) in enumerate(zip(self.available_gpus, gpu_workloads, [len(files) for files in gpu_file_lists])):
            logger.info(f"GPU {gpu_id}: 分配 {file_count} 个文件，预估工作负载: {workload:.2f}")
        
        return gpu_file_lists
    
    def process_files_isolated_gpu(self, file_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """
        隔离GPU处理模式 - 每个GPU处理相应比例的文件
        
        Args:
            file_pairs: 文件路径对列表
            
        Returns:
            处理结果列表
        """
        if not self.available_gpus:
            logger.error("没有可用的GPU")
            return []
        
        logger.info(f"开始隔离GPU处理，文件总数: {len(file_pairs)}")
        logger.info(f"使用GPU: {self.available_gpus}")
        
        # 使用智能分配策略
        gpu_file_lists = self.distribute_files_smartly(file_pairs)
        
        # 准备GPU批次数据
        gpu_file_batches = []
        for i, (gpu_id, gpu_files) in enumerate(zip(self.available_gpus, gpu_file_lists)):
            gpu_file_batches.append((gpu_files, gpu_id, i))
            logger.info(f"GPU {gpu_id}: 最终分配 {len(gpu_files)} 个文件")
        
        # 准备参数
        args_list = []
        for gpu_files, gpu_id, position in gpu_file_batches:
            args = (
                gpu_files,
                gpu_id,
                self.config.model_name,
                self.config.task,
                self.config.target_sr,
                self.config.input_dir,
                self.config.output_dir,
                position,
                self.config  # 传递完整的配置对象
            )
            args_list.append(args)
        
        # 使用 subprocess.Popen 启动完全独立的进程，确保 CUDA_VISIBLE_DEVICES 在模块导入前生效
        import tempfile, pickle, subprocess
        
        num_gpus = len(self.available_gpus)
        all_results = []
        
        logger.info(f"启动 {len(args_list)} 个独立GPU进程...")
        
        processes = []
        for idx, args in enumerate(args_list):
            gpu_id = args[1]
            gpu_files = args[0]
            
            # 保存参数到临时文件
            args_file = tempfile.mkstemp(suffix='.pkl', prefix=f'moss_args_gpu{gpu_id}_')[1]
            with open(args_file, 'wb') as f:
                pickle.dump(args, f)
            
            result_file = tempfile.mkstemp(suffix='.pkl', prefix=f'moss_result_gpu{gpu_id}_')[1]
            log_file = f"/tmp/mossformergan_gpu{gpu_id}.log"
            
            # 将子进程代码写入临时脚本文件
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            child_script = f'''
import os, sys, pickle
sys.path.insert(0, "{base_dir}")

log_f = open("{log_file}", 'w')
sys.stdout = log_f
sys.stderr = log_f

import speech_enhancement_process.mossformergan_batch_inference as _mb
import __main__ as _main
for _attr in dir(_mb):
    if not _attr.startswith('_') and _attr[0].isupper():
        try:
            setattr(_main, _attr, getattr(_mb, _attr))
        except:
            pass

with open("{args_file}", 'rb') as f:
    args = pickle.load(f)

results = _mb.process_gpu_batch_isolated(args)

with open("{result_file}", 'wb') as f:
    pickle.dump(results, f)

log_f.close()
'''
            child_script_file = tempfile.mkstemp(suffix='.py', prefix=f'moss_worker_gpu{gpu_id}_')[1]
            with open(child_script_file, 'w') as f:
                f.write(child_script)
            
            logger.info(f"  启动 GPU {gpu_id} 进程 ({idx+1}/{len(args_list)}), {len(gpu_files)} 文件, log={log_file}")
            env = os.environ.copy()
            env.pop('http_proxy', None)
            env.pop('https_proxy', None)
            env.pop('HTTP_PROXY', None)
            env.pop('HTTPS_PROXY', None)
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            
            p = subprocess.Popen(
                ['python', '-u', child_script_file],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                env=env
            )
            # 存储以便后续等待和清理
            processes.append((gpu_id, p, args_file, result_file, log_file, child_script_file))
            time.sleep(5)
        
        # 收集结果 — 每个 GPU 重置超时，不累计
        timeout_minutes = getattr(self.config, 'process_timeout_minutes', 30)
        
        for gpu_id, p, args_file, result_file, log_file, child_script_file in processes:
            remaining = timeout_minutes * 60
            try:
                ret = p.wait(timeout=remaining)
                logger.info(f"GPU {gpu_id} 进程已完成 (exit={ret}), log={log_file}")
            except subprocess.TimeoutExpired:
                logger.error(f"GPU {gpu_id} 进程超时 ({timeout_minutes}m)，强制终止")
                p.kill()
                p.wait()
            
            # 读取结果
            try:
                with open(result_file, 'rb') as f:
                    batch = pickle.load(f)
                all_results.extend(batch)
                logger.info(f"  GPU {gpu_id}: 读取到 {len(batch)} 条结果")
            except Exception as e:
                logger.error(f"  GPU {gpu_id}: 无法读取结果文件: {e}")
            finally:
                for fpath in [args_file, result_file, child_script_file]:
                    try:
                        os.unlink(fpath)
                    except:
                        pass
        
        logger.info(f"隔离GPU处理完成，总结果: {len(all_results)} 个")
        return all_results
    
    def process_files_parallel(self, file_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """并行处理文件"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            # 提交任务
            future_to_file = {
                executor.submit(self.enhance_single_file, input_file, output_file): 
                (input_file, output_file) for input_file, output_file in file_pairs
            }
            
            # 处理结果
            with tqdm(total=len(file_pairs), desc="处理音频文件", unit="文件") as pbar:
                for future in as_completed(future_to_file):
                    input_file, output_file = future_to_file[future]
                    try:
                        success, message = future.result()
                        results.append((success, message))
                        
                        # 更新进度条
                        if success:
                            pbar.set_postfix({"状态": "成功", "文件": os.path.basename(input_file)})
                        else:
                            pbar.set_postfix({"状态": "失败", "文件": os.path.basename(input_file)})
                            
                    except Exception as e:
                        results.append((False, f"处理异常 {input_file}: {e}"))
                        pbar.set_postfix({"状态": "异常", "文件": os.path.basename(input_file)})
                    
                    pbar.update(1)
        
        return results
    
    def process_files_batch(self, file_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """
        批量处理文件 - 根据模式选择处理方式
        
        Args:
            file_pairs: 文件路径对列表
            
        Returns:
            处理结果列表
        """
        # 检查是否使用隔离GPU模式
        if self.config.batch_processing_mode == 'isolated_gpu':
            return self.process_files_isolated_gpu(file_pairs)
        
        # 其他批处理模式的实现可以在这里添加
        logger.warning("暂不支持的批处理模式，使用并行处理")
        return self.process_files_parallel(file_pairs)
    
    def monitor_gpu_memory(self, gpu_id: int = None) -> Dict[str, float]:
        """
        监控GPU内存使用情况
        
        Args:
            gpu_id: 要监控的GPU ID，如果为None则监控所有可用GPU
            
        Returns:
            包含内存使用信息的字典
        """
        memory_info = {}
        
        if not torch.cuda.is_available():
            return {"error": "CUDA不可用"}
        
        if gpu_id is not None:
            gpu_ids = [gpu_id]
        else:
            gpu_ids = self.available_gpus
        
        for gid in gpu_ids:
            try:
                torch.cuda.set_device(gid)
                
                # 获取内存信息
                total_memory = torch.cuda.get_device_properties(gid).total_memory
                allocated_memory = torch.cuda.memory_allocated(gid)
                reserved_memory = torch.cuda.memory_reserved(gid)
                free_memory = total_memory - allocated_memory
                
                memory_info[f"GPU_{gid}"] = {
                    "total_gb": total_memory / (1024**3),
                    "allocated_gb": allocated_memory / (1024**3),
                    "reserved_gb": reserved_memory / (1024**3),
                    "free_gb": free_memory / (1024**3),
                    "utilization_percent": (allocated_memory / total_memory) * 100
                }
                
            except Exception as e:
                memory_info[f"GPU_{gid}"] = {"error": str(e)}
        
        return memory_info
    
    def print_gpu_memory_status(self):
        """打印GPU内存状态"""
        memory_info = self.monitor_gpu_memory()
        
        print("\n" + "="*60)
        print("GPU 内存状态监控")
        print("="*60)
        
        for gpu_key, info in memory_info.items():
            if "error" in info:
                print(f"{gpu_key}: 错误 - {info['error']}")
            else:
                print(f"{gpu_key}:")
                print(f"  总内存:     {info['total_gb']:.1f} GB")
                print(f"  已分配:     {info['allocated_gb']:.2f} GB")
                print(f"  已保留:     {info['reserved_gb']:.2f} GB")
                print(f"  可用:       {info['free_gb']:.1f} GB")
                print(f"  使用率:     {info['utilization_percent']:.1f}%")
                
                # 内存使用警告
                if info['utilization_percent'] > 90:
                    print(f"  ⚠️  内存使用过高，可能导致显存不足")
                elif info['utilization_percent'] > 70:
                    print(f"  ⚠️  内存使用较高，注意大文件处理")
                print()
        
        print("="*60)
    
    def test_model_functionality(self):
        """测试模型功能"""
        if not torch.cuda.is_available():
            logger.info("CUDA不可用，跳过模型测试")
            return
        
        logger.info("测试模型功能...")
        
        # 显示测试前的GPU内存状态
        self.print_gpu_memory_status()
        
        # 创建一个测试音频文件
        test_audio = np.random.randn(16000).astype(np.float32)  # 1秒的随机音频
        test_file = "/tmp/test_audio.wav"
        sf.write(test_file, test_audio, 16000)
        
        try:
            for i, (clearvoice_instance, gpu_id) in enumerate(self.clearvoice_instances):
                logger.info(f"测试ClearVoice实例 {i} (GPU {gpu_id})...")
                
                # 设置目标GPU
                if gpu_id is not None:
                    torch.cuda.set_device(gpu_id)
                
                # 记录处理前的GPU内存使用
                if gpu_id is not None:
                    mem_before = torch.cuda.memory_allocated(gpu_id)
                    logger.info(f"GPU {gpu_id} 处理前内存: {mem_before / 1024 / 1024:.2f} MB")
                
                # 在指定GPU上进行测试处理（文件进文件出）
                with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                    enhanced_audio = clearvoice_instance(test_file, online_write=False)
                    
                    # 记录处理后的GPU内存使用
                    if gpu_id is not None:
                        mem_after = torch.cuda.memory_allocated(gpu_id)
                        mem_diff = mem_after - mem_before
                        logger.info(f"GPU {gpu_id} 处理后内存: {mem_after / 1024 / 1024:.2f} MB")
                        logger.info(f"GPU {gpu_id} 内存增加: {mem_diff / 1024 / 1024:.2f} MB")
                
                # 检查结果
                if enhanced_audio is not None:
                    logger.info(f"✓ ClearVoice实例 {i} 测试成功")
                else:
                    logger.error(f"✗ ClearVoice实例 {i} 测试失败")
                
        except Exception as e:
            logger.error(f"模型测试失败: {e}")
        finally:
            # 清理测试文件
            if os.path.exists(test_file):
                os.remove(test_file)
        
        # 显示测试后的GPU内存状态
        self.print_gpu_memory_status()
        logger.info("模型测试完成")


class LargeFileProtectionStrategy:
    """大文件处理保护策略类"""
    
    def __init__(self, config: MossFormerGANConfig):
        self.config = config
        self.large_file_stats = {
            "total_large_files": 0,
            "skipped_by_size": 0,
            "skipped_by_duration": 0,
            "failed_by_memory": 0,
            "failed_by_timeout": 0,
            "successfully_processed": 0
        }
    
    def check_file_safety(self, file_path: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        检查文件是否安全处理
        
        Returns:
            (is_safe, reason, file_info)
        """
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            file_info = {
                "size_mb": file_size_mb,
                "estimated_duration_min": file_size_mb * 0.8,  # 保守估算
                "memory_requirement_gb": file_size_mb * 0.005,  # 粗略估算内存需求
            }
            
            # 检查文件大小
            if file_size_mb > self.config.max_file_size_mb:
                self.large_file_stats["skipped_by_size"] += 1
                return False, f"文件过大 ({file_size_mb:.1f}MB > {self.config.max_file_size_mb}MB)", file_info
            
            # 检查预估时长
            if file_info["estimated_duration_min"] > self.config.max_audio_duration_minutes:
                self.large_file_stats["skipped_by_duration"] += 1
                return False, f"预估时长过长 ({file_info['estimated_duration_min']:.1f}min > {self.config.max_audio_duration_minutes}min)", file_info
            
            # 检查GPU内存需求
            if torch.cuda.is_available():
                gpu_total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_limit = gpu_total_memory_gb * getattr(self.config, 'gpu_memory_usage_limit', 0.8)
                if file_info["memory_requirement_gb"] > memory_limit:
                    return False, f"预估GPU内存需求过高 ({file_info['memory_requirement_gb']:.1f}GB > {memory_limit:.1f}GB)", file_info
            
            # 标记为大文件（但可以处理）
            large_file_threshold = getattr(self.config, 'large_file_warning_mb', 200.0)
            if file_size_mb > large_file_threshold:
                self.large_file_stats["total_large_files"] += 1
            
            return True, "文件检查通过", file_info
            
        except Exception as e:
            return False, f"文件检查失败: {e}", {}
    
    def get_processing_recommendations(self, file_info: Dict[str, Any]) -> List[str]:
        """根据文件信息提供处理建议"""
        recommendations = []
        
        size_mb = file_info.get("size_mb", 0)
        
        if size_mb > 100:
            recommendations.append("🔍 建议使用隔离GPU模式处理大文件")
        
        if size_mb > 300:
            recommendations.append("⚠️ 处理前确保GPU有足够显存 (>4GB)")
            recommendations.append("💾 建议关闭其他GPU程序释放显存")
        
        if size_mb > 500:
            recommendations.append("🚨 极大文件，建议分段处理或使用更大显存的GPU")
            recommendations.append("⏰ 预计处理时间较长，请耐心等待")
        
        return recommendations
    
    def handle_memory_error(self, file_path: str, error_type: str) -> str:
        """处理内存错误，返回处理建议"""
        file_size_mb = 0
        try:
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        except:
            pass
        
        suggestions = []
        
        if error_type == "cuda_memory":
            self.large_file_stats["failed_by_memory"] += 1
            suggestions.extend([
                f"🚨 GPU显存不足，无法处理 {file_size_mb:.1f}MB 大文件",
                "💡 建议解决方案:",
                "   1. 使用显存更大的GPU (建议>8GB)",
                "   2. 关闭其他占用GPU的程序",
                "   3. 将文件分割成较小片段处理",
                "   4. 调整配置降低max_file_size_mb限制"
            ])
        
        elif error_type == "system_memory":
            self.large_file_stats["failed_by_memory"] += 1
            suggestions.extend([
                f"🚨 系统内存不足，无法处理 {file_size_mb:.1f}MB 大文件",
                "💡 建议解决方案:",
                "   1. 增加系统内存 (建议>16GB)",
                "   2. 关闭其他占用内存的程序",
                "   3. 重启系统清理内存",
                "   4. 使用内存更大的服务器"
            ])
        
        elif error_type == "timeout":
            self.large_file_stats["failed_by_timeout"] += 1
            suggestions.extend([
                f"🚨 处理超时，{file_size_mb:.1f}MB 文件耗时过长",
                "💡 建议解决方案:",
                "   1. 增加process_timeout_minutes配置值",
                "   2. 使用更快的GPU (如RTX 4090, A100等)",
                "   3. 检查文件是否损坏",
                "   4. 考虑分段处理长音频"
            ])
        
        return "\n".join(suggestions)
    
    def print_protection_summary(self):
        """打印保护策略执行摘要"""
        if self.large_file_stats["total_large_files"] == 0:
            return
        
        print("\n" + "="*60)
        print("🛡️ 大文件保护策略执行摘要")
        print("="*60)
        
        total = self.large_file_stats["total_large_files"]
        skipped_size = self.large_file_stats["skipped_by_size"]
        skipped_duration = self.large_file_stats["skipped_by_duration"]
        failed_memory = self.large_file_stats["failed_by_memory"]
        failed_timeout = self.large_file_stats["failed_by_timeout"]
        successful = self.large_file_stats["successfully_processed"]
        
        print(f"检测到大文件总数: {total}")
        print(f"按大小跳过:       {skipped_size} 个 (>{self.config.max_file_size_mb}MB)")
        print(f"按时长跳过:       {skipped_duration} 个 (>{self.config.max_audio_duration_minutes}min)")
        print(f"内存不足失败:     {failed_memory} 个")
        print(f"处理超时失败:     {failed_timeout} 个")
        print(f"成功处理:         {successful} 个")
        
        if skipped_size + skipped_duration + failed_memory + failed_timeout > 0:
            print(f"\n💡 优化建议:")
            if skipped_size > 0:
                print(f"   - 考虑提高max_file_size_mb限制 (当前{self.config.max_file_size_mb}MB)")
            if failed_memory > 0:
                print(f"   - 使用更大显存的GPU或增加系统内存")
            if failed_timeout > 0:
                print(f"   - 增加process_timeout_minutes配置 (当前{self.config.process_timeout_minutes}min)")
        
        print("="*60)


class SystemDiagnostics:
    """系统诊断类"""
    
    @staticmethod
    def diagnose_performance(config=None):
        """诊断系统性能并提供优化建议"""
        print("\n" + "=" * 60)
        print("性能诊断")
        print("=" * 60)
        
        # 检查GPU信息
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"检测到 {num_gpus} 个GPU:")
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # 检查CPU信息
        try:
            import psutil
            cpu_count = psutil.cpu_count(logical=False)
            cpu_count_logical = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            print(f"CPU: {cpu_count} 物理核心, {cpu_count_logical} 逻辑核心")
            print(f"内存: {memory_gb:.1f} GB")
        except ImportError:
            cpu_count = os.cpu_count()
            memory_gb = 32  # 默认假设32GB
            print(f"CPU: {cpu_count} 核心 (无法获取详细信息)")
            print(f"内存: 无法检测 (假设32GB)")
        
        # 提供优化建议
        print("\n优化建议:")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus >= 4:
            print("✓ 检测到多个GPU，推荐使用隔离GPU处理模式")
        elif num_gpus >= 2:
            print("✓ 检测到多个GPU，可以使用多GPU并行处理")
        else:
            print("! 单GPU环境，建议增加batch_size")
        
        print("\n💡 性能优化提示:")
        print("  - 隔离GPU模式提供最佳的GPU利用率")
        print("  - ClearVoice模型支持实时语音增强")
        print("  - 调整batch_size以优化内存使用")
        print("  - 使用多进程模式避免GIL限制")
        
        print("\n🛡️ 大文件保护机制:")
        if config is not None:
            print(f"  - 自动跳过超大文件(>{config.max_file_size_mb}MB或>{config.max_audio_duration_minutes}分钟)")
            print("  - GPU内存实时监控，不足时自动清理")
            print(f"  - 进程崩溃自动重试(最多{config.max_retry_attempts}次)")
            print(f"  - 进程超时保护({config.process_timeout_minutes}分钟)")
            print(f"  - 单文件处理超时保护({config.gpu_timeout_seconds//60}分钟)")
        else:
            print("  - 自动跳过超大文件")
            print("  - GPU内存实时监控，不足时自动清理")
            print("  - 进程崩溃自动重试")
            print("  - 进程超时保护")
            print("  - 单文件处理超时保护")
        print("=" * 60)


def create_default_config() -> MossFormerGANConfig:
    """创建默认配置"""
    return MossFormerGANConfig()


def display_file_processing_status(files_to_process: List[Tuple[str, str]], 
                                   files_to_skip: List[Tuple[str, str]], 
                                   max_display: int = 10,
                                   config: MossFormerGANConfig = None):
    """
    显示文件处理状态
    
    Args:
        files_to_process: 要处理的文件列表
        files_to_skip: 要跳过的文件列表
        max_display: 最大显示文件数量
    """
    print("\n" + "=" * 80)
    print("文件处理状态")
    print("=" * 80)
    
    # 显示要处理的文件
    if files_to_process:
        print(f"\n📋 要处理的文件 ({len(files_to_process)} 个):")
        print("-" * 50)
        for i, (input_file, output_file) in enumerate(files_to_process[:max_display]):
            rel_input = os.path.relpath(input_file)
            rel_output = os.path.relpath(output_file)
            print(f"  {i+1:2d}. {rel_input}")
            print(f"      -> {rel_output}")
        
        if len(files_to_process) > max_display:
            print(f"      ... 还有 {len(files_to_process) - max_display} 个文件")
    else:
        print("\n📋 要处理的文件: 无")
    
    # 显示要跳过的文件
    if files_to_skip:
        print(f"\n⏭️  要跳过的文件 ({len(files_to_skip)} 个):")
        print("-" * 50)
        for i, (input_file, output_file) in enumerate(files_to_skip[:max_display]):
            rel_input = os.path.relpath(input_file)
            rel_output = os.path.relpath(output_file)
            print(f"  {i+1:2d}. {rel_input} (已存在)")
            print(f"      -> {rel_output}")
        
        if len(files_to_skip) > max_display:
            print(f"      ... 还有 {len(files_to_skip) - max_display} 个文件")
    else:
        print("\n⏭️  要跳过的文件: 无")
    
    # 显示总结
    total_files = len(files_to_process) + len(files_to_skip)
    print(f"\n📊 文件统计:")
    print(f"   总文件数: {total_files}")
    print(f"   需处理:   {len(files_to_process)}")
    print(f"   跳过:     {len(files_to_skip)}")
    
    if total_files > 0:
        process_rate = (len(files_to_process) / total_files) * 100
        print(f"   处理率:   {process_rate:.1f}%")
    
    # 如果有跳过的文件，提示用户如何重新处理
    if files_to_skip:
        print(f"\n💡 提示: 如需重新处理已存在的文件，请修改配置:")
        print(f"   在第92行将 skip_existing 改为 False")
    
    # 大文件处理警告和建议
    large_files = []
    for input_file, output_file in files_to_process:
        try:
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
            large_file_threshold = getattr(config, 'large_file_warning_mb', 200.0)
            if file_size_mb > large_file_threshold:
                large_files.append((input_file, file_size_mb))
        except:
            pass
    
    if large_files:
        large_file_threshold = getattr(config, 'large_file_warning_mb', 200.0)
        print(f"\n⚠️ 检测到 {len(large_files)} 个大文件 (>{large_file_threshold}MB):")
        print("=" * 50)
        for i, (file_path, size_mb) in enumerate(large_files[:5]):  # 只显示前5个
            print(f"   {i+1}. {os.path.basename(file_path)} ({size_mb:.1f}MB)")
        if len(large_files) > 5:
            print(f"   ... 还有 {len(large_files)-5} 个大文件")
        
        print(f"\n🛡️ 大文件保护策略已启用:")
        print(f"   ✓ 自动跳过超大文件 (>{config.max_file_size_mb}MB或>{config.max_audio_duration_minutes}分钟)")
        print(f"   ✓ GPU内存实时监控和自动清理")
        print(f"   ✓ 进程崩溃自动重试 (最多2次)")
        print(f"   ✓ 进程/文件处理超时保护")
        print(f"   ✓ 详细错误分类和处理建议")
    
    print("=" * 80)


def main():
    """主函数"""
    print("MossFormerGAN批量语音增强处理器（重构版）")
    print("=" * 60)
    
    try:
        # 创建配置
        config = create_default_config()
        
        # 性能诊断
        SystemDiagnostics.diagnose_performance(config)
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 创建处理器
        processor = MossFormerGANBatchProcessor(config)
        
        # 如果不是隔离GPU模式，进行模型测试
        if config.batch_processing_mode != 'isolated_gpu':
            processor.test_model_functionality()
        
        # 获取所有音频文件
        print("扫描音频文件...")
        files_to_process, files_to_skip, all_files = processor.get_audio_files()
        
        # 显示文件处理状态
        display_file_processing_status(files_to_process, files_to_skip, config=config)
        
        # 检查是否有文件需要处理
        if not files_to_process:
            print("没有文件需要处理，程序结束")
            return 0
        
        processor.stats.total_files = len(files_to_process) + len(files_to_skip)
        processor.stats.skipped_files = len(files_to_skip)  # 预设跳过的文件数量
        print(f"找到 {len(files_to_process)} 个音频文件需要处理，{len(files_to_skip)} 个文件已跳过")
        
        # 显示配置信息
        print(f"\n配置信息:")
        print(f"输入目录: {config.input_dir}")
        print(f"输出目录: {config.output_dir}")
        print(f"模型: {config.model_name}")
        print(f"任务: {config.task}")
        print(f"使用设备: {config.device}")
        print(f"可用GPU: {processor.available_gpus if processor.available_gpus else 'CPU'}")
        print(f"多GPU模式: {'启用' if config.use_multi_gpu else '禁用'}")
        print(f"处理模式: {config.batch_processing_mode}")
        print(f"并行模式: {'多进程' if config.use_multiprocess else '多线程'}")
        print(f"批处理大小: {config.batch_size}")
        print(f"目标采样率: {config.target_sr}Hz")
        print(f"异步处理管道: {'启用' if config.enable_async_pipeline else '禁用'}")
        print(f"管道队列大小: {config.pipeline_queue_size} (预处理缓冲)")
        print(f"每GPU工作线程: {config.max_workers_per_gpu}")
        print(f"GPU处理超时: {config.gpu_timeout_seconds}秒")
        print(f"FFT下采样: {'启用' if config.use_fft_downsample else '禁用'}")
        print(f"🛡️ 大文件保护: 单文件限制 {config.max_file_size_mb}MB / {config.max_audio_duration_minutes}分钟")
        print(f"🔄 进程恢复: {'启用' if config.enable_process_recovery else '禁用'} (最多重试{config.max_retry_attempts}次)")
        print(f"⏱️ 进程超时: {config.process_timeout_minutes}分钟")
        
        # 根据GPU显存给出配置建议
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"\n💡 根据您的GPU显存 ({gpu_memory_gb:.1f}GB) 的配置建议:")
            
            if gpu_memory_gb >= 24:  # 高端GPU (如A100, RTX 4090)
                print(f"   🚀 高端GPU配置: 可处理更大文件")
                print(f"   建议配置: max_file_size_mb=1000, max_audio_duration_minutes=120")
            elif gpu_memory_gb >= 12:  # 中高端GPU (如RTX 3080Ti, 4070Ti)
                print(f"   ⚡ 中高端GPU配置: 当前配置适中")
                print(f"   建议配置: max_file_size_mb=800, max_audio_duration_minutes=90")
            elif gpu_memory_gb >= 8:  # 中端GPU (如RTX 3070, 4060Ti)
                print(f"   ✅ 中端GPU配置: 当前配置合理")
                print(f"   建议配置: max_file_size_mb=500, max_audio_duration_minutes=60 (当前设置)")
            elif gpu_memory_gb >= 4:  # 入门GPU (如GTX 1660, RTX 3050)
                print(f"   ⚠️ 入门GPU配置: 建议降低限制")
                print(f"   建议配置: max_file_size_mb=200, max_audio_duration_minutes=30")
            else:  # 低端GPU
                print(f"   🔴 低端GPU配置: 建议显著降低限制")
                print(f"   建议配置: max_file_size_mb=100, max_audio_duration_minutes=15")
            
            print(f"   📝 修改方法: 在配置类 (第110-116行) 中调整相应参数")
        
        # 开始处理
        print("\n开始批量处理...")
        start_time = time.time()
        
        # 选择处理方式
        if config.use_parallel:
            print("使用并行处理模式...")
            results = processor.process_files_parallel(files_to_process)
        else:
            print("使用批处理模式...")
            results = processor.process_files_batch(files_to_process)
        
        # 记录处理时间
        processor.stats.processing_time = time.time() - start_time
        
        # 显示结果
        processor.stats.print_summary(config)
        
        # 显示大文件保护策略摘要
        processor.protection_strategy.print_protection_summary()
        
        # 显示失败的文件和处理建议
        failed_files = [message for success, message in results if not success]
        if failed_files:
            print(f"\n❌ 失败的文件 ({len(failed_files)}):")
            print("-" * 60)
            
            # 分类错误类型
            memory_errors = []
            timeout_errors = []
            large_file_errors = []
            other_errors = []
            
            for message in failed_files:
                message_lower = message.lower()
                if "显存不足" in message_lower or "内存不足" in message_lower or "out of memory" in message_lower:
                    memory_errors.append(message)
                elif "超时" in message_lower or "timeout" in message_lower:
                    timeout_errors.append(message)
                elif "文件过大" in message_lower or "时长过长" in message_lower:
                    large_file_errors.append(message)
                else:
                    other_errors.append(message)
            
            # 显示分类错误
            if memory_errors:
                print(f"\n🚨 内存相关错误 ({len(memory_errors)} 个):")
                for i, message in enumerate(memory_errors[:5]):
                    print(f"   {i+1}. {message}")
                if len(memory_errors) > 5:
                    print(f"   ... 还有 {len(memory_errors)-5} 个内存错误")
                    
                print(f"\n💡 内存错误解决建议:")
                print(f"   • 使用更大显存的GPU (推荐 ≥8GB)")
                print(f"   • 关闭其他占用GPU/内存的程序")
                print(f"   • 降低配置中的max_file_size_mb限制")
                print(f"   • 将大音频文件分割成小段处理")
            
            if timeout_errors:
                print(f"\n⏰ 超时错误 ({len(timeout_errors)} 个):")
                for i, message in enumerate(timeout_errors[:5]):
                    print(f"   {i+1}. {message}")
                if len(timeout_errors) > 5:
                    print(f"   ... 还有 {len(timeout_errors)-5} 个超时错误")
                    
                print(f"\n💡 超时错误解决建议:")
                print(f"   • 增加配置中的process_timeout_minutes (当前{config.process_timeout_minutes}分钟)")
                print(f"   • 使用更快的GPU硬件")
                print(f"   • 检查音频文件是否损坏")
                print(f"   • 考虑分批处理大文件")
            
            if large_file_errors:
                print(f"\n📏 大文件错误 ({len(large_file_errors)} 个):")
                for i, message in enumerate(large_file_errors[:5]):
                    print(f"   {i+1}. {message}")
                if len(large_file_errors) > 5:
                    print(f"   ... 还有 {len(large_file_errors)-5} 个大文件错误")
                    
                print(f"\n💡 大文件错误解决建议:")
                print(f"   • 提高配置中的max_file_size_mb限制 (当前{config.max_file_size_mb}MB)")
                print(f"   • 提高配置中的max_audio_duration_minutes限制 (当前{config.max_audio_duration_minutes}分钟)")
                print(f"   • 使用专业音频处理服务器")
                print(f"   • 考虑音频压缩或重采样预处理")
            
            if other_errors:
                print(f"\n❓ 其他错误 ({len(other_errors)} 个):")
                for i, message in enumerate(other_errors[:5]):
                    print(f"   {i+1}. {message}")
                if len(other_errors) > 5:
                    print(f"   ... 还有 {len(other_errors)-5} 个其他错误")
            
            print(f"\n🔧 通用恢复步骤:")
            print(f"   1. 重启程序清理内存缓存")
            print(f"   2. 检查输入文件的完整性")
            print(f"   3. 调整保护策略配置参数")
            print(f"   4. 考虑升级硬件配置")
            print(f"   5. 分批处理减少单次负载")
        
        print("\n处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 