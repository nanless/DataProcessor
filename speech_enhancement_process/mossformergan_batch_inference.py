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
    input_dir: str = "/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads"
    output_dir: str = "/root/code/github_repos/DataCrawler/ximalaya_downloader/downloads_mossformer_enhanced"
    
    # 模型配置
    model_name: str = "MossFormerGAN_SE_16K"
    task: str = 'speech_enhancement'
    
    # 硬件配置
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
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
    完全隔离的GPU处理函数
    
    Args:
        args: 包含文件列表、GPU ID和其他参数的元组
        
    Returns:
        处理结果列表
    """
    file_list, gpu_id, model_name, task, target_sr, input_dir, output_dir, position = args
    
    try:
        # 强制设置GPU设备 - 完全隔离
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 重新导入所需库 - 这很重要，必须在设置环境变量后导入
        import torch
        import numpy as np
        import soundfile as sf
        from tqdm import tqdm
        
        # 等待一小段时间让CUDA正确初始化
        import time
        time.sleep(1)
        
        # 设置线程数
        torch.set_num_threads(8)
        
        # 验证CUDA可用性并初始化
        if not torch.cuda.is_available():
            print(f"GPU {gpu_id} 进程: CUDA不可用")
            return [(False, "CUDA不可用") for _ in file_list]
        
        # 现在GPU 0对应于实际的指定GPU
        device_count = torch.cuda.device_count()
        if device_count == 0:
            print(f"GPU {gpu_id} 进程: 没有检测到GPU设备")
            return [(False, "没有检测到GPU设备") for _ in file_list]
        
        # 设置设备并清空缓存
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        
        # 验证设备信息
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
        
        # 在进程中导入ClearVoice
        try:
            from clearvoice import ClearVoice
            
            # 创建ClearVoice实例
            clearvoice_instance = ClearVoice(task=task, model_names=[model_name])
            
            print(f"GPU {gpu_id} 模型加载成功，开始处理 {len(file_list)} 个文件")
            
            # 创建进度条
            pbar = tqdm(file_list, desc=f"GPU {gpu_id}", position=position, leave=True)
            
            results = []
            for i, (input_file, output_file) in enumerate(pbar):
                try:
                    # 检查是否跳过已存在的文件
                    if os.path.exists(output_file):
                        results.append((True, f"跳过已存在文件: {input_file}"))
                        continue
                    
                    # 确保输出目录存在
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    
                    # 加载和验证音频文件 - 支持FFT下采样
                    from types import SimpleNamespace
                    gpu_config = SimpleNamespace()
                    gpu_config.use_fft_downsample = True  # 启用FFT下采样
                    gpu_config.fft_downsample_quality = "high"  # 高质量
                    gpu_config.anti_alias_filter = True  # 抗混叠滤波
                    gpu_config.target_sr = target_sr  # 目标采样率
                    
                    audio, sr = AudioProcessor.load_and_validate_audio(input_file, gpu_config)
                    
                    if len(audio) == 0:
                        raise ValueError("音频文件为空")
                    
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
                    
                    # 调试：检查增强音频的类型和属性
                    print(f"调试信息 - 增强音频类型: {type(enhanced_audio)}")
                    if hasattr(enhanced_audio, 'shape'):
                        print(f"调试信息 - 增强音频形状: {enhanced_audio.shape}")
                    if hasattr(enhanced_audio, 'dtype'):
                        print(f"调试信息 - 增强音频数据类型: {enhanced_audio.dtype}")
                    
                    # 保存结果 - 不使用ClearVoice的write方法，使用自定义保存确保采样率控制 (16kHz)
                    if enhanced_audio is not None:
                        try:
                            # 使用模型的采样率 (16kHz)
                            model_sr = 16000
                            
                            # 处理不同格式的增强音频数据
                            audio_data = None
                            
                            # 检查ClearVoice返回的数据类型
                            import numpy as np
                            import torch
                            
                            if isinstance(enhanced_audio, np.ndarray):
                                audio_data = enhanced_audio.astype(np.float32)
                                print(f"调试信息 - 检测到numpy数组，形状: {audio_data.shape}")
                            elif isinstance(enhanced_audio, torch.Tensor):
                                audio_data = enhanced_audio.detach().cpu().numpy().astype(np.float32)
                                print(f"调试信息 - 检测到torch张量，转换后形状: {audio_data.shape}")
                            elif isinstance(enhanced_audio, list):
                                audio_data = np.array(enhanced_audio, dtype=np.float32)
                                print(f"调试信息 - 检测到列表，转换后形状: {audio_data.shape}")
                            elif hasattr(enhanced_audio, 'numpy'):
                                # 某些类型可能有numpy()方法
                                audio_data = enhanced_audio.numpy().astype(np.float32)
                                print(f"调试信息 - 使用numpy()方法，形状: {audio_data.shape}")
                            else:
                                # 尝试直接转换
                                audio_data = np.array(enhanced_audio, dtype=np.float32)
                                print(f"调试信息 - 直接转换，形状: {audio_data.shape}")
                            
                            # 验证音频数据
                            if audio_data is None or len(audio_data) == 0:
                                raise ValueError("转换后的音频数据为空")
                            
                            # 处理多维数组 - 确保是1D
                            if audio_data.ndim > 1:
                                if audio_data.shape[0] == 1:
                                    audio_data = audio_data.squeeze(0)
                                elif audio_data.shape[1] == 1:
                                    audio_data = audio_data.squeeze(1)
                                else:
                                    # 多通道，取第一个通道
                                    audio_data = audio_data[0] if audio_data.shape[0] < audio_data.shape[1] else audio_data[:, 0]
                                
                                print(f"调试信息 - 降维后形状: {audio_data.shape}")
                            
                            # 使用AudioProcessor的save_audio方法保存
                            AudioProcessor.save_audio(audio_data, output_file, model_sr)
                        except Exception as save_error:
                            # 如果自定义保存失败，尝试使用soundfile直接保存
                            print(f"主要保存方法失败: {save_error}")
                            try:
                                import soundfile as sf
                                import numpy as np
                                import time
                                model_sr = 16000
                                
                                # 重新处理音频数据（复用上面的逻辑）
                                backup_audio_data = None
                                if isinstance(enhanced_audio, np.ndarray):
                                    backup_audio_data = enhanced_audio.astype(np.float32)
                                elif isinstance(enhanced_audio, torch.Tensor):
                                    backup_audio_data = enhanced_audio.detach().cpu().numpy().astype(np.float32)
                                elif isinstance(enhanced_audio, list):
                                    backup_audio_data = np.array(enhanced_audio, dtype=np.float32)
                                else:
                                    backup_audio_data = np.array(enhanced_audio, dtype=np.float32)
                                
                                # 处理多维数组
                                if backup_audio_data.ndim > 1:
                                    if backup_audio_data.shape[0] == 1:
                                        backup_audio_data = backup_audio_data.squeeze(0)
                                    elif backup_audio_data.shape[1] == 1:
                                        backup_audio_data = backup_audio_data.squeeze(1)
                                    else:
                                        backup_audio_data = backup_audio_data[0] if backup_audio_data.shape[0] < backup_audio_data.shape[1] else backup_audio_data[:, 0]
                                
                                # 归一化
                                if np.abs(backup_audio_data).max() > 1.0:
                                    backup_audio_data = backup_audio_data / np.abs(backup_audio_data).max()
                                
                                # 尝试多种保存格式
                                formats_to_try = [
                                    ('WAV', 'PCM_16'),
                                    ('WAV', 'PCM_24'),
                                    ('WAV', 'FLOAT'),
                                    (None, None)  # 让soundfile自动选择
                                ]
                                
                                saved_successfully = False
                                for fmt, subtype in formats_to_try:
                                    try:
                                        # 创建备用文件路径（简化文件名）
                                        backup_path = os.path.join(
                                            os.path.dirname(output_file),
                                            f"backup_gpu{gpu_id}_{int(time.time())}_{os.getpid()}.wav"
                                        )
                                        
                                        if fmt and subtype:
                                            sf.write(backup_path, backup_audio_data, model_sr, format=fmt, subtype=subtype)
                                        else:
                                            sf.write(backup_path, backup_audio_data, model_sr)
                                        
                                        # 如果成功保存，尝试重命名
                                        if os.path.exists(backup_path):
                                            try:
                                                import shutil
                                                shutil.move(backup_path, output_file)
                                                saved_successfully = True
                                                print(f"备用保存成功，格式: {fmt}/{subtype}")
                                                break
                                            except Exception as rename_error:
                                                print(f"重命名失败，但文件已保存到: {backup_path}")
                                                saved_successfully = True
                                                break
                                    except Exception as format_error:
                                        print(f"格式 {fmt}/{subtype} 保存失败: {format_error}")
                                        continue
                                
                                if not saved_successfully:
                                    raise Exception("所有备用保存格式都失败了")
                                        
                            except Exception as final_error:
                                print(f"所有保存方法都失败: 主要错误={save_error}, 最终错误={final_error}")
                                # 尝试保存原始音频数据用于调试
                                try:
                                    debug_path = os.path.join(
                                        os.path.dirname(output_file),
                                        f"debug_gpu{gpu_id}_{int(time.time())}_{os.getpid()}.txt"
                                    )
                                    with open(debug_path, 'w') as f:
                                        f.write(f"Enhanced audio type: {type(enhanced_audio)}\n")
                                        f.write(f"Enhanced audio: {str(enhanced_audio)[:500]}...\n")
                                    print(f"调试信息已保存到: {debug_path}")
                                except:
                                    pass
                                raise
                        
                        # 检查结果
                        if os.path.exists(output_file):
                            results.append((True, f"成功处理: {input_file}"))
                            pbar.set_postfix({"状态": "成功"})
                        else:
                            results.append((False, f"处理失败，输出文件未生成: {input_file}"))
                            pbar.set_postfix({"状态": "失败"})
                    else:
                        results.append((False, f"处理失败，无音频数据: {input_file}"))
                        pbar.set_postfix({"状态": "失败"})
                    
                except Exception as e:
                    results.append((False, f"处理失败 {input_file}: {e}"))
                    pbar.set_postfix({"状态": "异常"})
            
            pbar.close()
            
            # 统计结果
            success_count = sum(1 for success, _ in results if success)
            print(f"GPU {gpu_id} 处理完成: {success_count}/{len(file_list)} 成功")
            
            return results
            
        except Exception as e:
            print(f"GPU {gpu_id} 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return [(False, f"模型初始化失败: {e}") for _ in file_list]
            
    except Exception as e:
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return [(False, f"进程初始化失败: {e}") for _ in file_list]


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
    
    def update_from_results(self, results: List[Tuple[bool, str]]):
        """从结果更新统计信息"""
        for success, message in results:
            if success:
                if "跳过已存在文件" in message:
                    self.skipped_files += 1
                else:
                    self.processed_files += 1
            else:
                self.failed_files += 1
    
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
        
        # 显示GPU使用情况
        if config.use_multi_gpu:
            print(f"处理模式:     {'隔离GPU模式' if config.batch_processing_mode == 'isolated_gpu' else '多GPU模式'}")
            print(f"使用GPU数:    {len(config.gpu_ids)}")
            print(f"GPU列表:      {config.gpu_ids}")
        
        print("=" * 60)


class MossFormerGANBatchProcessor:
    """MossFormerGAN批量处理器"""
    
    def __init__(self, config: MossFormerGANConfig):
        self.config = config
        self.stats = ProcessingStats()
        
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
            audio, sr = AudioProcessor.load_and_validate_audio(input_file, self.config)
            
            # 在指定GPU上处理 - 使用ClearVoice进行语音增强（文件进文件出）
            # 创建临时文件保存下采样后的音频
            
            # 创建临时文件
            temp_dir = tempfile.gettempdir()
            temp_filename = f"temp_audio_gpu{gpu_id}_{int(time.time())}_{os.getpid()}_single.wav"
            temp_input_path = os.path.join(temp_dir, temp_filename)
            
            try:
                # 保存音频到临时文件
                sf.write(temp_input_path, audio, sr, format='WAV', subtype='PCM_16')
                
                # 使用ClearVoice处理临时文件（文件进文件出）
                with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                    enhanced_audio = clearvoice_instance(temp_input_path, online_write=False)
                    
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
            return False, f"处理失败 {input_file}: {e}"
    
    def get_audio_files(self) -> List[Tuple[str, str]]:
        """获取所有音频文件路径"""
        audio_files = []
        
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
                    
                    audio_files.append((input_path, output_path))
        
        return audio_files
    
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
        
        # 打乱文件列表以实现GPU负载均衡
        shuffled_file_pairs = file_pairs.copy()
        random.shuffle(shuffled_file_pairs)
        logger.info("已打乱文件列表以优化GPU负载均衡")
        
        # 将文件分配给各个GPU
        num_gpus = len(self.available_gpus)
        files_per_gpu = len(shuffled_file_pairs) // num_gpus
        remainder = len(shuffled_file_pairs) % num_gpus
        
        gpu_file_batches = []
        start_idx = 0
        
        for i, gpu_id in enumerate(self.available_gpus):
            # 计算当前GPU应该处理的文件数量
            current_batch_size = files_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            # 分配文件给当前GPU
            gpu_files = shuffled_file_pairs[start_idx:end_idx]
            gpu_file_batches.append((gpu_files, gpu_id, i))
            
            logger.info(f"GPU {gpu_id}: 分配 {len(gpu_files)} 个文件")
            start_idx = end_idx
        
        # 使用多进程处理
        all_results = []
        with mp.Pool(processes=num_gpus) as pool:
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
                    position
                )
                args_list.append(args)
            
            # 提交任务
            results = pool.map(process_gpu_batch_isolated, args_list)
            
            # 合并结果
            for result_batch in results:
                all_results.extend(result_batch)
        
        # 更新统计信息
        self.stats.update_from_results(all_results)
        
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
    
    def test_model_functionality(self):
        """测试模型功能"""
        if not torch.cuda.is_available():
            logger.info("CUDA不可用，跳过模型测试")
            return
        
        logger.info("测试模型功能...")
        
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
        
        logger.info("模型测试完成")


class SystemDiagnostics:
    """系统诊断类"""
    
    @staticmethod
    def diagnose_performance():
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
        print("=" * 60)


def create_default_config() -> MossFormerGANConfig:
    """创建默认配置"""
    return MossFormerGANConfig()


def main():
    """主函数"""
    print("MossFormerGAN批量语音增强处理器（重构版）")
    print("=" * 60)
    
    try:
        # 创建配置
        config = create_default_config()
        
        # 性能诊断
        SystemDiagnostics.diagnose_performance()
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 创建处理器
        processor = MossFormerGANBatchProcessor(config)
        
        # 如果不是隔离GPU模式，进行模型测试
        if config.batch_processing_mode != 'isolated_gpu':
            processor.test_model_functionality()
        
        # 获取所有音频文件
        print("扫描音频文件...")
        audio_files = processor.get_audio_files()
        
        if not audio_files:
            print("未找到任何音频文件")
            return
        
        processor.stats.total_files = len(audio_files)
        print(f"找到 {len(audio_files)} 个音频文件")
        
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
        
        # 开始处理
        print("\n开始批量处理...")
        start_time = time.time()
        
        # 选择处理方式
        if config.use_parallel:
            print("使用并行处理模式...")
            results = processor.process_files_parallel(audio_files)
        else:
            print("使用批处理模式...")
            results = processor.process_files_batch(audio_files)
        
        # 记录处理时间
        processor.stats.processing_time = time.time() - start_time
        
        # 显示结果
        processor.stats.print_summary(config)
        
        # 显示失败的文件
        failed_files = [message for success, message in results if not success]
        if failed_files:
            print(f"\n失败的文件 ({len(failed_files)}):")
            for i, message in enumerate(failed_files[:10]):  # 只显示前10个
                print(f"  {i+1}. {message}")
            if len(failed_files) > 10:
                print(f"  ... 还有 {len(failed_files)-10} 个失败的文件")
        
        print("\n处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 