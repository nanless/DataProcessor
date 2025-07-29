#!/usr/bin/env python3
"""
Resemble-Enhance批量语音增强处理器

该脚本提供了一个高效的批量语音增强解决方案，支持多GPU并行处理，
具有动态长度处理、内存优化等特性。

主要功能:
- 批量处理多种音频格式 (wav, mp3, flac, m4a, mp4)
- 多GPU并行处理
- 动态长度处理避免不必要的音频填充
- 内存优化和异步I/O
- 支持分块处理长音频
- 详细的性能统计和诊断
- 多通道音频自动转换为单声道（取第一个通道）
- 自动转换到模型采样率并保存（44.1kHz）
- 智能文件名处理（支持长文件名和特殊字符清理，空格替换为下划线）
- 正确处理多维音频张量格式

依赖要求:
- 处理MP4和M4A文件需要安装: pip install pydub
- 输出采样率: 44.1kHz (模型固定采样率，与输入采样率无关)
- 自动处理长文件名和特殊字符，避免保存错误
- 修复多维音频张量导致的保存错误

作者: AI Assistant
版本: 2.3
"""

import os
import multiprocessing as mp
import time
import threading
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pydub import AudioSegment
from tqdm import tqdm, trange
from torch.nn.utils.parametrize import remove_parametrizations
from torchaudio.transforms import MelSpectrogram

# 设置multiprocessing启动方法为spawn（CUDA多进程必需）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，会抛出RuntimeError，可以忽略
    pass

# 配置日志
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """处理配置类"""
    
    # 路径配置
    input_dir: str = "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid"
    output_dir: str = "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_resemble_enhanced_2"
    run_dir: str = "/root/data/pretrained_models/resemble-enhance/enhancer_stage2"
    
    # 硬件配置
    device: str = "cuda"
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    
    # 音频配置
    target_sr: int = 44100  # 模型的采样率（输出音频将以此采样率保存）
    skip_existing: bool = True  # 跳过已存在的文件
    
    # FFT下采样配置
    use_fft_downsample: bool = True  # 是否使用FFT下采样
    fft_downsample_quality: str = "high"  # FFT下采样质量: "high", "medium", "low"
    anti_alias_filter: bool = True  # 是否使用抗混叠滤波器
    
    # resemble_enhance 参数
    nfe: int = 64
    solver: str = "midpoint"
    lambd: float = 0.5
    tau: float = 0.5
    
    # 性能优化参数
    batch_size: int = 64  # 批处理大小
    max_length: int = 2205000  # 最大长度（50秒@44kHz）
    min_chunk_length: int = 1323000  # 分块阈值（30秒）
    chunk_seconds: float = 20.0  # 分块大小（秒）
    overlap_seconds: float = 0.5  # 重叠时间（秒）
    num_workers: int = 8  # I/O线程数
    prefetch_factor: int = 2  # 预取因子
    use_mixed_precision: bool = False  # 混合精度（resemble_enhance不支持复数半精度）
    optimize_memory: bool = True  # 内存优化
    dynamic_length: bool = True  # 动态长度处理
    
    def __post_init__(self):
        """初始化后的验证"""
        if not os.path.exists(self.input_dir):
            raise ValueError(f"输入目录不存在: {self.input_dir}")
        
        if not os.path.exists(self.run_dir):
            raise ValueError(f"模型目录不存在: {self.run_dir}")
        
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA不可用")
        
        # 验证GPU ID
        available_gpus = list(range(torch.cuda.device_count()))
        invalid_gpus = [gpu_id for gpu_id in self.gpu_ids if gpu_id not in available_gpus]
        if invalid_gpus:
            raise ValueError(f"无效的GPU ID: {invalid_gpus}, 可用GPU: {available_gpus}")


class AudioProcessor:
    """音频处理工具类"""
    
    SUPPORTED_FORMATS = ('.wav', '.mp3', '.flac', '.m4a', '.mp4')
    
    @staticmethod
    def load_audio(file_path: str, config: Optional['ProcessingConfig'] = None) -> Tuple[torch.Tensor, int]:
        """
        加载音频文件
        多通道音频将被转换为单声道（取第一个通道）
        支持FFT下采样到目标采样率
        
        Args:
            file_path: 音频文件路径
            config: 配置对象，包含FFT下采样参数
            
        Returns:
            音频数据和采样率的元组
        """
        try:
            if file_path.lower().endswith(('.m4a', '.mp4')):
                # 处理m4a和mp4文件  
                try:
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
                    
                    # 转换为torch tensor
                    dwav = torch.from_numpy(audio_data)
                    sr = audio.frame_rate
                except Exception as e:
                    raise RuntimeError(f"处理MP4/M4A文件失败: {e}")
            else:
                # 直接加载其他格式的音频文件
                dwav, sr = torchaudio.load(file_path)
                
                # 转换为单声道 - 只取第一个通道
                if dwav.shape[0] > 1:
                    dwav = dwav[0]  # 只取第一个通道
                else:
                    dwav = dwav.squeeze(0)
            
            # 应用FFT下采样（如果配置启用且需要）
            if config and config.use_fft_downsample and sr != config.target_sr:
                logger.info(f"对文件 {os.path.basename(file_path)} 执行FFT下采样: {sr}Hz -> {config.target_sr}Hz")
                # 将torch tensor转换为numpy数组进行下采样
                audio_np = dwav.numpy() if isinstance(dwav, torch.Tensor) else dwav
                downsampled_audio = AudioProcessor.fft_downsample(
                    audio_np, sr, config.target_sr, 
                    quality=config.fft_downsample_quality,
                    anti_alias=config.anti_alias_filter
                )
                # 转换回torch tensor
                dwav = torch.from_numpy(downsampled_audio)
                sr = config.target_sr
            
            return dwav, sr
            
        except Exception as e:
            raise RuntimeError(f"加载音频文件失败 {file_path}: {e}")
    
    @staticmethod
    def save_audio(audio_data: torch.Tensor, file_path: str, sample_rate: int):
        """
        保存音频文件
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
            if audio_data is None or audio_data.numel() == 0:
                raise ValueError("音频数据为空")
            
            # 确保音频数据是torch.Tensor
            if not isinstance(audio_data, torch.Tensor):
                audio_data = torch.tensor(audio_data, dtype=torch.float32)
            
            # 确保数据类型正确
            if audio_data.dtype != torch.float32:
                audio_data = audio_data.float()
            
            # 确保音频数据在合理范围内
            max_val = torch.abs(audio_data).max()
            if max_val > 1.0:
                audio_data = audio_data / max_val
            
            # 确保音频数据维度正确
            print(f"调试信息 - 保存前音频形状: {audio_data.shape}")
            
            # 处理不同维度的数据
            if audio_data.dim() == 3:
                # 3D数组，取第一个batch和第一个通道
                audio_data = audio_data[0, 0]  
                print(f"调试信息 - 3D数组降维，新形状: {audio_data.shape}")
            elif audio_data.dim() == 2:
                if audio_data.shape[0] == 1:
                    # (1, N) -> (N,)，然后再转为(1, N)用于torchaudio
                    audio_data = audio_data.squeeze(0).unsqueeze(0)
                elif audio_data.shape[1] == 1:
                    # (N, 1) -> (N,)，然后再转为(1, N)用于torchaudio  
                    audio_data = audio_data.squeeze(1).unsqueeze(0)
                # 如果是(C, N)形状且C>1，保持原样（多通道）
                print(f"调试信息 - 2D数组处理，新形状: {audio_data.shape}")
            elif audio_data.dim() == 1:
                # 1D数组转为2D (1, N)
                audio_data = audio_data.unsqueeze(0)
                print(f"调试信息 - 1D数组转2D，新形状: {audio_data.shape}")
            
            print(f"调试信息 - 最终保存的音频形状: {audio_data.shape}, 数据类型: {audio_data.dtype}")
            
            # 使用torchaudio保存
            torchaudio.save(file_path, audio_data, sample_rate, format="wav", encoding="PCM_S", bits_per_sample=16)
            
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
        
        return new_file_path
    
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
                filter_type = 'kaiser_best'
            elif quality == "medium":
                res_type = 'fft'
                filter_type = 'kaiser_fast'
            else:  # low
                res_type = 'fft'
                filter_type = 'linear'
            
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
    
    @staticmethod
    def resample_audio(audio_data: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
        """
        重采样音频
        
        Args:
            audio_data: 音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            重采样后的音频数据
        """
        if orig_sr == target_sr:
            return audio_data
        
        # 确保在CPU上进行重采样
        audio_data = audio_data.cpu()
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        return resampler(audio_data)
    
    @staticmethod
    def normalize_audio(audio_data: torch.Tensor) -> torch.Tensor:
        """
        归一化音频数据
        
        Args:
            audio_data: 音频数据
            
        Returns:
            归一化后的音频数据
        """
        max_val = audio_data.abs().max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        return audio_data
    
    @staticmethod
    def validate_audio_data(audio_data: torch.Tensor, file_path: str = "") -> bool:
        """
        验证音频数据的有效性
        
        Args:
            audio_data: 音频数据
            file_path: 文件路径（用于错误信息）
            
        Returns:
            是否有效
        """
        if torch.isnan(audio_data).any() or torch.isinf(audio_data).any():
            logger.warning(f"音频数据包含NaN或Inf: {file_path}")
            return False
        return True
    
    @staticmethod
    def prepare_audio_for_inference(
        audio_data: torch.Tensor, 
        orig_sr: int, 
        target_sr: int, 
        max_length: int,
        dynamic_length: bool = True
    ) -> torch.Tensor:
        """
        为推理准备音频数据
        
        Args:
            audio_data: 音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            max_length: 最大长度
            dynamic_length: 是否使用动态长度
            
        Returns:
            准备好的音频数据
        """
        # 重采样
        audio_data = AudioProcessor.resample_audio(audio_data, orig_sr, target_sr)
        
        # 归一化
        audio_data = AudioProcessor.normalize_audio(audio_data)
        
        # 长度处理
        if dynamic_length:
            # 动态长度模式：只截断过长的音频
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
                logger.info(f"音频过长，截断到 {max_length} 样本")
        else:
            # 固定长度模式
            if len(audio_data) > max_length:
                audio_data = audio_data[:max_length]
            elif len(audio_data) < max_length:
                audio_data = torch.nn.functional.pad(
                    audio_data, 
                    (0, max_length - len(audio_data))
                )
        
        return audio_data
    
    @staticmethod
    def fft_downsample(audio_data: torch.Tensor, orig_sr: int, target_sr: int, 
                      quality: str = "high", anti_alias: bool = True) -> torch.Tensor:
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
            
            # 转换为numpy以便librosa处理
            if isinstance(audio_data, torch.Tensor):
                audio_np = audio_data.cpu().numpy()
                is_tensor = True
            else:
                audio_np = audio_data
                is_tensor = False
            
            # 确保输入是1D数组
            if audio_np.ndim > 1:
                if audio_np.shape[0] == 1:
                    audio_np = audio_np.squeeze(0)
                elif audio_np.shape[1] == 1:
                    audio_np = audio_np.squeeze(1)
                else:
                    audio_np = audio_np[0]  # 取第一个通道
            
            # 根据质量设置参数
            if quality == "high":
                res_type = 'fft'  # 使用FFT方法
            elif quality == "medium":
                res_type = 'polyphase'
            else:  # low
                res_type = 'linear'
            
            # 使用librosa进行重采样
            resampled_audio = librosa.resample(
                audio_np, 
                orig_sr=orig_sr, 
                target_sr=target_sr, 
                res_type=res_type,
                fix=True,
                scale=False
            )
            
            # 转换回原来的数据类型
            if is_tensor:
                result = torch.from_numpy(resampled_audio).to(audio_data.device)
                return result.type_as(audio_data)
            else:
                return resampled_audio.astype(audio_data.dtype)
            
        except ImportError:
            logger.error("librosa未安装，请执行: pip install librosa")
            # 回退到torchaudio重采样
            return AudioProcessor.resample_audio(audio_data, orig_sr, target_sr)
        except Exception as e:
            logger.error(f"librosa FFT下采样失败: {e}，回退到torchaudio重采样")
            return AudioProcessor.resample_audio(audio_data, orig_sr, target_sr)


class DeviceSafeInference:
    """设备安全的推理类"""
    
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.logger = logging.getLogger(__name__)
    
    def inference_chunk(self, model, audio_data: torch.Tensor, sample_rate: int, npad: int = 441) -> torch.Tensor:
        """
        对单个音频块进行推理
        
        Args:
            model: 推理模型
            audio_data: 音频数据
            sample_rate: 采样率
            npad: 填充长度
            
        Returns:
            增强后的音频数据
        """
        assert model.hp.wav_rate == sample_rate, f"期望采样率 {model.hp.wav_rate} Hz, 实际 {sample_rate} Hz"
        
        length = audio_data.shape[-1]
        abs_max = audio_data.abs().max().clamp(min=1e-7)
        
        assert audio_data.dim() == 1, f"期望1D音频，实际 {audio_data.dim()}D"
        
        # 移动到设备并归一化
        audio_data = audio_data.to(self.device)
        audio_data = audio_data / abs_max
        audio_data = torch.nn.functional.pad(audio_data, (0, npad))
        
        # 推理
        with torch.no_grad():
            enhanced_audio = model(audio_data[None])[0]
            enhanced_audio = enhanced_audio[:length]
            enhanced_audio = enhanced_audio * abs_max
        
        return enhanced_audio
    
    def compute_correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """计算两个信号的相关性"""
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()
    
    def compute_offset(self, chunk1: torch.Tensor, chunk2: torch.Tensor, sr: int = 44100) -> int:
        """计算两个音频块之间的偏移"""
        hop_length = sr // 200
        win_length = hop_length * 4
        n_fft = 2 ** (win_length - 1).bit_length()
        
        # 确保chunks在同一设备上
        device = chunk1.device
        chunk2 = chunk2.to(device)
        
        mel_fn = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=80,
            f_min=0.0,
            f_max=sr // 2,
        ).to(device)
        
        spec1 = mel_fn(chunk1).log1p()
        spec2 = mel_fn(chunk2).log1p()
        
        corr = self.compute_correlation(spec1, spec2)
        corr = corr.mean(dim=0)
        
        argmax = corr.argmax().item()
        
        if argmax > len(corr) // 2:
            argmax -= len(corr)
        
        offset = -argmax * hop_length
        return offset
    
    def merge_chunks(
        self, 
        chunks: List[torch.Tensor], 
        chunk_length: int, 
        hop_length: int, 
        sr: int = 44100, 
        length: Optional[int] = None
    ) -> torch.Tensor:
        """合并音频块"""
        if not chunks:
            return torch.zeros(0)
        
        # 确保所有chunks在同一设备上
        device = chunks[0].device
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk.to(device)
        
        signal_length = (len(chunks) - 1) * hop_length + chunk_length
        overlap_length = chunk_length - hop_length
        signal = torch.zeros(signal_length, device=device)
        
        # 创建淡入淡出效果
        fadein = torch.linspace(0, 1, overlap_length, device=device)
        fadein = torch.cat([fadein, torch.ones(hop_length, device=device)])
        fadeout = torch.linspace(1, 0, overlap_length, device=device)
        fadeout = torch.cat([torch.ones(hop_length, device=device), fadeout])
        
        for i, chunk in enumerate(chunks):
            start = i * hop_length
            end = start + chunk_length
            
            if len(chunk) < chunk_length:
                chunk = torch.nn.functional.pad(chunk, (0, chunk_length - len(chunk)))
            
            # 计算偏移
            if i > 0:
                pre_region = chunks[i - 1][-overlap_length:]
                cur_region = chunk[:overlap_length]
                offset = self.compute_offset(pre_region, cur_region, sr=sr)
                start -= offset
                end -= offset
            
            # 应用淡入淡出
            if i == 0:
                chunk = chunk * fadeout
            elif i == len(chunks) - 1:
                chunk = chunk * fadein
            else:
                chunk = chunk * fadein * fadeout
            
            signal[start:end] += chunk[: len(signal[start:end])]
        
        if length is not None:
            signal = signal[:length]
        
        return signal
    
    def inference(
        self,
        model,
        audio_data: torch.Tensor,
        sample_rate: int,
        chunk_seconds: float = 20.0,
        overlap_seconds: float = 0.5,
        min_chunk_length: int = 1323000
    ) -> Tuple[torch.Tensor, int]:
        """
        执行推理
        
        Args:
            model: 推理模型
            audio_data: 音频数据
            sample_rate: 采样率
            chunk_seconds: 分块大小（秒）
            overlap_seconds: 重叠时间（秒）
            min_chunk_length: 最小分块长度
            
        Returns:
            增强后的音频数据和采样率
        """
        # 移除权重归一化
        def remove_weight_norm_recursively(module):
            for _, module in module.named_modules():
                try:
                    remove_parametrizations(module, "weight")
                except Exception:
                    pass
        
        remove_weight_norm_recursively(model)
        
        hp = model.hp
        
        # 确保输入在正确的设备上
        audio_data = audio_data.to(self.device)
        
        # 重采样（如果需要）
        if sample_rate != hp.wav_rate:
            audio_data = AudioProcessor.resample_audio(audio_data, sample_rate, hp.wav_rate)
            audio_data = audio_data.to(self.device)
        
        sample_rate = hp.wav_rate
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        # 动态长度处理
        if audio_data.shape[-1] <= min_chunk_length:
            # 短音频直接处理
            self.logger.info(f"短音频 ({audio_data.shape[-1]} 样本)，直接处理")
            enhanced_audio = self.inference_chunk(model, audio_data, sample_rate)
        else:
            # 长音频分块处理
            self.logger.info(f"长音频 ({audio_data.shape[-1]} 样本)，分块处理")
            chunk_length = int(sample_rate * chunk_seconds)
            overlap_length = int(sample_rate * overlap_seconds)
            hop_length = chunk_length - overlap_length
            
            chunks = []
            for start in tqdm(range(0, audio_data.shape[-1], hop_length), desc="处理音频块"):
                chunk_data = audio_data[start : start + chunk_length]
                processed_chunk = self.inference_chunk(model, chunk_data, sample_rate)
                chunks.append(processed_chunk)
            
            enhanced_audio = self.merge_chunks(
                chunks, chunk_length, hop_length, sr=sample_rate, length=audio_data.shape[-1]
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        elapsed_time = time.perf_counter() - start_time
        self.logger.info(f"推理时间: {elapsed_time:.3f}s, 速度: {enhanced_audio.shape[-1] / elapsed_time / 1000:.3f} kHz")
        
        # 移动到CPU
        enhanced_audio = enhanced_audio.cpu()
        
        return enhanced_audio, sample_rate


def preprocess_audio_batch(audio_files: List[Tuple[str, str]], config: ProcessingConfig) -> List[Tuple[str, str]]:
    """
    批量预处理音频文件 - 仅返回有效的文件路径
    
    Args:
        audio_files: 音频文件路径列表
        config: 处理配置
        
    Returns:
        有效的文件路径列表
    """
    processed_batch = []
    
    for input_file, output_file in audio_files:
        try:
            # 检查是否跳过已存在的文件
            if config.skip_existing and os.path.exists(output_file):
                continue
                
            # 简单检查文件是否存在和可读
            if not os.path.exists(input_file):
                logger.warning(f"文件不存在: {input_file}")
                continue
                
            processed_batch.append((input_file, output_file))
            
        except Exception as e:
            logger.error(f"预处理失败 {input_file}: {e}")
            continue
    
    return processed_batch


def process_gpu_batch_resemble_optimized(args) -> List[Tuple[bool, str]]:
    """
    优化的GPU处理函数 - 使用批处理和异步I/O
    
    Args:
        args: 包含文件列表、GPU ID、配置和位置的元组
        
    Returns:
        处理结果列表
    """
    file_list, gpu_id, config, position = args
    
    # 强制设置GPU设备
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 重新导入所需库（多进程环境需要）
    import torch
    import torchaudio
    from concurrent.futures import ThreadPoolExecutor
    
    # 设置线程数和优化
    torch.set_num_threads(4)
    if config.optimize_memory:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    
    # 设置GPU设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU {gpu_id} 进程启动，设备: {device_name}")
    
    # 在进程中导入resemble_enhance
    try:
        from resemble_enhance.enhancer.train import Enhancer, HParams
        
        # 加载模型
        run_dir = Path(config.run_dir)
        hp = HParams.load(run_dir)
        enhancer = Enhancer(hp)
        path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(path, map_location="cpu")["module"]
        enhancer.load_state_dict(state_dict)
        enhancer.eval()
        enhancer.to("cuda:0")
        
        # 获取模型期望的采样率
        model_sr = hp.wav_rate
        print(f"GPU {gpu_id} 模型期望采样率: {model_sr}Hz")
        
        # 配置增强参数
        enhancer.configurate_(
            nfe=config.nfe, 
            solver=config.solver, 
            lambd=config.lambd, 
            tau=config.tau
        )
        
        print(f"GPU {gpu_id} 模型加载成功，开始处理 {len(file_list)} 个文件")
        
        # 创建进度条
        pbar = tqdm(total=len(file_list), desc=f"GPU {gpu_id}", position=position, leave=True)
        
        # 创建推理器
        inference_engine = DeviceSafeInference("cuda:0")
        
        results = []
        
        # 使用线程池进行异步I/O
        with ThreadPoolExecutor(max_workers=config.num_workers) as executor:
            # 批处理文件
            for i in range(0, len(file_list), config.batch_size):
                batch_files = file_list[i:i+config.batch_size]
                
                # 异步预处理音频
                future_preprocess = executor.submit(preprocess_audio_batch, batch_files, config)
                processed_batch = future_preprocess.result()
                
                if not processed_batch:
                    pbar.update(len(batch_files))
                    continue
                
                # 在GPU进程内部处理每个音频文件
                enhanced_batch = []
                batch_info = []
                
                with torch.no_grad():
                    for input_file, output_file in processed_batch:
                        try:
                            # 加载音频
                            dwav, sr = AudioProcessor.load_audio(input_file)
                            
                            # 验证音频数据
                            if not AudioProcessor.validate_audio_data(dwav, input_file):
                                continue
                            
                            # 准备音频数据
                            dwav = AudioProcessor.prepare_audio_for_inference(
                                dwav, sr, model_sr, config.max_length, config.dynamic_length
                            )
                            
                            # 再次验证处理后的数据
                            if not AudioProcessor.validate_audio_data(dwav, input_file):
                                continue
                            
                            # 确保数据连续性
                            if not dwav.is_contiguous():
                                dwav = dwav.contiguous()
                            
                            # 执行推理
                            hwav, output_sr = inference_engine.inference(
                                model=enhancer,
                                audio_data=dwav,
                                sample_rate=model_sr,
                                chunk_seconds=config.chunk_seconds,
                                overlap_seconds=config.overlap_seconds,
                                min_chunk_length=config.min_chunk_length
                            )
                            
                            # 验证输出数据
                            if not AudioProcessor.validate_audio_data(hwav, input_file):
                                continue
                            
                            enhanced_batch.append((hwav, output_sr))
                            batch_info.append((input_file, output_file))
                            
                        except Exception as e:
                            logger.error(f"处理音频文件失败 {input_file}: {e}")
                            continue
                
                # 异步保存音频文件 - 确保使用模型的采样率 (44.1kHz)
                save_futures = []
                model_sr = 44100  # Resemble-Enhance模型的采样率
                
                for j, (input_file, output_file) in enumerate(batch_info):
                    if j < len(enhanced_batch):
                        hwav, _ = enhanced_batch[j]  # 忽略output_sr，使用模型采样率
                        future_save = executor.submit(AudioProcessor.save_audio, hwav, output_file, model_sr)
                        save_futures.append((future_save, input_file))
                
                # 等待保存完成
                for future_save, input_file in save_futures:
                    try:
                        future_save.result()
                        results.append((True, f"成功处理: {input_file}"))
                    except Exception as e:
                        results.append((False, f"保存失败 {input_file}: {e}"))
                
                pbar.update(len(batch_files))
                pbar.set_postfix({"状态": f"成功处理{len(enhanced_batch)}/{len(batch_files)}"})
                
                # 清理GPU内存
                if config.optimize_memory:
                    torch.cuda.empty_cache()
        
        pbar.close()
        
        # 统计结果
        success_count = sum(1 for success, _ in results if success)
        print(f"GPU {gpu_id} 处理完成: {success_count}/{len(file_list)} 成功")
        
        return results
        
    except Exception as e:
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        return [(False, f"进程初始化失败: {e}") for _ in file_list]


class ProcessingStats:
    """处理统计信息类"""
    
    def __init__(self):
        self.total_files = 0
        self.processed_files = 0
        self.skipped_files = 0
        self.failed_files = 0
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
    
    def print_summary(self, config: ProcessingConfig):
        """打印统计摘要"""
        print("\n" + "=" * 70)
        print("处理统计")
        print("=" * 70)
        print(f"总文件数:     {self.total_files}")
        print(f"成功处理:     {self.processed_files}")
        print(f"跳过文件:     {self.skipped_files}")
        print(f"失败文件:     {self.failed_files}")
        
        if self.total_files > 0:
            success_rate = (self.processed_files / self.total_files) * 100
            print(f"成功率:       {success_rate:.1f}%")
        
        print(f"处理时间:     {self.processing_time:.2f}秒")
        
        # 性能指标
        if self.processing_time > 0:
            files_per_sec = self.processed_files / self.processing_time
            print(f"处理速度:     {files_per_sec:.2f} 文件/秒")
        
        # 配置信息
        print(f"处理模式:     优化I/O批处理模式")
        print(f"使用GPU数:    {len(config.gpu_ids)}")
        print(f"GPU列表:      {config.gpu_ids}")
        print(f"批处理大小:   {config.batch_size}")
        print(f"I/O线程数:    {config.num_workers}")
        print(f"混合精度:     {'开启' if config.use_mixed_precision else '关闭'}")
        print(f"内存优化:     {'开启' if config.optimize_memory else '关闭'}")
        print(f"动态长度:     {'开启' if config.dynamic_length else '关闭'}")
        print(f"分块阈值:     {config.min_chunk_length} 样本 ({config.min_chunk_length/config.target_sr:.1f}秒)")
        print(f"分块大小:     {config.chunk_seconds}秒")
        print(f"重叠时间:     {config.overlap_seconds}秒")
        print("=" * 70)


class ResembleEnhancerBatchProcessor:
    """Resemble-Enhance批量处理器"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.stats = ProcessingStats()
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.available_gpus = [i for i in config.gpu_ids if i < torch.cuda.device_count()]
            if not self.available_gpus:
                raise RuntimeError("指定的GPU不可用")
            logger.info(f"可用GPU: {self.available_gpus}")
        else:
            raise RuntimeError("CUDA不可用")
    
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
        logger.info(f"开始隔离GPU处理，文件总数: {len(file_pairs)}")
        logger.info(f"使用GPU: {self.available_gpus}")
        
        # 将文件分配给各个GPU
        num_gpus = len(self.available_gpus)
        files_per_gpu = len(file_pairs) // num_gpus
        remainder = len(file_pairs) % num_gpus
        
        gpu_file_batches = []
        start_idx = 0
        
        for i, gpu_id in enumerate(self.available_gpus):
            # 计算当前GPU应该处理的文件数量
            current_batch_size = files_per_gpu + (1 if i < remainder else 0)
            end_idx = start_idx + current_batch_size
            
            # 分配文件给当前GPU
            gpu_files = file_pairs[start_idx:end_idx]
            gpu_file_batches.append((gpu_files, gpu_id, i))
            
            logger.info(f"GPU {gpu_id}: 分配 {len(gpu_files)} 个文件")
            start_idx = end_idx
        
        # 使用多进程处理
        all_results = []
        with mp.Pool(processes=num_gpus) as pool:
            # 准备参数
            args_list = []
            for gpu_files, gpu_id, position in gpu_file_batches:
                args = (gpu_files, gpu_id, self.config, position)
                args_list.append(args)
            
            # 提交任务
            results = pool.map(process_gpu_batch_resemble_optimized, args_list)
            
            # 合并结果
            for result_batch in results:
                all_results.extend(result_batch)
        
        # 更新统计信息
        self.stats.update_from_results(all_results)
        
        logger.info(f"隔离GPU处理完成，总结果: {len(all_results)} 个")
        return all_results


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
            cpu_count_logical = cpu_count
            memory_gb = 32  # 默认假设32GB
            print(f"CPU: {cpu_count} 核心 (无法获取详细信息)")
            print(f"内存: 无法检测 (假设32GB)")
        
        # 提供优化建议
        print("\n优化建议:")
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        
        if num_gpus >= 4:
            print("✓ 检测到多个GPU，当前配置适合多GPU处理")
        elif num_gpus >= 2:
            print("! 建议调整batch_size到64-128来充分利用GPU")
        else:
            print("! 单GPU环境，建议增加batch_size到64-128")
        
        if memory_gb >= 64:
            print("✓ 内存充足，可以增加batch_size到128和num_workers到12")
        elif memory_gb >= 32:
            print("✓ 内存充足，当前配置合适")
        else:
            print("! 内存较少，建议减少batch_size到32-48")
        
        if cpu_count >= 16:
            print("✓ CPU核心充足，可以增加num_workers到12-16")
        elif cpu_count >= 8:
            print("✓ CPU核心充足，当前num_workers设置合适")
        else:
            print("! CPU核心较少，建议减少num_workers到4-6")
        
        print("\n💡 性能优化提示:")
        print("  - 使用动态长度处理避免不必要的音频填充")
        print("  - 短音频(<30秒)不分块处理，提高效率")
        print("  - 减少NFE参数以提高推理速度")
        print("  - 优化批处理大小以充分利用GPU内存")
        print("=" * 60)
    
    @staticmethod
    def provide_performance_tips(stats: ProcessingStats):
        """提供性能提示"""
        if stats.processing_time > 0:
            total_processed = stats.processed_files
            processing_time = stats.processing_time
            files_per_sec = total_processed / processing_time
            
            print(f"\n性能提示:")
            if files_per_sec < 0.5:
                print("• 处理速度很慢，建议：")
                print("  - 增加batch_size到128（如果GPU内存充足）")
                print("  - 增加num_workers到12-16")
                print("  - 检查是否有短音频被不必要地分块处理")
                print("  - 考虑进一步减少NFE参数到16-24")
            elif files_per_sec < 2:
                print("• 处理速度较慢，可以考虑：")
                print("  - 增加batch_size到96-128")
                print("  - 增加I/O线程数 (num_workers)")
                print("  - 确保动态长度处理已启用")
            elif files_per_sec < 5:
                print("• 处理速度适中，可以考虑：")
                print("  - 微调batch_size以优化GPU利用率")
                print("  - 调整分块阈值以适应您的音频长度分布")
            else:
                print("• 处理速度良好！")
            
            print(f"\n📊 性能统计:")
            print(f"  - 处理速度: {files_per_sec:.2f} 文件/秒")
            remaining_files = stats.total_files - total_processed
            if remaining_files > 0:
                print(f"  - 预计剩余时间: {remaining_files / files_per_sec / 3600:.1f} 小时")
            if stats.total_files > 0:
                print(f"  - 进度: {(total_processed / stats.total_files) * 100:.1f}%")


def create_default_config() -> ProcessingConfig:
    """创建默认配置"""
    return ProcessingConfig()


def main():
    """主函数"""
    print("Resemble-Enhance批量语音增强处理器（重构版）")
    print("=" * 60)
    
    try:
        # 创建配置
        config = create_default_config()
        
        # 性能诊断
        SystemDiagnostics.diagnose_performance()
        
        # 创建输出目录
        os.makedirs(config.output_dir, exist_ok=True)
        
        # 创建处理器
        processor = ResembleEnhancerBatchProcessor(config)
        
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
        print(f"设备: {config.device}")
        print(f"可用GPU: {processor.available_gpus}")
        print(f"目标采样率: {config.target_sr}Hz")
        print(f"NFE: {config.nfe}")
        print(f"Solver: {config.solver}")
        print(f"Lambda: {config.lambd}")
        print(f"Tau: {config.tau}")
        
        print(f"\n优化参数:")
        print(f"批处理大小: {config.batch_size}")
        print(f"最大音频长度: {config.max_length} 样本 ({config.max_length/config.target_sr:.1f}秒)")
        print(f"分块阈值: {config.min_chunk_length} 样本 ({config.min_chunk_length/config.target_sr:.1f}秒)")
        print(f"分块大小: {config.chunk_seconds}秒")
        print(f"重叠时间: {config.overlap_seconds}秒")
        print(f"I/O线程数: {config.num_workers}")
        print(f"混合精度: {'开启' if config.use_mixed_precision else '关闭'}")
        print(f"内存优化: {'开启' if config.optimize_memory else '关闭'}")
        print(f"动态长度: {'开启' if config.dynamic_length else '关闭'}")
        
        # 开始处理
        print("\n开始批量处理...")
        start_time = time.time()
        
        results = processor.process_files_isolated_gpu(audio_files)
        
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
        
        # 性能提示
        SystemDiagnostics.provide_performance_tips(processor.stats)
        
        print("\n处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
