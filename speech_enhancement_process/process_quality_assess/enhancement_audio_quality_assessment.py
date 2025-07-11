#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频质量评估脚本 (多卡并行版本)
使用Kimi-Audio模型对降噪前后的音频进行识别和WER计算
使用TEN-VAD进行VAD，基于语音段能量计算gain值
支持多GPU并行处理
通过HTTP API调用LLM服务进行文本标准化
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
import librosa
import torch
from jiwer import wer, cer
from datetime import datetime
import warnings
import struct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import pickle
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# 添加TEN VAD路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../include")))
from ten_vad import TenVad

# 设置代理绕过，确保可以访问本地LLM服务
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    # 多目录配置：每个元素包含一个目录组的配置
    'directory_configs': [
        {
            'name': 'default',  # 配置名称
            'original_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid",
            'enhanced_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced",
            'output_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment",
            'volume_matching': True,  # 是否启用音量匹配
            'volume_matching_method': 'ten_vad_energy',  # 音量匹配方法
        }
    ],
    # 全局配置
    'kimi_model_path': "/root/data/pretrained_models/Kimi-Audio-7B-Instruct",
    'kimi_audio_dir': "/root/code/github_repos/Kimi-Audio",
    'device': "cuda:0",
    'target_sr': 16000,  # Kimi-Audio期望的采样率
    'batch_size': 8,
    'num_workers': 4,
    'skip_existing': False,
    'max_audio_length': 600,  # 最大音频长度（秒）
    'ten_vad_hop_size': 256,  # TEN VAD帧跳跃大小（样本数，256样本=16ms@16kHz）
    'ten_vad_threshold': 0.5,  # TEN VAD阈值
    'num_gpus': 3,  # GPU数量（GPU 1,2,3用于ASR，GPU 0用于LLM服务）
    'gpu_ids': [1, 2, 3],  # GPU ID列表
    'text_normalization': 'llm',  # 文本标准化级别: 'basic', 'advanced', 'llm'
    'remove_punctuation': True,  # 是否移除标点符号
    'normalize_spacing': True,  # 是否标准化空格
    'handle_spelling': True,  # 是否处理拼读问题
    'spelling_max_length': 4,  # 拼读检测的最大长度
    'preserve_common_words': True,  # 是否保留常见单词不拆分
    'use_llm_normalization': True,  # 是否使用LLM进行文本标准化
    'llm_service_url': 'http://localhost:8000',  # LLM服务URL
    'llm_timeout': 30,  # LLM服务超时时间（秒）
    'llm_max_retries': 3,  # LLM服务最大重试次数
}

class AudioVolumeProcessor:
    """音频音量处理器"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.eps = 1e-8  # 避免除零错误
        self.hop_size = CONFIG['ten_vad_hop_size']  # TEN VAD帧跳跃大小
        self.threshold = CONFIG['ten_vad_threshold']  # TEN VAD阈值
        self.vad = TenVad(self.hop_size, self.threshold)
        
        logger.info(f"初始化TEN VAD: hop_size={self.hop_size}, threshold={self.threshold}")
    
    def audio_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """将浮点音频转换为int16格式"""
        # 确保音频在[-1, 1]范围内
        audio = np.clip(audio, -1.0, 1.0)
        # 转换为int16
        return (audio * 32767).astype(np.int16)
    
    def perform_vad(self, audio: np.ndarray) -> List[bool]:
        """执行语音活动检测 - 使用TEN VAD"""
        try:
            # 转换为int16格式
            audio_int16 = self.audio_to_int16(audio)
            
            # 计算帧数
            num_frames = len(audio_int16) // self.hop_size
            
            # 对每帧进行VAD
            vad_results = []
            for i in range(num_frames):
                # 提取音频帧
                start_idx = i * self.hop_size
                end_idx = start_idx + self.hop_size
                
                if end_idx <= len(audio_int16):
                    audio_frame = audio_int16[start_idx:end_idx]
                    
                    # 进行TEN VAD检测
                    out_probability, out_flag = self.vad.process(audio_frame)
                    vad_results.append(bool(out_flag))
                else:
                    # 处理最后一帧（可能不足hop_size长度）
                    audio_frame = audio_int16[start_idx:]
                    if len(audio_frame) > 0:
                        # 填充到hop_size长度
                        padded_frame = np.zeros(self.hop_size, dtype=np.int16)
                        padded_frame[:len(audio_frame)] = audio_frame
                        
                        out_probability, out_flag = self.vad.process(padded_frame)
                        vad_results.append(bool(out_flag))
            
            logger.debug(f"TEN VAD检测完成: 总帧数={num_frames}, 语音帧数={sum(vad_results)}")
            return vad_results
            
        except Exception as e:
            logger.warning(f"TEN VAD检测失败: {e}")
            # 如果VAD失败，返回全部为语音的结果
            num_frames = len(audio) // self.hop_size
            return [True] * num_frames
    
    def extract_speech_segments(self, audio: np.ndarray) -> np.ndarray:
        """提取语音段"""
        try:
            # 执行VAD
            vad_results = self.perform_vad(audio)
            
            if not any(vad_results):
                logger.warning("未检测到语音段，返回原始音频")
                return audio
            
            # 根据VAD结果提取语音段
            speech_segments = []
            for i, is_speech in enumerate(vad_results):
                if is_speech:
                    start_idx = i * self.hop_size
                    end_idx = (i + 1) * self.hop_size
                    if end_idx <= len(audio):
                        speech_segments.append(audio[start_idx:end_idx])
            
            if speech_segments:
                # 连接所有语音段
                speech_audio = np.concatenate(speech_segments)
                logger.info(f"TEN VAD检测: 原始长度 {len(audio)/self.sr:.2f}s, 语音段长度 {len(speech_audio)/self.sr:.2f}s")
                return speech_audio
            else:
                logger.warning("未找到有效语音段，返回原始音频")
                return audio
                
        except Exception as e:
            logger.warning(f"提取语音段失败: {e}")
            return audio
    
    def calculate_energy(self, audio: np.ndarray) -> float:
        """计算音频能量"""
        if len(audio) == 0:
            return 0.0
        return np.sum(audio ** 2) / len(audio)
    
    def calculate_gain_vad_energy(self, reference: np.ndarray, enhanced: np.ndarray) -> float:
        """基于VAD后的语音段能量计算gain值"""
        try:
            # 确保信号长度相同
            min_len = min(len(reference), len(enhanced))
            reference = reference[:min_len]
            enhanced = enhanced[:min_len]
            
            # 提取语音段
            ref_speech = self.extract_speech_segments(reference)
            enh_speech = self.extract_speech_segments(enhanced)
            
            # 计算语音段能量
            ref_energy = self.calculate_energy(ref_speech)
            enh_energy = self.calculate_energy(enh_speech)
            
            # 计算gain
            if enh_energy > self.eps:
                gain = np.sqrt(ref_energy / enh_energy)
            else:
                gain = 1.0
            
            # 确保gain在合理范围内
            gain = np.clip(gain, 0.1, 10.0)
            
            logger.info(f"TEN VAD能量计算: 参考能量={ref_energy:.6f}, 增强能量={enh_energy:.6f}, gain={gain:.4f}")
            
            return gain
            
        except Exception as e:
            logger.warning(f"TEN VAD能量gain计算失败: {e}")
            return 1.0
    
    def apply_gain(self, audio: np.ndarray, gain: float) -> Tuple[np.ndarray, float]:
        """对音频应用gain"""
        try:
            # 应用gain
            processed_audio = audio * gain
            
            # 防止溢出
            max_val = np.max(np.abs(processed_audio))
            if max_val > 0.95:
                processed_audio = processed_audio * (0.95 / max_val)
                actual_gain = gain * (0.95 / max_val)
            else:
                actual_gain = gain
            
            logger.info(f"应用gain: {actual_gain:.4f}")
            
            return processed_audio, actual_gain
            
        except Exception as e:
            logger.warning(f"应用gain失败: {e}")
            return audio, 1.0

class HTTPTextNormalizer:
    """基于HTTP API的文本标准化器"""
    
    def __init__(self, service_url: str, timeout: int = 30, max_retries: int = 3, gpu_id: int = 0):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.gpu_id = gpu_id
        
        # 创建HTTP会话，配置重试策略和代理绕过
        self.session = requests.Session()
        
        # 设置代理绕过
        self.session.proxies = {
            'http': None,
            'https': None
        }
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # 设置请求头
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': f'AudioQualityAssessment/1.0 (GPU-{gpu_id})'
        })
        
        logger.info(f"GPU {gpu_id}: 初始化HTTP文本标准化器")
        logger.info(f"GPU {gpu_id}: 服务URL: {self.service_url}")
        logger.info(f"GPU {gpu_id}: 超时时间: {self.timeout}s")
        logger.info(f"GPU {gpu_id}: 最大重试: {self.max_retries}")
        
        # 测试连接
        self._test_connection()
    
    def _test_connection(self):
        """测试与LLM服务的连接"""
        try:
            health_url = f"{self.service_url}/health"
            response = self.session.get(health_url, timeout=10)
            response.raise_for_status()
            
            logger.info(f"GPU {self.gpu_id}: ✓ LLM服务连接正常")
            
            # 获取模型信息
            try:
                model_info_url = f"{self.service_url}/model_info"
                response = self.session.get(model_info_url, timeout=10)
                response.raise_for_status()
                info = response.json()
                logger.info(f"GPU {self.gpu_id}: LLM模型信息: {info}")
            except Exception as e:
                logger.warning(f"GPU {self.gpu_id}: 无法获取模型信息: {e}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"GPU {self.gpu_id}: LLM服务连接失败: {e}")
            raise
    
    def normalize_text_pair(self, text1: str, text2: str) -> Tuple[str, str]:
        """使用HTTP API标准化文本对"""
        try:
            # 准备请求数据
            request_data = {
                "text1": text1,
                "text2": text2
            }
            
            # 发送请求
            normalize_url = f"{self.service_url}/normalize"
            response = self.session.post(
                normalize_url,
                json=request_data,
                timeout=self.timeout
            )
            
            # 检查响应
            response.raise_for_status()
            result = response.json()
            
            # 提取结果
            normalized_text1 = result.get("normalized_text1", "")
            normalized_text2 = result.get("normalized_text2", "")
            success = result.get("success", False)
            error_message = result.get("error_message")
            
            if not success and error_message:
                logger.warning(f"GPU {self.gpu_id}: LLM标准化失败: {error_message}")
            
            logger.info(f"GPU {self.gpu_id}: HTTP LLM标准化完成")
            logger.info(f"GPU {self.gpu_id}: 原始文本1: '{text1}' -> 标准化: '{normalized_text1}'")
            logger.info(f"GPU {self.gpu_id}: 原始文本2: '{text2}' -> 标准化: '{normalized_text2}'")
            
            return normalized_text1, normalized_text2
            
        except requests.exceptions.Timeout:
            logger.error(f"GPU {self.gpu_id}: LLM服务请求超时")
            return self.basic_normalize(text1), self.basic_normalize(text2)
        except requests.exceptions.RequestException as e:
            logger.error(f"GPU {self.gpu_id}: LLM服务请求失败: {e}")
            return self.basic_normalize(text1), self.basic_normalize(text2)
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: HTTP LLM标准化失败: {e}")
            return self.basic_normalize(text1), self.basic_normalize(text2)
    
    def basic_normalize(self, text: str) -> str:
        """基础文本标准化（降级方法）"""
        import re
        import string
        
        if not text:
            return ""
        
        # 转换为小写
        text = text.lower().strip()
        
        # 移除标点符号
        punctuation = string.punctuation + '，。！？；：""''（）【】《》〈〉「」『』〖〗〔〕［］｛｝'
        for p in punctuation:
            text = text.replace(p, '')
        
        # 标准化空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class KimiAudioProcessor:
    """Kimi-Audio处理器"""
    
    def __init__(self, model_path: str, device: str = "cuda:0", kimi_audio_dir: str = None, gpu_id: int = 0):
        self.model_path = model_path
        self.device = device
        self.gpu_id = gpu_id
        self.model = None
        self.kimi_audio_dir = kimi_audio_dir or "/root/code/github_repos/Kimi-Audio"
        self.original_cwd = os.getcwd()
        
        # 延迟加载模型，避免在主进程中初始化CUDA
        # 在子进程中才会真正加载模型
    
    def load_model(self):
        """加载Kimi-Audio模型"""
        if self.model is not None:
            return  # 模型已经加载过了
        
        try:
            # 在子进程中设置CUDA设备
            if torch.cuda.is_available():
                # 由于CUDA_VISIBLE_DEVICES已经在进程中设置，这里使用0
                torch.cuda.set_device(0)
                logger.info(f"GPU {self.gpu_id}: 设置CUDA设备")
            
            # 切换到Kimi-Audio目录
            if os.path.exists(self.kimi_audio_dir):
                os.chdir(self.kimi_audio_dir)
                logger.info(f"GPU {self.gpu_id}: 切换到Kimi-Audio目录: {self.kimi_audio_dir}")
                
                # 添加Kimi-Audio目录到Python路径
                if self.kimi_audio_dir not in sys.path:
                    sys.path.insert(0, self.kimi_audio_dir)
                    logger.info(f"GPU {self.gpu_id}: 添加到Python路径: {self.kimi_audio_dir}")
            else:
                logger.warning(f"GPU {self.gpu_id}: Kimi-Audio目录不存在: {self.kimi_audio_dir}")
            
            from kimia_infer.api.kimia import KimiAudio
            
            # 加载模型
            self.model = KimiAudio(
                model_path=self.model_path,
                load_detokenizer=False,
            )
            
            logger.info(f"GPU {self.gpu_id}: ✓ 成功加载Kimi-Audio模型: {self.model_path}")
            
        except ImportError as e:
            logger.error(f"GPU {self.gpu_id}: 无法导入Kimi-Audio: {e}")
            logger.error("请确保已正确安装Kimi-Audio并设置正确的路径")
            raise
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 加载Kimi-Audio模型失败: {e}")
            raise
        finally:
            # 恢复原始工作目录
            os.chdir(self.original_cwd)
    
    def transcribe_audio(self, audio_path: str) -> str:
        """使用Kimi-Audio进行语音识别"""
        try:
            # 确保模型已经加载
            if self.model is None:
                self.load_model()
            
            # 切换到Kimi-Audio目录进行推理
            current_dir = os.getcwd()
            if os.path.exists(self.kimi_audio_dir):
                os.chdir(self.kimi_audio_dir)
                
                # 确保Python路径包含Kimi-Audio目录
                if self.kimi_audio_dir not in sys.path:
                    sys.path.insert(0, self.kimi_audio_dir)
            
            # 准备消息
            messages = [
                {"role": "user", "message_type": "text", "content": "请将音频内容转换为文字。"},
                {"role": "user", "message_type": "audio", "content": audio_path}
            ]
            
            # 生成参数
            sampling_params = {
                "audio_temperature": 0.8,
                "audio_top_k": 10,
                "text_temperature": 0.0,
                "text_top_k": 5,
                "audio_repetition_penalty": 1.0,
                "audio_repetition_window_size": 64,
                "text_repetition_penalty": 1.0,
                "text_repetition_window_size": 16,
            }
            
            # 生成转录
            _, text_output = self.model.generate(
                messages, 
                **sampling_params, 
                output_type="text"
            )
            
            # 清理文本
            if text_output:
                text_output = text_output.strip()
                # 移除常见的转录前缀
                prefixes_to_remove = [
                    "The audio says:",
                    "The transcription is:",
                    "Transcription:",
                    "Audio content:",
                    "音频内容是：",
                    "转录结果：",
                    "转录内容：",
                ]
                for prefix in prefixes_to_remove:
                    if text_output.startswith(prefix):
                        text_output = text_output[len(prefix):].strip()
            
            return text_output or ""
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 语音识别失败 {audio_path}: {e}")
            return ""
        finally:
            # 恢复原始工作目录
            os.chdir(current_dir)
    
    def batch_transcribe(self, audio_files: List[str], desc: str = "转录音频") -> List[str]:
        """批量转录音频"""
        results = []
        
        for audio_file in tqdm(audio_files, desc=f"GPU {self.gpu_id}: {desc}"):
            if os.path.exists(audio_file):
                transcription = self.transcribe_audio(audio_file)
                results.append(transcription)
            else:
                logger.warning(f"GPU {self.gpu_id}: 音频文件不存在: {audio_file}")
                results.append("")
        
        return results

class QualityAssessmentProcessor:
    """质量评估处理器"""
    
    def __init__(self, config: Dict, dir_config: Dict, gpu_id: int = 0):
        self.config = config
        self.dir_config = dir_config  # 当前目录配置
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.volume_processor = AudioVolumeProcessor(sr=config['target_sr'])
        self.kimi_processor = KimiAudioProcessor(
            model_path=config['kimi_model_path'],
            device=self.device,
            kimi_audio_dir=config['kimi_audio_dir'],
            gpu_id=gpu_id
        )
        
        # 从目录配置中获取volume_matching设置
        self.volume_matching = dir_config.get('volume_matching', True)
        self.volume_matching_method = dir_config.get('volume_matching_method', 'vad_energy')
        
        # 初始化HTTP文本标准化器（如果启用）
        self.http_normalizer = None
        if config.get('use_llm_normalization', False) and config.get('text_normalization') == 'llm':
            try:
                self.http_normalizer = HTTPTextNormalizer(
                    service_url=config['llm_service_url'],
                    timeout=config['llm_timeout'],
                    max_retries=config['llm_max_retries'],
                    gpu_id=gpu_id
                )
            except Exception as e:
                logger.warning(f"GPU {gpu_id}: 初始化HTTP文本标准化器失败: {e}")
                logger.warning(f"GPU {gpu_id}: 将使用传统标准化方法")
                self.http_normalizer = None
        
        # 统计信息
        self.stats = {
            'total_pairs': 0,
            'processed_pairs': 0,
            'failed_pairs': 0,
            'total_wer': 0.0,
            'total_cer': 0.0,
            'processing_time': 0.0,
        }
    
    def preprocess_audio(self, audio_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """预处理音频"""
        if target_sr is None:
            target_sr = self.config['target_sr']
        
        try:
            # 读取音频
            audio, sr = sf.read(audio_path)
            
            # 转换为单声道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # 重采样
            if sr != target_sr:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_vhq")
            
            # 限制长度
            max_samples = int(self.config['max_audio_length'] * target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                logger.warning(f"音频长度超过限制，截断至{self.config['max_audio_length']}秒")
            
            return audio, target_sr
            
        except Exception as e:
            logger.error(f"音频预处理失败 {audio_path}: {e}")
            raise
    
    def save_processed_audio(self, audio: np.ndarray, output_path: str, sr: int):
        """保存处理后的音频"""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            sf.write(output_path, audio, sr)
            return True
        except Exception as e:
            logger.error(f"保存音频失败 {output_path}: {e}")
            return False
    
    def get_common_words(self) -> set:
        """获取常见英文单词集合"""
        # 常见的英文单词列表（可以根据需要扩展）
        common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
            'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'she', 'or', 'an', 'will',
            'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get',
            'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see', 'other', 'than', 'then',
            'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
            'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any', 'these', 'give', 'day',
            'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had', 'were', 'said', 'each', 'which', 'their',
            'said', 'she', 'use', 'how', 'many', 'oil', 'sit', 'set', 'run', 'eat', 'far', 'sea', 'eye', 'ago',
            'off', 'far', 'set', 'own', 'under', 'last', 'right', 'move', 'thing', 'general', 'school', 'never',
            'same', 'another', 'begin', 'while', 'number', 'part', 'turn', 'real', 'leave', 'might', 'great',
            'little', 'world', 'public', 'read', 'such', 'where', 'much', 'family', 'long', 'both', 'leave',
            'put', 'end', 'why', 'let', 'home', 'big', 'find', 'came', 'every', 'side', 'tried', 'told', 'men',
            'women', 'child', 'children', 'got', 'life', 'called', 'need', 'may', 'ask', 'went', 'say', 'kind',
            'must', 'house', 'picture', 'try', 'again', 'change', 'play', 'small', 'spell', 'move', 'live',
            'place', 'sound', 'great', 'again', 'still', 'every', 'large', 'must', 'big', 'even', 'such', 'because',
            'turn', 'here', 'why', 'ask', 'went', 'men', 'read', 'need', 'land', 'different', 'home', 'move',
            'try', 'kind', 'hand', 'picture', 'again', 'change', 'off', 'play', 'spell', 'air', 'away', 'animal',
            'house', 'point', 'page', 'letter', 'mother', 'answer', 'found', 'study', 'still', 'learn', 'should',
            'america', 'world', 'high', 'every', 'near', 'add', 'food', 'between', 'own', 'below', 'country',
            'plant', 'last', 'school', 'father', 'keep', 'tree', 'never', 'start', 'city', 'earth', 'eye', 'light',
            'thought', 'head', 'under', 'story', 'saw', 'left', 'dont', 'few', 'while', 'along', 'might', 'close',
            'something', 'seem', 'next', 'hard', 'open', 'example', 'begin', 'life', 'always', 'those', 'both',
            'paper', 'together', 'got', 'group', 'often', 'run', 'important', 'until', 'children', 'side', 'feet',
            'car', 'mile', 'night', 'walk', 'white', 'sea', 'began', 'grow', 'took', 'river', 'four', 'carry',
            'state', 'once', 'book', 'hear', 'stop', 'without', 'second', 'later', 'miss', 'idea', 'enough', 'eat',
            'face', 'watch', 'far', 'indian', 'really', 'almost', 'let', 'above', 'girl', 'sometimes', 'mountain',
            'cut', 'young', 'talk', 'soon', 'list', 'song', 'being', 'leave', 'family', 'its'
        }
        return common_words
    
    def normalize_text(self, text: str) -> str:
        """文本标准化，处理标点、空格、连词等问题"""
        import re
        import string
        
        if not text:
            return ""
        
        # 根据配置选择标准化级别
        normalization_level = self.config.get('text_normalization', 'advanced')
        
        if normalization_level == 'basic':
            # 基础标准化：处理大小写、标点符号和多余空格
            text = text.lower().strip()
            
            # 移除标点符号（如果启用）
            if self.config.get('remove_punctuation', True):
                # 包括中文标点符号
                punctuation = string.punctuation + '，。！？；：""''（）【】《》〈〉「」『』〖〗〔〕［］｛｝'
                for p in punctuation:
                    text = text.replace(p, '')
                
                # 处理特殊字符和符号
                text = re.sub(r'[^\w\s]', '', text)
            
            # 标准化空格
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # 转换为小写
        text = text.lower().strip()
        
        # 移除标点符号（如果启用）
        if self.config.get('remove_punctuation', True):
            # 包括中文标点符号
            punctuation = string.punctuation + '，。！？；：""''（）【】《》〈〉「」『』〖〗〔〕［］｛｝'
            for p in punctuation:
                text = text.replace(p, '')
            
            # 处理特殊字符和符号
            text = re.sub(r'[^\w\s]', '', text)
        
        # 统一空格处理（如果启用）
        if self.config.get('normalize_spacing', True):
            # 将多个空格替换为单个空格
            text = re.sub(r'\s+', ' ', text)
        
        # 处理拼读问题（如果启用）
        if self.config.get('handle_spelling', True) and normalization_level == 'advanced':
            text = self._handle_spelling_normalization(text)
        
        # 最终清理空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _handle_spelling_normalization(self, text: str) -> str:
        """处理拼读标准化"""
        import re
        
        # 获取常见单词集合
        common_words = self.get_common_words() if self.config.get('preserve_common_words', True) else set()
        max_length = self.config.get('spelling_max_length', 4)
        
        # 先处理明显的拼读模式（单个字符用空格分隔）
        # 匹配单个字符后跟空格的模式，这通常是拼读
        if re.search(r'\b[a-z]\s+[a-z]\b', text):
            # 这看起来已经是拼读模式，保持原样
            return text
        
        # 尝试识别可能的拼读模式
        words = text.split()
        processed_words = []
        
        for word in words:
            # 如果单词长度>1且全是字母或数字，可能需要拆分
            if len(word) > 1 and (word.isalpha() or word.isdigit()):
                should_split = False
                
                if word.isdigit():
                    # 数字序列，如果长度在合理范围内就拆分
                    if len(word) <= max_length:
                        should_split = True
                elif word.isalpha():
                    # 字母序列
                    if word.lower() in common_words:
                        # 是常见单词，不拆分
                        should_split = False
                    elif len(word) <= max_length:
                        # 短字母序列，可能是拼读
                        # 额外检查：如果包含元音且长度>2，可能是正常单词
                        if len(word) > 2 and any(vowel in word for vowel in 'aeiou'):
                            # 进一步检查是否符合英文单词的常见模式
                            # 简单规则：如果元音和辅音交替出现，可能是正常单词
                            vowels = 'aeiou'
                            vowel_pattern = sum(1 for c in word if c in vowels)
                            if vowel_pattern >= len(word) // 3:  # 至少1/3是元音
                                should_split = False
                            else:
                                should_split = True
                        else:
                            should_split = True
                
                if should_split:
                    processed_words.append(' '.join(word))
                else:
                    processed_words.append(word)
            else:
                processed_words.append(word)
        
        return ' '.join(processed_words)
    
    def calculate_wer_cer(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """计算WER和CER（文本应该已经标准化）"""
        try:
            if not reference or not hypothesis:
                return 1.0, 1.0  # 如果有空文本，返回最大错误率
            
            # 计算WER
            word_error_rate = wer(reference, hypothesis)
            
            # 计算CER
            char_error_rate = cer(reference, hypothesis)
            
            return word_error_rate, char_error_rate
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 计算WER/CER失败: {e}")
            return 1.0, 1.0
    
    def process_audio_pair(self, original_path: str, enhanced_path: str, 
                          output_dir: str, base_original_dir: str) -> Dict:
        """处理单个音频对"""
        
        result = {
            'original_path': original_path,
            'enhanced_path': enhanced_path,
            'processed_enhanced_path': '',
            'relative_path': '',  # 相对路径
            'original_transcription': '',
            'enhanced_transcription': '',
            'original_transcription_normalized': '',  # TN后的原始文本
            'enhanced_transcription_normalized': '',  # TN后的增强文本
            'wer': 0.0,
            'cer': 0.0,
            'is_usable': False,  # CER<5%的音频才可用
            'volume_scale_factor': 1.0,
            'volume_matching_enabled': self.volume_matching,
            'volume_matching_method': self.volume_matching_method,
            'processing_time': 0.0,
            'success': False,
            'error_message': '',
            'timestamp': datetime.now().isoformat(),
            'text_normalization_config': {
                'level': self.config.get('text_normalization', 'llm'),
                'remove_punctuation': self.config.get('remove_punctuation', True),
                'normalize_spacing': self.config.get('normalize_spacing', True),
                'handle_spelling': self.config.get('handle_spelling', True),
                'spelling_max_length': self.config.get('spelling_max_length', 4),
                'preserve_common_words': self.config.get('preserve_common_words', True),
                'use_llm_normalization': self.config.get('use_llm_normalization', False),
                'llm_service_url': self.config.get('llm_service_url', ''),
                'llm_enabled': self.http_normalizer is not None
            },
            'directory_config': {
                'name': self.dir_config.get('name', 'unknown'),
                'original_dir': self.dir_config.get('original_dir', ''),
                'enhanced_dir': self.dir_config.get('enhanced_dir', ''),
                'output_dir': self.dir_config.get('output_dir', ''),
            }
        }
        
        try:
            import time
            start_time = time.time()
            
            # 检查输入文件
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"原始音频文件不存在: {original_path}")
            if not os.path.exists(enhanced_path):
                raise FileNotFoundError(f"增强音频文件不存在: {enhanced_path}")
            
            # 计算相对路径，用于保持目录结构
            relative_path = os.path.relpath(original_path, base_original_dir)
            result['relative_path'] = relative_path
            
            # 创建对应的输出目录结构
            relative_dir = os.path.dirname(relative_path)
            structured_output_dir = os.path.join(output_dir, relative_dir) if relative_dir else output_dir
            os.makedirs(structured_output_dir, exist_ok=True)
            
            # 预处理音频
            logger.info(f"GPU {self.gpu_id}: 预处理音频: {original_path}")
            original_audio, sr = self.preprocess_audio(original_path)
            logger.info(f"GPU {self.gpu_id}: 预处理音频: {enhanced_path}")
            enhanced_audio, sr = self.preprocess_audio(enhanced_path)
            
            # 计算gain并应用到增强音频
            logger.info(f"GPU {self.gpu_id}: 计算并应用gain...")
            if self.volume_matching:
                if self.volume_matching_method == 'ten_vad_energy':
                    gain = self.volume_processor.calculate_gain_vad_energy(original_audio, enhanced_audio)
                    matched_audio, actual_gain = self.volume_processor.apply_gain(enhanced_audio, gain)
                    result['volume_scale_factor'] = actual_gain
                else:
                    # 其他方法暂时直接返回原音频
                    matched_audio = enhanced_audio
                    result['volume_scale_factor'] = 1.0
            else:
                # 不进行音量匹配
                matched_audio = enhanced_audio
                result['volume_scale_factor'] = 1.0
                logger.info(f"GPU {self.gpu_id}: 跳过音量匹配（已禁用）")
            
            result['volume_matching_enabled'] = self.volume_matching
            result['volume_matching_method'] = self.volume_matching_method
            
            # 保存处理后的音频（按原始目录结构）
            audio_basename = os.path.splitext(os.path.basename(relative_path))[0]
            matched_filename = f"{audio_basename}_gain_applied.wav"
            matched_path = os.path.join(structured_output_dir, matched_filename)
            
            if self.save_processed_audio(matched_audio, matched_path, sr):
                result['processed_enhanced_path'] = matched_path
            
            # 创建临时文件进行识别
            temp_original = os.path.join(structured_output_dir, f"temp_original_{audio_basename}.wav")
            temp_enhanced = matched_path
            
            # 保存原始音频（重采样后）
            if self.save_processed_audio(original_audio, temp_original, sr):
                
                # 语音识别
                logger.info(f"GPU {self.gpu_id}: 开始语音识别...")
                
                # 识别原始音频
                logger.info(f"GPU {self.gpu_id}: 识别原始音频...")
                original_transcription = self.kimi_processor.transcribe_audio(temp_original)
                result['original_transcription'] = original_transcription
                logger.info(f"GPU {self.gpu_id}: 原始音频识别结果: {original_transcription}")
                
                # 识别增强音频
                logger.info(f"GPU {self.gpu_id}: 识别增强音频...")
                enhanced_transcription = self.kimi_processor.transcribe_audio(temp_enhanced)
                result['enhanced_transcription'] = enhanced_transcription
                logger.info(f"GPU {self.gpu_id}: 增强音频识别结果: {enhanced_transcription}")
                
                # 计算WER和CER
                if original_transcription and enhanced_transcription:
                    # 先保存TN前的文本
                    result['original_transcription_normalized'] = original_transcription
                    result['enhanced_transcription_normalized'] = enhanced_transcription
                    
                    # 进行文本标准化
                    if self.http_normalizer is not None and self.config.get('text_normalization') == 'llm':
                        # 使用HTTP LLM标准化
                        logger.info(f"GPU {self.gpu_id}: 使用HTTP LLM进行文本标准化")
                        normalized_original, normalized_enhanced = self.http_normalizer.normalize_text_pair(
                            original_transcription, enhanced_transcription
                        )
                        result['original_transcription_normalized'] = normalized_original
                        result['enhanced_transcription_normalized'] = normalized_enhanced
                    else:
                        # 使用传统标准化
                        normalized_original = self.normalize_text(original_transcription)
                        normalized_enhanced = self.normalize_text(enhanced_transcription)
                        result['original_transcription_normalized'] = normalized_original
                        result['enhanced_transcription_normalized'] = normalized_enhanced
                    
                    # 计算WER和CER（使用标准化后的文本）
                    if normalized_original and normalized_enhanced:
                        wer_score = wer(normalized_original, normalized_enhanced)
                        cer_score = cer(normalized_original, normalized_enhanced)
                        result['wer'] = wer_score
                        result['cer'] = cer_score
                        
                        # 判断音频是否可用（CER < 5%）
                        result['is_usable'] = cer_score < 0.05
                        
                        logger.info(f"GPU {self.gpu_id}: WER: {wer_score:.4f}, CER: {cer_score:.4f}, 可用: {result['is_usable']}")
                    else:
                        logger.warning(f"GPU {self.gpu_id}: 标准化后文本为空，无法计算WER/CER")
                        result['is_usable'] = False
                else:
                    logger.warning(f"GPU {self.gpu_id}: 转录结果为空，无法计算WER/CER")
                    result['is_usable'] = False
                
                # 保存文本结果（按原始目录结构）
                self.save_text_results(result, structured_output_dir, audio_basename)
                
                # 清理临时文件
                try:
                    if os.path.exists(temp_original):
                        os.remove(temp_original)
                except:
                    pass
                
                result['success'] = True
                
            result['processing_time'] = time.time() - start_time
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"GPU {self.gpu_id}: 处理音频对失败: {e}")
        
        return result
    
    def run_assessment_subset(self, audio_pairs: List[Tuple[str, str]], subset_id: int):
        """运行质量评估子集"""
        logger.info(f"GPU {self.gpu_id}: 开始处理子集 {subset_id}，共 {len(audio_pairs)} 个音频对")
        
        # 创建子集输出目录
        subset_output_dir = os.path.join(self.dir_config['output_dir'], f"subset_{subset_id}")
        os.makedirs(subset_output_dir, exist_ok=True)
        
        # 处理音频对
        results = []
        
        for i, (original_path, enhanced_path) in enumerate(tqdm(audio_pairs, desc=f"GPU {self.gpu_id}: 处理子集{subset_id}")):
            
            logger.info(f"GPU {self.gpu_id}: 处理第{i+1}/{len(audio_pairs)}对音频")
            
            # 检查是否跳过已存在的结果
            if self.config['skip_existing']:
                # 计算相对路径和基础文件名
                relative_path = os.path.relpath(original_path, self.dir_config['original_dir'])
                audio_basename = os.path.splitext(os.path.basename(relative_path))[0]
                result_file = os.path.join(
                    subset_output_dir,
                    f"{audio_basename}_assessment.json"
                )
                if os.path.exists(result_file):
                    try:
                        with open(result_file, 'r', encoding='utf-8') as f:
                            existing_result = json.load(f)
                        results.append(existing_result)
                        logger.info(f"GPU {self.gpu_id}: 跳过已存在的结果: {result_file}")
                        continue
                    except:
                        logger.warning(f"GPU {self.gpu_id}: 读取已存在结果失败: {result_file}")
            
            # 处理音频对
            result = self.process_audio_pair(
                original_path, enhanced_path, subset_output_dir, self.dir_config['original_dir']
            )
            results.append(result)
            
            # 保存单个结果到子集目录
            audio_basename = os.path.splitext(os.path.basename(result.get('relative_path', enhanced_path)))[0]
            result_file = os.path.join(
                subset_output_dir,
                f"{audio_basename}_assessment.json"
            )
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 保存子集结果
        subset_results_file = os.path.join(subset_output_dir, f"subset_{subset_id}_results.json")
        with open(subset_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GPU {self.gpu_id}: 子集 {subset_id} 处理完成，结果保存在: {subset_output_dir}")
        
        return results

    def get_audio_pairs(self, dir_config: Dict) -> List[Tuple[str, str]]:
        """获取音频对列表"""
        audio_pairs = []
        
        # 遍历原始音频目录
        for root, dirs, files in os.walk(dir_config['original_dir']):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    original_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(original_path, dir_config['original_dir'])
                    
                    # 构造增强音频路径
                    enhanced_path = os.path.join(
                        dir_config['enhanced_dir'],
                        os.path.splitext(rel_path)[0] + '.wav'
                    )
                    
                    # 检查增强音频是否存在
                    if os.path.exists(enhanced_path):
                        audio_pairs.append((original_path, enhanced_path))
                    else:
                        logger.warning(f"增强音频不存在: {enhanced_path}")
        
        return audio_pairs
    
    def save_text_results(self, result: Dict, output_dir: str, audio_basename: str):
        """保存文本结果到JSON文件"""
        text_result_file = os.path.join(output_dir, f"{audio_basename}_text_results.json")
        with open(text_result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"GPU {self.gpu_id}: 文本结果已保存到: {text_result_file}")


def split_audio_pairs(audio_pairs: List[Tuple[str, str]], num_splits: int) -> List[List[Tuple[str, str]]]:
    """将音频对列表分成多个子集"""
    total_pairs = len(audio_pairs)
    pairs_per_split = total_pairs // num_splits
    remainder = total_pairs % num_splits
    
    splits = []
    start_idx = 0
    
    for i in range(num_splits):
        # 分配剩余的音频对到前几个子集
        current_split_size = pairs_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size
        
        splits.append(audio_pairs[start_idx:end_idx])
        start_idx = end_idx
    
    return splits

def process_gpu_subset(args_tuple):
    """处理单个GPU子集的函数"""
    gpu_id, audio_pairs_subset, config, dir_config, subset_id = args_tuple
    
    try:
        # 设置进程的GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 确保在子进程中重新初始化CUDA
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 因为CUDA_VISIBLE_DEVICES已经设置，所以这里用0
            
        # 创建处理器
        processor = QualityAssessmentProcessor(config, dir_config, gpu_id)
        
        # 处理子集
        results = processor.run_assessment_subset(audio_pairs_subset, subset_id)
        
        return results
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 处理子集失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def merge_results(output_dir: str, num_gpus: int, config: Dict, dir_config: Dict):
    """合并所有GPU的结果"""
    logger.info(f"开始合并结果 - 目录: {dir_config['name']}")
    
    all_results = []
    
    # 收集所有子集的结果
    for subset_id in range(num_gpus):
        subset_dir = os.path.join(output_dir, f"subset_{subset_id}")
        subset_results_file = os.path.join(subset_dir, f"subset_{subset_id}_results.json")
        
        if os.path.exists(subset_results_file):
            try:
                with open(subset_results_file, 'r', encoding='utf-8') as f:
                    subset_results = json.load(f)
                all_results.extend(subset_results)
                logger.info(f"合并子集 {subset_id} 的 {len(subset_results)} 个结果")
            except Exception as e:
                logger.error(f"读取子集 {subset_id} 结果失败: {e}")
    
    if not all_results:
        logger.error("没有找到任何结果文件")
        return
    
    # 保存合并后的结果
    logger.info(f"总共合并了 {len(all_results)} 个结果")
    
    # 保存详细结果
    results_file = os.path.join(output_dir, "quality_assessment_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存统计摘要
    successful_results = [r for r in all_results if r['success']]
    usable_results = [r for r in successful_results if r.get('is_usable', False)]
    
    if successful_results:
        wer_scores = [r['wer'] for r in successful_results]
        cer_scores = [r['cer'] for r in successful_results]
        
        # 可用音频的统计
        usable_wer_scores = [r['wer'] for r in usable_results]
        usable_cer_scores = [r['cer'] for r in usable_results]
        
        summary = {
            'directory_config': dir_config,
            'total_pairs': len(all_results),
            'successful_pairs': len(successful_results),
            'failed_pairs': len(all_results) - len(successful_results),
            'usable_pairs': len(usable_results),
            'usable_rate': len(usable_results) / len(successful_results) if successful_results else 0.0,
            'average_wer': float(np.mean(wer_scores)) if wer_scores else 0.0,
            'average_cer': float(np.mean(cer_scores)) if cer_scores else 0.0,
            'median_wer': float(np.median(wer_scores)) if wer_scores else 0.0,
            'median_cer': float(np.median(cer_scores)) if cer_scores else 0.0,
            'wer_std': float(np.std(wer_scores)) if wer_scores else 0.0,
            'cer_std': float(np.std(cer_scores)) if cer_scores else 0.0,
            'min_wer': float(np.min(wer_scores)) if wer_scores else 0.0,
            'max_wer': float(np.max(wer_scores)) if wer_scores else 0.0,
            'min_cer': float(np.min(cer_scores)) if cer_scores else 0.0,
            'max_cer': float(np.max(cer_scores)) if cer_scores else 0.0,
            # 可用音频统计
            'usable_average_wer': float(np.mean(usable_wer_scores)) if usable_wer_scores else 0.0,
            'usable_average_cer': float(np.mean(usable_cer_scores)) if usable_cer_scores else 0.0,
            'usable_median_wer': float(np.median(usable_wer_scores)) if usable_wer_scores else 0.0,
            'usable_median_cer': float(np.median(usable_cer_scores)) if usable_cer_scores else 0.0,
            'global_config': config,
            'timestamp': datetime.now().isoformat()
        }
    else:
        summary = {
            'directory_config': dir_config,
            'total_pairs': len(all_results),
            'successful_pairs': 0,
            'failed_pairs': len(all_results),
            'usable_pairs': 0,
            'usable_rate': 0.0,
            'average_wer': 0.0,
            'average_cer': 0.0,
            'median_wer': 0.0,
            'median_cer': 0.0,
            'wer_std': 0.0,
            'cer_std': 0.0,
            'min_wer': 0.0,
            'max_wer': 0.0,
            'min_cer': 0.0,
            'max_cer': 0.0,
            'usable_average_wer': 0.0,
            'usable_average_cer': 0.0,
            'usable_median_wer': 0.0,
            'usable_median_cer': 0.0,
            'global_config': config,
            'timestamp': datetime.now().isoformat()
        }
    
    summary_file = os.path.join(output_dir, "assessment_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 保存CSV格式
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(output_dir, "quality_assessment_results.csv")
        df.to_csv(csv_file, index=False)
    except ImportError:
        logger.warning("pandas未安装，跳过CSV保存")
    
    logger.info(f"合并结果保存到: {output_dir}")
    
    # 打印统计摘要
    print_summary(all_results, dir_config)
    
    return all_results

def print_summary(results: List[Dict], dir_config: Dict):
    """打印统计摘要"""
    successful_results = [r for r in results if r['success']]
    usable_results = [r for r in successful_results if r.get('is_usable', False)]
    
    print("\n" + "=" * 80)
    print(f"音频质量评估结果摘要 - 目录: {dir_config['name']}")
    print("=" * 80)
    print(f"总音频对数:     {len(results)}")
    print(f"成功处理:       {len(successful_results)}")
    print(f"失败处理:       {len(results) - len(successful_results)}")
    print(f"可用音频数:     {len(usable_results)} (CER<5%)")
    print(f"可用率:         {len(usable_results)/len(successful_results)*100:.1f}%" if successful_results else "0.0%")
    
    if successful_results:
        wer_scores = [r['wer'] for r in successful_results]
        cer_scores = [r['cer'] for r in successful_results]
        usable_wer_scores = [r['wer'] for r in usable_results]
        usable_cer_scores = [r['cer'] for r in usable_results]
        
        print(f"\n词错误率 (WER):")
        print(f"  平均值:       {np.mean(wer_scores):.4f}")
        print(f"  中位数:       {np.median(wer_scores):.4f}")
        print(f"  标准差:       {np.std(wer_scores):.4f}")
        print(f"  最小值:       {np.min(wer_scores):.4f}")
        print(f"  最大值:       {np.max(wer_scores):.4f}")
        
        print(f"\n字符错误率 (CER):")
        print(f"  平均值:       {np.mean(cer_scores):.4f}")
        print(f"  中位数:       {np.median(cer_scores):.4f}")
        print(f"  标准差:       {np.std(cer_scores):.4f}")
        print(f"  最小值:       {np.min(cer_scores):.4f}")
        print(f"  最大值:       {np.max(cer_scores):.4f}")
        
        # 可用音频统计
        if usable_results:
            print(f"\n可用音频质量 (CER<5%):")
            print(f"  WER平均值:    {np.mean(usable_wer_scores):.4f}")
            print(f"  WER中位数:    {np.median(usable_wer_scores):.4f}")
            print(f"  CER平均值:    {np.mean(usable_cer_scores):.4f}")
            print(f"  CER中位数:    {np.median(usable_cer_scores):.4f}")
        
        # WER分布
        print(f"\nWER分布:")
        excellent = sum(1 for w in wer_scores if w < 0.1)
        good = sum(1 for w in wer_scores if 0.1 <= w < 0.2)
        fair = sum(1 for w in wer_scores if 0.2 <= w < 0.3)
        poor = sum(1 for w in wer_scores if w >= 0.3)
        
        print(f"  优秀 (WER < 0.1):   {excellent} ({excellent/len(wer_scores)*100:.1f}%)")
        print(f"  良好 (0.1 ≤ WER < 0.2): {good} ({good/len(wer_scores)*100:.1f}%)")
        print(f"  一般 (0.2 ≤ WER < 0.3): {fair} ({fair/len(wer_scores)*100:.1f}%)")
        print(f"  较差 (WER ≥ 0.3):   {poor} ({poor/len(wer_scores)*100:.1f}%)")
        
        # CER分布
        print(f"\nCER分布 (可用性判断):")
        usable_cer = sum(1 for c in cer_scores if c < 0.05)
        marginal_cer = sum(1 for c in cer_scores if 0.05 <= c < 0.10)
        poor_cer = sum(1 for c in cer_scores if c >= 0.10)
        
        print(f"  可用 (CER < 5%):    {usable_cer} ({usable_cer/len(cer_scores)*100:.1f}%)")
        print(f"  边际 (5% ≤ CER < 10%): {marginal_cer} ({marginal_cer/len(cer_scores)*100:.1f}%)")
        print(f"  不可用 (CER ≥ 10%): {poor_cer} ({poor_cer/len(cer_scores)*100:.1f}%)")
        
        # 增益应用方法统计
        volume_methods = [r['volume_matching_method'] for r in successful_results]
        print(f"\n增益应用方法: {volume_methods[0] if volume_methods else 'N/A'}")
        
        # 处理时间统计
        processing_times = [r['processing_time'] for r in successful_results]
        print(f"\n处理时间统计:")
        print(f"  平均处理时间:   {np.mean(processing_times):.2f} 秒")
        print(f"  总处理时间:     {np.sum(processing_times):.2f} 秒")
    
    print("=" * 80)

def get_audio_pairs(dir_config: Dict) -> List[Tuple[str, str]]:
    """获取音频对列表"""
    audio_pairs = []
    
    # 遍历原始音频目录
    for root, dirs, files in os.walk(dir_config['original_dir']):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                original_path = os.path.join(root, file)
                
                # 计算相对路径
                rel_path = os.path.relpath(original_path, dir_config['original_dir'])
                
                # 构造增强音频路径
                enhanced_path = os.path.join(
                    dir_config['enhanced_dir'],
                    os.path.splitext(rel_path)[0] + '.wav'
                )
                
                # 检查增强音频是否存在
                if os.path.exists(enhanced_path):
                    audio_pairs.append((original_path, enhanced_path))
                else:
                    logger.warning(f"增强音频不存在: {enhanced_path}")
    
    return audio_pairs

def main():
    """主函数"""
    # 设置multiprocessing启动方法为spawn（用于CUDA兼容性）
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        # 如果已经设置过了，就忽略
        pass
    
    parser = argparse.ArgumentParser(description="音频质量评估脚本 (多GPU并行) - 支持多目录处理")
    
    # 多目录配置相关参数
    parser.add_argument("--config_file", type=str, 
                       help="配置文件路径（JSON格式），优先级高于命令行参数")
    parser.add_argument("--original_dirs", type=str, nargs='+',
                       help="原始音频目录列表")
    parser.add_argument("--enhanced_dirs", type=str, nargs='+',
                       help="增强音频目录列表")
    parser.add_argument("--output_dirs", type=str, nargs='+',
                       help="输出目录列表")
    parser.add_argument("--config_names", type=str, nargs='+',
                       help="配置名称列表")
    parser.add_argument("--volume_matching", type=str, nargs='+',
                       choices=['true', 'false'],
                       help="每个目录是否启用音量匹配（true/false）")
    parser.add_argument("--volume_methods", type=str, nargs='+',
                       choices=['ten_vad_energy'],
                       help="每个目录的音量匹配方法")
    
    # 向后兼容的单目录参数
    parser.add_argument("--original_dir", type=str,
                       help="原始音频目录（单目录模式）")
    parser.add_argument("--enhanced_dir", type=str,
                       help="增强音频目录（单目录模式）")
    parser.add_argument("--output_dir", type=str,
                       help="输出目录（单目录模式）")
    parser.add_argument("--volume_method", type=str,
                       choices=['ten_vad_energy'],
                       help="音量匹配方法（单目录模式）")
    
    # 全局配置参数
    parser.add_argument("--kimi_model_path", type=str,
                       default=CONFIG['kimi_model_path'],
                       help="Kimi-Audio模型路径")
    parser.add_argument("--kimi_audio_dir", type=str,
                       default=CONFIG['kimi_audio_dir'],
                       help="Kimi-Audio代码目录")
    parser.add_argument("--skip_existing", action="store_true",
                       default=CONFIG['skip_existing'],
                       help="跳过已存在的结果")
    parser.add_argument("--max_audio_length", type=int,
                       default=CONFIG['max_audio_length'],
                       help="最大音频长度（秒）")
    parser.add_argument("--ten_vad_hop_size", type=int,
                       default=CONFIG['ten_vad_hop_size'],
                       help="TEN VAD帧跳跃大小（样本数，256样本=16ms@16kHz）")
    parser.add_argument("--ten_vad_threshold", type=float,
                       default=CONFIG['ten_vad_threshold'],
                       help="TEN VAD阈值（0.0-1.0）")
    parser.add_argument("--num_gpus", type=int,
                       default=CONFIG['num_gpus'],
                       help="GPU数量（GPU 1,2,3用于ASR，GPU 0用于LLM服务）")
    parser.add_argument("--gpu_ids", type=int, nargs='+',
                       default=CONFIG['gpu_ids'],
                       help="GPU ID列表（建议使用 1,2,3，GPU 0留给LLM服务）")
    parser.add_argument("--text_normalization", type=str,
                       default=CONFIG['text_normalization'],
                       choices=['basic', 'advanced', 'llm'],
                       help="文本标准化级别")
    parser.add_argument("--remove_punctuation", action="store_true",
                       default=CONFIG['remove_punctuation'],
                       help="移除标点符号")
    parser.add_argument("--normalize_spacing", action="store_true",
                       default=CONFIG['normalize_spacing'],
                       help="标准化空格")
    parser.add_argument("--handle_spelling", action="store_true",
                       default=CONFIG['handle_spelling'],
                       help="处理拼读问题")
    parser.add_argument("--spelling_max_length", type=int,
                       default=CONFIG['spelling_max_length'],
                       help="拼读检测的最大长度")
    parser.add_argument("--preserve_common_words", action="store_true",
                       default=CONFIG['preserve_common_words'],
                       help="保留常见单词不拆分")
    parser.add_argument("--llm_service_url", type=str,
                       default=CONFIG['llm_service_url'],
                       help="LLM服务URL")
    parser.add_argument("--llm_timeout", type=int,
                       default=CONFIG['llm_timeout'],
                       help="LLM服务超时时间（秒）")
    parser.add_argument("--llm_max_retries", type=int,
                       default=CONFIG['llm_max_retries'],
                       help="LLM服务最大重试次数")
    parser.add_argument("--use_llm_normalization", action="store_true",
                       default=CONFIG['use_llm_normalization'],
                       help="使用LLM进行文本标准化")
    
    args = parser.parse_args()
    
    # 处理配置文件
    if args.config_file:
        try:
            with open(args.config_file, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 合并配置文件到全局配置
            for key, value in file_config.items():
                if key in CONFIG:
                    CONFIG[key] = value
            
            logger.info(f"从配置文件加载配置: {args.config_file}")
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return
    
    # 处理目录配置
    directory_configs = []
    
    # 检查是否提供了多目录参数
    if args.original_dirs:
        # 多目录模式
        num_configs = len(args.original_dirs)
        
        # 验证参数长度一致性
        if not args.enhanced_dirs or len(args.enhanced_dirs) != num_configs:
            logger.error("enhanced_dirs数量必须与original_dirs数量一致")
            return
        if not args.output_dirs or len(args.output_dirs) != num_configs:
            logger.error("output_dirs数量必须与original_dirs数量一致")
            return
        
        # 设置默认值
        config_names = args.config_names or [f"config_{i+1}" for i in range(num_configs)]
        volume_matching = args.volume_matching or ['true'] * num_configs
        volume_methods = args.volume_methods or ['ten_vad_energy'] * num_configs
        
        # 验证默认值长度
        if len(config_names) != num_configs:
            config_names = [f"config_{i+1}" for i in range(num_configs)]
        if len(volume_matching) != num_configs:
            volume_matching = ['true'] * num_configs
        if len(volume_methods) != num_configs:
            volume_methods = ['ten_vad_energy'] * num_configs
        
        # 创建目录配置
        for i in range(num_configs):
            directory_configs.append({
                'name': config_names[i],
                'original_dir': args.original_dirs[i],
                'enhanced_dir': args.enhanced_dirs[i],
                'output_dir': args.output_dirs[i],
                'volume_matching': volume_matching[i].lower() == 'true',
                'volume_matching_method': volume_methods[i],
            })
    
    elif args.original_dir:
        # 单目录模式（向后兼容）
        directory_configs.append({
            'name': 'single_dir',
            'original_dir': args.original_dir,
            'enhanced_dir': args.enhanced_dir,
            'output_dir': args.output_dir,
            'volume_matching': True,
            'volume_matching_method': args.volume_method or 'ten_vad_energy',
        })
    
    else:
        # 使用默认配置
        directory_configs = CONFIG['directory_configs']
    
    # 更新CONFIG
    CONFIG['directory_configs'] = directory_configs
    
    # 更新全局配置
    global_config_keys = ['kimi_model_path', 'kimi_audio_dir', 'skip_existing', 
                         'max_audio_length', 'ten_vad_hop_size', 'ten_vad_threshold',
                         'num_gpus', 'gpu_ids', 'text_normalization', 'remove_punctuation',
                         'normalize_spacing', 'handle_spelling', 'spelling_max_length',
                         'preserve_common_words', 'llm_service_url', 'llm_timeout',
                         'llm_max_retries', 'use_llm_normalization']
    
    for key in global_config_keys:
        if hasattr(args, key) and getattr(args, key) is not None:
            CONFIG[key] = getattr(args, key)
    
    # 确保GPU数量和ID列表一致
    if len(CONFIG['gpu_ids']) != CONFIG['num_gpus']:
        CONFIG['gpu_ids'] = list(range(1, CONFIG['num_gpus'] + 1))  # 默认使用GPU 1,2,3...
    
    # 打印配置信息
    print("音频质量评估脚本 (多GPU并行版本 - 支持多目录处理)")
    print("=" * 80)
    print(f"目录配置数量: {len(CONFIG['directory_configs'])}")
    for i, dir_config in enumerate(CONFIG['directory_configs']):
        print(f"\n目录配置 {i+1}: {dir_config['name']}")
        print(f"  原始音频目录: {dir_config['original_dir']}")
        print(f"  增强音频目录: {dir_config['enhanced_dir']}")
        print(f"  输出目录: {dir_config['output_dir']}")
        print(f"  音量匹配: {'启用' if dir_config['volume_matching'] else '禁用'}")
        if dir_config['volume_matching']:
            print(f"  音量匹配方法: {dir_config['volume_matching_method']}")
    
    print(f"\n全局配置:")
    print(f"Kimi-Audio模型: {CONFIG['kimi_model_path']}")
    print(f"Kimi-Audio目录: {CONFIG['kimi_audio_dir']}")
    print(f"GPU配置: GPU 0 (LLM服务), GPU {CONFIG['gpu_ids']} (ASR)")
    print(f"ASR GPU数量: {CONFIG['num_gpus']}")
    print(f"文本标准化级别: {CONFIG['text_normalization']}")
    print(f"TEN VAD配置: hop_size={CONFIG['ten_vad_hop_size']}, threshold={CONFIG['ten_vad_threshold']}")
    print(f"跳过已存在: {CONFIG['skip_existing']}")
    print(f"最大音频长度: {CONFIG['max_audio_length']} 秒")
    print("=" * 80)
    
    # 验证基本配置
    if not os.path.exists(CONFIG['kimi_model_path']):
        logger.error(f"Kimi-Audio模型路径不存在: {CONFIG['kimi_model_path']}")
        return
    
    if not os.path.exists(CONFIG['kimi_audio_dir']):
        logger.error(f"Kimi-Audio代码目录不存在: {CONFIG['kimi_audio_dir']}")
        return
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法使用GPU")
        return
    
    available_gpus = torch.cuda.device_count()
    max_required_gpu = max(CONFIG['gpu_ids']) if CONFIG['gpu_ids'] else 0
    if available_gpus <= max_required_gpu:
        logger.error(f"可用GPU数量({available_gpus})不足，需要GPU {max_required_gpu}")
        return
    
    # 检查LLM服务连接（如果启用）
    if CONFIG.get('use_llm_normalization', False) and CONFIG.get('text_normalization') == 'llm':
        try:
            import requests
            session = requests.Session()
            session.proxies = {'http': None, 'https': None}
            
            health_url = f"{CONFIG['llm_service_url']}/health"
            response = session.get(health_url, timeout=10)
            response.raise_for_status()
            print(f"✓ LLM服务连接正常: {CONFIG['llm_service_url']}")
        except Exception as e:
            logger.error(f"LLM服务连接失败: {e}")
            return
    
    # 处理每个目录配置
    total_start_time = time.time()
    all_directory_results = []
    
    for dir_idx, dir_config in enumerate(CONFIG['directory_configs']):
        print(f"\n{'='*80}")
        print(f"开始处理目录配置 {dir_idx+1}/{len(CONFIG['directory_configs'])}: {dir_config['name']}")
        print(f"{'='*80}")
        
        try:
            # 验证目录存在
            if not os.path.exists(dir_config['original_dir']):
                logger.error(f"原始音频目录不存在: {dir_config['original_dir']}")
                continue
            
            if not os.path.exists(dir_config['enhanced_dir']):
                logger.error(f"增强音频目录不存在: {dir_config['enhanced_dir']}")
                continue
            
            # 创建输出目录
            os.makedirs(dir_config['output_dir'], exist_ok=True)
            
            # 获取音频对列表
            logger.info(f"获取音频对列表: {dir_config['name']}")
            audio_pairs = get_audio_pairs(dir_config)
            
            if not audio_pairs:
                logger.warning(f"目录 {dir_config['name']} 中未找到音频对")
                continue
            
            logger.info(f"找到 {len(audio_pairs)} 个音频对")
            
            # 分割音频对
            logger.info(f"将音频对分成 {CONFIG['num_gpus']} 个子集...")
            audio_pairs_splits = split_audio_pairs(audio_pairs, CONFIG['num_gpus'])
            
            # 准备多进程参数
            process_args = []
            for i, (gpu_id, audio_pairs_subset) in enumerate(zip(CONFIG['gpu_ids'], audio_pairs_splits)):
                process_args.append((gpu_id, audio_pairs_subset, CONFIG, dir_config, i))
            
            # 启动多进程处理
            logger.info(f"启动多GPU并行处理: {dir_config['name']}")
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=CONFIG['num_gpus']) as executor:
                futures = []
                for args in process_args:
                    future = executor.submit(process_gpu_subset, args)
                    futures.append(future)
                
                # 等待所有进程完成
                results = []
                for future in futures:
                    try:
                        result = future.result()
                        results.extend(result)
                    except Exception as e:
                        logger.error(f"进程执行失败: {e}")
            
            processing_time = time.time() - start_time
            logger.info(f"目录 {dir_config['name']} 处理完成，耗时: {processing_time:.2f} 秒")
            
            # 合并结果
            dir_results = merge_results(dir_config['output_dir'], CONFIG['num_gpus'], CONFIG, dir_config)
            all_directory_results.extend(dir_results)
            
        except Exception as e:
            logger.error(f"处理目录 {dir_config['name']} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 总结所有目录的结果
    total_processing_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("所有目录处理完成！")
    print(f"总处理时间: {total_processing_time:.2f} 秒")
    print(f"处理的目录数: {len(CONFIG['directory_configs'])}")
    print(f"总音频对数: {len(all_directory_results)}")
    print(f"{'='*80}")
    
    # 保存总体摘要
    if all_directory_results:
        total_summary = {
            'total_directories': len(CONFIG['directory_configs']),
            'total_audio_pairs': len(all_directory_results),
            'total_processing_time': total_processing_time,
            'directory_configs': CONFIG['directory_configs'],
            'global_config': CONFIG,
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到第一个输出目录的上级目录
        if CONFIG['directory_configs']:
            summary_dir = os.path.dirname(CONFIG['directory_configs'][0]['output_dir'])
            summary_file = os.path.join(summary_dir, "multi_directory_summary.json")
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(total_summary, f, indent=2, ensure_ascii=False)
            print(f"总体摘要保存到: {summary_file}")
    
    return

if __name__ == "__main__":
    main() 