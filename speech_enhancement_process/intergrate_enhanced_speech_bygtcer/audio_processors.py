#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频处理模块
包含语音识别、文本标准化、音量处理等核心组件
"""

import os
import sys
import json
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import fcntl

# 导入本地TEN VAD
from ten_vad import TenVad

# 设置日志
logger = logging.getLogger(__name__)

class AudioVolumeProcessor:
    """音频音量处理器"""
    
    def __init__(self, sr: int = 16000, ten_vad_config: Dict = None):
        self.sr = sr
        self.eps = 1e-8  # 避免除零错误
        
        # 从配置中获取TEN VAD参数
        if ten_vad_config is None:
            ten_vad_config = {'hop_size': 256, 'threshold': 0.5}
            
        self.hop_size = ten_vad_config.get('hop_size', 256)
        self.threshold = ten_vad_config.get('threshold', 0.5)
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
    """基于HTTP API的文本标准化器 - 带服务自动恢复功能"""
    
    def __init__(self, service_url: str, timeout: int = 30, max_retries: int = 3, gpu_id: int = 0):
        self.service_url = service_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        self.gpu_id = gpu_id
        self.service_restart_attempts = 0
        self.max_service_restarts = 3
        self.restart_lock_file = "/tmp/llm_service_restart.lock"
        
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
        
        # 测试连接（严格检查，失败时抛出异常）
        self._test_connection()
    
    def _restart_llm_service(self) -> bool:
        """重启LLM服务（带锁机制避免多进程冲突）"""
        import subprocess
        
        if self.service_restart_attempts >= self.max_service_restarts:
            logger.error(f"GPU {self.gpu_id}: 达到最大服务重启次数限制 ({self.max_service_restarts})")
            return False
        
        # 尝试获取重启锁
        try:
            lock_fd = os.open(self.restart_lock_file, os.O_CREAT | os.O_TRUNC | os.O_RDWR)
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            logger.info(f"GPU {self.gpu_id}: 获得服务重启锁")
        except (OSError, IOError):
            logger.info(f"GPU {self.gpu_id}: 其他进程正在重启服务，等待完成...")
            # 等待其他进程完成重启
            for wait_attempt in range(30):  # 最多等待60秒
                time.sleep(2)
                if self._test_connection_quiet():
                    logger.info(f"GPU {self.gpu_id}: 服务已被其他进程恢复")
                    return True
            logger.warning(f"GPU {self.gpu_id}: 等待其他进程重启超时")
            return False
        
        try:
            self.service_restart_attempts += 1
            logger.warning(f"GPU {self.gpu_id}: 开始重启LLM服务 (第{self.service_restart_attempts}次)")
            
            # 清除可能的proxy干扰
            env = os.environ.copy()
            env.pop('http_proxy', None)
            env.pop('https_proxy', None)
            env.pop('HTTP_PROXY', None)
            env.pop('HTTPS_PROXY', None)
            
            # 停止现有服务
            logger.info(f"GPU {self.gpu_id}: 停止现有LLM服务...")
            stop_result = subprocess.run(
                ["./stop_multi_llm_services.sh"],
                env=env,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # 等待服务完全停止
            time.sleep(5)
            
            # 重新启动服务
            logger.info(f"GPU {self.gpu_id}: 重新启动LLM服务...")
            start_result = subprocess.run(
                ["./auto_start_llm_services.sh"],
                env=env,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if start_result.returncode == 0:
                # 等待服务初始化
                logger.info(f"GPU {self.gpu_id}: 等待服务初始化...")
                time.sleep(20)
                
                # 测试服务是否恢复
                if self._test_connection_quiet():
                    logger.info(f"GPU {self.gpu_id}: LLM服务重启成功")
                    return True
                else:
                    logger.error(f"GPU {self.gpu_id}: LLM服务重启后仍然不可用")
                    return False
            else:
                logger.error(f"GPU {self.gpu_id}: LLM服务启动失败: {start_result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 重启LLM服务时出错: {e}")
            return False
        finally:
            # 释放锁
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
                os.remove(self.restart_lock_file)
                logger.info(f"GPU {self.gpu_id}: 释放服务重启锁")
            except:
                pass
    
    def _test_connection_quiet(self) -> bool:
        """静默测试连接（不打印日志）"""
        try:
            health_url = f"{self.service_url}/health"
            response = self.session.get(health_url, timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _test_connection(self):
        """测试与LLM服务的连接 - 带重试机制"""
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                health_url = f"{self.service_url}/health"
                response = self.session.get(health_url, timeout=10)
                response.raise_for_status()
                
                # 验证响应内容
                try:
                    response_data = response.json()
                    if response_data.get("status") == "healthy":
                        logger.info(f"GPU {self.gpu_id}: ✓ LLM服务连接正常")
                        break
                except:
                    if "healthy" in response.text.lower():
                        logger.info(f"GPU {self.gpu_id}: ✓ LLM服务连接正常")
                        break
                    else:
                        raise requests.exceptions.RequestException("健康检查响应格式异常")
                
            except requests.exceptions.RequestException as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"GPU {self.gpu_id}: LLM服务连接失败 (尝试 {attempt+1}/{max_attempts}): {e}")
                    logger.info(f"GPU {self.gpu_id}: 等待 3 秒后重试...")
                    time.sleep(3)
                    continue
                else:
                    logger.error(f"GPU {self.gpu_id}: LLM服务连接失败，已尝试 {max_attempts} 次: {e}")
                    raise
        
        # 获取模型信息（非关键，失败不影响主要功能）
        try:
            model_info_url = f"{self.service_url}/model_info"
            response = self.session.get(model_info_url, timeout=10)
            response.raise_for_status()
            info = response.json()
            logger.info(f"GPU {self.gpu_id}: LLM模型信息: {info}")
        except Exception as e:
            logger.debug(f"GPU {self.gpu_id}: 无法获取模型信息 (不影响主要功能): {e}")
    
    def normalize_text_pair(self, text1: str, text2: str) -> Tuple[str, str]:
        """使用HTTP API标准化文本对 - 带自动服务恢复功能"""
        
        for attempt in range(self.max_retries + 1):
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
                
                # 成功时重置重启计数器
                self.service_restart_attempts = 0
                return normalized_text1, normalized_text2
                
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout, 
                    requests.exceptions.RequestException) as e:
                
                logger.warning(f"GPU {self.gpu_id}: LLM服务连接失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")
                
                # 如果不是最后一次尝试
                if attempt < self.max_retries:
                    # 尝试重启服务（仅在前几次尝试时）
                    if attempt == 0:
                        logger.warning(f"GPU {self.gpu_id}: 检测到服务不可用，尝试自动恢复...")
                        if self._restart_llm_service():
                            logger.info(f"GPU {self.gpu_id}: 服务恢复成功，继续重试...")
                            continue
                        else:
                            logger.error(f"GPU {self.gpu_id}: 服务恢复失败")
                    
                    # 等待后重试
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    logger.info(f"GPU {self.gpu_id}: 等待 {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 最后一次尝试失败
                    logger.error(f"GPU {self.gpu_id}: 所有重试均失败，LLM服务不可用")
                    raise RuntimeError(f"LLM服务不可用，已尝试 {self.max_retries + 1} 次")
                    
            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: HTTP LLM标准化失败: {e}")
                if attempt < self.max_retries:
                    time.sleep(2)
                    continue
                else:
                    raise RuntimeError(f"HTTP LLM标准化失败: {e}")
        
        # 这里不应该到达，但以防万一
        raise RuntimeError("未知错误：标准化过程意外结束")


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
        # 初始化current_dir，确保在任何异常发生前都已定义
        current_dir = os.getcwd()
        
        try:
            # 确保模型已经加载
            if self.model is None:
                self.load_model()
            
            # 切换到Kimi-Audio目录进行推理
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
            try:
                os.chdir(current_dir)
            except Exception as e:
                logger.warning(f"GPU {self.gpu_id}: 恢复工作目录失败: {e}")
                # 尝试恢复到原始工作目录
                try:
                    os.chdir(self.original_cwd)
                except:
                    pass
    
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