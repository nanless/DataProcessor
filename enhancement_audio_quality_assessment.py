#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频质量评估脚本 (多卡并行版本)
使用Kimi-Audio模型对降噪前后的音频进行识别和WER计算
使用webrtcvad进行VAD，基于语音段能量计算gain值
支持多GPU并行处理
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
import webrtcvad
import struct
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import pickle

warnings.filterwarnings("ignore")

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'original_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid",
    'enhanced_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced",
    'output_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment",
    'kimi_model_path': "/root/data/pretrained_models/Kimi-Audio-7B-Instruct",
    'kimi_audio_dir': "/root/code/github_repos/Kimi-Audio",
    'device': "cuda:0",
    'target_sr': 16000,  # Kimi-Audio期望的采样率
    'batch_size': 8,
    'num_workers': 4,
    'skip_existing': True,
    'max_audio_length': 600,  # 最大音频长度（秒）
    'volume_matching_method': 'vad_energy',  # 音量匹配方法: 'vad_energy'
    'vad_aggressiveness': 3,  # VAD积极性等级 (0-3)
    'vad_frame_duration': 30,  # VAD帧长度（毫秒）
    'num_gpus': 4,  # GPU数量
    'gpu_ids': [0, 1, 2, 3],  # GPU ID列表
}

class AudioVolumeProcessor:
    """音频音量处理器"""
    
    def __init__(self, sr: int = 16000):
        self.sr = sr
        self.eps = 1e-8  # 避免除零错误
        self.vad = webrtcvad.Vad(CONFIG['vad_aggressiveness'])
        self.frame_duration = CONFIG['vad_frame_duration']  # 毫秒
        self.frame_size = int(sr * self.frame_duration / 1000)  # 样本数
    
    def audio_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """将浮点音频转换为int16格式"""
        # 确保音频在[-1, 1]范围内
        audio = np.clip(audio, -1.0, 1.0)
        # 转换为int16
        return (audio * 32767).astype(np.int16)
    
    def perform_vad(self, audio: np.ndarray) -> List[bool]:
        """执行语音活动检测"""
        try:
            # 转换为int16格式
            audio_int16 = self.audio_to_int16(audio)
            
            # 确保音频长度是帧大小的整数倍
            num_frames = len(audio_int16) // self.frame_size
            audio_padded = audio_int16[:num_frames * self.frame_size]
            
            # 将音频分帧
            frames = audio_padded.reshape(-1, self.frame_size)
            
            # 对每帧进行VAD
            vad_results = []
            for frame in frames:
                # 转换为bytes
                frame_bytes = frame.tobytes()
                # 进行VAD检测
                is_speech = self.vad.is_speech(frame_bytes, self.sr)
                vad_results.append(is_speech)
            
            return vad_results
            
        except Exception as e:
            logger.warning(f"VAD检测失败: {e}")
            # 如果VAD失败，返回全部为语音的结果
            num_frames = len(audio) // self.frame_size
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
                    start_idx = i * self.frame_size
                    end_idx = (i + 1) * self.frame_size
                    if end_idx <= len(audio):
                        speech_segments.append(audio[start_idx:end_idx])
            
            if speech_segments:
                # 连接所有语音段
                speech_audio = np.concatenate(speech_segments)
                logger.info(f"VAD检测: 原始长度 {len(audio)/self.sr:.2f}s, 语音段长度 {len(speech_audio)/self.sr:.2f}s")
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
            
            logger.info(f"VAD能量计算: 参考能量={ref_energy:.6f}, 增强能量={enh_energy:.6f}, gain={gain:.4f}")
            
            return gain
            
        except Exception as e:
            logger.warning(f"VAD能量gain计算失败: {e}")
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
    
    def __init__(self, config: Dict, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        self.volume_processor = AudioVolumeProcessor(sr=config['target_sr'])
        self.kimi_processor = KimiAudioProcessor(
            model_path=config['kimi_model_path'],
            device=self.device,
            kimi_audio_dir=config['kimi_audio_dir'],
            gpu_id=gpu_id
        )
        
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
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            
            # 限制长度
            max_samples = int(self.config['max_audio_length'] * target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                logger.warning(f"音频长度超过限制，截断至{self.config['max_audio_length']}秒")
            
            # 归一化
            if np.max(np.abs(audio)) > 0:
                audio = audio / np.max(np.abs(audio)) * 0.95
            
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
    
    def calculate_wer_cer(self, reference: str, hypothesis: str) -> Tuple[float, float]:
        """计算WER和CER"""
        try:
            if not reference or not hypothesis:
                return 1.0, 1.0  # 如果有空文本，返回最大错误率
            
            # 文本预处理
            reference = reference.strip().lower()
            hypothesis = hypothesis.strip().lower()
            
            # 计算WER
            word_error_rate = wer(reference, hypothesis)
            
            # 计算CER
            char_error_rate = cer(reference, hypothesis)
            
            return word_error_rate, char_error_rate
            
        except Exception as e:
            logger.error(f"计算WER/CER失败: {e}")
            return 1.0, 1.0
    
    def process_audio_pair(self, original_path: str, enhanced_path: str, 
                          output_dir: str) -> Dict:
        """处理单个音频对"""
        
        result = {
            'original_path': original_path,
            'enhanced_path': enhanced_path,
            'processed_enhanced_path': '',
            'original_transcription': '',
            'enhanced_transcription': '',
            'wer': 0.0,
            'cer': 0.0,
            'volume_scale_factor': 1.0,
            'volume_matching_method': self.config['volume_matching_method'],
            'processing_time': 0.0,
            'success': False,
            'error_message': '',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            import time
            start_time = time.time()
            
            # 检查输入文件
            if not os.path.exists(original_path):
                raise FileNotFoundError(f"原始音频文件不存在: {original_path}")
            if not os.path.exists(enhanced_path):
                raise FileNotFoundError(f"增强音频文件不存在: {enhanced_path}")
            
            # 预处理音频
            logger.info(f"GPU {self.gpu_id}: 预处理音频: {original_path}")
            original_audio, sr = self.preprocess_audio(original_path)
            logger.info(f"GPU {self.gpu_id}: 预处理音频: {enhanced_path}")
            enhanced_audio, sr = self.preprocess_audio(enhanced_path)
            
            # 计算gain并应用到增强音频
            logger.info(f"GPU {self.gpu_id}: 计算并应用gain...")
            if self.config['volume_matching_method'] == 'vad_energy':
                gain = self.volume_processor.calculate_gain_vad_energy(original_audio, enhanced_audio)
                matched_audio, actual_gain = self.volume_processor.apply_gain(enhanced_audio, gain)
                result['volume_scale_factor'] = actual_gain
            else:
                # 其他方法暂时直接返回原音频
                matched_audio = enhanced_audio
                result['volume_scale_factor'] = 1.0
            
            # 保存处理后的音频
            matched_filename = f"{Path(enhanced_path).stem}_gain_applied.wav"
            matched_path = os.path.join(output_dir, matched_filename)
            
            if self.save_processed_audio(matched_audio, matched_path, sr):
                result['processed_enhanced_path'] = matched_path
            
            # 创建临时文件进行识别
            temp_original = os.path.join(output_dir, f"temp_original_{Path(original_path).stem}.wav")
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
                    wer_score, cer_score = self.calculate_wer_cer(
                        original_transcription, enhanced_transcription
                    )
                    result['wer'] = wer_score
                    result['cer'] = cer_score
                    
                    logger.info(f"GPU {self.gpu_id}: WER: {wer_score:.4f}, CER: {cer_score:.4f}")
                else:
                    logger.warning(f"GPU {self.gpu_id}: 转录结果为空，无法计算WER/CER")
                
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
        subset_output_dir = os.path.join(self.config['output_dir'], f"subset_{subset_id}")
        os.makedirs(subset_output_dir, exist_ok=True)
        
        # 处理音频对
        results = []
        
        for i, (original_path, enhanced_path) in enumerate(tqdm(audio_pairs, desc=f"GPU {self.gpu_id}: 处理子集{subset_id}")):
            
            logger.info(f"GPU {self.gpu_id}: 处理第{i+1}/{len(audio_pairs)}对音频")
            
            # 检查是否跳过已存在的结果
            if self.config['skip_existing']:
                result_file = os.path.join(
                    subset_output_dir,
                    f"{Path(enhanced_path).stem}_assessment.json"
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
                original_path, enhanced_path, subset_output_dir
            )
            results.append(result)
            
            # 保存单个结果
            result_file = os.path.join(
                subset_output_dir,
                f"{Path(enhanced_path).stem}_assessment.json"
            )
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        # 保存子集结果
        subset_results_file = os.path.join(subset_output_dir, f"subset_{subset_id}_results.json")
        with open(subset_results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"GPU {self.gpu_id}: 子集 {subset_id} 处理完成，结果保存在: {subset_output_dir}")
        
        return results

    def get_audio_pairs(self) -> List[Tuple[str, str]]:
        """获取音频对列表"""
        audio_pairs = []
        
        # 遍历原始音频目录
        for root, dirs, files in os.walk(self.config['original_dir']):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    original_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(original_path, self.config['original_dir'])
                    
                    # 构造增强音频路径
                    enhanced_path = os.path.join(
                        self.config['enhanced_dir'],
                        os.path.splitext(rel_path)[0] + '.wav'
                    )
                    
                    # 检查增强音频是否存在
                    if os.path.exists(enhanced_path):
                        audio_pairs.append((original_path, enhanced_path))
                    else:
                        logger.warning(f"增强音频不存在: {enhanced_path}")
        
        return audio_pairs
    


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
    gpu_id, audio_pairs_subset, config, subset_id = args_tuple
    
    try:
        # 设置进程的GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 确保在子进程中重新初始化CUDA
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)  # 因为CUDA_VISIBLE_DEVICES已经设置，所以这里用0
            
        # 创建处理器
        processor = QualityAssessmentProcessor(config, gpu_id)
        
        # 处理子集
        results = processor.run_assessment_subset(audio_pairs_subset, subset_id)
        
        return results
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 处理子集失败: {e}")
        import traceback
        traceback.print_exc()
        return []

def merge_results(output_dir: str, num_gpus: int, config: Dict):
    """合并所有GPU的结果"""
    logger.info("开始合并结果...")
    
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
    if successful_results:
        wer_scores = [r['wer'] for r in successful_results]
        cer_scores = [r['cer'] for r in successful_results]
        
        summary = {
            'total_pairs': len(all_results),
            'successful_pairs': len(successful_results),
            'failed_pairs': len(all_results) - len(successful_results),
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
            'config': config,
            'timestamp': datetime.now().isoformat()
        }
    else:
        summary = {
            'total_pairs': len(all_results),
            'successful_pairs': 0,
            'failed_pairs': len(all_results),
            'average_wer': 0.0,
            'average_cer': 0.0,
            'config': config,
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
    print_summary(all_results)

def print_summary(results: List[Dict]):
    """打印统计摘要"""
    successful_results = [r for r in results if r['success']]
    
    print("\n" + "=" * 80)
    print("音频质量评估结果摘要 (多GPU并行)")
    print("=" * 80)
    print(f"总音频对数:     {len(results)}")
    print(f"成功处理:       {len(successful_results)}")
    print(f"失败处理:       {len(results) - len(successful_results)}")
    
    if successful_results:
        wer_scores = [r['wer'] for r in successful_results]
        cer_scores = [r['cer'] for r in successful_results]
        
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
        
        # 增益应用方法统计
        volume_methods = [r['volume_matching_method'] for r in successful_results]
        print(f"\n增益应用方法: {volume_methods[0] if volume_methods else 'N/A'}")
        
        # 处理时间统计
        processing_times = [r['processing_time'] for r in successful_results]
        print(f"\n处理时间统计:")
        print(f"  平均处理时间:   {np.mean(processing_times):.2f} 秒")
        print(f"  总处理时间:     {np.sum(processing_times):.2f} 秒")
    
    print("=" * 80)

def get_audio_pairs(config: Dict) -> List[Tuple[str, str]]:
    """获取音频对列表"""
    audio_pairs = []
    
    # 遍历原始音频目录
    for root, dirs, files in os.walk(config['original_dir']):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                original_path = os.path.join(root, file)
                
                # 计算相对路径
                rel_path = os.path.relpath(original_path, config['original_dir'])
                
                # 构造增强音频路径
                enhanced_path = os.path.join(
                    config['enhanced_dir'],
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
    
    parser = argparse.ArgumentParser(description="音频质量评估脚本 (多GPU并行)")
    parser.add_argument("--original_dir", type=str, 
                       default=CONFIG['original_dir'],
                       help="原始音频目录")
    parser.add_argument("--enhanced_dir", type=str,
                       default=CONFIG['enhanced_dir'],
                       help="增强音频目录")
    parser.add_argument("--output_dir", type=str,
                       default=CONFIG['output_dir'],
                       help="输出目录")
    parser.add_argument("--kimi_model_path", type=str,
                       default=CONFIG['kimi_model_path'],
                       help="Kimi-Audio模型路径")
    parser.add_argument("--kimi_audio_dir", type=str,
                       default=CONFIG['kimi_audio_dir'],
                       help="Kimi-Audio代码目录")
    parser.add_argument("--volume_method", type=str,
                       default=CONFIG['volume_matching_method'],
                       choices=['vad_energy'],
                       help="音量匹配方法")
    parser.add_argument("--skip_existing", action="store_true",
                       default=CONFIG['skip_existing'],
                       help="跳过已存在的结果")
    parser.add_argument("--max_audio_length", type=int,
                       default=CONFIG['max_audio_length'],
                       help="最大音频长度（秒）")
    parser.add_argument("--vad_aggressiveness", type=int,
                       default=CONFIG['vad_aggressiveness'],
                       choices=[0, 1, 2, 3],
                       help="VAD积极性等级 (0-3)")
    parser.add_argument("--num_gpus", type=int,
                       default=CONFIG['num_gpus'],
                       help="GPU数量")
    parser.add_argument("--gpu_ids", type=int, nargs='+',
                       default=CONFIG['gpu_ids'],
                       help="GPU ID列表")
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG.update(vars(args))
    
    # 确保GPU数量和ID列表一致
    if len(CONFIG['gpu_ids']) != CONFIG['num_gpus']:
        CONFIG['gpu_ids'] = list(range(CONFIG['num_gpus']))
    
    print("音频质量评估脚本 (多GPU并行版本 - 已修复CUDA兼容性)")
    print("=" * 70)
    print(f"原始音频目录: {CONFIG['original_dir']}")
    print(f"增强音频目录: {CONFIG['enhanced_dir']}")
    print(f"输出目录: {CONFIG['output_dir']}")
    print(f"Kimi-Audio模型: {CONFIG['kimi_model_path']}")
    print(f"Kimi-Audio目录: {CONFIG['kimi_audio_dir']}")
    print(f"GPU数量: {CONFIG['num_gpus']}")
    print(f"GPU ID列表: {CONFIG['gpu_ids']}")
    print(f"音量匹配方法: {CONFIG['volume_matching_method']}")
    print(f"VAD积极性等级: {CONFIG['vad_aggressiveness']}")
    print(f"跳过已存在: {CONFIG['skip_existing']}")
    print(f"最大音频长度: {CONFIG['max_audio_length']} 秒")
    print("已启用: spawn多进程模式 (CUDA兼容)")
    print("=" * 70)
    
    # 检查输入目录
    if not os.path.exists(CONFIG['original_dir']):
        logger.error(f"原始音频目录不存在: {CONFIG['original_dir']}")
        return
    
    if not os.path.exists(CONFIG['enhanced_dir']):
        logger.error(f"增强音频目录不存在: {CONFIG['enhanced_dir']}")
        return
    
    # 检查Kimi-Audio模型
    if not os.path.exists(CONFIG['kimi_model_path']):
        logger.error(f"Kimi-Audio模型路径不存在: {CONFIG['kimi_model_path']}")
        logger.error("请先下载Kimi-Audio模型：")
        logger.error("huggingface-cli download moonshotai/Kimi-Audio-7B-Instruct --local-dir /path/to/model")
        return
    
    # 检查Kimi-Audio代码目录
    if not os.path.exists(CONFIG['kimi_audio_dir']):
        logger.error(f"Kimi-Audio代码目录不存在: {CONFIG['kimi_audio_dir']}")
        logger.error("请先克隆Kimi-Audio代码仓库：")
        logger.error("git clone https://github.com/moonshotai/Kimi-Audio.git")
        return
    
    # 检查GPU可用性
    if not torch.cuda.is_available():
        logger.error("CUDA不可用，无法使用GPU")
        return
    
    available_gpus = torch.cuda.device_count()
    if available_gpus < CONFIG['num_gpus']:
        logger.error(f"可用GPU数量({available_gpus})小于所需数量({CONFIG['num_gpus']})")
        return
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 获取音频对列表
    logger.info("获取音频对列表...")
    audio_pairs = get_audio_pairs(CONFIG)
    
    if not audio_pairs:
        logger.error("未找到音频对")
        return
    
    logger.info(f"找到 {len(audio_pairs)} 个音频对")
    
    # 分割音频对
    logger.info(f"将音频对分成 {CONFIG['num_gpus']} 个子集...")
    audio_pairs_splits = split_audio_pairs(audio_pairs, CONFIG['num_gpus'])
    
    for i, split in enumerate(audio_pairs_splits):
        logger.info(f"子集 {i}: {len(split)} 个音频对")
    
    # 准备多进程参数
    process_args = []
    for i, (gpu_id, audio_pairs_subset) in enumerate(zip(CONFIG['gpu_ids'], audio_pairs_splits)):
        process_args.append((gpu_id, audio_pairs_subset, CONFIG, i))
    
    # 启动多进程处理
    try:
        logger.info("启动多GPU并行处理...")
        start_time = time.time()
        
        # 使用进程池并行处理，使用spawn方法
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
        logger.info(f"多GPU并行处理完成，耗时: {processing_time:.2f} 秒")
        
        # 合并结果
        merge_results(CONFIG['output_dir'], CONFIG['num_gpus'], CONFIG)
        
        print(f"\n评估完成！总耗时: {processing_time:.2f} 秒")
        print(f"结果保存在: {CONFIG['output_dir']}")
        
    except Exception as e:
        logger.error(f"多GPU并行处理失败: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main() 