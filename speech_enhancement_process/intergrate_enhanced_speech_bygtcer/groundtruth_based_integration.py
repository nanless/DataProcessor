#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Groundtruth的音频集成脚本

根据降噪前后音频与groundtruth文本的CER差异来判断降噪是否有效：
- 默认选择CER最低的降噪版本
- 只有当降噪版本的CER明显高于原音频（退化量>阈值）时才回退到原音频

支持的数据集：
- BAAI-ChildMandarin41.25H
- Chinese_English_Scripted_Speech_Corpus_Children  
- King-ASR-EN-Kid
- speechocean762
"""

import os
import sys
import json
import shutil
import logging
import argparse
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm
import torch
from jiwer import wer, cer
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import requests
import time
import librosa

# 导入本地音频处理模块
from audio_processors import KimiAudioProcessor, HTTPTextNormalizer, AudioVolumeProcessor

# 设置代理绕过
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "config.json"):
        self.config_file = Path(config_file)
        self.config = self.load_config()
    
    def load_config(self) -> Dict:
        """加载配置文件"""
        try:
            if not self.config_file.exists():
                logger.error(f"配置文件不存在: {self.config_file}")
                raise FileNotFoundError(f"配置文件不存在: {self.config_file}")
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            raise
    
    def get_global_config(self) -> Dict:
        """获取全局配置"""
        return self.config.get('global_config', {})
    
    def get_datasets(self) -> List[Dict]:
        """获取数据集配置"""
        return self.config.get('datasets', [])
    
    def get_dataset_by_name(self, name: str) -> Optional[Dict]:
        """根据名称获取数据集配置"""
        for dataset in self.get_datasets():
            if dataset.get('name') == name:
                return dataset
        return None
    
    def update_global_config(self, updates: Dict):
        """更新全局配置"""
        self.config['global_config'].update(updates)

class GroundtruthReader:
    """Groundtruth文本读取器"""
    
    def __init__(self, groundtruth_file: str):
        self.groundtruth_file = Path(groundtruth_file)
        self.groundtruth_dict = {}
        self.load_groundtruth()
    
    def load_groundtruth(self):
        """加载groundtruth文本"""
        try:
            logger.info(f"加载groundtruth文件: {self.groundtruth_file}")
            
            with open(self.groundtruth_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t', 1)  # 使用tab分割，更准确
                    if len(parts) == 2:
                        utterance_id = parts[0]
                        transcript = parts[1]
                        self.groundtruth_dict[utterance_id] = transcript
                    else:
                        # 如果tab分割失败，尝试空格分割（兼容性）
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            utterance_id = parts[0]
                            transcript = parts[1]
                            self.groundtruth_dict[utterance_id] = transcript
                        else:
                            logger.warning(f"跳过无效的groundtruth行: {line.strip()}")
            
        except FileNotFoundError:
            logger.error(f"Groundtruth文件不存在: {self.groundtruth_file}")
            raise
        except Exception as e:
            logger.error(f"读取groundtruth文件失败: {e}，但继续处理")
            
        logger.info(f"加载了 {len(self.groundtruth_dict)} 条groundtruth记录")
    
    def get_transcript(self, utterance_id: str) -> Optional[str]:
        """获取指定utterance的转录文本"""
        return self.groundtruth_dict.get(utterance_id)

class WavScpReader:
    """wav.scp文件读取器"""
    
    def __init__(self, wav_scp_file: str):
        self.wav_scp_file = Path(wav_scp_file)
        self.audio_path_dict = {}  # utterance_id -> audio_path
        self.reverse_path_dict = {}  # audio_path -> utterance_id
        self.filename_dict = {}  # filename -> utterance_id (for fuzzy matching)
        self.load_wav_scp()
    
    def load_wav_scp(self):
        """加载wav.scp文件"""
        try:
            logger.info(f"加载wav.scp文件: {self.wav_scp_file}")
            
            if not self.wav_scp_file.exists():
                logger.warning(f"wav.scp文件不存在: {self.wav_scp_file}")
                return
            
            with open(self.wav_scp_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        utterance_id = parts[0]
                        audio_path = ' '.join(parts[1:])  # 处理路径中可能包含空格的情况
                        
                        # 转换为绝对路径
                        if not os.path.isabs(audio_path):
                            # 如果是相对路径，相对于wav.scp文件所在目录
                            audio_path = str(self.wav_scp_file.parent / audio_path)
                        
                        self.audio_path_dict[utterance_id] = audio_path
                        
                        # 构建反向映射
                        abs_audio_path = os.path.abspath(audio_path)
                        self.reverse_path_dict[abs_audio_path] = utterance_id
                        
                        # 构建文件名映射（用于宽松匹配）
                        filename = Path(audio_path).name
                        if filename not in self.filename_dict:
                            self.filename_dict[filename] = utterance_id
                        else:
                            # 如果文件名重复，记录警告
                            logger.warning(f"重复的文件名 {filename}: {self.filename_dict[filename]} vs {utterance_id}")
                    else:
                        logger.warning(f"跳过无效的wav.scp行 {line_num}: {line}")
            
        except Exception as e:
            logger.warning(f"读取wav.scp文件失败: {e}，将使用目录遍历方式")
            
        logger.info(f"从wav.scp文件加载了 {len(self.audio_path_dict)} 条音频路径记录")
    
    def get_audio_path(self, utterance_id: str) -> Optional[str]:
        """获取指定utterance的音频文件路径"""
        return self.audio_path_dict.get(utterance_id)
    
    def get_all_utterance_ids(self) -> List[str]:
        """获取所有utterance ID"""
        return list(self.audio_path_dict.keys())
    
    def get_utterance_id_by_path(self, audio_path: str) -> Optional[str]:
        """根据音频路径获取utterance ID"""
        # 首先尝试精确匹配（绝对路径）
        abs_path = os.path.abspath(audio_path)
        if abs_path in self.reverse_path_dict:
            return self.reverse_path_dict[abs_path]
        
        # 然后尝试文件名匹配
        filename = Path(audio_path).name
        if filename in self.filename_dict:
            return self.filename_dict[filename]
        
        return None

class GroundtruthBasedProcessor:
    """基于Groundtruth的音频处理器"""
    
    def __init__(self, global_config: Dict, dataset_config: Dict, gpu_id: int = 0, llm_service_url: str = None):
        self.global_config = global_config
        self.dataset_config = dataset_config
        self.gpu_id = gpu_id
        self.device = f"cuda:{gpu_id}"
        
        # 初始化组件
        ten_vad_config = global_config.get('ten_vad_config', {})
        self.volume_processor = AudioVolumeProcessor(
            sr=global_config['target_sr'],
            ten_vad_config=ten_vad_config
        )
        
        self.kimi_processor = KimiAudioProcessor(
            model_path=global_config['kimi_model_path'],
            device=self.device,
            kimi_audio_dir=global_config['kimi_audio_dir'],
            gpu_id=gpu_id
        )
        
        # 初始化文本标准化器
        service_url = llm_service_url or global_config['llm_service_url']
        try:
            self.http_normalizer = HTTPTextNormalizer(
                service_url=service_url,
                timeout=global_config['llm_timeout'],
                max_retries=global_config['llm_max_retries'],
                gpu_id=gpu_id
            )
        except Exception as e:
            logger.error(f"GPU {gpu_id}: 初始化文本标准化器失败: {e}")
            raise
        
        # 初始化groundtruth读取器
        self.groundtruth_reader = GroundtruthReader(dataset_config['groundtruth_file'])
        
        # 初始化wav.scp读取器（如果存在）
        wav_scp_file = dataset_config.get('wav_scp_file')
        if wav_scp_file:
            self.wav_scp_reader = WavScpReader(wav_scp_file)
        else:
            self.wav_scp_reader = None
            logger.info("未配置wav.scp文件，将使用目录遍历方式")
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'original_selected': 0,
            'enhanced_selected': 0,
            'no_groundtruth': 0,
            'enhancement_methods': {},
            'cer_improvements': []
        }
    
    def preprocess_audio(self, audio_path: str, target_sr: int = None) -> Tuple[np.ndarray, int]:
        """预处理音频"""
        if target_sr is None:
            target_sr = self.global_config['target_sr']
        
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
            max_samples = int(self.global_config['max_audio_length'] * target_sr)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                logger.warning(f"音频长度超过限制，截断至{self.global_config['max_audio_length']}秒")
            
            return audio, target_sr
            
        except Exception as e:
            logger.error(f"音频预处理失败 {audio_path}: {e}")
            raise
    
    def extract_utterance_id(self, audio_path: str) -> str:
        """从音频路径提取utterance ID"""
        # 如果有wav.scp文件，则根据音频路径反向查找utterance ID
        if self.wav_scp_reader:
            utterance_id = self.wav_scp_reader.get_utterance_id_by_path(audio_path)
            if utterance_id:
                return utterance_id
            
            # 如果仍然没找到，记录警告但继续使用文件名
            logger.warning(f"GPU {self.gpu_id}: 在wav.scp中未找到音频路径 {audio_path}，使用文件名作为utterance ID")
        
        # 提取文件名（不含扩展名）作为utterance ID (fallback)
        return Path(audio_path).stem
    
    def find_enhanced_audio(self, original_path: str, enhancement_method: str) -> Optional[str]:
        """查找对应的降噪音频文件"""
        # 支持新的配置格式（字典）和旧的配置格式（字符串）
        method_name = enhancement_method if isinstance(enhancement_method, str) else enhancement_method.get('name', enhancement_method)
        
        base_dir = Path(self.dataset_config['base_dir'])
        original_rel_path = Path(original_path).relative_to(base_dir)
        
        # 构建降噪音频路径
        enhanced_dir = base_dir.parent / f"{base_dir.name}_{method_name}_enhanced"
        
        # 尝试查找.wav文件
        enhanced_path_wav = enhanced_dir / original_rel_path.with_suffix('.wav')
        if enhanced_path_wav.exists():
            return str(enhanced_path_wav)
        
        # 尝试查找.WAV文件（大写扩展名）
        enhanced_path_WAV = enhanced_dir / original_rel_path.with_suffix('.WAV')
        if enhanced_path_WAV.exists():
            return str(enhanced_path_WAV)
        
        return None
        
    def get_enhancement_method_config(self, enhancement_method) -> Dict:
        """获取增强方法的配置信息"""
        if isinstance(enhancement_method, dict):
            return enhancement_method
        else:
            # 向后兼容：如果是字符串，创建默认配置
            return {
                'name': enhancement_method,
                'use_ten_vad_volume_matching': True,
                'description': f'{enhancement_method}增强器（默认配置）'
            }
    
    def normalize_text_for_language(self, text1: str, text2: str, language: str) -> Tuple[str, str]:
        """根据语言类型标准化文本"""
        try:
            if self.http_normalizer:
                return self.http_normalizer.normalize_text_pair(text1, text2)
            else:
                # 简单的标准化处理
                if language == 'en':
                    # 英语：转小写，去标点
                    import re
                    def simple_normalize_en(text):
                        text = text.lower()
                        text = re.sub(r'[^\w\s]', '', text)
                        return text.strip()
                    return simple_normalize_en(text1), simple_normalize_en(text2)
                elif language == 'zh':
                    # 中文：去标点，去空格
                    import re
                    def simple_normalize_zh(text):
                        text = re.sub(r'[^\u4e00-\u9fff\w]', '', text)
                        return text.strip()
                    return simple_normalize_zh(text1), simple_normalize_zh(text2)
                else:
                    # mixed语言：使用LLM必须成功
                    raise RuntimeError("混合语言必须使用LLM标准化")
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 文本标准化失败: {e}")
            raise
    
    def evaluate_audio_vs_groundtruth(self, audio_path: str, groundtruth_text: str, language: str) -> Dict:
        """评估音频识别结果与groundtruth的差异"""
        try:
            # 语音识别
            transcription = self.kimi_processor.transcribe_audio(audio_path)
            
            if not transcription:
                return {
                    'transcription': '',
                    'normalized_transcription': '',
                    'normalized_groundtruth': '',
                    'cer': 1.0,
                    'wer': 1.0,
                    'success': False,
                    'error': 'ASR识别失败'
                }
            
            # 文本标准化
            normalized_transcription, normalized_groundtruth = self.normalize_text_for_language(
                transcription, groundtruth_text, language
            )
            
            # 计算CER和WER
            cer_score = cer(normalized_groundtruth, normalized_transcription)
            wer_score = wer(normalized_groundtruth, normalized_transcription)
            
            return {
                'transcription': transcription,
                'normalized_transcription': normalized_transcription,
                'normalized_groundtruth': normalized_groundtruth,
                'cer': cer_score,
                'wer': wer_score,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 音频评估失败 {audio_path}: {e}")
            return {
                'transcription': '',
                'normalized_transcription': '',
                'normalized_groundtruth': '',
                'cer': 1.0,
                'wer': 1.0,
                'success': False,
                'error': str(e)
            }
    
    def evaluate_audio_vs_groundtruth_with_id(self, audio_path: str, groundtruth_text: str, language: str, utterance_id: str) -> Dict:
        """评估音频识别结果与groundtruth的差异（使用预定义的utterance ID）"""
        try:
            # 语音识别
            transcription = self.kimi_processor.transcribe_audio(audio_path)
            
            if not transcription:
                return {
                    'transcription': '',
                    'normalized_transcription': '',
                    'normalized_groundtruth': '',
                    'cer': 1.0,
                    'wer': 1.0,
                    'success': False,
                    'error': 'ASR识别失败',
                    'utterance_id': utterance_id
                }
            
            # 文本标准化
            normalized_transcription, normalized_groundtruth = self.normalize_text_for_language(
                transcription, groundtruth_text, language
            )
            
            # 计算CER和WER
            cer_score = cer(normalized_groundtruth, normalized_transcription)
            wer_score = wer(normalized_groundtruth, normalized_transcription)
            
            return {
                'transcription': transcription,
                'normalized_transcription': normalized_transcription,
                'normalized_groundtruth': normalized_groundtruth,
                'cer': cer_score,
                'wer': wer_score,
                'success': True,
                'error': None,
                'utterance_id': utterance_id
            }
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 音频评估失败 {audio_path} (utterance: {utterance_id}): {e}")
            return {
                'transcription': '',
                'normalized_transcription': '',
                'normalized_groundtruth': '',
                'cer': 1.0,
                'wer': 1.0,
                'success': False,
                'error': str(e),
                'utterance_id': utterance_id
            }
    
    def process_audio_file(self, original_path: str, output_dir: str) -> Dict:
        """处理单个音频文件"""
        result = {
            'original_path': original_path,
            'utterance_id': '',
            'groundtruth_text': '',
            'language': self.dataset_config['language'],
            'selected_method': 'original',
            'selected_path': original_path,
            'original_evaluation': {},
            'enhanced_evaluations': {},
            'decision_info': {},
            'copy_success': False,
            'processing_time': 0.0,
            'success': False,
            'error_message': '',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            start_time = time.time()
            
            # 提取utterance ID - 这里original_path一定是来自wav.scp的原始路径
            utterance_id = self.extract_utterance_id(original_path)
            result['utterance_id'] = utterance_id
            
            # 获取groundtruth文本
            groundtruth_text = self.groundtruth_reader.get_transcript(utterance_id)
            if not groundtruth_text:
                result['error_message'] = f'未找到utterance {utterance_id} 的groundtruth文本'
                self.stats['no_groundtruth'] += 1
                logger.warning(f"GPU {self.gpu_id}: {result['error_message']}")
                return result
            
            result['groundtruth_text'] = groundtruth_text
            
            # 评估原音频
            logger.info(f"GPU {self.gpu_id}: 评估原音频: {original_path}")
            original_eval = self.evaluate_audio_vs_groundtruth(
                original_path, groundtruth_text, self.dataset_config['language']
            )
            result['original_evaluation'] = original_eval
            
            if not original_eval['success']:
                result['error_message'] = f"原音频评估失败: {original_eval['error']}"
                return result
            
            # 评估所有可用的降噪版本
            best_enhanced_method = None
            best_enhanced_cer = float('inf')
            best_enhanced_eval = None
            best_enhanced_config = None
            
            # 定义增强方法优先级（在CER相等时使用）
            method_priority = {
                'mtfaa': 0,           # 最高优先级
                'mossformer': 1,      # 第二优先级
                'zipenhancer': 2,     # 第三优先级（已弃用）
                'resemble': 3,        # 第四优先级
            }
            
            for method in self.dataset_config['enhancement_methods']:
                method_config = self.get_enhancement_method_config(method)
                method_name = method_config['name']
                
                enhanced_path = self.find_enhanced_audio(original_path, method)
                if not enhanced_path:
                    logger.debug(f"GPU {self.gpu_id}: 未找到降噪音频: {method_name}")
                    continue
                
                logger.info(f"GPU {self.gpu_id}: 评估降噪音频: {enhanced_path}")
                # 评估增强音频时，使用原始音频对应的utterance_id，而不是从增强音频路径提取
                enhanced_eval = self.evaluate_audio_vs_groundtruth_with_id(
                    enhanced_path, groundtruth_text, self.dataset_config['language'], utterance_id
                )
                
                # 添加方法配置信息到评估结果中
                enhanced_eval['method_config'] = method_config
                result['enhanced_evaluations'][method_name] = enhanced_eval
                
                if enhanced_eval['success']:
                    current_cer = enhanced_eval['cer']
                    current_priority = method_priority.get(method_name, 999)  # 未知方法给最低优先级
                    best_priority = method_priority.get(best_enhanced_method, 999) if best_enhanced_method else 999
                    
                    # 选择逻辑：
                    # 1. CER更低时直接选择
                    # 2. CER相等时选择优先级更高的方法（数字越小优先级越高）
                    should_select = (
                        current_cer < best_enhanced_cer or  # CER更低
                        (abs(current_cer - best_enhanced_cer) < 1e-8 and current_priority < best_priority)  # CER相等但优先级更高
                    )
                    
                    if should_select:
                        best_enhanced_cer = current_cer
                        best_enhanced_method = method_name
                        best_enhanced_eval = enhanced_eval
                        best_enhanced_config = method_config
                        
                        # 记录选择原因
                        if abs(current_cer - enhanced_eval.get('previous_best_cer', float('inf'))) < 1e-6:
                            logger.info(f"GPU {self.gpu_id}: CER相等({current_cer:.6f})，优先选择 {method_name}")
                        else:
                            logger.info(f"GPU {self.gpu_id}: 选择CER更低的方法 {method_name}: {current_cer:.6f}")
            
            # 决策逻辑：默认使用降噪版本，只有当降噪版本明显更差时才用原音频
            original_cer = original_eval['cer']
            max_degradation = self.global_config['max_degradation']
            
            if best_enhanced_method and best_enhanced_eval:
                cer_degradation = best_enhanced_cer - original_cer  # 正值表示降噪版本更差
                
                # 默认使用降噪版本，只有当降噪版本明显更差时才回退到原音频
                if cer_degradation > max_degradation:
                    # 降噪版本明显更差，回退到原音频
                    result['decision_info'] = {
                        'reason': 'enhanced_degraded_significantly',
                        'original_cer': original_cer,
                        'enhanced_cer': best_enhanced_cer,
                        'cer_degradation': cer_degradation,
                        'max_degradation_threshold': max_degradation,
                        'fallback_method': 'original'
                    }
                    
                    self.stats['original_selected'] += 1
                    
                else:
                    # 使用最佳降噪版本（默认选择）
                    result['selected_method'] = best_enhanced_method
                    result['selected_path'] = self.find_enhanced_audio(original_path, {'name': best_enhanced_method})
                    
                    # 检查是否有CER相等的其他方法
                    equal_cer_methods = []
                    for method_name, eval_result in result['enhanced_evaluations'].items():
                        if eval_result['success'] and abs(eval_result['cer'] - best_enhanced_cer) < 1e-8:
                            equal_cer_methods.append(method_name)
                    
                    # 构建决策信息
                    decision_reason = 'enhanced_selected_as_default'
                    additional_info = {}
                    
                    if len(equal_cer_methods) > 1:
                        decision_reason = 'enhanced_selected_by_priority'
                        additional_info = {
                            'equal_cer_methods': equal_cer_methods,
                            'priority_selected': best_enhanced_method,
                            'selection_note': f'CER相等时优先选择{best_enhanced_method}'
                        }
                    
                    result['decision_info'] = {
                        'reason': decision_reason,
                        'original_cer': original_cer,
                        'enhanced_cer': best_enhanced_cer,
                        'cer_change': cer_degradation,  # 负值表示改善，正值表示轻微退化但可接受
                        'method': best_enhanced_method,
                        'method_config': best_enhanced_config,
                        **additional_info
                    }
                    
                    self.stats['enhanced_selected'] += 1
                    if best_enhanced_method not in self.stats['enhancement_methods']:
                        self.stats['enhancement_methods'][best_enhanced_method] = 0
                    self.stats['enhancement_methods'][best_enhanced_method] += 1
                    
                    # 记录CER变化（改善为负值，退化为正值）
                    self.stats['cer_improvements'].append(-cer_degradation)
                    
            else:
                # 没有可用的降噪版本，使用原音频
                result['decision_info'] = {
                    'reason': 'no_enhanced_available',
                    'original_cer': original_cer
                }
                
                self.stats['original_selected'] += 1
            
            # 拷贝选择的音频到输出目录
            relative_path = Path(original_path).relative_to(self.dataset_config['base_dir'])
            target_path = Path(output_dir) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if result['selected_path'] != original_path:
                    # 选择了增强版本，检查是否需要音量匹配
                    use_volume_matching = True
                    if best_enhanced_config:
                        use_volume_matching = best_enhanced_config.get('use_ten_vad_volume_matching', True)
                    
                    if use_volume_matching:
                        # 使用TEN-VAD音量匹配
                        original_audio, sr = self.preprocess_audio(original_path)
                        enhanced_audio, sr = self.preprocess_audio(result['selected_path'])
                        
                        # 计算并应用gain
                        gain = self.volume_processor.calculate_gain_vad_energy(original_audio, enhanced_audio)
                        matched_audio, actual_gain = self.volume_processor.apply_gain(enhanced_audio, gain)
                        
                        # 保存匹配后的音频
                        sf.write(target_path, matched_audio, sr)
                        result['volume_gain_applied'] = actual_gain
                        result['volume_matching_method'] = 'TEN-VAD'
                        logger.info(f"GPU {self.gpu_id}: 应用TEN-VAD音量匹配，gain={actual_gain:.4f}")
                    else:
                        # 直接拷贝增强音频，不做音量匹配
                        shutil.copy2(result['selected_path'], target_path)
                        result['volume_gain_applied'] = 1.0
                        result['volume_matching_method'] = 'None'
                        logger.info(f"GPU {self.gpu_id}: 直接拷贝增强音频，无音量匹配")
                else:
                    # 直接拷贝原音频
                    shutil.copy2(result['selected_path'], target_path)
                    result['volume_gain_applied'] = 1.0
                    result['volume_matching_method'] = 'None'
                
                result['copy_success'] = True
                result['output_audio_path'] = str(target_path)
                logger.info(f"GPU {self.gpu_id}: 文件拷贝成功: {target_path}")
                
                # 生成对应的JSON文件
                if self.global_config.get('output_individual_json', True):
                    self.save_individual_json(result, target_path)
                
            except Exception as e:
                logger.error(f"GPU {self.gpu_id}: 文件拷贝失败: {e}")
                result['copy_success'] = False
                result['error_message'] = f"文件拷贝失败: {str(e)}"
            
            result['processing_time'] = time.time() - start_time
            result['success'] = True
            
        except Exception as e:
            result['error_message'] = str(e)
            logger.error(f"GPU {self.gpu_id}: 处理音频文件失败: {e}")
        
        return result
    
    def save_individual_json(self, result: Dict, audio_path: Path):
        """为每个音频文件保存对应的JSON详细信息文件"""
        try:
            # 生成JSON文件路径（与音频文件同名，但扩展名为.json）
            json_path = audio_path.with_suffix('.json')
            
            # 创建简化的结果信息，专门用于单个文件的JSON
            individual_result = {
                'audio_file_info': {
                    'utterance_id': result['utterance_id'],
                    'original_path': result['original_path'],
                    'selected_method': result['selected_method'],
                    'selected_path': result['selected_path'],
                    'output_audio_path': result.get('output_audio_path', str(audio_path)),
                    'language': result['language'],
                    'volume_matching_method': result.get('volume_matching_method', 'Unknown'),
                    'volume_gain_applied': result.get('volume_gain_applied', 1.0)
                },
                
                'groundtruth_info': {
                    'groundtruth_text': result['groundtruth_text']
                },
                
                'asr_results': {
                    'original_evaluation': {
                        'transcription': result['original_evaluation'].get('transcription', ''),
                        'normalized_transcription': result['original_evaluation'].get('normalized_transcription', ''),
                        'normalized_groundtruth': result['original_evaluation'].get('normalized_groundtruth', ''),
                        'cer': result['original_evaluation'].get('cer', 1.0),
                        'wer': result['original_evaluation'].get('wer', 1.0),
                        'success': result['original_evaluation'].get('success', False)
                    },
                    'enhanced_evaluations': {}
                },
                
                'decision_analysis': {
                    'decision_info': result.get('decision_info', {}),
                    'selected_reason': result.get('decision_info', {}).get('reason', 'unknown'),
                    'cer_comparison': {
                        'original_cer': result.get('decision_info', {}).get('original_cer', 1.0),
                        'selected_cer': result.get('decision_info', {}).get('enhanced_cer', 
                                                  result.get('decision_info', {}).get('original_cer', 1.0)),
                        'cer_improvement': result.get('decision_info', {}).get('cer_change', 0.0)
                    }
                },
                
                'processing_metadata': {
                    'processing_time': result.get('processing_time', 0.0),
                    'timestamp': result.get('timestamp', ''),
                    'gpu_id': self.gpu_id,
                    'success': result.get('success', False),
                    'copy_success': result.get('copy_success', False),
                    'error_message': result.get('error_message', '')
                }
            }
            
            # 添加所有增强方法的评估结果
            for method_name, enhanced_eval in result.get('enhanced_evaluations', {}).items():
                individual_result['asr_results']['enhanced_evaluations'][method_name] = {
                    'transcription': enhanced_eval.get('transcription', ''),
                    'normalized_transcription': enhanced_eval.get('normalized_transcription', ''),
                    'cer': enhanced_eval.get('cer', 1.0),
                    'wer': enhanced_eval.get('wer', 1.0),
                    'success': enhanced_eval.get('success', False),
                    'method_config': enhanced_eval.get('method_config', {}),
                    'enhancement_applied': method_name == result['selected_method']
                }
            
            # 添加配置信息
            individual_result['config_info'] = {
                'dataset_name': self.dataset_config['name'],
                'max_degradation_threshold': self.global_config['max_degradation'],
                'available_enhancement_methods': [
                    self.get_enhancement_method_config(method)['name'] 
                    for method in self.dataset_config['enhancement_methods']
                ]
            }
            
            # 保存JSON文件
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(individual_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"GPU {self.gpu_id}: 生成单独JSON文件: {json_path}")
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: 生成单独JSON文件失败 {audio_path}: {e}")
    
    def process_audio_subset(self, audio_files: List[str], subset_id: int, output_dir: str) -> List[Dict]:
        """处理音频文件子集"""
        logger.info(f"GPU {self.gpu_id}: 开始处理子集 {subset_id}，共 {len(audio_files)} 个文件")
        
        results = []
        for i, audio_file in enumerate(tqdm(audio_files, desc=f"GPU {self.gpu_id}: 子集{subset_id}")):
            logger.info(f"GPU {self.gpu_id}: 处理第{i+1}/{len(audio_files)}个文件")
            
            result = self.process_audio_file(audio_file, output_dir)
            results.append(result)
            
            # 更新统计
            self.stats['total_files'] += 1
            if result['success']:
                self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        return results

def collect_audio_files(dataset_config: Dict) -> List[str]:
    """收集数据集中的所有音频文件"""
    # 优先使用wav.scp文件
    wav_scp_file = dataset_config.get('wav_scp_file')
    if wav_scp_file and Path(wav_scp_file).exists():
        logger.info(f"使用wav.scp文件收集音频: {wav_scp_file}")
        wav_scp_reader = WavScpReader(wav_scp_file)
        audio_files = []
        
        for utterance_id in wav_scp_reader.get_all_utterance_ids():
            audio_path = wav_scp_reader.get_audio_path(utterance_id)
            if audio_path and Path(audio_path).exists():
                audio_files.append(audio_path)
            else:
                logger.warning(f"音频文件不存在: {audio_path} (utterance: {utterance_id})")
        
        logger.info(f"从wav.scp文件收集到 {len(audio_files)} 个音频文件")
        return sorted(audio_files)
    
    # 降级到目录遍历方式
    logger.info("使用目录遍历方式收集音频文件")
    audio_files = []
    base_dir = Path(dataset_config['base_dir'])
    
    # 音频扩展名
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac'}
    
    for root, dirs, files in os.walk(base_dir):
        # 跳过kaldi_files目录
        if 'kaldi_files' in Path(root).parts:
            continue
            
        for file in files:
            if Path(file).suffix.lower() in audio_extensions:
                audio_files.append(os.path.join(root, file))
    
    logger.info(f"从目录遍历收集到 {len(audio_files)} 个音频文件")
    return sorted(audio_files)

def split_audio_files(audio_files: List[str], num_splits: int) -> List[List[str]]:
    """将音频文件列表分成多个子集"""
    total_files = len(audio_files)
    files_per_split = total_files // num_splits
    remainder = total_files % num_splits
    
    splits = []
    start_idx = 0
    
    for i in range(num_splits):
        current_split_size = files_per_split + (1 if i < remainder else 0)
        end_idx = start_idx + current_split_size
        
        splits.append(audio_files[start_idx:end_idx])
        start_idx = end_idx
    
    return splits

def process_gpu_subset(args_tuple):
    """处理单个GPU子集的函数"""
    gpu_id, audio_files_subset, global_config, dataset_config, subset_id, output_dir, llm_service_url = args_tuple
    
    try:
        # 设置进程的GPU环境变量
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 确保在子进程中重新初始化CUDA
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        
        # 创建处理器
        processor = GroundtruthBasedProcessor(global_config, dataset_config, gpu_id, llm_service_url)
        
        # 处理子集
        results = processor.process_audio_subset(audio_files_subset, subset_id, output_dir)
        
        return results, processor.stats
        
    except Exception as e:
        logger.error(f"GPU {gpu_id}: 处理子集失败: {e}")
        import traceback
        traceback.print_exc()
        return [], {}

def merge_results(results_list: List[List[Dict]], stats_list: List[Dict], 
                 output_dir: str, dataset_config: Dict):
    """合并所有GPU的结果"""
    logger.info(f"合并结果 - 数据集: {dataset_config['name']}")
    
    # 合并结果
    all_results = []
    for results in results_list:
        all_results.extend(results)
    
    # 合并统计
    merged_stats = {
        'total_files': 0,
        'processed_files': 0,
        'failed_files': 0,
        'original_selected': 0,
        'enhanced_selected': 0,
        'no_groundtruth': 0,
        'enhancement_methods': {},
        'cer_improvements': []
    }
    
    for stats in stats_list:
        for key in ['total_files', 'processed_files', 'failed_files', 
                   'original_selected', 'enhanced_selected', 'no_groundtruth']:
            merged_stats[key] += stats.get(key, 0)
        
        for method, count in stats.get('enhancement_methods', {}).items():
            if method not in merged_stats['enhancement_methods']:
                merged_stats['enhancement_methods'][method] = 0
            merged_stats['enhancement_methods'][method] += count
        
        merged_stats['cer_improvements'].extend(stats.get('cer_improvements', []))
    
    # 保存详细结果
    results_file = Path(output_dir) / "groundtruth_based_integration_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 保存统计摘要
    summary = {
        'dataset_config': dataset_config,
        'statistics': merged_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    # 计算额外统计
    if merged_stats['cer_improvements']:
        summary['statistics']['cer_improvement_stats'] = {
            'mean': float(np.mean(merged_stats['cer_improvements'])),
            'median': float(np.median(merged_stats['cer_improvements'])),
            'std': float(np.std(merged_stats['cer_improvements'])),
            'min': float(np.min(merged_stats['cer_improvements'])),
            'max': float(np.max(merged_stats['cer_improvements']))
        }
    
    # 计算选择率
    total_successful = merged_stats['processed_files']
    if total_successful > 0:
        summary['statistics']['selection_rates'] = {
            'original_rate': merged_stats['original_selected'] / total_successful,
            'enhanced_rate': merged_stats['enhanced_selected'] / total_successful,
            'success_rate': total_successful / merged_stats['total_files'] if merged_stats['total_files'] > 0 else 0.0
        }
    
    summary_file = Path(output_dir) / "integration_summary.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    logger.info(f"结果保存到: {output_dir}")
    print_summary(merged_stats, dataset_config)
    
    return all_results

def print_summary(stats: Dict, dataset_config: Dict):
    """打印统计摘要"""
    print("\n" + "=" * 80)
    print(f"基于Groundtruth的音频集成结果 - 数据集: {dataset_config['name']}")
    print("=" * 80)
    print(f"总音频文件数:     {stats['total_files']}")
    print(f"成功处理:         {stats['processed_files']}")
    print(f"失败处理:         {stats['failed_files']}")
    print(f"无groundtruth:    {stats['no_groundtruth']}")
    print(f"\n选择结果:")
    print(f"  使用原音频:     {stats['original_selected']} ({stats['original_selected']/max(1,stats['processed_files'])*100:.1f}%)")
    print(f"  使用降噪音频:   {stats['enhanced_selected']} ({stats['enhanced_selected']/max(1,stats['processed_files'])*100:.1f}%)")
    
    if stats['enhancement_methods']:
        print(f"\n降噪方法使用统计:")
        for method, count in stats['enhancement_methods'].items():
            print(f"  {method}: {count} 次")
    
    if stats['cer_improvements']:
        improvements = stats['cer_improvements']
        print(f"\nCER改善情况:")
        print(f"  平均改善:       {np.mean(improvements):.4f}")
        print(f"  中位数改善:     {np.median(improvements):.4f}")
        print(f"  最大改善:       {np.max(improvements):.4f}")
    
    print("=" * 80)

def detect_gpu_config():
    """检测GPU配置"""
    available_gpus = torch.cuda.device_count()
    
    if available_gpus < 1:
        logger.error("至少需要1张GPU")
        raise RuntimeError("至少需要1张GPU")
    
    # 每张GPU卡都独立运行
    gpu_groups = []
    for gpu_id in range(available_gpus):
        gpu_groups.append({
            'gpu_id': gpu_id,
            'llm_service_url': f'http://localhost:{8000 + gpu_id}',
        })
    
    return gpu_groups

def main():
    """主函数"""
    # 设置multiprocessing启动方法
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    parser = argparse.ArgumentParser(description="基于Groundtruth的音频集成脚本")
    
    parser.add_argument("--config", type=str, default="config.json",
                       help="配置文件路径")
    parser.add_argument("--datasets", type=str, nargs='+',
                       help="要处理的数据集名称")
    parser.add_argument("--max_degradation", type=float,
                       help="最大CER退化量，超过此值才回退到原音频")
    # 保持向后兼容性
    parser.add_argument("--min_improvement", type=float, dest='max_degradation',
                       help="(已弃用) 使用--max_degradation代替")
    parser.add_argument("--num_gpus", type=int,
                       help="使用的GPU数量")
    
    args = parser.parse_args()
    
    # 加载配置
    config_manager = ConfigManager(args.config)
    global_config = config_manager.get_global_config()
    datasets = config_manager.get_datasets()
    
    # 更新配置
    if args.max_degradation is not None:
        global_config['max_degradation'] = args.max_degradation
    
    # 检测GPU配置
    gpu_groups = detect_gpu_config()
    num_gpus = args.num_gpus if args.num_gpus else len(gpu_groups)
    if num_gpus > len(gpu_groups):
        num_gpus = len(gpu_groups)
    
    logger.info(f"使用 {num_gpus} 张GPU进行处理")
    
    # 选择要处理的数据集
    datasets_to_process = []
    if args.datasets:
        for dataset_name in args.datasets:
            dataset_config = config_manager.get_dataset_by_name(dataset_name)
            if dataset_config:
                datasets_to_process.append(dataset_config)
            else:
                logger.warning(f"未找到数据集配置: {dataset_name}")
    else:
        datasets_to_process = datasets
    
    print("基于Groundtruth的音频集成脚本")
    print("=" * 80)
    print(f"配置文件: {args.config}")
    print(f"要处理的数据集: {[d['name'] for d in datasets_to_process]}")
    print(f"CER退化阈值: {global_config['max_degradation']:.4f}")
    print(f"使用GPU数量: {num_gpus}")
    print("=" * 80)
    
    # 检查LLM服务
    session = requests.Session()
    # 彻底清除代理设置
    session.proxies = {
        'http': None,
        'https': None,
        'no_proxy': 'localhost,127.0.0.1,::1'
    }
    
    # 清除可能影响的环境变量
    original_env = {}
    proxy_vars = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'all_proxy', 'ALL_PROXY']
    for var in proxy_vars:
        if var in os.environ:
            original_env[var] = os.environ[var]
            del os.environ[var]
    
    # 设置no_proxy环境变量
    os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
    os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
    
    print("等待LLM服务完全启动...")
    time.sleep(20)  # 给服务更多启动时间
    
    available_services = 0
    failed_services = []
    
    print("检查LLM服务连接状态...")
    for i in range(num_gpus):
        url = f"http://localhost:{8000 + i}"
        service_ready = False
        
        # 每个服务尝试多次连接
        for attempt in range(15):  # 增加到15次尝试
            try:
                response = session.get(f"{url}/health", timeout=8)
                if response.status_code == 200:
                    # 验证响应内容
                    try:
                        response_data = response.json()
                        if response_data.get("status") == "healthy":
                            available_services += 1
                            print(f"✓ LLM服务正常: {url}")
                            service_ready = True
                            break
                    except:
                        # 如果不是JSON响应，检查文本内容
                        if "healthy" in response.text.lower():
                            available_services += 1
                            print(f"✓ LLM服务正常: {url}")
                            service_ready = True
                            break
                else:
                    logger.debug(f"LLM服务响应异常: {url}, 状态码: {response.status_code}, 尝试 {attempt+1}/15")
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                logger.debug(f"LLM服务连接失败: {url}, 错误: {type(e).__name__}, 尝试 {attempt+1}/15")
            except Exception as e:
                logger.debug(f"LLM服务检查失败: {url}, 错误: {e}, 尝试 {attempt+1}/15")
            
            # 如果不是最后一次尝试，等待后重试
            if attempt < 14:
                time.sleep(2)
        
        # 如果所有尝试都失败了
        if not service_ready:
            failed_services.append(f"{url} (15次尝试后仍无法连接)")
            logger.warning(f"LLM服务连接失败: {url}, 已尝试15次")
    
    # 恢复原始环境变量
    for var, value in original_env.items():
        os.environ[var] = value
    
    if available_services == 0:
        logger.error("没有可用的LLM服务，请先启动")
        if failed_services:
            logger.error("失败的服务列表:")
            for service in failed_services:
                logger.error(f"  - {service}")
        logger.error("请运行以下命令启动LLM服务:")
        logger.error("  ./auto_start_llm_services.sh")
        logger.error("或者检查服务日志: tail -f logs/llm_service_*.log")
        return
    
    print(f"✓ 发现 {available_services}/{num_gpus} 个可用的LLM服务")
    if available_services < num_gpus:
        logger.warning(f"部分LLM服务不可用 ({available_services}/{num_gpus})，将使用可用的服务继续处理")
    
    # 处理每个数据集
    total_start_time = time.time()
    
    for dataset_config in datasets_to_process:
        print(f"\n{'='*80}")
        print(f"开始处理数据集: {dataset_config['name']}")
        print(f"{'='*80}")
        
        try:
            # 检查数据集目录和groundtruth文件
            if not Path(dataset_config['base_dir']).exists():
                logger.error(f"数据集目录不存在: {dataset_config['base_dir']}")
                continue
            
            if not Path(dataset_config['groundtruth_file']).exists():
                logger.error(f"Groundtruth文件不存在: {dataset_config['groundtruth_file']}")
                continue
            
            # 创建输出目录
            output_dir = Path(dataset_config['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # 收集音频文件
            logger.info("收集音频文件...")
            audio_files = collect_audio_files(dataset_config)
            if not audio_files:
                logger.warning(f"数据集 {dataset_config['name']} 中未找到音频文件")
                continue
            
            logger.info(f"找到 {len(audio_files)} 个音频文件")
            
            # 分割音频文件给多个GPU
            audio_files_splits = split_audio_files(audio_files, num_gpus)
            
            # 准备多进程参数
            process_args = []
            for i in range(num_gpus):
                if i < len(audio_files_splits):
                    gpu_id = gpu_groups[i]['gpu_id']
                    llm_service_url = gpu_groups[i]['llm_service_url']
                    process_args.append((
                        gpu_id, audio_files_splits[i], global_config, dataset_config, 
                        i, str(output_dir), llm_service_url
                    ))
            
            # 启动多进程处理
            logger.info(f"启动多GPU并行处理...")
            start_time = time.time()
            
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for args in process_args:
                    future = executor.submit(process_gpu_subset, args)
                    futures.append(future)
                
                # 等待所有进程完成
                results_list = []
                stats_list = []
                for future in futures:
                    try:
                        results, stats = future.result()
                        results_list.append(results)
                        stats_list.append(stats)
                    except Exception as e:
                        logger.error(f"进程执行失败: {e}")
                        results_list.append([])
                        stats_list.append({})
            
            processing_time = time.time() - start_time
            logger.info(f"数据集 {dataset_config['name']} 处理完成，耗时: {processing_time:.2f} 秒")
            
            # 合并结果
            merge_results(results_list, stats_list, str(output_dir), dataset_config)
            
        except Exception as e:
            logger.error(f"处理数据集 {dataset_config['name']} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_processing_time = time.time() - total_start_time
    
    print(f"\n{'='*80}")
    print("所有数据集处理完成！")
    print(f"总处理时间: {total_processing_time:.2f} 秒")
    print(f"处理的数据集数: {len(datasets_to_process)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main() 