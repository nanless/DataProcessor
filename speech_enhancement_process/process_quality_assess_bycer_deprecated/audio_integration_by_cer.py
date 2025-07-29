#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频整合脚本 - 基于CER判断最佳版本

根据ASR和LLM TN后的CER结果，判断降噪前的音频是否可以由降噪后的音频替换。
支持多个降噪模型的比较，自动选择最佳版本进行整合。

功能：
1. 读取原音频和多个降噪模型的质量评估结果
2. 根据CER阈值判断是否使用降噪版本
3. 多个降噪模型可用时，选择CER最低的版本
4. 保持原有目录结构，拷贝最佳音频到新目录
5. 生成详细的选择记录JSON文件
"""

import os
import sys
import json
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioIntegrator:
    """音频整合器"""
    
    def __init__(self, original_dir: str, enhanced_dirs: List[str], output_dir: str, 
                 cer_threshold: float = 0.05, prefer_original: bool = True):
        """
        初始化音频整合器
        
        Args:
            original_dir: 原音频目录
            enhanced_dirs: 降噪后音频目录列表
            output_dir: 输出目录
            cer_threshold: CER阈值，低于此值认为音频可用
            prefer_original: 当CER相近时是否优先选择原音频
        """
        self.original_dir = Path(original_dir)
        self.enhanced_dirs = [Path(d) for d in enhanced_dirs]
        self.output_dir = Path(output_dir)
        self.cer_threshold = cer_threshold
        self.prefer_original = prefer_original
        
        # 验证输入路径
        if not self.original_dir.exists():
            raise ValueError(f"原音频目录不存在: {original_dir}")
        
        for enhanced_dir in self.enhanced_dirs:
            if not enhanced_dir.exists():
                raise ValueError(f"降噪音频目录不存在: {enhanced_dir}")
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'original_selected': 0,
            'enhanced_selected': 0,
            'enhancement_methods': {},
            'failed_files': 0,
            'cer_improvements': []
        }
        
        logger.info(f"音频整合器初始化完成")
        logger.info(f"原音频目录: {self.original_dir}")
        logger.info(f"降噪目录数量: {len(self.enhanced_dirs)}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"CER阈值: {self.cer_threshold:.1%}")
        logger.info(f"优先原音频: {self.prefer_original}")
    
    def load_assessment_results(self, assessment_dir: Path) -> Dict[str, Dict]:
        """加载质量评估结果"""
        try:
            # 查找quality_assessment_results.json文件
            results_file = assessment_dir / "quality_assessment_results.json"
            
            if not results_file.exists():
                logger.warning(f"评估结果文件不存在: {results_file}")
                return {}
            
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            # 将结果转换为以相对路径为key的字典
            results_dict = {}
            for result in results:
                if 'relative_path' in result and result.get('success', False):
                    relative_path = result['relative_path']
                    results_dict[relative_path] = result
            
            logger.info(f"从 {assessment_dir} 加载了 {len(results_dict)} 个评估结果")
            return results_dict
            
        except Exception as e:
            logger.error(f"加载评估结果失败 {assessment_dir}: {e}")
            return {}
    
    def find_enhancement_method_name(self, enhanced_dir: Path) -> str:
        """从目录名推断降噪方法名称"""
        dir_name = enhanced_dir.name.lower()
        
        # 常见的降噪方法标识
        method_patterns = {
            'resemble': 'Resemble-Enhance',
            'zipenhancer': 'ZipEnhancer', 
            'mossformer': 'MossFormer',
            'speechenhance': 'SpeechEnhance',
            'dfnet': 'DFNet',
            'deepfilternet': 'DeepFilterNet',
            'noisereduce': 'NoiseReduce',
            'spectral': 'SpectralSubtraction',
            'wiener': 'WienerFilter',
            'nsnet': 'NSNet',
            'dns': 'DNS',
            'fullsubnet': 'FullSubNet',
            'denoiser': 'Denoiser',
            'rnnoise': 'RNNoise'
        }
        
        for pattern, method_name in method_patterns.items():
            if pattern in dir_name:
                return method_name
        
        # 如果没有匹配，使用目录名的简化版本
        return enhanced_dir.name.replace('_enhanced', '').replace('_', ' ').title()
    
    def compare_audio_versions(self, relative_path: str, 
                              assessment_results: Dict[str, Dict[str, Dict]]) -> Tuple[str, Dict]:
        """
        比较音频的不同版本，选择最佳版本
        
        Args:
            relative_path: 音频的相对路径
            assessment_results: 所有评估结果 {method_name: {relative_path: result}}
            
        Returns:
            (selected_method, detailed_info): 选择的方法名和详细信息
        """
        candidates = []
        
        # 收集所有可用的降噪版本信息
        # 注意：评估结果中的CER是比较原音频转录和降噪音频转录的结果
        for method_name, results in assessment_results.items():
            if relative_path in results:
                result = results[relative_path]
                if result.get('success', False):
                    cer = result.get('cer', 1.0)
                    wer = result.get('wer', 1.0)
                    is_usable = result.get('is_usable', False)
                    
                    candidates.append({
                        'method': method_name,
                        'cer': cer,
                        'wer': wer,
                        'is_usable': is_usable,
                        'result_data': result
                    })
        
        if not candidates:
            return 'original', {
                'reason': 'no_assessment_data',
                'message': '没有找到评估数据，使用原音频'
            }
        
        # 筛选可用的降噪版本（CER < threshold）
        usable_candidates = [c for c in candidates if c['cer'] < self.cer_threshold]
        
        # 决策逻辑
        if not usable_candidates:
            # 没有可用的降噪版本，使用原音频
            min_cer = min(c['cer'] for c in candidates)
            return 'original', {
                'reason': 'no_usable_enhanced',
                'message': f'所有降噪版本CER均高于阈值 {self.cer_threshold:.1%}，最低CER={min_cer:.3f}',
                'all_candidates': candidates,
                'min_enhanced_cer': min_cer
            }
        
        # 选择CER最低的降噪版本
        best_enhanced = min(usable_candidates, key=lambda x: x['cer'])
        
        # 如果设置了优先原音频，需要考虑原音频的"隐含质量"
        # 在这种情况下，我们假设如果降噪版本的CER很高，说明原音频可能质量更好
        if self.prefer_original:
            enhanced_cer = best_enhanced['cer']
            
            # 如果最佳降噪版本的CER接近阈值上限，可能原音频更好
            # 这里使用一个更保守的策略：如果CER > 3%，优先考虑原音频
            if enhanced_cer > 0.03:
                return 'original', {
                    'reason': 'prefer_original_conservative',
                    'message': f'最佳降噪版本CER={enhanced_cer:.3f}较高，优先选择源音频',
                    'best_enhanced_cer': enhanced_cer,
                    'best_enhanced_method': best_enhanced['method']
                }
        
        # 选择最佳降噪版本
        return best_enhanced['method'], {
            'reason': 'enhanced_better',
            'message': f'选择最佳降噪版本: {best_enhanced["method"]}',
            'selected_cer': best_enhanced['cer'],
            'selected_wer': best_enhanced['wer'],
            'improvement_note': f'CER={best_enhanced["cer"]:.3f} < 阈值{self.cer_threshold:.1%}',
            'all_usable_candidates': usable_candidates
        }
    
    def copy_audio_file(self, relative_path: str, selected_method: str) -> bool:
        """拷贝选择的音频文件到输出目录"""
        try:
            # 确定源文件路径
            if selected_method == 'original':
                source_file = self.original_dir / relative_path
            else:
                # 找到对应的降噪目录
                source_file = None
                for enhanced_dir in self.enhanced_dirs:
                    method_name = self.find_enhancement_method_name(enhanced_dir)
                    if method_name == selected_method:
                        # 降噪后的文件通常是.wav格式
                        base_name = Path(relative_path).stem
                        source_file = enhanced_dir / Path(relative_path).parent / f"{base_name}.wav"
                        break
                
                if source_file is None:
                    logger.error(f"找不到降噪方法 {selected_method} 对应的目录")
                    return False
            
            if not source_file.exists():
                logger.error(f"源文件不存在: {source_file}")
                return False
            
            # 确定目标文件路径
            target_file = self.output_dir / relative_path
            
            # 如果源文件是降噪后的.wav，但目标应该保持原始扩展名
            if selected_method != 'original':
                original_ext = Path(relative_path).suffix
                if source_file.suffix != original_ext:
                    # 保持原始扩展名
                    target_file = target_file.with_suffix('.wav')  # 降噪后通常是wav
            
            # 创建目标目录
            target_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 拷贝文件
            shutil.copy2(source_file, target_file)
            logger.debug(f"拷贝文件: {source_file} -> {target_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"拷贝文件失败 {relative_path}: {e}")
            return False
    
    def collect_all_audio_files(self) -> List[str]:
        """收集所有需要处理的音频文件（相对路径）"""
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg'}
        audio_files = []
        
        for root, dirs, files in os.walk(self.original_dir):
            for file in files:
                if Path(file).suffix.lower() in audio_extensions:
                    relative_path = os.path.relpath(
                        os.path.join(root, file), 
                        self.original_dir
                    )
                    audio_files.append(relative_path)
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        return sorted(audio_files)
    
    def process_integration(self) -> Dict:
        """执行音频整合处理"""
        logger.info("开始音频整合处理...")
        
        # 加载所有评估结果
        assessment_results = {}
        
        # 加载原音频的评估结果（如果存在）
        for enhanced_dir in self.enhanced_dirs:
            # 查找对应的评估结果目录
            assessment_dir = enhanced_dir.parent / f"{enhanced_dir.name}_quality_assessment"
            if assessment_dir.exists():
                method_name = self.find_enhancement_method_name(enhanced_dir)
                assessment_results[method_name] = self.load_assessment_results(assessment_dir)
                logger.info(f"加载降噪方法 '{method_name}' 的评估结果")
        
        # 假设原音频的评估结果在某个地方（可能需要调整）
        # 这里暂时跳过原音频评估结果的加载，因为通常我们只有降噪后的评估
        
        if not assessment_results:
            logger.error("没有找到任何评估结果，无法进行整合")
            return {'success': False, 'error': '没有找到评估结果'}
        
        # 收集所有音频文件
        audio_files = self.collect_all_audio_files()
        self.stats['total_files'] = len(audio_files)
        
        # 处理每个音频文件
        integration_records = []
        
        for relative_path in tqdm(audio_files, desc="处理音频文件"):
            try:
                # 比较不同版本，选择最佳版本
                selected_method, decision_info = self.compare_audio_versions(
                    relative_path, assessment_results
                )
                
                # 拷贝选择的文件
                copy_success = self.copy_audio_file(relative_path, selected_method)
                
                # 收集详细信息
                record = {
                    'relative_path': relative_path,
                    'selected_method': selected_method,
                    'is_enhanced': selected_method != 'original',
                    'copy_success': copy_success,
                    'decision_info': decision_info,
                    'timestamp': datetime.now().isoformat()
                }
                
                # 添加ASR和CER信息
                if selected_method in assessment_results:
                    results = assessment_results[selected_method]
                    if relative_path in results:
                        result_data = results[relative_path]
                        record['asr_info'] = {
                            'original_transcription': result_data.get('original_transcription', ''),
                            'enhanced_transcription': result_data.get('enhanced_transcription', ''),
                            'original_transcription_normalized': result_data.get('original_transcription_normalized', ''),
                            'enhanced_transcription_normalized': result_data.get('enhanced_transcription_normalized', ''),
                            'wer': result_data.get('wer', None),
                            'cer': result_data.get('cer', None),
                            'is_usable': result_data.get('is_usable', False)
                        }
                
                integration_records.append(record)
                
                # 更新统计
                if copy_success:
                    if selected_method == 'original':
                        self.stats['original_selected'] += 1
                    else:
                        self.stats['enhanced_selected'] += 1
                        # 统计降噪方法使用次数
                        if selected_method not in self.stats['enhancement_methods']:
                            self.stats['enhancement_methods'][selected_method] = 0
                        self.stats['enhancement_methods'][selected_method] += 1
                        
                        # 记录CER改善情况
                        if 'improvement' in decision_info and decision_info['improvement']:
                            self.stats['cer_improvements'].append(decision_info['improvement'])
                else:
                    self.stats['failed_files'] += 1
                    
            except Exception as e:
                logger.error(f"处理文件失败 {relative_path}: {e}")
                self.stats['failed_files'] += 1
                
                # 记录失败信息
                integration_records.append({
                    'relative_path': relative_path,
                    'selected_method': 'error',
                    'is_enhanced': False,
                    'copy_success': False,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        # 保存整合记录
        self.save_integration_records(integration_records)
        
        # 保存统计摘要
        self.save_integration_summary()
        
        logger.info("音频整合处理完成")
        return {
            'success': True,
            'stats': self.stats,
            'records_count': len(integration_records)
        }
    
    def save_integration_records(self, records: List[Dict]):
        """保存详细的整合记录"""
        records_file = self.output_dir / "audio_integration_records.json"
        
        try:
            with open(records_file, 'w', encoding='utf-8') as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            
            logger.info(f"整合记录已保存: {records_file}")
            
        except Exception as e:
            logger.error(f"保存整合记录失败: {e}")
    
    def save_integration_summary(self):
        """保存整合统计摘要"""
        # 计算统计数据
        summary = {
            'integration_config': {
                'original_dir': str(self.original_dir),
                'enhanced_dirs': [str(d) for d in self.enhanced_dirs],
                'output_dir': str(self.output_dir),
                'cer_threshold': self.cer_threshold,
                'prefer_original': self.prefer_original
            },
            'statistics': self.stats.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        # 计算额外统计
        if self.stats['cer_improvements']:
            summary['statistics']['cer_improvement_stats'] = {
                'mean': float(np.mean(self.stats['cer_improvements'])),
                'median': float(np.median(self.stats['cer_improvements'])),
                'std': float(np.std(self.stats['cer_improvements'])),
                'min': float(np.min(self.stats['cer_improvements'])),
                'max': float(np.max(self.stats['cer_improvements']))
            }
        
        # 计算选择率
        total_successful = self.stats['total_files'] - self.stats['failed_files']
        if total_successful > 0:
            summary['statistics']['selection_rates'] = {
                'original_rate': self.stats['original_selected'] / total_successful,
                'enhanced_rate': self.stats['enhanced_selected'] / total_successful,
                'success_rate': total_successful / self.stats['total_files']
            }
        
        # 保存摘要
        summary_file = self.output_dir / "integration_summary.json"
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"整合摘要已保存: {summary_file}")
            
        except Exception as e:
            logger.error(f"保存整合摘要失败: {e}")
    
    def print_summary(self):
        """打印整合结果摘要"""
        print("\n" + "=" * 80)
        print("音频整合结果摘要")
        print("=" * 80)
        
        print(f"总音频文件数:     {self.stats['total_files']}")
        print(f"成功处理:         {self.stats['total_files'] - self.stats['failed_files']}")
        print(f"失败处理:         {self.stats['failed_files']}")
        print(f"\n选择结果:")
        print(f"  使用原音频:     {self.stats['original_selected']} ({self.stats['original_selected']/max(1,self.stats['total_files'])*100:.1f}%)")
        print(f"  使用降噪音频:   {self.stats['enhanced_selected']} ({self.stats['enhanced_selected']/max(1,self.stats['total_files'])*100:.1f}%)")
        
        if self.stats['enhancement_methods']:
            print(f"\n降噪方法使用统计:")
            for method, count in self.stats['enhancement_methods'].items():
                print(f"  {method}: {count} 次")
        
        if self.stats['cer_improvements']:
            improvements = self.stats['cer_improvements']
            print(f"\nCER改善情况:")
            print(f"  平均改善:       {np.mean(improvements):.4f}")
            print(f"  中位数改善:     {np.median(improvements):.4f}")
            print(f"  最大改善:       {np.max(improvements):.4f}")
        
        print(f"\n配置信息:")
        print(f"  CER阈值:        {self.cer_threshold:.1%}")
        print(f"  优先原音频:     {self.prefer_original}")
        print(f"  输出目录:       {self.output_dir}")
        
        print("=" * 80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="音频整合脚本 - 基于CER选择最佳音频版本")
    
    # 必需参数
    parser.add_argument("--original_dir", type=str, required=True,
                       help="原音频目录路径")
    parser.add_argument("--enhanced_dirs", type=str, nargs='+', required=True,
                       help="降噪音频目录路径列表（支持多个降噪模型）")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="整合后音频输出目录")
    
    # 可选参数
    parser.add_argument("--cer_threshold", type=float, default=0.05,
                       help="CER阈值，低于此值认为音频可用 (默认: 0.05)")
    parser.add_argument("--prefer_original", action="store_true", default=True,
                       help="当CER相近时优先选择原音频 (默认: True)")
    parser.add_argument("--no_prefer_original", action="store_false", dest="prefer_original",
                       help="禁用优先选择原音频")
    
    args = parser.parse_args()
    
    print("音频整合脚本")
    print("=" * 50)
    print(f"原音频目录:       {args.original_dir}")
    print(f"降噪目录数量:     {len(args.enhanced_dirs)}")
    for i, enhanced_dir in enumerate(args.enhanced_dirs):
        print(f"  降噪目录{i+1}:     {enhanced_dir}")
    print(f"输出目录:         {args.output_dir}")
    print(f"CER阈值:          {args.cer_threshold:.1%}")
    print(f"优先原音频:       {args.prefer_original}")
    print("=" * 50)
    
    try:
        # 创建整合器
        integrator = AudioIntegrator(
            original_dir=args.original_dir,
            enhanced_dirs=args.enhanced_dirs,
            output_dir=args.output_dir,
            cer_threshold=args.cer_threshold,
            prefer_original=args.prefer_original
        )
        
        # 执行整合
        result = integrator.process_integration()
        
        if result['success']:
            # 打印摘要
            integrator.print_summary()
            
            print("\n✅ 音频整合完成！")
            print(f"📁 整合后的音频保存在: {args.output_dir}")
            print(f"📄 详细记录文件: {args.output_dir}/audio_integration_records.json")
            print(f"📊 统计摘要文件: {args.output_dir}/integration_summary.json")
        else:
            print(f"\n❌ 音频整合失败: {result.get('error', '未知错误')}")
            return 1
            
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 