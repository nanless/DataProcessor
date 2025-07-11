import os
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import multiprocessing as mp
import time
import logging
from typing import List, Tuple, Dict, Optional

# 设置multiprocessing启动方法为spawn（CUDA多进程必需）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，会抛出RuntimeError，可以忽略
    pass

# 设置pytorch线程数
import torch
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入ClearVoice
try:
    from clearvoice import ClearVoice
    CLEARVOICE_AVAILABLE = True
except ImportError as e:
    logger.error(f"ClearVoice未安装成功: {e}")
    logger.error("请执行: pip install clearvoice")
    CLEARVOICE_AVAILABLE = False

# 配置参数
CONFIG = {
    'input_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid",
    'output_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_mossformer_enhanced",
    'model_name': "MossFormerGAN_SE_16K",
    'task': 'speech_enhancement',
    'device': "cuda",
    'gpu_ids': [0, 1, 2, 3],  # 使用的GPU ID列表
    'batch_size': 32,  # 批处理大小
    'num_workers': 8,  # 并行工作线程数
    'target_sr': 16000,  # 目标采样率
    'use_parallel': False,  # 修改为False，使用批处理模式来启用多GPU
    'use_multi_gpu': True,  # 是否使用多GPU
    'use_multiprocess': True,  # 是否使用多进程（否则使用多线程）
    'batch_processing_mode': 'isolated_gpu',  # 新增隔离GPU模式
    'skip_existing': True,  # 跳过已存在的文件
}

# 全局函数用于多进程处理
def process_gpu_batch_isolated(args):
    """完全隔离的GPU处理函数"""
    file_list, gpu_id, model_name, task, target_sr, input_dir, output_dir, position = args
    
    # 强制设置GPU设备 - 完全隔离
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 重新导入所需库
    import torch
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
    
    # 只设置线程数，避免interop threads的错误
    torch.set_num_threads(8)
    
    # 现在GPU 0对应于实际的指定GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU {gpu_id} 进程启动，设备: {device_name}")
    
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
                
                # 验证音频文件
                audio, sr = sf.read(input_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if len(audio) == 0:
                    raise ValueError("音频文件为空")
                
                # 处理音频 - 使用ClearVoice
                enhanced_audio = clearvoice_instance(input_path=input_file, online_write=False)
                
                # 保存结果
                if enhanced_audio is not None:
                    clearvoice_instance.write(enhanced_audio, output_path=output_file)
                    
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
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        return [(False, f"进程初始化失败: {e}") for _ in file_list]

def process_batch_on_gpu_worker(args):
    """多进程工作函数（保留原有功能）"""
    batch_files, gpu_id, model_name, task, target_sr, config = args
    
    # 强制设置GPU设备
    if gpu_id is not None and torch.cuda.is_available():
        # 设置CUDA_VISIBLE_DEVICES，让进程只看到指定GPU
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # 在spawn模式下，不需要torch.cuda.init()
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
        for audio_file in batch_files:
            try:
                # 验证音频文件
                audio, sr = sf.read(audio_file)
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                if len(audio) == 0:
                    raise ValueError("音频文件为空")
                
                # 处理音频
                enhanced_audio = clearvoice_instance(input_path=audio_file, online_write=False)
                
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

class MossFormerGANBatchProcessor:
    """MossFormerGAN批量处理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.model_name = config['model_name']
        self.task = config['task']
        self.device = config['device']
        self.batch_size = config['batch_size']
        self.target_sr = config['target_sr']
        self.gpu_ids = config.get('gpu_ids', [0])
        self.use_multi_gpu = config.get('use_multi_gpu', False)
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.available_gpus = [i for i in self.gpu_ids if i < torch.cuda.device_count()]
            if not self.available_gpus:
                logger.warning("指定的GPU不可用，使用CPU")
                self.device = "cpu"
                self.use_multi_gpu = False
            else:
                logger.info(f"可用GPU: {self.available_gpus}")
        else:
            logger.warning("CUDA不可用，使用CPU")
            self.device = "cpu"
            self.use_multi_gpu = False
        
        # 根据批处理模式决定是否初始化模型
        if self.config.get('batch_processing_mode') == 'isolated_gpu':
            # 隔离GPU模式不在主进程中初始化模型
            self.clearvoice_instances = []
            logger.info("隔离GPU模式：将在子进程中初始化模型")
        else:
            # 初始化模型
            self.clearvoice_instances = []
            self.init_model()
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'total_duration': 0,
            'processing_time': 0,
        }
    
    def init_model(self):
        """初始化MossFormerGAN模型"""
        if not CLEARVOICE_AVAILABLE:
            raise ImportError("ClearVoice未安装，无法使用MossFormerGAN")
        
        try:
            if self.use_multi_gpu and len(self.available_gpus) > 1:
                # 多GPU模式：为每个GPU创建一个ClearVoice实例
                logger.info(f"初始化多GPU模式，使用GPU: {self.available_gpus}")
                for gpu_id in self.available_gpus:
                    # 设置当前GPU设备
                    torch.cuda.set_device(gpu_id)
                    
                    # 创建ClearVoice实例
                    clearvoice_instance = ClearVoice(task=self.task, model_names=[self.model_name])
                    
                    # 存储实例和对应的GPU ID
                    self.clearvoice_instances.append((clearvoice_instance, gpu_id))
                    logger.info(f"✓ 在GPU {gpu_id}上加载模型成功")
            else:
                # 单GPU或CPU模式
                if self.available_gpus:
                    torch.cuda.set_device(self.available_gpus[0])
                
                clearvoice_instance = ClearVoice(task=self.task, model_names=[self.model_name])
                self.clearvoice_instances.append((clearvoice_instance, self.available_gpus[0] if self.available_gpus else None))
                logger.info(f"✓ 在设备 {self.device} 上加载模型成功")
            
            logger.info(f"✓ 成功加载MossFormerGAN模型: {self.model_name}")
            logger.info(f"✓ 总共加载了 {len(self.clearvoice_instances)} 个模型实例")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def preprocess_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """预处理音频文件（简化版，因为ClearVoice可以直接处理）"""
        try:
            # 读取音频文件进行验证
            audio, sr = sf.read(audio_path)
            
            # 转换为单声道
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            
            # 基本验证
            if len(audio) == 0:
                raise ValueError("音频文件为空")
            
            # 返回音频数据用于验证，实际处理时会直接传文件路径给ClearVoice
            return audio, sr
            
        except Exception as e:
            logger.error(f"音频预处理失败 {audio_path}: {e}")
            raise
    
    def enhance_audio_batch(self, audio_files: List[str]) -> List[Tuple[bool, str, Optional[np.ndarray]]]:
        """批量增强音频文件 - 支持多GPU"""
        results = []
        
        logger.info(f"enhance_audio_batch: 处理 {len(audio_files)} 个文件")
        logger.info(f"批处理模式: {self.config['batch_processing_mode']}")
        logger.info(f"ClearVoice实例数量: {len(self.clearvoice_instances)}")
        
        if self.config['batch_processing_mode'] == 'torch' and len(self.clearvoice_instances) > 1:
            # 多GPU批处理模式
            logger.info("🚀 启用多GPU批处理模式")
            return self.enhance_audio_batch_multigpu(audio_files)
        else:
            # 单GPU批处理模式
            logger.info("⚠️  使用单GPU批处理模式")
            return self.enhance_audio_batch_single(audio_files)
    
    def enhance_audio_batch_multigpu(self, audio_files: List[str]) -> List[Tuple[bool, str, Optional[np.ndarray]]]:
        """多GPU批处理"""
        results = []
        num_gpus = len(self.clearvoice_instances)
        
        logger.info(f"📊 多GPU处理: {len(audio_files)} 个文件，{num_gpus} 个GPU")
        
        # 将文件分配到不同GPU
        gpu_batches = [[] for _ in range(num_gpus)]
        for i, audio_file in enumerate(audio_files):
            gpu_batches[i % num_gpus].append(audio_file)
        
        # 显示分配情况
        for gpu_idx, batch in enumerate(gpu_batches):
            if batch:
                logger.info(f"GPU {gpu_idx}: 分配 {len(batch)} 个文件")
        
        # 选择执行器类型
        if self.config.get('use_multiprocess', False):
            # 使用多进程执行器
            logger.info("🔄 使用多进程模式进行GPU并行处理")
            with ProcessPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for gpu_idx, batch in enumerate(gpu_batches):
                    if batch:  # 只处理非空批次
                        logger.info(f"提交GPU {gpu_idx} 任务: {len(batch)} 个文件")
                        future = executor.submit(process_batch_on_gpu_worker, (batch, self.clearvoice_instances[gpu_idx][1], self.model_name, self.task, self.target_sr, self.config))
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        logger.info(f"收到批处理结果: {len(batch_results)} 个")
                    except Exception as e:
                        logger.error(f"多进程GPU批处理失败: {e}")
        else:
            # 使用多线程执行器
            logger.info("🔄 使用多线程模式进行GPU并行处理")
            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                futures = []
                for gpu_idx, batch in enumerate(gpu_batches):
                    if batch:  # 只处理非空批次
                        logger.info(f"提交GPU {gpu_idx} 任务: {len(batch)} 个文件")
                        future = executor.submit(self.process_batch_on_gpu, batch, gpu_idx)
                        futures.append(future)
                
                # 收集结果
                for future in as_completed(futures):
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        logger.info(f"收到批处理结果: {len(batch_results)} 个")
                    except Exception as e:
                        logger.error(f"多线程GPU批处理失败: {e}")
        
        logger.info(f"✅ 多GPU处理完成，总结果: {len(results)} 个")
        return results
    
    def enhance_audio_batch_single(self, audio_files: List[str]) -> List[Tuple[bool, str, Optional[np.ndarray]]]:
        """单GPU批处理"""
        results = []
        clearvoice_instance, gpu_id = self.clearvoice_instances[0]
        
        # 确保设置正确的GPU
        if gpu_id is not None:
            torch.cuda.set_device(gpu_id)
        
        for audio_file in audio_files:
            try:
                # 验证音频文件
                audio, sr = self.preprocess_audio(audio_file)
                
                # 在指定GPU上处理
                with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                    enhanced_audio = clearvoice_instance(input_path=audio_file, online_write=False)
                
                # 处理结果
                if enhanced_audio is not None:
                    results.append((True, f"成功处理: {audio_file}", enhanced_audio))
                else:
                    results.append((False, f"处理失败: {audio_file}", None))
                
            except Exception as e:
                logger.error(f"处理失败 {audio_file}: {e}")
                results.append((False, f"处理失败 {audio_file}: {e}", None))
        
        return results
    
    def set_gpu_device(self, gpu_id: int):
        """强制设置GPU设备"""
        if gpu_id is not None and torch.cuda.is_available():
            # 设置当前设备
            torch.cuda.set_device(gpu_id)
            # 清空GPU缓存
            torch.cuda.empty_cache()
            logger.info(f"已设置GPU设备: {gpu_id}")
            return True
        return False

    def process_batch_on_gpu(self, batch_files: List[str], gpu_idx: int) -> List[Tuple[bool, str, Optional[np.ndarray]]]:
        """在指定GPU上处理批次"""
        results = []
        clearvoice_instance, gpu_id = self.clearvoice_instances[gpu_idx]
        
        # 强制设置GPU
        self.set_gpu_device(gpu_id)
        
        # 验证当前GPU
        if gpu_id is not None and torch.cuda.is_available():
            current_device = torch.cuda.current_device()
            logger.info(f"当前GPU设备: {current_device}, 目标GPU: {gpu_id}")
            if current_device != gpu_id:
                logger.warning(f"GPU设备不匹配! 当前: {current_device}, 目标: {gpu_id}")
        
        logger.info(f"GPU {gpu_id} 开始处理 {len(batch_files)} 个文件")
        
        for audio_file in batch_files:
            try:
                # 验证音频文件
                audio, sr = self.preprocess_audio(audio_file)
                
                # 在指定GPU上处理
                if gpu_id is not None:
                    with torch.cuda.device(gpu_id):
                        enhanced_audio = clearvoice_instance(input_path=audio_file, online_write=False)
                else:
                    enhanced_audio = clearvoice_instance(input_path=audio_file, online_write=False)
                
                # 处理结果
                if enhanced_audio is not None:
                    results.append((True, f"成功处理: {audio_file} (GPU {gpu_id})", enhanced_audio))
                else:
                    results.append((False, f"处理失败: {audio_file} (GPU {gpu_id})", None))
                
            except Exception as e:
                logger.error(f"GPU {gpu_id} 处理失败 {audio_file}: {e}")
                results.append((False, f"处理失败 {audio_file}: {e}", None))
        
        logger.info(f"GPU {gpu_id} 完成处理")
        return results
    
    def enhance_single_file(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """增强单个文件"""
        try:
            # 检查是否跳过已存在的文件
            if self.config['skip_existing'] and os.path.exists(output_file):
                self.stats['skipped_files'] += 1
                return True, f"跳过已存在文件: {input_file}"
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 使用第一个ClearVoice实例处理
            clearvoice_instance, gpu_id = self.clearvoice_instances[0]
            
            # 确保设置正确的GPU
            if gpu_id is not None:
                torch.cuda.set_device(gpu_id)
            
            # 在指定GPU上处理
            with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                enhanced_audio = clearvoice_instance(input_path=input_file, online_write=False)
            
            # 保存结果
            if enhanced_audio is not None:
                clearvoice_instance.write(enhanced_audio, output_path=output_file)
                
                # 检查结果
                if os.path.exists(output_file):
                    self.stats['processed_files'] += 1
                    return True, f"成功处理: {input_file}"
                else:
                    self.stats['failed_files'] += 1
                    return False, f"处理失败，输出文件未生成: {input_file}"
            else:
                self.stats['failed_files'] += 1
                return False, f"处理失败，无音频数据: {input_file}"
                
        except Exception as e:
            self.stats['failed_files'] += 1
            return False, f"处理失败 {input_file}: {e}"
    
    def get_audio_files(self, input_dir: str) -> List[Tuple[str, str]]:
        """获取所有音频文件路径"""
        audio_files = []
        
        for root, dirs, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    input_path = os.path.join(root, file)
                    
                    # 计算相对路径
                    rel_path = os.path.relpath(input_path, input_dir)
                    
                    # 构造输出路径，确保为wav格式
                    output_path = os.path.join(
                        self.config['output_dir'], 
                        os.path.splitext(rel_path)[0] + '.wav'
                    )
                    
                    audio_files.append((input_path, output_path))
        
        return audio_files
    
    def process_files_isolated_gpu(self, file_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """隔离GPU处理模式 - 每个GPU处理四分之一的文件"""
        if not self.available_gpus:
            logger.error("没有可用的GPU")
            return []
        
        logger.info(f"开始隔离GPU处理，文件总数: {len(file_pairs)}")
        logger.info(f"使用GPU: {self.available_gpus}")
        
        # 将文件分成等份
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
            
            logger.info(f"GPU {gpu_id}: 分配 {len(gpu_files)} 个文件 (索引 {start_idx}-{end_idx-1})")
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
                    self.model_name,
                    self.task,
                    self.target_sr,
                    self.config['input_dir'],
                    self.config['output_dir'],
                    position
                )
                args_list.append(args)
            
            # 提交任务
            results = pool.map(process_gpu_batch_isolated, args_list)
            
            # 合并结果
            for result_batch in results:
                all_results.extend(result_batch)
        
        # 更新统计信息
        for success, message in all_results:
            if success:
                if "跳过已存在文件" in message:
                    self.stats['skipped_files'] += 1
                else:
                    self.stats['processed_files'] += 1
            else:
                self.stats['failed_files'] += 1
        
        logger.info(f"隔离GPU处理完成，总结果: {len(all_results)} 个")
        return all_results
    
    def process_files_parallel(self, file_pairs: List[Tuple[str, str]]) -> List[Tuple[bool, str]]:
        """并行处理文件"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config['num_workers']) as executor:
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
        """批量处理文件 - 根据模式选择处理方式"""
        
        # 检查是否使用隔离GPU模式
        if self.config.get('batch_processing_mode') == 'isolated_gpu':
            return self.process_files_isolated_gpu(file_pairs)
        
        # 原有的批处理逻辑
        results = []
        
        with tqdm(total=len(file_pairs), desc="批量处理音频文件", unit="文件") as pbar:
            for i in range(0, len(file_pairs), self.batch_size):
                batch = file_pairs[i:i + self.batch_size]
                
                # 准备批处理数据
                input_files = [pair[0] for pair in batch]
                output_files = [pair[1] for pair in batch]
                
                # 批量增强
                batch_results = self.enhance_audio_batch(input_files)
                
                # 保存结果
                for j, (success, message, enhanced_data) in enumerate(batch_results):
                    input_file = input_files[j]
                    output_file = output_files[j]
                    
                    if success:
                        try:
                            # 检查是否跳过已存在的文件
                            if self.config['skip_existing'] and os.path.exists(output_file):
                                results.append((True, f"跳过已存在文件: {input_file}"))
                                self.stats['skipped_files'] += 1
                                pbar.set_postfix({"状态": "跳过"})
                                continue
                            
                            # 确保输出目录存在
                            os.makedirs(os.path.dirname(output_file), exist_ok=True)
                            
                            # 保存音频数据
                            if enhanced_data is not None:
                                # 使用ClearVoice的write方法保存
                                clearvoice_instance = self.clearvoice_instances[0][0]
                                clearvoice_instance.write(enhanced_data, output_path=output_file)
                            
                            # 验证输出文件是否存在
                            if os.path.exists(output_file):
                                results.append((True, f"成功处理: {input_file}"))
                                self.stats['processed_files'] += 1
                                pbar.set_postfix({"状态": "成功"})
                            else:
                                results.append((False, f"输出文件未生成: {input_file}"))
                                self.stats['failed_files'] += 1
                                pbar.set_postfix({"状态": "失败"})
                            
                        except Exception as e:
                            results.append((False, f"保存失败 {input_file}: {e}"))
                            self.stats['failed_files'] += 1
                            pbar.set_postfix({"状态": "保存失败"})
                    else:
                        results.append((False, message))
                        self.stats['failed_files'] += 1
                        pbar.set_postfix({"状态": "处理失败"})
                    
                    pbar.update(1)
        
        return results
    
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
                
                # 在指定GPU上进行测试处理
                with torch.cuda.device(gpu_id) if gpu_id is not None else torch.no_grad():
                    enhanced_audio = clearvoice_instance(input_path=test_file, online_write=False)
                    
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

    def print_stats(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("处理统计")
        print("=" * 60)
        print(f"总文件数:     {self.stats['total_files']}")
        print(f"成功处理:     {self.stats['processed_files']}")
        print(f"跳过文件:     {self.stats['skipped_files']}")
        print(f"失败文件:     {self.stats['failed_files']}")
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            print(f"成功率:       {success_rate:.1f}%")
        print(f"处理时间:     {self.stats['processing_time']:.2f}秒")
        
        # 显示GPU使用情况
        if self.use_multi_gpu:
            print(f"处理模式:     隔离GPU模式" if self.config.get('batch_processing_mode') == 'isolated_gpu' else "多GPU模式")
            print(f"使用GPU数:    {len(self.available_gpus)}")
            print(f"GPU列表:      {self.available_gpus}")
        
        print("=" * 60)

def main():
    """主函数"""
    print("MossFormerGAN批量语音增强脚本")
    print("=" * 60)
    
    # 检查ClearVoice是否可用
    if not CLEARVOICE_AVAILABLE:
        print("错误: ClearVoice未安装")
        print("请执行: pip install clearvoice")
        return
    
    # 检查输入目录
    if not os.path.exists(CONFIG['input_dir']):
        print(f"错误: 输入目录不存在: {CONFIG['input_dir']}")
        return
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 创建处理器
    try:
        processor = MossFormerGANBatchProcessor(CONFIG)
        
        # 如果不是隔离GPU模式，进行模型测试
        if CONFIG.get('batch_processing_mode') != 'isolated_gpu':
            # 测试模型功能
            processor.test_model_functionality()
        
    except Exception as e:
        print(f"初始化处理器失败: {e}")
        return
    
    # 获取所有音频文件
    print("扫描音频文件...")
    audio_files = processor.get_audio_files(CONFIG['input_dir'])
    
    if not audio_files:
        print("未找到任何音频文件")
        return
    
    processor.stats['total_files'] = len(audio_files)
    print(f"找到 {len(audio_files)} 个音频文件")
    
    # 开始处理
    print(f"使用设备: {CONFIG['device']}")
    print(f"可用GPU: {processor.available_gpus if processor.available_gpus else 'CPU'}")
    print(f"多GPU模式: {'启用' if processor.use_multi_gpu else '禁用'}")
    print(f"处理模式: {CONFIG['batch_processing_mode']}")
    print(f"并行模式: {'多进程' if CONFIG.get('use_multiprocess', False) else '多线程'}")
    print(f"批处理大小: {CONFIG['batch_size']}")
    print(f"目标采样率: {CONFIG['target_sr']}Hz")
    print(f"模型: {CONFIG['model_name']}")
    
    start_time = time.time()
    
    # 选择处理方式
    if CONFIG['use_parallel']:
        print("使用并行处理模式...")
        results = processor.process_files_parallel(audio_files)
    else:
        print("使用批处理模式...")
        results = processor.process_files_batch(audio_files)
    
    # 记录处理时间
    processor.stats['processing_time'] = time.time() - start_time
    
    # 显示结果
    processor.print_stats()
    
    # 显示失败的文件
    failed_files = [message for success, message in results if not success]
    if failed_files:
        print(f"\n失败的文件 ({len(failed_files)}):")
        for i, message in enumerate(failed_files[:10]):  # 只显示前10个
            print(f"  {i+1}. {message}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files)-10} 个失败的文件")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main() 