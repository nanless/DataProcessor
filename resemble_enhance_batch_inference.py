import os
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
import torch
import torchaudio
from pydub import AudioSegment
import soundfile as sf
import numpy as np
from typing import List, Tuple
import logging
import time

# 设置multiprocessing启动方法为spawn（CUDA多进程必需）
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # 如果已经设置过，会抛出RuntimeError，可以忽略
    pass

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    'input_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid",
    'output_dir': "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_resemble_enhanced",
    'device': "cuda",
    'gpu_ids': [0, 1, 2, 3],  # 使用的GPU ID列表
    'target_sr': 16000,  # 目标采样率
    'skip_existing': True,  # 跳过已存在的文件
    # resemble_enhance 参数
    'nfe': 64,
    'solver': "midpoint", 
    'lambd': 1.0,
    'tau': 0.5,
    'run_dir': "/root/data/pretrained_models/resemble-enhance/enhancer_stage2"
}

def process_gpu_batch_resemble(args):
    """完全隔离的GPU处理函数 - 使用resemble_enhance"""
    file_list, gpu_id, config, position = args
    
    # 强制设置GPU设备 - 完全隔离
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 重新导入所需库
    import torch
    import torchaudio
    import numpy as np
    import soundfile as sf
    from tqdm import tqdm
    from pydub import AudioSegment
    from pathlib import Path
    
    # 只设置线程数，避免interop threads的错误
    torch.set_num_threads(8)
    
    # 现在GPU 0对应于实际的指定GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        device_name = torch.cuda.get_device_name(0)
        print(f"GPU {gpu_id} 进程启动，设备: {device_name}")
    
    # 在进程中导入resemble_enhance
    try:
        from resemble_enhance.enhancer.inference import inference
        from resemble_enhance.enhancer.download import download
        from resemble_enhance.enhancer.train import Enhancer, HParams
        
        # 加载模型
        run_dir = Path(config['run_dir'])
        hp = HParams.load(run_dir)
        enhancer = Enhancer(hp)
        path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
        state_dict = torch.load(path, map_location="cpu")["module"]
        enhancer.load_state_dict(state_dict)
        enhancer.eval()
        enhancer.to("cuda:0")  # 使用cuda:0，因为CUDA_VISIBLE_DEVICES已设置
        
        # 配置增强参数
        enhancer.configurate_(
            nfe=config['nfe'], 
            solver=config['solver'], 
            lambd=config['lambd'], 
            tau=config['tau']
        )
        
        print(f"GPU {gpu_id} 模型加载成功，开始处理 {len(file_list)} 个文件")
        
        # 创建进度条
        pbar = tqdm(file_list, desc=f"GPU {gpu_id}", position=position, leave=True)
        
        results = []
        for i, (input_file, output_file) in enumerate(pbar):
            try:
                # 检查是否跳过已存在的文件
                if config['skip_existing'] and os.path.exists(output_file):
                    results.append((True, f"跳过已存在文件: {input_file}"))
                    continue
                
                # 确保输出目录存在
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # 读取音频文件
                if input_file.lower().endswith('.m4a'):
                    # 处理m4a文件
                    audio = AudioSegment.from_file(input_file)
                    mono_audio = audio.split_to_mono()[0]  # 取第一个通道
                    
                    # 创建临时wav文件
                    temp_wav = input_file.replace('.m4a', '_temp.wav')
                    mono_audio.export(temp_wav, format="wav")
                    
                    # 加载wav文件
                    dwav, sr = torchaudio.load(temp_wav)
                    
                    # 清理临时文件
                    os.remove(temp_wav)
                else:
                    # 直接加载其他格式的音频文件
                    dwav, sr = torchaudio.load(input_file)
                
                # 转换为单声道
                if dwav.shape[0] > 1:
                    dwav = dwav.mean(0)
                else:
                    dwav = dwav.squeeze(0)
                
                # 重采样到目标采样率
                if sr != config['target_sr']:
                    resampler = torchaudio.transforms.Resample(sr, config['target_sr'])
                    dwav = resampler(dwav)
                    sr = config['target_sr']
                
                # 使用resemble_enhance进行增强
                hwav, sr = inference(model=enhancer, dwav=dwav, sr=sr, device="cuda:0")
                
                # 保存增强后的音频
                torchaudio.save(output_file, hwav.unsqueeze(0), sr)
                
                results.append((True, f"成功处理: {input_file}"))
                pbar.set_postfix({"状态": "成功"})
                
            except Exception as e:
                results.append((False, f"处理失败 {input_file}: {e}"))
                pbar.set_postfix({"状态": "异常"})
                logger.error(f"GPU {gpu_id} 处理失败 {input_file}: {e}")
        
        pbar.close()
        
        # 统计结果
        success_count = sum(1 for success, _ in results if success)
        print(f"GPU {gpu_id} 处理完成: {success_count}/{len(file_list)} 成功")
        
        return results
        
    except Exception as e:
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        return [(False, f"进程初始化失败: {e}") for _ in file_list]

class ResembleEnhancerBatchProcessor:
    """Resemble-Enhance批量处理器"""
    
    def __init__(self, config):
        self.config = config
        self.gpu_ids = config['gpu_ids']
        
        # 检查GPU可用性
        if torch.cuda.is_available():
            self.available_gpus = [i for i in self.gpu_ids if i < torch.cuda.device_count()]
            if not self.available_gpus:
                logger.error("指定的GPU不可用")
                raise RuntimeError("没有可用的GPU")
            else:
                logger.info(f"可用GPU: {self.available_gpus}")
        else:
            logger.error("CUDA不可用")
            raise RuntimeError("CUDA不可用")
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'failed_files': 0,
            'processing_time': 0,
        }
    
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
        logger.info(f"开始隔离GPU处理，文件总数: {len(file_pairs)}")
        logger.info(f"使用GPU: {self.available_gpus}")
        
        # 将文件分成4等份
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
                args = (gpu_files, gpu_id, self.config, position)
                args_list.append(args)
            
            # 提交任务
            results = pool.map(process_gpu_batch_resemble, args_list)
            
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
        print(f"处理模式:     隔离GPU模式")
        print(f"使用GPU数:    {len(self.available_gpus)}")
        print(f"GPU列表:      {self.available_gpus}")
        print("=" * 60)

def main():
    """主函数"""
    print("Resemble-Enhance批量语音增强脚本")
    print("=" * 60)
    
    # 检查输入目录
    if not os.path.exists(CONFIG['input_dir']):
        print(f"错误: 输入目录不存在: {CONFIG['input_dir']}")
        return
    
    # 创建输出目录
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    
    # 创建处理器
    try:
        processor = ResembleEnhancerBatchProcessor(CONFIG)
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
    print(f"可用GPU: {processor.available_gpus}")
    print(f"目标采样率: {CONFIG['target_sr']}Hz")
    print(f"NFE: {CONFIG['nfe']}")
    print(f"Solver: {CONFIG['solver']}")
    print(f"Lambda: {CONFIG['lambd']}")
    print(f"Tau: {CONFIG['tau']}")
    
    start_time = time.time()
    
    # 处理文件
    print("开始批量处理...")
    results = processor.process_files_isolated_gpu(audio_files)
    
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
