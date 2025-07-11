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
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    'target_sr': 44100,  # 目标采样率（修改为44100以匹配模型期望）
    'skip_existing': True,  # 跳过已存在的文件
    # resemble_enhance 参数
    'nfe': 64,
    'solver': "midpoint", 
    'lambd': 1.0,
    'tau': 0.5,
    'run_dir': "/root/data/pretrained_models/resemble-enhance/enhancer_stage2",
    # 性能优化参数
    'batch_size': 16,  # 批处理大小（异步I/O批处理）
    'max_length': 4410000,  # 最大音频长度（100秒@44kHz）
    'num_workers': 4,  # I/O线程数
    'prefetch_factor': 2,  # 预取因子
    'use_mixed_precision': False,  # 使用混合精度（resemble_enhance不支持复数半精度）
    'optimize_memory': True,  # 内存优化
}

def preprocess_audio_batch(audio_files: List[Tuple[str, str]], config: dict) -> List[Tuple[str, str]]:
    """批量预处理音频文件 - 仅返回有效的文件路径"""
    processed_batch = []
    
    for input_file, output_file in audio_files:
        try:
            # 检查是否跳过已存在的文件
            if config['skip_existing'] and os.path.exists(output_file):
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

def device_safe_inference(model, dwav, sr, device, chunk_seconds: float = 30.0, overlap_seconds: float = 1.0):
    """设备安全的inference函数，修复resemble_enhance库中的设备不匹配问题"""
    import torch
    import torch.nn.functional as F
    from torch.nn.utils.parametrize import remove_parametrizations
    from torchaudio.transforms import MelSpectrogram
    from tqdm import trange
    import time
    import logging
    
    logger = logging.getLogger(__name__)
    
    def device_safe_inference_chunk(model, dwav, sr, device, npad=441):
        """设备安全的inference_chunk函数"""
        assert model.hp.wav_rate == sr, f"Expected {model.hp.wav_rate} Hz, got {sr} Hz"
        
        length = dwav.shape[-1]
        abs_max = dwav.abs().max().clamp(min=1e-7)
        
        assert dwav.dim() == 1, f"Expected 1D waveform, got {dwav.dim()}D"
        dwav = dwav.to(device)
        dwav = dwav / abs_max
        dwav = F.pad(dwav, (0, npad))
        
        with torch.no_grad():
            hwav = model(dwav[None])[0]  # 保持在GPU上，不移动到CPU
            hwav = hwav[:length]
            hwav = hwav * abs_max
        
        return hwav  # 返回GPU上的tensor
    
    def device_safe_compute_corr(x, y):
        """设备安全的compute_corr函数"""
        return torch.fft.ifft(torch.fft.fft(x) * torch.fft.fft(y).conj()).abs()
    
    def device_safe_compute_offset(chunk1, chunk2, sr=44100):
        """设备安全的compute_offset函数"""
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
        
        corr = device_safe_compute_corr(spec1, spec2)
        corr = corr.mean(dim=0)
        
        argmax = corr.argmax().item()
        
        if argmax > len(corr) // 2:
            argmax -= len(corr)
        
        offset = -argmax * hop_length
        
        return offset
    
    def device_safe_merge_chunks(chunks, chunk_length, hop_length, sr=44100, length=None):
        """设备安全的merge_chunks函数"""
        if not chunks:
            return torch.zeros(0)
        
        # 确保所有chunks在同一设备上
        device = chunks[0].device
        for i, chunk in enumerate(chunks):
            chunks[i] = chunk.to(device)
        
        signal_length = (len(chunks) - 1) * hop_length + chunk_length
        overlap_length = chunk_length - hop_length
        signal = torch.zeros(signal_length, device=device)
        
        fadein = torch.linspace(0, 1, overlap_length, device=device)
        fadein = torch.cat([fadein, torch.ones(hop_length, device=device)])
        fadeout = torch.linspace(1, 0, overlap_length, device=device)
        fadeout = torch.cat([torch.ones(hop_length, device=device), fadeout])
        
        for i, chunk in enumerate(chunks):
            start = i * hop_length
            end = start + chunk_length
            
            if len(chunk) < chunk_length:
                chunk = F.pad(chunk, (0, chunk_length - len(chunk)))
            
            if i > 0:
                pre_region = chunks[i - 1][-overlap_length:]
                cur_region = chunk[:overlap_length]
                offset = device_safe_compute_offset(pre_region, cur_region, sr=sr)
                start -= offset
                end -= offset
            
            if i == 0:
                chunk = chunk * fadeout
            elif i == len(chunks) - 1:
                chunk = chunk * fadein
            else:
                chunk = chunk * fadein * fadeout
            
            signal[start:end] += chunk[: len(signal[start:end])]
        
        signal = signal[:length]
        
        return signal
    
    # 主inference函数逻辑
    from torch.nn.utils.parametrize import remove_parametrizations
    
    def remove_weight_norm_recursively(module):
        """移除权重归一化的递归函数"""
        for _, module in module.named_modules():
            try:
                remove_parametrizations(module, "weight")
            except Exception:
                pass
    
    remove_weight_norm_recursively(model)
    
    hp = model.hp
    
    # 确保输入在正确的设备上
    dwav = dwav.to(device)
    
    # 重采样（如果需要）
    if sr != hp.wav_rate:
        import torchaudio.functional
        dwav = torchaudio.functional.resample(
            dwav,
            orig_freq=sr,
            new_freq=hp.wav_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser",
            beta=14.769656459379492,
        )
        dwav = dwav.to(device)  # 确保重采样后仍在GPU上
    
    sr = hp.wav_rate
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.perf_counter()
    
    chunk_length = int(sr * chunk_seconds)
    overlap_length = int(sr * overlap_seconds)
    hop_length = chunk_length - overlap_length
    
    chunks = []
    for start in trange(0, dwav.shape[-1], hop_length, desc="Processing chunks"):
        chunk_data = dwav[start : start + chunk_length]
        processed_chunk = device_safe_inference_chunk(model, chunk_data, sr, device)
        chunks.append(processed_chunk)
    
    hwav = device_safe_merge_chunks(chunks, chunk_length, hop_length, sr=sr, length=dwav.shape[-1])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Elapsed time: {elapsed_time:.3f} s, {hwav.shape[-1] / elapsed_time / 1000:.3f} kHz")
    
    # 最终移动到CPU
    hwav = hwav.cpu()
    
    return hwav, sr

def process_gpu_batch_resemble_optimized(args):
    """优化的GPU处理函数 - 使用批处理和异步I/O"""
    file_list, gpu_id, config, position = args
    
    # 强制设置GPU设备
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # 重新导入所需库
    import torch
    import torchaudio
    import numpy as np
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor
    from pydub import AudioSegment
    
    # 设置线程数和优化
    torch.set_num_threads(4)
    if config['optimize_memory']:
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
        run_dir = Path(config['run_dir'])
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
            nfe=config['nfe'], 
            solver=config['solver'], 
            lambd=config['lambd'], 
            tau=config['tau']
        )
        
        # 设置混合精度（resemble_enhance不支持复数半精度，暂时禁用）
        # if config['use_mixed_precision']:
        #     enhancer.half()
        
        print(f"GPU {gpu_id} 模型加载成功，开始处理 {len(file_list)} 个文件")
        
        # 创建进度条
        pbar = tqdm(total=len(file_list), desc=f"GPU {gpu_id}", position=position, leave=True)
        
        results = []
        batch_size = config['batch_size']
        
        # 使用线程池进行异步I/O
        with ThreadPoolExecutor(max_workers=config['num_workers']) as executor:
            # 批处理文件
            for i in range(0, len(file_list), batch_size):
                batch_files = file_list[i:i+batch_size]
                
                # 异步预处理音频（仅检查文件有效性）
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
                            # 在GPU进程内部加载和处理音频
                            if input_file.lower().endswith('.m4a'):
                                # 处理m4a文件
                                audio = AudioSegment.from_file(input_file)
                                mono_audio = audio.split_to_mono()[0]
                                
                                # 转换为numpy数组
                                audio_data = np.array(mono_audio.get_array_of_samples(), dtype=np.float32)
                                audio_data = audio_data / (1 << 15)  # 16位音频归一化
                                
                                # 转换为torch tensor
                                dwav = torch.from_numpy(audio_data)
                                sr = mono_audio.frame_rate
                            else:
                                # 直接加载其他格式的音频文件
                                dwav, sr = torchaudio.load(input_file)
                                
                                # 转换为单声道
                                if dwav.shape[0] > 1:
                                    dwav = dwav.mean(0)
                                else:
                                    dwav = dwav.squeeze(0)
                            
                            # 重采样到模型期望的采样率（在CPU上完成）
                            # 注意：我们始终使用模型期望的采样率，而不是配置中的target_sr
                            if sr != model_sr:
                                # 确保dwav在CPU上进行重采样
                                dwav = dwav.cpu()
                                resampler = torchaudio.transforms.Resample(sr, model_sr)
                                dwav = resampler(dwav)
                                sr = model_sr
                            
                            # 检查数据有效性
                            if torch.isnan(dwav).any() or torch.isinf(dwav).any():
                                print(f"警告: 输入数据包含NaN或Inf，跳过文件 {input_file}")
                                continue
                            
                            # 截断或填充到固定长度（在CPU上完成）
                            dwav = dwav.cpu()  # 确保在CPU上
                            if len(dwav) > config['max_length']:
                                dwav = dwav[:config['max_length']]
                            elif len(dwav) < config['max_length']:
                                dwav = torch.nn.functional.pad(dwav, (0, config['max_length'] - len(dwav)))
                            
                            # 归一化音频（在CPU上完成）
                            if dwav.abs().max() > 1.0:
                                dwav = dwav / dwav.abs().max()
                            
                            # 最后将处理好的数据移到GPU
                            dwav = dwav.to("cuda:0")
                            
                            # 确保数据在正确的设备上
                            if dwav.device != torch.device("cuda:0"):
                                print(f"警告: dwav在错误的设备上 {dwav.device}, 重新移动到cuda:0")
                                dwav = dwav.to("cuda:0")
                            
                            # 检查模型参数的设备（仅在错误时打印）
                            model_device = next(enhancer.parameters()).device
                            if model_device != dwav.device:
                                print(f"设备不匹配警告: 模型设备={model_device}, 数据设备={dwav.device}")
                                print(f"数据形状={dwav.shape}, 类型={dwav.dtype}")
                            
                            # 再次检查数据有效性
                            if torch.isnan(dwav).any() or torch.isinf(dwav).any():
                                print(f"警告: GPU上的数据包含NaN或Inf，跳过文件 {input_file}")
                                continue
                            
                            # 确保dwav是连续的tensor
                            if not dwav.is_contiguous():
                                dwav = dwav.contiguous()
                            
                            # 单个推理 - 使用设备安全的inference函数，避免设备不匹配
                            try:
                                hwav, output_sr = device_safe_inference(
                                    model=enhancer, 
                                    dwav=dwav, 
                                    sr=model_sr,
                                    device="cuda:0"
                                )
                            except Exception as e:
                                print(f"推理失败 {input_file}: {e}")
                                continue
                            
                            # 检查输出数据有效性
                            if torch.isnan(hwav).any() or torch.isinf(hwav).any():
                                print(f"警告: 输出数据包含NaN或Inf，跳过文件 {input_file}")
                                continue
                            
                            enhanced_batch.append((hwav.cpu(), output_sr))
                            batch_info.append((input_file, output_file))
                            
                        except Exception as e:
                            print(f"处理音频文件失败 {input_file}: {e}")
                            continue
                
                # 异步保存音频文件
                save_futures = []
                
                for j, (input_file, output_file) in enumerate(batch_info):
                    if j < len(enhanced_batch):
                        hwav, output_sr = enhanced_batch[j]
                        future_save = executor.submit(save_audio_file, hwav, output_file, output_sr)
                        save_futures.append((future_save, input_file))
                    else:
                        # 没有对应的增强音频，跳过
                        results.append((False, f"跳过无效文件: {input_file}"))
                
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
                if config['optimize_memory']:
                    torch.cuda.empty_cache()
        
        pbar.close()
        
        # 统计结果
        success_count = sum(1 for success, _ in results if success)
        print(f"GPU {gpu_id} 处理完成: {success_count}/{len(file_list)} 成功")
        
        return results
        
    except Exception as e:
        print(f"GPU {gpu_id} 进程初始化失败: {e}")
        return [(False, f"进程初始化失败: {e}") for _ in file_list]

def save_audio_file(hwav, output_file, sr):
    """保存音频文件的辅助函数"""
    try:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 保存增强后的音频
        if hwav.dim() == 1:
            hwav = hwav.unsqueeze(0)
        
        torchaudio.save(output_file, hwav, sr)
        return True
    except Exception as e:
        raise Exception(f"保存音频文件失败: {e}")

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
            results = pool.map(process_gpu_batch_resemble_optimized, args_list)
            
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
        print("\n" + "=" * 70)
        print("处理统计")
        print("=" * 70)
        print(f"总文件数:     {self.stats['total_files']}")
        print(f"成功处理:     {self.stats['processed_files']}")
        print(f"跳过文件:     {self.stats['skipped_files']}")
        print(f"失败文件:     {self.stats['failed_files']}")
        if self.stats['total_files'] > 0:
            success_rate = (self.stats['processed_files'] / self.stats['total_files']) * 100
            print(f"成功率:       {success_rate:.1f}%")
        print(f"处理时间:     {self.stats['processing_time']:.2f}秒")
        
        # 计算性能指标
        if self.stats['processing_time'] > 0:
            files_per_sec = self.stats['processed_files'] / self.stats['processing_time']
            print(f"处理速度:     {files_per_sec:.2f} 文件/秒")
        
        print(f"处理模式:     优化I/O批处理模式")
        print(f"使用GPU数:    {len(self.available_gpus)}")
        print(f"GPU列表:      {self.available_gpus}")
        print(f"批处理大小:   {self.config['batch_size']} (I/O批处理)")
        print(f"I/O线程数:    {self.config['num_workers']}")
        print(f"混合精度:     {'开启' if self.config['use_mixed_precision'] else '关闭（不支持复数半精度）'}")
        print(f"内存优化:     {'开启' if self.config['optimize_memory'] else '关闭'}")
        print("=" * 70)

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
        print("! 建议调整batch_size到32-64来充分利用GPU")
    else:
        print("! 单GPU环境，建议增加batch_size到32-64")
    
    if memory_gb >= 64:
        print("✓ 内存充足，可以增加batch_size到64和num_workers到8")
    elif memory_gb >= 32:
        print("! 内存适中，当前配置合适")
    else:
        print("! 内存较少，建议减少batch_size到8-16")
    
    if cpu_count >= 16:
        print("✓ CPU核心充足，可以增加num_workers到8-16")
    elif cpu_count >= 8:
        print("! CPU核心适中，当前num_workers设置合适")
    else:
        print("! CPU核心较少，建议减少num_workers到2-4")
    
    print("=" * 60)

def main():
    """主函数"""
    print("Resemble-Enhance批量语音增强脚本（优化版）")
    print("=" * 60)
    
    # 性能诊断
    diagnose_performance()
    
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
    
    print("\n优化参数:")
    print(f"批处理大小: {CONFIG['batch_size']} (I/O批处理)")
    print(f"最大音频长度: {CONFIG['max_length']} 样本 ({CONFIG['max_length']/CONFIG['target_sr']:.1f}秒)")
    print(f"I/O线程数: {CONFIG['num_workers']}")
    print(f"混合精度: {'开启' if CONFIG['use_mixed_precision'] else '关闭（resemble_enhance不支持复数半精度）'}")
    print(f"内存优化: {'开启' if CONFIG['optimize_memory'] else '关闭'}")
    
    start_time = time.time()
    
    # 处理文件
    print("\n开始优化批量处理...")
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
    
    # 性能提示
    if processor.stats['processing_time'] > 0:
        total_processed = processor.stats['processed_files']
        processing_time = processor.stats['processing_time']
        files_per_sec = total_processed / processing_time
        
        print(f"\n性能提示:")
        if files_per_sec < 1:
            print("• 处理速度较慢，建议：")
            print("  - 增加batch_size到32-64")
            print("  - 增加num_workers到6-8")
            print("  - 减少音频最大长度 (max_length)")
        elif files_per_sec < 5:
            print("• 处理速度适中，可以考虑：")
            print("  - 增加I/O线程数 (num_workers)")
            print("  - 调整batch_size")
        else:
            print("• 处理速度良好！")
    
    print("\n处理完成！")

if __name__ == "__main__":
    main()
