import os
from pydub import AudioSegment
from resemble_enhance.enhancer.inference import inference
from resemble_enhance.enhancer.download import download
from resemble_enhance.enhancer.train import Enhancer, HParams
import torchaudio
from tqdm import tqdm
import torch

# 设置输入和输出目录
input_dir = "/mnt/kemove_data1/data/speech/crawled/temp/temp_ximalaya_for_process"
output_dir = "/mnt/kemove_data1/data/speech/crawled/temp/temp_ximalaya_for_process_enhanced"
device = "cuda"  # 使用cuda进行增强
nfe=64
solver="midpoint"
lambd=1.0
tau=0.5
run_dir="downloaded_models"

def walk_m4a_files(input_dir):
    """
    遍历目录中的所有m4a文件
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".m4a"):
                yield os.path.join(root, file)

def load_enhancer(run_dir: str, device):
    run_dir = download(run_dir)
    hp = HParams.load(run_dir)
    enhancer = Enhancer(hp)
    path = run_dir / "ds" / "G" / "default" / "mp_rank_00_model_states.pt"
    state_dict = torch.load(path, map_location="cpu")["module"]
    enhancer.load_state_dict(state_dict)
    enhancer.eval()
    enhancer.to(device)
    return enhancer


if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 遍历目录中的所有m4a文件
for file_path in tqdm(walk_m4a_files(input_dir)):
    if file_path.endswith(".m4a"):
        # 转换为单通道wav文件，取第一个通道
        audio = AudioSegment.from_file(file_path)
        mono_audio = audio.split_to_mono()[0]  # 取第一个通道
        wav_file_path = file_path.replace(".m4a", ".wav")
        mono_audio.export(wav_file_path, format="wav")
        
        # 使用resemble-enhance进行降噪和增强
        enhanced_wav_file_path = wav_file_path.replace(input_dir, output_dir)
        if not os.path.exists(os.path.dirname(enhanced_wav_file_path)):
            os.makedirs(os.path.dirname(enhanced_wav_file_path))
        dwav, sr = torchaudio.load(wav_file_path)
        dwav = dwav.mean(0)
        enhancer = load_enhancer(run_dir, device)
        enhancer.configurate_(nfe=nfe, solver=solver, lambd=lambd, tau=tau)
        hwav, sr = inference(model=enhancer, dwav=dwav, sr=sr, device=device)
        torchaudio.save(enhanced_wav_file_path, hwav.unsqueeze(0), sr)
        

print("All files processed.")
