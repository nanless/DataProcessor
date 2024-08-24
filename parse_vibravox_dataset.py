import os
import struct
import wave
import io
import pandas as pd
import numpy as np
from tqdm import tqdm

def parse_wave_bytes(byte_data, save_path):
    # 创建一个字节流对象
    byte_stream = io.BytesIO(byte_data)

    # 解析 RIFF 头
    riff_header = byte_stream.read(12)
    riff_id, file_size, wave_id = struct.unpack('<4sI4s', riff_header)

    print(f"RIFF ID: {riff_id}")
    print(f"File Size: {file_size}")
    print(f"WAVE ID: {wave_id}")

    if riff_id != b'RIFF' or wave_id != b'WAVE':
        raise ValueError("Not a valid WAV file")

    # 继续解析 FMT 子块
    fmt_header = byte_stream.read(24)
    fmt_id, fmt_size, audio_format, num_channels, sample_rate, byte_rate, block_align, bits_per_sample = struct.unpack('<4sIHHIIHH', fmt_header)

    print(f"\nFMT ID: {fmt_id}")
    print(f"FMT Size: {fmt_size}")
    print(f"Audio Format: {audio_format}")
    print(f"Number of Channels: {num_channels}")
    print(f"Sample Rate: {sample_rate}")
    print(f"Byte Rate: {byte_rate}")
    print(f"Block Align: {block_align}")
    print(f"Bits per Sample: {bits_per_sample}")

    # 解析 DATA 子块
    # 首先跳过 Fact 和 PEAK 子块
    while True:
        chunk_header = byte_stream.read(8)
        if len(chunk_header) < 8:
            break
        chunk_id, chunk_size = struct.unpack('<4sI', chunk_header)
        
        if chunk_id == b'data':
            data_size = chunk_size
            audio_data = byte_stream.read(data_size)
            print(f"\nDATA ID: {chunk_id}")
            print(f"Data Size: {data_size}")
            break
        else:
            print(f"\nSkipping chunk {chunk_id}, Size: {chunk_size}")
            byte_stream.seek(chunk_size, io.SEEK_CUR)

    # 检查并处理音频数据格式
    if audio_format == 3:  # 如果音频格式为 32-bit float
        audio_samples = np.frombuffer(audio_data, dtype=np.float32)
        # 将浮点数据转换为 16-bit PCM 数据
        audio_samples = np.int16(audio_samples * 32767)  # 浮点数 [-1, 1] 范围转换为 [-32767, 32767]
        audio_data = audio_samples.tobytes()

        # 更新采样宽度为 16-bit
        bits_per_sample = 16
        byte_rate = sample_rate * num_channels * bits_per_sample // 8
        block_align = num_channels * bits_per_sample // 8

    # 将音频数据写入新的 WAV 文件
    with wave.open(save_path, 'wb') as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(bits_per_sample // 8)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)

    print(f"\nWAV file created as '{save_path}'")

original_folder = "/data1/data/speech/vibravox/speech_clean"
save_folder = "/data1/data/speech/vibravox/speech_clean_wav"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

for file_name in tqdm(os.listdir(original_folder)):
    if file_name.endswith(".parquet"):
        df = pd.read_parquet(os.path.join(original_folder, file_name))
        for i in range(len(df)):
            audio_bytes = df.iloc[i]['audio.headset_microphone']['bytes']
            save_name = df.iloc[i]['audio.headset_microphone']['path']
            save_path = os.path.join(save_folder, save_name)
            parse_wave_bytes(audio_bytes, save_path)

# df = pd.read_parquet("/data1/data/speech/vibravox/speech_clean/train-00165-of-00219.parquet")

# print(df.head())
# print(df.columns)
# print(df.iloc[2]['audio.headset_microphone'].keys())
# print(df.iloc[2]['audio.headset_microphone']['path'])
# print(df.iloc[2]['audio.headset_microphone']['bytes'][:100])
# print(df.iloc[2]['audio.headset_microphone']['bytes'][-100:])
# parse_wave_bytes(df.iloc[2]['audio.headset_microphone']['bytes'])

