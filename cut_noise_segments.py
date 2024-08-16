import os
import argparse
from pydub import AudioSegment
from tqdm import tqdm

def slice_audio(input_dir, output_dir, duration):
    files_to_process = []
    
    # 先收集所有需要处理的文件
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".wav") or file.endswith(".flac"):
                files_to_process.append(os.path.join(root, file))
    
    # 处理并显示进度条
    for file_path in tqdm(files_to_process, desc="Processing files"):
        try:
            audio = AudioSegment.from_file(file_path)
            file_duration = len(audio) / 1000  # Convert to seconds
            file_name, file_ext = os.path.splitext(os.path.basename(file_path))
            rel_dir = os.path.relpath(os.path.dirname(file_path), input_dir)
            output_subdir = os.path.join(output_dir, rel_dir)
            os.makedirs(output_subdir, exist_ok=True)
            
            for i in range(0, int(file_duration), duration):
                start = i * 1000
                end = min((i + duration) * 1000, len(audio))
                audio_slice = audio[start:end]
                output_file_path = os.path.join(output_subdir, f"{file_name}_slice{i//duration}{file_ext}")
                audio_slice.export(output_file_path, format=file_ext[1:]) 
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Slice .wav or .flac files into smaller segments.")
    parser.add_argument("--input_dir", type=str, help="Input directory containing .wav or .flac files.")
    parser.add_argument("--output_dir", type=str, help="Output directory to save sliced audio files.")
    parser.add_argument("--duration", type=int, default=10, help="Duration of each slice in seconds (default: 10).")
    
    args = parser.parse_args()
    
    slice_audio(args.input_dir, args.output_dir, args.duration)

if __name__ == "__main__":
    main()
