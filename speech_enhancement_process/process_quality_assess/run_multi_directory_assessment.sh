#!/bin/bash

# 音频质量评估脚本 - 多目录处理示例
# 
# 使用方法：
# 1. 修改下面的路径配置
# 2. 确保LLM服务已启动（如果使用LLM文本标准化）
# 3. 运行脚本：bash run_multi_directory_assessment.sh

# 设置基本路径
BASE_DIR="/root/group-shared/voiceprint/data/speech/speaker_verification"
SCRIPT_DIR="/root/code/github_repos/DataProcessor/speech_enhancement_process/process_quality_assess"

# 检查LLM服务是否运行（可选）
echo "检查LLM服务状态..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "警告: LLM服务未运行，将使用基础文本标准化"
    USE_LLM="false"
else
    echo "✓ LLM服务正常运行"
    USE_LLM="true"
fi

# 方法1: 使用配置文件（推荐）
echo "方法1: 使用配置文件运行..."
cd $SCRIPT_DIR
python enhancement_audio_quality_assessment.py \
    --config_file multi_directory_config_example.json

# 方法2: 使用命令行参数
echo "方法2: 使用命令行参数运行..."
# cd $SCRIPT_DIR
# python enhancement_audio_quality_assessment.py \
#     --original_dirs \
#         "${BASE_DIR}/King-ASR-EN-Kid" \
#         "${BASE_DIR}/King-ASR-EN-Kid" \
#         "${BASE_DIR}/King-ASR-EN-Kid" \
#     --enhanced_dirs \
#         "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced" \
#         "${BASE_DIR}/King-ASR-EN-Kid_demucs_enhanced" \
#         "${BASE_DIR}/King-ASR-EN-Kid_speechbrain_enhanced" \
#     --output_dirs \
#         "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment" \
#         "${BASE_DIR}/King-ASR-EN-Kid_demucs_enhanced_quality_assessment" \
#         "${BASE_DIR}/King-ASR-EN-Kid_speechbrain_enhanced_quality_assessment" \
#     --config_names \
#         "zipenhancer" \
#         "demucs" \
#         "speechbrain" \
#     --volume_matching \
#         "true" \
#         "true" \
#         "false" \
#     --volume_methods \
#         "ten_vad_energy" \
#         "ten_vad_energy" \
#         "ten_vad_energy" \
#     --num_gpus 3 \
#     --gpu_ids 1 2 3 \
#     --text_normalization "llm" \
#     --use_llm_normalization \
#     --llm_service_url "http://localhost:8000" \
#     --ten_vad_hop_size 256 \
#     --ten_vad_threshold 0.5 \
#     --skip_existing

# 方法3: 单目录模式（向后兼容）
echo "方法3: 单目录模式运行..."
# cd $SCRIPT_DIR
# python enhancement_audio_quality_assessment.py \
#     --original_dir "${BASE_DIR}/King-ASR-EN-Kid" \
#     --enhanced_dir "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced" \
#     --output_dir "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment" \
#     --volume_method "ten_vad_energy" \
#     --num_gpus 3 \
#     --gpu_ids 1 2 3 \
#     --text_normalization "llm" \
#     --use_llm_normalization \
#     --llm_service_url "http://localhost:8000" \
#     --ten_vad_hop_size 256 \
#     --ten_vad_threshold 0.5 \
#     --skip_existing

echo "评估完成！"
echo "检查输出目录以查看结果。" 