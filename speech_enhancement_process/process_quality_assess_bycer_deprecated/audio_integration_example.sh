#!/bin/bash

# 音频整合脚本使用示例
# 演示如何使用音频整合脚本来选择最佳音频版本

# 设置路径变量
BASE_DIR="/root/group-shared/voiceprint/data/speech/speaker_verification"
SCRIPT_DIR="/root/code/github_repos/DataProcessor/speech_enhancement_process/process_quality_assess"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    音频整合脚本使用示例${NC}"
echo -e "${BLUE}========================================${NC}"

cd "$SCRIPT_DIR"

echo -e "\n${YELLOW}示例1: 比较两种降噪方法${NC}"
echo "比较ZipEnhancer和MossFormer两种降噪方法的效果"

# 示例1: 基本使用
echo -e "\n${GREEN}命令:${NC}"
cat << 'EOF'
bash run_audio_integration.sh \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dirs \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced" \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_mossformer_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated"
EOF

echo -e "\n${YELLOW}示例2: 使用严格的CER阈值${NC}"
echo "设置更严格的CER阈值（3%），只选择高质量的降噪音频"

echo -e "\n${GREEN}命令:${NC}"
cat << 'EOF'
bash run_audio_integration.sh \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dirs \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_resemble_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_strict" \
    --cer_threshold 0.03
EOF

echo -e "\n${YELLOW}示例3: 禁用优先原音频${NC}"
echo "总是选择最佳的降噪版本，不考虑原音频偏好"

echo -e "\n${GREEN}命令:${NC}"
cat << 'EOF'
bash run_audio_integration.sh \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dirs \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_enhanced_only" \
    --no_prefer_original
EOF

echo -e "\n${YELLOW}示例4: 多模型综合比较${NC}"
echo "同时比较三种不同的降噪方法"

echo -e "\n${GREEN}命令:${NC}"
cat << 'EOF'
bash run_audio_integration.sh \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dirs \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced" \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_mossformer_enhanced" \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_resemble_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated_multi"
EOF

echo -e "\n${YELLOW}使用前提醒:${NC}"
echo "1. 确保已经运行过质量评估脚本，生成了CER数据"
echo "2. 检查对应的 *_quality_assessment 目录是否存在"
echo "3. 确认有足够的磁盘空间用于整合后的音频文件"

echo -e "\n${YELLOW}查看结果:${NC}"
echo "整合完成后，检查以下文件："
echo "• 整合后的音频文件（保持原有目录结构）"
echo "• audio_integration_records.json（详细的选择记录）"
echo "• integration_summary.json（统计摘要）"

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}    示例展示完成${NC}"
echo -e "${BLUE}========================================${NC}"

# 如果用户想要运行第一个示例
echo -e "\n是否运行示例1？(需要确保相关目录和评估结果存在) [y/N]:"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo -e "\n${GREEN}运行示例1...${NC}"
    
    # 检查必要的目录是否存在
    original_dir="$BASE_DIR/King-ASR-EN-Kid"
    enhanced_dir1="$BASE_DIR/King-ASR-EN-Kid_zipenhancer_enhanced"
    enhanced_dir2="$BASE_DIR/King-ASR-EN-Kid_mossformer_enhanced"
    output_dir="$BASE_DIR/King-ASR-EN-Kid_integrated"
    
    if [ ! -d "$original_dir" ]; then
        echo -e "${RED}错误: 原音频目录不存在: $original_dir${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}检查目录和评估结果...${NC}"
    available_dirs=()
    
    for enhanced_dir in "$enhanced_dir1" "$enhanced_dir2"; do
        if [ -d "$enhanced_dir" ]; then
            assessment_dir="${enhanced_dir}_quality_assessment"
            if [ -f "$assessment_dir/quality_assessment_results.json" ]; then
                available_dirs+=("$enhanced_dir")
                echo -e "${GREEN}✓ 找到: $(basename "$enhanced_dir")${NC}"
            else
                echo -e "${YELLOW}⚠ 缺少评估结果: $(basename "$enhanced_dir")${NC}"
            fi
        else
            echo -e "${YELLOW}⚠ 目录不存在: $(basename "$enhanced_dir")${NC}"
        fi
    done
    
    if [ ${#available_dirs[@]} -eq 0 ]; then
        echo -e "${RED}错误: 没有找到可用的降噪目录${NC}"
        exit 1
    fi
    
    # 运行整合脚本
    bash run_audio_integration.sh \
        --original_dir "$original_dir" \
        --enhanced_dirs "${available_dirs[@]}" \
        --output_dir "$output_dir"
        
else
    echo -e "\n${YELLOW}您可以根据需要修改路径后手动运行上述示例${NC}"
fi 