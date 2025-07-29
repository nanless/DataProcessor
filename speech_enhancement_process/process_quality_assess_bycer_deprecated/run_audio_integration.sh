#!/bin/bash

# 音频整合脚本 - 基于CER选择最佳音频版本
# 
# 使用方法：
# bash run_audio_integration.sh --original_dir 原音频目录 --enhanced_dirs 降噪目录1 降噪目录2 --output_dir 输出目录

# 设置基本路径
BASE_DIR="/root/group-shared/voiceprint/data/speech/speaker_verification"
SCRIPT_DIR="/root/code/github_repos/DataProcessor/speech_enhancement_process/process_quality_assess"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认参数
DEFAULT_CER_THRESHOLD=0.05
DEFAULT_PREFER_ORIGINAL=true

# 显示帮助信息
show_help() {
    echo "音频整合脚本 - 基于CER选择最佳音频版本"
    echo ""
    echo "用法:"
    echo "  $0 --original_dir <原音频目录> --enhanced_dirs <降噪目录1> [降噪目录2] [降噪目录3] --output_dir <输出目录> [选项]"
    echo ""
    echo "必需参数:"
    echo "  --original_dir      原音频目录路径"
    echo "  --enhanced_dirs     降噪音频目录路径列表（可指定多个）"
    echo "  --output_dir        整合后音频输出目录"
    echo ""
    echo "可选参数:"
    echo "  --cer_threshold     CER阈值，低于此值认为音频可用 (默认: $DEFAULT_CER_THRESHOLD)"
    echo "  --prefer_original   当CER相近时优先选择原音频 (默认: $DEFAULT_PREFER_ORIGINAL)"
    echo "  --no_prefer_original 禁用优先选择原音频"
    echo "  -h, --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  # 基本使用 - 比较两种降噪方法"
    echo "  $0 \\"
    echo "    --original_dir \"$BASE_DIR/King-ASR-EN-Kid\" \\"
    echo "    --enhanced_dirs \"$BASE_DIR/King-ASR-EN-Kid_zipenhancer_enhanced\" \"$BASE_DIR/King-ASR-EN-Kid_mossformer_enhanced\" \\"
    echo "    --output_dir \"$BASE_DIR/King-ASR-EN-Kid_integrated\""
    echo ""
    echo "  # 使用自定义CER阈值"
    echo "  $0 \\"
    echo "    --original_dir \"$BASE_DIR/King-ASR-EN-Kid\" \\"
    echo "    --enhanced_dirs \"$BASE_DIR/King-ASR-EN-Kid_resemble_enhanced\" \\"
    echo "    --output_dir \"$BASE_DIR/King-ASR-EN-Kid_integrated\" \\"
    echo "    --cer_threshold 0.03"
    echo ""
    echo "  # 禁用优先原音频"
    echo "  $0 \\"
    echo "    --original_dir \"$BASE_DIR/King-ASR-EN-Kid\" \\"
    echo "    --enhanced_dirs \"$BASE_DIR/King-ASR-EN-Kid_zipenhancer_enhanced\" \\"
    echo "    --output_dir \"$BASE_DIR/King-ASR-EN-Kid_integrated\" \\"
    echo "    --no_prefer_original"
    echo ""
    echo "注意："
    echo "  - 脚本会自动查找对应的质量评估结果文件（*_quality_assessment目录）"
    echo "  - 支持多个降噪模型比较，会自动选择CER最低的版本"
    echo "  - 整合后的音频会保持原有的目录结构"
    echo "  - 详细的选择记录会保存在输出目录的JSON文件中"
}

# 解析命令行参数
ORIGINAL_DIR=""
ENHANCED_DIRS=()
OUTPUT_DIR=""
CER_THRESHOLD=$DEFAULT_CER_THRESHOLD
PREFER_ORIGINAL=$DEFAULT_PREFER_ORIGINAL

while [[ $# -gt 0 ]]; do
    case $1 in
        --original_dir)
            ORIGINAL_DIR="$2"
            shift 2
            ;;
        --enhanced_dirs)
            shift
            # 收集所有降噪目录，直到遇到下一个选项或结束
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                ENHANCED_DIRS+=("$1")
                shift
            done
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --cer_threshold)
            CER_THRESHOLD="$2"
            shift 2
            ;;
        --prefer_original)
            PREFER_ORIGINAL=true
            shift
            ;;
        --no_prefer_original)
            PREFER_ORIGINAL=false
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 验证必需参数
if [ -z "$ORIGINAL_DIR" ]; then
    echo -e "${RED}错误: 必须指定 --original_dir 参数${NC}"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

if [ ${#ENHANCED_DIRS[@]} -eq 0 ]; then
    echo -e "${RED}错误: 必须指定至少一个 --enhanced_dirs 参数${NC}"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

if [ -z "$OUTPUT_DIR" ]; then
    echo -e "${RED}错误: 必须指定 --output_dir 参数${NC}"
    echo "使用 -h 或 --help 查看帮助"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    音频整合脚本 - 基于CER选择${NC}"
echo -e "${BLUE}========================================${NC}"

# 切换到脚本目录
cd "$SCRIPT_DIR" || {
    echo -e "${RED}错误: 无法切换到脚本目录 $SCRIPT_DIR${NC}"
    exit 1
}

# 激活conda环境
echo -e "${YELLOW}激活conda环境...${NC}"
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 无法激活conda环境 kimi-audio${NC}"
    exit 1
fi

echo -e "${GREEN}✓ conda环境已激活${NC}"

# 显示配置信息
echo -e "\n${YELLOW}配置信息:${NC}"
echo -e "原音频目录:       $ORIGINAL_DIR"
echo -e "降噪目录数量:     ${#ENHANCED_DIRS[@]}"
for i in "${!ENHANCED_DIRS[@]}"; do
    echo -e "  降噪目录$((i+1)):     ${ENHANCED_DIRS[i]}"
done
echo -e "输出目录:         $OUTPUT_DIR"
echo -e "CER阈值:          $CER_THRESHOLD"
echo -e "优先原音频:       $PREFER_ORIGINAL"

# 验证输入目录是否存在
echo -e "\n${YELLOW}验证输入目录...${NC}"

if [ ! -d "$ORIGINAL_DIR" ]; then
    echo -e "${RED}错误: 原音频目录不存在: $ORIGINAL_DIR${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 原音频目录存在${NC}"

for enhanced_dir in "${ENHANCED_DIRS[@]}"; do
    if [ ! -d "$enhanced_dir" ]; then
        echo -e "${RED}错误: 降噪音频目录不存在: $enhanced_dir${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 降噪目录存在: $(basename "$enhanced_dir")${NC}"
done

# 检查质量评估结果是否存在
echo -e "\n${YELLOW}检查质量评估结果...${NC}"
assessment_found=false

for enhanced_dir in "${ENHANCED_DIRS[@]}"; do
    assessment_dir="${enhanced_dir}_quality_assessment"
    if [ -d "$assessment_dir" ]; then
        if [ -f "$assessment_dir/quality_assessment_results.json" ]; then
            echo -e "${GREEN}✓ 找到评估结果: $(basename "$assessment_dir")${NC}"
            assessment_found=true
        else
            echo -e "${YELLOW}⚠ 评估结果目录存在但缺少结果文件: $(basename "$assessment_dir")${NC}"
        fi
    else
        echo -e "${YELLOW}⚠ 未找到评估结果目录: $(basename "$assessment_dir")${NC}"
    fi
done

if [ "$assessment_found" = false ]; then
    echo -e "${RED}错误: 没有找到任何质量评估结果${NC}"
    echo -e "${YELLOW}提示: 请先运行质量评估脚本生成评估结果${NC}"
    exit 1
fi

# 构建Python脚本参数
PYTHON_ARGS=(
    "--original_dir" "$ORIGINAL_DIR"
    "--enhanced_dirs" "${ENHANCED_DIRS[@]}"
    "--output_dir" "$OUTPUT_DIR"
    "--cer_threshold" "$CER_THRESHOLD"
)

if [ "$PREFER_ORIGINAL" = "true" ]; then
    PYTHON_ARGS+=("--prefer_original")
else
    PYTHON_ARGS+=("--no_prefer_original")
fi

# 运行音频整合脚本
echo -e "\n${YELLOW}开始音频整合...${NC}"
echo -e "${BLUE}========================================${NC}"

python audio_integration_by_cer.py "${PYTHON_ARGS[@]}"

INTEGRATION_EXIT_CODE=$?

# 显示结果
echo -e "\n${BLUE}========================================${NC}"

if [ $INTEGRATION_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ 音频整合完成！${NC}"
    echo -e "\n${YELLOW}结果文件:${NC}"
    echo -e "📁 整合音频目录: $OUTPUT_DIR"
    
    if [ -f "$OUTPUT_DIR/audio_integration_records.json" ]; then
        echo -e "📄 详细记录文件: $OUTPUT_DIR/audio_integration_records.json"
    fi
    
    if [ -f "$OUTPUT_DIR/integration_summary.json" ]; then
        echo -e "📊 统计摘要文件: $OUTPUT_DIR/integration_summary.json"
    fi
    
    # 显示基本统计信息
    if [ -f "$OUTPUT_DIR/integration_summary.json" ]; then
        echo -e "\n${YELLOW}快速统计:${NC}"
        if command -v jq &> /dev/null; then
            total_files=$(jq -r '.statistics.total_files // 0' "$OUTPUT_DIR/integration_summary.json")
            original_selected=$(jq -r '.statistics.original_selected // 0' "$OUTPUT_DIR/integration_summary.json")
            enhanced_selected=$(jq -r '.statistics.enhanced_selected // 0' "$OUTPUT_DIR/integration_summary.json")
            
            echo -e "  总文件数: $total_files"
            echo -e "  使用原音频: $original_selected"
            echo -e "  使用降噪音频: $enhanced_selected"
        else
            echo -e "  (安装jq工具可显示详细统计信息)"
        fi
    fi
    
else
    echo -e "${RED}❌ 音频整合失败！${NC}"
    echo -e "\n${YELLOW}故障排除:${NC}"
    echo -e "• 检查输入目录是否存在"
    echo -e "• 确认质量评估结果文件完整"
    echo -e "• 检查磁盘空间是否足够"
    echo -e "• 查看错误日志以获取详细信息"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    脚本执行完成${NC}"
echo -e "${BLUE}========================================${NC}"

exit $INTEGRATION_EXIT_CODE 