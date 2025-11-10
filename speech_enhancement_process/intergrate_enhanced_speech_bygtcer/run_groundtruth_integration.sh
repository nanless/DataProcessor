#!/bin/bash

# 基于Groundtruth的音频集成脚本启动器
# 自动启动LLM服务并运行音频集成

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    基于Groundtruth的音频集成${NC}"
echo -e "${BLUE}========================================${NC}"

# 激活conda环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

cd "$SCRIPT_DIR"

# 检查CUDA可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，无法使用GPU${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${YELLOW}检测到 ${GPU_COUNT} 张GPU${NC}"

if [ "$GPU_COUNT" -lt 1 ]; then
    echo -e "${RED}至少需要1张GPU${NC}"
    exit 1
fi

# 检查LLM服务状态
echo -e "${YELLOW}检查LLM服务状态...${NC}"
RUNNING_SERVICES=0

# 临时清除代理环境变量以确保能正确连接localhost服务
OLD_HTTP_PROXY="$http_proxy"
OLD_HTTPS_PROXY="$https_proxy"
OLD_HTTP_PROXY_UPPER="$HTTP_PROXY"
OLD_HTTPS_PROXY_UPPER="$HTTPS_PROXY"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

for ((i=0; i<GPU_COUNT; i++)); do
    port=$((8000 + i))
    echo -e "${YELLOW}  检查端口 $port...${NC}"
    
    # 多次尝试连接服务
    service_available=false
    for attempt in {1..5}; do
        if HEALTH_RESPONSE=$(curl -s --connect-timeout 3 --max-time 8 "http://localhost:$port/health" 2>/dev/null); then
            if echo "$HEALTH_RESPONSE" | grep -q "healthy\|OK\|status"; then
                echo -e "${GREEN}    ✓ LLM服务 (端口 $port) 正常运行${NC}"
                RUNNING_SERVICES=$((RUNNING_SERVICES + 1))
                service_available=true
                break
            fi
        fi
        
        if [ $attempt -lt 5 ]; then
            echo -e "${YELLOW}    等待响应... (尝试 $attempt/5)${NC}"
            sleep 2
        fi
    done
    
    if [ "$service_available" = false ]; then
        echo -e "${YELLOW}    ⚠ LLM服务 (端口 $port) 未运行或不可用${NC}"
        if [[ -n "$HEALTH_RESPONSE" ]]; then
            echo -e "${YELLOW}      响应: ${HEALTH_RESPONSE:0:100}${NC}"
        fi
    fi
done

# 恢复原始代理设置
[[ -n "$OLD_HTTP_PROXY" ]] && export http_proxy="$OLD_HTTP_PROXY"
[[ -n "$OLD_HTTPS_PROXY" ]] && export https_proxy="$OLD_HTTPS_PROXY"
[[ -n "$OLD_HTTP_PROXY_UPPER" ]] && export HTTP_PROXY="$OLD_HTTP_PROXY_UPPER"
[[ -n "$OLD_HTTPS_PROXY_UPPER" ]] && export HTTPS_PROXY="$OLD_HTTPS_PROXY_UPPER"

# 如果没有LLM服务运行，启动它们
if [ "$RUNNING_SERVICES" -eq 0 ]; then
    echo -e "${YELLOW}启动LLM服务...${NC}"
    
    # 启动LLM服务
    if ./auto_start_llm_services.sh; then
        echo -e "${GREEN}✓ LLM服务启动成功${NC}"
    else
        echo -e "${RED}✗ LLM服务启动失败${NC}"
        exit 1
    fi
elif [ "$RUNNING_SERVICES" -lt "$GPU_COUNT" ]; then
    echo -e "${YELLOW}部分LLM服务未运行，建议重新启动所有服务${NC}"
    read -p "是否重新启动所有LLM服务? [y/N]: " -r
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}重新启动LLM服务...${NC}"
        ./stop_multi_llm_services.sh
        sleep 5
        if ./auto_start_llm_services.sh; then
            echo -e "${GREEN}✓ LLM服务重新启动成功${NC}"
        else
            echo -e "${RED}✗ LLM服务重新启动失败${NC}"
            exit 1
        fi
    fi
else
    echo -e "${GREEN}✓ 所有LLM服务正常运行${NC}"
fi

# 解析命令行参数
CONFIG_FILE="config.json"
DATASETS=""
MAX_DEGRADATION=""
NUM_GPUS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --datasets)
            shift
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                if [ -z "$DATASETS" ]; then
                    DATASETS="$1"
                else
                    DATASETS="$DATASETS $1"
                fi
                shift
            done
            ;;
        --max_degradation)
            MAX_DEGRADATION="$2"
            shift 2
            ;;
        --min_improvement)
            # 向后兼容性
            MAX_DEGRADATION="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config <文件>               配置文件路径 (默认: config.json)"
            echo "  --datasets <数据集名称...>    要处理的数据集名称列表"
            echo "                               可选: BAAI-ChildMandarin41.25H"
            echo "                                    Chinese_English_Scripted_Speech_Corpus_Children"
            echo "                                    King-ASR-EN-Kid"
            echo "                                    speechocean762"
            echo "                               注意: 现在使用mtfaa和mossformer增强方法"
            echo "  --max_degradation <数值>      最大CER退化量，超过此值才回退到原音频"
            echo "  --min_improvement <数值>      (已弃用) 使用--max_degradation代替"
            echo "  --num_gpus <数量>             使用的GPU数量 (默认: 自动检测)"
            echo "  -h, --help                   显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0 --datasets King-ASR-EN-Kid speechocean762"
            echo "  $0 --config my_config.json --datasets BAAI-ChildMandarin41.25H --max_degradation 0.03"
            echo "  $0  # 使用默认配置处理所有数据集"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 -h 或 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 构建Python命令
PYTHON_CMD="python3 groundtruth_based_integration.py"

PYTHON_CMD="$PYTHON_CMD --config \"$CONFIG_FILE\""

if [ -n "$DATASETS" ]; then
    PYTHON_CMD="$PYTHON_CMD --datasets $DATASETS"
fi

if [ -n "$MAX_DEGRADATION" ]; then
    PYTHON_CMD="$PYTHON_CMD --max_degradation $MAX_DEGRADATION"
fi

if [ -n "$NUM_GPUS" ]; then
    PYTHON_CMD="$PYTHON_CMD --num_gpus $NUM_GPUS"
fi

# 打印配置信息
echo -e "\n${YELLOW}配置信息:${NC}"
echo -e "配置文件: $CONFIG_FILE"
if [ -n "$DATASETS" ]; then
    echo -e "数据集: $DATASETS"
else
    echo -e "数据集: 配置文件中的所有数据集"
fi
if [ -n "$MAX_DEGRADATION" ]; then
    echo -e "最大CER退化量: $MAX_DEGRADATION"
else
    echo -e "最大CER退化量: 配置文件中的设置"
fi
echo -e "GPU数量: ${NUM_GPUS:-自动检测($GPU_COUNT)}"

echo -e "\n${YELLOW}执行命令:${NC}"
echo -e "$PYTHON_CMD"

echo -e "\n${YELLOW}开始处理...${NC}"
echo "=" * 80

# 执行Python脚本
eval $PYTHON_CMD

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}✅ 基于Groundtruth的音频集成完成！${NC}"
else
    echo -e "\n${RED}❌ 处理过程中出现错误 (退出码: $EXIT_CODE)${NC}"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}    处理完成${NC}"
echo -e "${BLUE}========================================${NC}"

exit $EXIT_CODE 