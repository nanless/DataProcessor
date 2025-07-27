#!/bin/bash

# 音频质量评估脚本 - 自动GPU配置版本
# 
# 使用方法：
# 1. 确保配置文件正确
# 2. 运行脚本：bash run_multi_directory_assessment_auto_gpu.sh

# 设置基本路径
BASE_DIR="/root/group-shared/voiceprint/data/speech/speaker_verification"
SCRIPT_DIR="/root/code/github_repos/DataProcessor/speech_enhancement_process/process_quality_assess"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    音频质量评估 - 自动GPU配置${NC}"
echo -e "${BLUE}========================================${NC}"

# 切换到脚本目录
cd $SCRIPT_DIR

# 激活conda环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

# 检测GPU数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -1)
    echo -e "${YELLOW}检测到 ${GPU_COUNT} 张GPU${NC}"
else
    echo -e "${RED}错误: 无法检测GPU，请确保NVIDIA驱动正常工作${NC}"
    exit 1
fi

# 显示GPU配置说明
if [ "$GPU_COUNT" -eq 4 ]; then
    echo -e "${GREEN}4卡配置：${NC}"
    echo -e "  • GPU 0: LLM服务 (端口8000)"
    echo -e "  • GPU 1,2,3: ASR处理"
elif [ "$GPU_COUNT" -eq 8 ]; then
    echo -e "${GREEN}8卡配置：${NC}"
    echo -e "  • 组1: GPU 0(LLM服务 端口8000) + GPU 1,2,3(ASR处理)"
    echo -e "  • 组2: GPU 4(LLM服务 端口8001) + GPU 5,6,7(ASR处理)"
else
    echo -e "${YELLOW}${GPU_COUNT}卡配置：使用传统模式${NC}"
    echo -e "  • GPU 0: LLM服务 (端口8000)"
    echo -e "  • GPU 1-${GPU_COUNT}: ASR处理"
fi

echo ""

# 启动LLM服务
echo -e "${YELLOW}步骤1: 启动LLM服务...${NC}"
./auto_start_llm_services.sh

if [ $? -ne 0 ]; then
    echo -e "${RED}错误: LLM服务启动失败${NC}"
    exit 1
fi

echo -e "${GREEN}✓ LLM服务启动成功${NC}"
echo ""

# 启动LLM服务监控（后台运行）
echo -e "${YELLOW}启动LLM服务监控...${NC}"
nohup ./monitor_and_restart_llm.sh > ./logs/llm_monitor.log 2>&1 &
MONITOR_PID=$!
echo $MONITOR_PID > ./logs/llm_monitor.pid
echo -e "${GREEN}✓ LLM监控已启动 (PID: $MONITOR_PID)${NC}"

# 等待服务完全启动并验证健康状态
echo -e "${YELLOW}等待服务初始化并验证健康状态...${NC}"

# 清除可能影响本地连接的proxy设置
unset http_proxy
unset https_proxy
unset HTTP_PROXY  
unset HTTPS_PROXY

# 给服务更多时间启动
sleep 15

# 验证LLM服务健康状态（带重试）
echo -e "${YELLOW}验证LLM服务健康状态...${NC}"
health_check_passed=true

for port in 8000 8001; do
    service_ok=false
    
    # 尝试3次健康检查
    for attempt in 1 2 3; do
        echo -e "${YELLOW}检查LLM服务(端口$port) - 第${attempt}次尝试...${NC}"
        
        if http_proxy="" https_proxy="" curl -s -f --connect-timeout 5 --max-time 10 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            echo -e "${GREEN}✓ LLM服务(端口$port)健康检查通过${NC}"
            service_ok=true
            break
        fi
        
        if [ $attempt -lt 3 ]; then
            echo -e "${YELLOW}等待5秒后重试...${NC}"
            sleep 5
        fi
    done
    
    if [ "$service_ok" = false ]; then
        echo -e "${RED}✗ LLM服务(端口$port)健康检查失败 (3次尝试)${NC}"
        health_check_passed=false
    fi
done

if [ "$health_check_passed" = false ]; then
    echo -e "${RED}错误: LLM服务健康检查失败，停止监控并退出${NC}"
    echo -e "${YELLOW}提示: 监控脚本将在60秒后开始检查并自动重启服务${NC}"
    kill $MONITOR_PID 2>/dev/null || true
    rm -f ./logs/llm_monitor.pid
    exit 1
fi

# 运行音频质量评估
echo -e "${YELLOW}步骤2: 开始音频质量评估...${NC}"

# 使用配置文件方式运行
if [ -f "multi_directory_config_example.json" ]; then
    echo -e "${YELLOW}使用配置文件: multi_directory_config_example.json${NC}"
    python enhancement_audio_quality_assessment.py \
        --config_file multi_directory_config_example.json
else
    echo -e "${YELLOW}配置文件不存在，使用默认参数...${NC}"
    python enhancement_audio_quality_assessment.py \
        --original_dir "${BASE_DIR}/King-ASR-EN-Kid" \
        --enhanced_dir "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced" \
        --output_dir "${BASE_DIR}/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment" \
        --text_normalization "llm" \
        --use_llm_normalization \
        --ten_vad_hop_size 256 \
        --ten_vad_threshold 0.5 \
        --skip_existing
fi

ASSESSMENT_EXIT_CODE=$?

# 停止LLM监控
echo -e "\n${YELLOW}停止LLM服务监控...${NC}"
if [ -f "./logs/llm_monitor.pid" ]; then
    MONITOR_PID=$(cat ./logs/llm_monitor.pid)
    kill $MONITOR_PID 2>/dev/null || true
    rm -f ./logs/llm_monitor.pid
    echo -e "${GREEN}✓ LLM监控已停止${NC}"
fi

# 显示结果
if [ $ASSESSMENT_EXIT_CODE -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}    音频质量评估完成！${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    echo -e "\n${YELLOW}后续操作:${NC}"
    echo -e "• 查看结果: 检查输出目录中的结果文件"
    echo -e "• 停止服务: ./stop_multi_llm_services.sh"
    echo -e "• 查看日志: tail -f ./logs/llm_service_*.log"
    echo -e "• 查看监控日志: tail -f ./logs/llm_monitor.log"
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}    音频质量评估失败！${NC}"
    echo -e "${RED}========================================${NC}"
    
    echo -e "\n${YELLOW}故障排除:${NC}"
    echo -e "• 检查GPU内存是否足够"
    echo -e "• 检查输入目录是否存在"
    echo -e "• 查看错误日志和监控日志"
    echo -e "• 查看监控日志: tail -f ./logs/llm_monitor.log"
    echo -e "• 停止服务: ./stop_multi_llm_services.sh"
fi

# 询问是否停止服务（仅在交互模式下）
echo ""
if [ -t 0 ]; then
    # 标准输入是终端，可以进行交互
    read -p "是否现在停止LLM服务? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}停止LLM服务...${NC}"
        ./stop_multi_llm_services.sh
        echo -e "${GREEN}✓ 服务已停止${NC}"
    else
        echo -e "${YELLOW}LLM服务保持运行状态${NC}"
        echo -e "手动停止: ./stop_multi_llm_services.sh"
    fi
else
    # 非交互模式（如nohup），默认不停止服务
    echo -e "${YELLOW}LLM服务保持运行状态${NC}"
    echo -e "手动停止: ./stop_multi_llm_services.sh"
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}    脚本执行完成${NC}"
echo -e "${BLUE}========================================${NC}" 