#!/bin/bash

# 自动启动LLM服务脚本
# 根据GPU数量自动配置和启动相应的LLM服务

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    自动启动LLM服务（自适应GPU配置）${NC}"
echo -e "${BLUE}========================================${NC}"

# 激活conda环境
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio

# 创建日志目录
mkdir -p ./logs

# 设置代理绕过
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# 检测GPU数量
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi未找到，无法检测GPU${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
echo -e "${YELLOW}检测到 ${GPU_COUNT} 张GPU${NC}"

# 根据GPU数量配置服务
if [ "$GPU_COUNT" -eq 4 ]; then
    # 4卡配置
    echo -e "${GREEN}使用4卡配置: GPU 0(LLM) + GPU 1,2,3(ASR)${NC}"
    
    # 启动组1的Ollama服务
    echo -e "${YELLOW}启动Ollama服务 (GPU 0, 端口 11434)...${NC}"
    CUDA_VISIBLE_DEVICES=0 nohup ollama serve > ./logs/ollama_0.log 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAMA_PID > ./logs/ollama_0.pid
    
    # 等待Ollama服务启动
    sleep 10
    
    # 拉取模型
    echo -e "${YELLOW}拉取模型 qwen2.5:32b...${NC}"
    CUDA_VISIBLE_DEVICES=0 ollama pull qwen2.5:32b
    
    # 启动LLM HTTP服务
    echo -e "${YELLOW}启动LLM HTTP服务 (端口 8000)...${NC}"
    CUDA_VISIBLE_DEVICES=0 nohup python3 llm_service.py \
        --model_name qwen2.5:32b \
        --ollama_host localhost:11434 \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        > ./logs/llm_service_0.log 2>&1 &
    LLM_PID=$!
    echo $LLM_PID > ./logs/llm_service_0.pid
    
    echo -e "${GREEN}4卡配置启动完成${NC}"
    
elif [ "$GPU_COUNT" -eq 8 ]; then
    # 8卡配置
    echo -e "${GREEN}使用8卡配置: 组1[GPU 0(LLM) + GPU 1,2,3(ASR)] + 组2[GPU 4(LLM) + GPU 5,6,7(ASR)]${NC}"
    
    # 启动组1的Ollama服务
    echo -e "${YELLOW}启动组1 Ollama服务 (GPU 0, 端口 11434)...${NC}"
    CUDA_VISIBLE_DEVICES=0 nohup ollama serve > ./logs/ollama_0.log 2>&1 &
    OLLAMA_PID_0=$!
    echo $OLLAMA_PID_0 > ./logs/ollama_0.pid
    
    # 启动组2的Ollama服务
    echo -e "${YELLOW}启动组2 Ollama服务 (GPU 4, 端口 11435)...${NC}"
    CUDA_VISIBLE_DEVICES=4 OLLAMA_HOST=0.0.0.0:11435 nohup ollama serve > ./logs/ollama_4.log 2>&1 &
    OLLAMA_PID_4=$!
    echo $OLLAMA_PID_4 > ./logs/ollama_4.pid
    
    # 等待Ollama服务启动
    sleep 15
    
    # 拉取模型（两个GPU都拉取）
    echo -e "${YELLOW}拉取模型 qwen2.5:32b (GPU 0)...${NC}"
    CUDA_VISIBLE_DEVICES=0 ollama pull qwen2.5:32b
    
    echo -e "${YELLOW}拉取模型 qwen2.5:32b (GPU 4)...${NC}"
    CUDA_VISIBLE_DEVICES=4 OLLAMA_HOST=localhost:11435 ollama pull qwen2.5:32b
    
    # 启动组1的LLM HTTP服务
    echo -e "${YELLOW}启动组1 LLM HTTP服务 (端口 8000)...${NC}"
    CUDA_VISIBLE_DEVICES=0 nohup python3 llm_service.py \
        --model_name qwen2.5:32b \
        --ollama_host localhost:11434 \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 1 \
        > ./logs/llm_service_0.log 2>&1 &
    LLM_PID_0=$!
    echo $LLM_PID_0 > ./logs/llm_service_0.pid
    
    # 启动组2的LLM HTTP服务
    echo -e "${YELLOW}启动组2 LLM HTTP服务 (端口 8001)...${NC}"
    CUDA_VISIBLE_DEVICES=4 nohup python3 llm_service.py \
        --model_name qwen2.5:32b \
        --ollama_host localhost:11435 \
        --host 0.0.0.0 \
        --port 8001 \
        --workers 1 \
        > ./logs/llm_service_4.log 2>&1 &
    LLM_PID_4=$!
    echo $LLM_PID_4 > ./logs/llm_service_4.pid
    
    echo -e "${GREEN}8卡配置启动完成${NC}"
    
else
    echo -e "${RED}不支持的GPU数量: ${GPU_COUNT}${NC}"
    echo -e "${YELLOW}支持的配置: 4卡或8卡${NC}"
    exit 1
fi

# 等待服务完全启动
echo -e "${YELLOW}等待服务完全启动...${NC}"
sleep 15

# 检查服务状态
echo -e "${BLUE}检查服务状态:${NC}"

if [ "$GPU_COUNT" -eq 4 ]; then
    # 检查4卡配置的服务
    if http_proxy="" https_proxy="" curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✓ LLM服务 (端口 8000) 正常运行${NC}"
    else
        echo -e "${RED}✗ LLM服务 (端口 8000) 连接失败${NC}"
    fi
    
elif [ "$GPU_COUNT" -eq 8 ]; then
    # 检查8卡配置的服务
    if http_proxy="" https_proxy="" curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✓ 组1 LLM服务 (端口 8000) 正常运行${NC}"
    else
        echo -e "${RED}✗ 组1 LLM服务 (端口 8000) 连接失败${NC}"
    fi
    
    if http_proxy="" https_proxy="" curl -s http://localhost:8001/health > /dev/null; then
        echo -e "${GREEN}✓ 组2 LLM服务 (端口 8001) 正常运行${NC}"
    else
        echo -e "${RED}✗ 组2 LLM服务 (端口 8001) 连接失败${NC}"
    fi
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}    自动启动完成${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "${YELLOW}使用提示:${NC}"
echo -e "• 查看日志: tail -f ./logs/*.log"
if [ "$GPU_COUNT" -eq 4 ]; then
    echo -e "• LLM服务地址: http://localhost:8000"
    echo -e "• 停止服务: kill \$(cat ./logs/ollama_0.pid) \$(cat ./logs/llm_service_0.pid)"
elif [ "$GPU_COUNT" -eq 8 ]; then
    echo -e "• 组1 LLM服务地址: http://localhost:8000"
    echo -e "• 组2 LLM服务地址: http://localhost:8001"
    echo -e "• 停止服务: kill \$(cat ./logs/ollama_*.pid) \$(cat ./logs/llm_service_*.pid)"
fi
echo -e "• 运行评估: python3 enhancement_audio_quality_assessment.py" 