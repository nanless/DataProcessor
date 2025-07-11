#!/bin/bash

# LLM服务启动脚本
# 使用Ollama部署Qwen3-8B模型
source /root/miniforge3/etc/profile.d/conda.sh
conda activate kimi-audio
mkdir -p ./logs

# 设置代理绕过，确保可以访问本地Ollama服务
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"

# 设置默认参数
MODEL_NAME="${MODEL_NAME:-qwen3:8b}"
OLLAMA_HOST="${OLLAMA_HOST:-localhost:11434}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-1}"

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    LLM文本标准化服务启动脚本${NC}"
echo -e "${BLUE}    使用Ollama + Qwen3-8B${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误: Python3未安装${NC}"
    exit 1
fi

# 检查必要的Python包
echo -e "${YELLOW}检查Python依赖...${NC}"
python3 -c "import fastapi, uvicorn, pydantic, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}错误: 缺少必要的Python包${NC}"
    echo -e "${YELLOW}请安装以下包:${NC}"
    echo "pip install fastapi uvicorn pydantic requests"
    exit 1
fi

# 检查ollama Python包
python3 -c "import ollama" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}安装ollama Python包...${NC}"
    pip install ollama
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法安装ollama Python包${NC}"
        exit 1
    fi
fi

# 检查Ollama是否已安装
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}错误: Ollama未安装${NC}"
    echo -e "${YELLOW}请按照以下步骤安装Ollama:${NC}"
    echo "curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# 检查Ollama服务是否运行
echo -e "${YELLOW}检查Ollama服务状态...${NC}"

# 使用ollama命令检查服务状态
if ! ollama list > /dev/null 2>&1; then
    echo -e "${YELLOW}Ollama服务未运行，正在启动...${NC}"
    
    # 启动Ollama服务
    if [ "$1" == "--daemon" ]; then
        # 后台启动模式
        echo -e "${YELLOW}在后台启动Ollama服务...${NC}"
        nohup ollama serve > ./logs/ollama.log 2>&1 &
        OLLAMA_PID=$!
        echo $OLLAMA_PID > ./logs/ollama.pid
        echo -e "${GREEN}Ollama服务已在后台启动 (PID: $OLLAMA_PID)${NC}"
        
        # 等待Ollama服务启动并可用
        echo -e "${YELLOW}等待Ollama服务初始化...${NC}"
        for i in {1..60}; do
            if ollama list > /dev/null 2>&1; then
                echo -e "${GREEN}✓ Ollama服务启动成功${NC}"
                break
            fi
            echo -n "."
            sleep 2
        done
        
        if ! ollama list > /dev/null 2>&1; then
            echo -e "${RED}Ollama服务启动失败或超时${NC}"
            echo -e "${YELLOW}请检查日志: tail -f ./logs/ollama.log${NC}"
            exit 1
        fi
    else
        # 前台启动模式，需要在另一个终端运行
        echo -e "${RED}请在另一个终端运行以下命令启动Ollama服务:${NC}"
        echo "ollama serve"
        echo -e "${YELLOW}等待服务启动后，按任意键继续...${NC}"
        read -n 1 -s
        
        # 再次检查服务是否可用
        if ! ollama list > /dev/null 2>&1; then
            echo -e "${RED}Ollama服务仍然不可用${NC}"
            echo -e "${YELLOW}请确保在另一个终端运行了 'ollama serve'${NC}"
            exit 1
        fi
    fi
else
    echo -e "${GREEN}✓ Ollama服务已运行${NC}"
fi

# 验证Ollama服务完全可用
echo -e "${YELLOW}验证Ollama服务...${NC}"
if ollama --version > /dev/null 2>&1; then
    OLLAMA_VERSION=$(ollama --version)
    echo -e "${GREEN}✓ Ollama版本: ${OLLAMA_VERSION}${NC}"
else
    echo -e "${RED}✗ Ollama命令不可用${NC}"
    exit 1
fi

# 检查并拉取模型
echo -e "${YELLOW}检查模型 ${MODEL_NAME}...${NC}"
if ! ollama list | grep -q "${MODEL_NAME}"; then
    echo -e "${YELLOW}正在拉取模型 ${MODEL_NAME}...${NC}"
    ollama pull "${MODEL_NAME}"
    if [ $? -ne 0 ]; then
        echo -e "${RED}错误: 无法拉取模型 ${MODEL_NAME}${NC}"
        echo -e "${YELLOW}请检查模型名称是否正确${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ 模型 ${MODEL_NAME} 拉取成功${NC}"
else
    echo -e "${GREEN}✓ 模型 ${MODEL_NAME} 已存在${NC}"
fi

# 显示配置信息
echo -e "${BLUE}服务配置:${NC}"
echo -e "  模型名称: ${GREEN}$MODEL_NAME${NC}"
echo -e "  Ollama服务地址: ${GREEN}$OLLAMA_HOST${NC}"
echo -e "  服务地址: ${GREEN}$HOST:$PORT${NC}"
echo -e "  工作进程数: ${GREEN}$WORKERS${NC}"

# 检查端口是否被占用
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${RED}错误: 端口 $PORT 已被占用${NC}"
    echo -e "${YELLOW}请使用其他端口或停止占用该端口的进程${NC}"
    exit 1
fi

# 创建日志目录
LOG_DIR="./logs"
mkdir -p $LOG_DIR

# 启动服务
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}启动LLM服务...${NC}"
echo -e "${BLUE}========================================${NC}"

# 设置环境变量
export MODEL_NAME=$MODEL_NAME
export OLLAMA_HOST=$OLLAMA_HOST

# 设置代理绕过，确保可以访问本地Ollama服务
export no_proxy="localhost,127.0.0.1,::1"
export NO_PROXY="localhost,127.0.0.1,::1"
echo -e "${YELLOW}已设置代理绕过: $no_proxy${NC}"

# 启动服务（后台运行）
if [ "$1" == "--daemon" ]; then
    echo -e "${YELLOW}后台启动模式${NC}"
    nohup python3 llm_service.py \
        --model_name "$MODEL_NAME" \
        --ollama_host "$OLLAMA_HOST" \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS \
        > $LOG_DIR/llm_service.log 2>&1 &
    
    LLM_PID=$!
    echo $LLM_PID > $LOG_DIR/llm_service.pid
    
    echo -e "${GREEN}LLM服务已在后台启动${NC}"
    echo -e "PID: $LLM_PID"
    echo -e "日志文件: $LOG_DIR/llm_service.log"
    echo -e "停止服务: kill $LLM_PID 或 ./stop_llm_service.sh"
    
    # 等待服务启动
    echo -e "${YELLOW}等待服务启动...${NC}"
    for i in {1..30}; do
        if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
            echo -e "${GREEN}✓ LLM服务启动成功${NC}"
            echo -e "服务地址: http://localhost:$PORT"
            echo -e "健康检查: http://localhost:$PORT/health"
            echo -e "模型信息: http://localhost:$PORT/model_info"
            exit 0
        fi
        echo -n "."
        sleep 2
    done
    
    echo -e "${RED}服务启动超时，请检查日志文件${NC}"
    exit 1
else
    # 前台运行
    echo -e "${YELLOW}前台运行模式（按Ctrl+C停止）${NC}"
    python3 llm_service.py \
        --model_name "$MODEL_NAME" \
        --ollama_host "$OLLAMA_HOST" \
        --host $HOST \
        --port $PORT \
        --workers $WORKERS
fi 