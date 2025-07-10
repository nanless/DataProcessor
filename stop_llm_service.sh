#!/bin/bash

# LLM服务停止脚本
# 停止LLM服务和Ollama服务

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

LOG_DIR="./logs"
LLM_PID_FILE="$LOG_DIR/llm_service.pid"
OLLAMA_PID_FILE="$LOG_DIR/ollama.pid"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    停止LLM文本标准化服务${NC}"
echo -e "${BLUE}    停止Ollama服务${NC}"
echo -e "${BLUE}========================================${NC}"

# 函数：停止指定PID的进程
stop_process() {
    local pid=$1
    local process_name=$2
    
    if [ -z "$pid" ]; then
        echo -e "${YELLOW}PID为空，跳过停止${NC}"
        return 1
    fi
    
    echo -e "${YELLOW}停止${process_name}进程 (PID: $pid)...${NC}"
    
    # 检查进程是否存在
    if ! kill -0 $pid 2>/dev/null; then
        echo -e "${YELLOW}进程 $pid 不存在，可能已经停止${NC}"
        return 0
    fi
    
    # 优雅关闭
    echo -e "${YELLOW}发送SIGTERM信号...${NC}"
    kill $pid
    
    # 等待进程结束
    for i in {1..10}; do
        if ! kill -0 $pid 2>/dev/null; then
            echo -e "${GREEN}✓ ${process_name}已停止${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    # 如果进程仍然存在，强制终止
    echo -e "${RED}进程未响应，强制终止...${NC}"
    kill -9 $pid
    
    # 再次检查
    sleep 2
    if ! kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}✓ ${process_name}已强制停止${NC}"
        return 0
    else
        echo -e "${RED}无法停止${process_name}进程${NC}"
        return 1
    fi
}

# 停止LLM服务
echo -e "${YELLOW}1. 停止LLM服务...${NC}"
if [ -f "$LLM_PID_FILE" ]; then
    LLM_PID=$(cat "$LLM_PID_FILE")
    if stop_process "$LLM_PID" "LLM服务"; then
        rm -f "$LLM_PID_FILE"
    fi
else
    echo -e "${YELLOW}LLM服务PID文件不存在，尝试查找运行中的进程...${NC}"
    
    # 查找运行中的llm_service.py进程
    LLM_PIDS=$(pgrep -f "llm_service.py")
    
    if [ -z "$LLM_PIDS" ]; then
        echo -e "${GREEN}未找到运行中的LLM服务${NC}"
    else
        echo -e "${YELLOW}找到运行中的LLM服务进程: $LLM_PIDS${NC}"
        
        # 终止进程
        for pid in $LLM_PIDS; do
            stop_process "$pid" "LLM服务"
        done
    fi
fi

# 停止Ollama服务
echo -e "${YELLOW}2. 停止Ollama服务...${NC}"
if [ -f "$OLLAMA_PID_FILE" ]; then
    OLLAMA_PID=$(cat "$OLLAMA_PID_FILE")
    if stop_process "$OLLAMA_PID" "Ollama服务"; then
        rm -f "$OLLAMA_PID_FILE"
    fi
else
    echo -e "${YELLOW}Ollama服务PID文件不存在，尝试查找运行中的进程...${NC}"
    
    # 查找运行中的ollama serve进程
    OLLAMA_PIDS=$(pgrep -f "ollama serve")
    
    if [ -z "$OLLAMA_PIDS" ]; then
        # 尝试查找ollama进程
        OLLAMA_PIDS=$(pgrep ollama)
    fi
    
    if [ -z "$OLLAMA_PIDS" ]; then
        echo -e "${GREEN}未找到运行中的Ollama服务${NC}"
    else
        echo -e "${YELLOW}找到运行中的Ollama服务进程: $OLLAMA_PIDS${NC}"
        
        # 终止进程
        for pid in $OLLAMA_PIDS; do
            stop_process "$pid" "Ollama服务"
        done
    fi
fi

# 检查端口占用情况
echo -e "${YELLOW}3. 检查端口占用情况...${NC}"

# 检查LLM服务端口 (默认8000)
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}端口 8000 仍被占用${NC}"
    PORT_PIDS=$(lsof -Pi :8000 -sTCP:LISTEN -t)
    echo -e "${YELLOW}占用进程: $PORT_PIDS${NC}"
else
    echo -e "${GREEN}✓ 端口 8000 已释放${NC}"
fi

# 检查Ollama服务端口 (默认11434)
if lsof -Pi :11434 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}端口 11434 仍被占用${NC}"
    PORT_PIDS=$(lsof -Pi :11434 -sTCP:LISTEN -t)
    echo -e "${YELLOW}占用进程: $PORT_PIDS${NC}"
else
    echo -e "${GREEN}✓ 端口 11434 已释放${NC}"
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}    服务停止完成${NC}"
echo -e "${BLUE}========================================${NC}"

echo -e "${YELLOW}提示:${NC}"
echo -e "• 如果需要彻底清理，可以运行: pkill -f ollama"
echo -e "• 重新启动服务: ./start_llm_service.sh --daemon"
echo -e "• 查看日志: tail -f ./logs/llm_service.log" 