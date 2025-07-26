#!/bin/bash

# LLM服务监控和自动重启脚本
# 用于大批量处理时确保LLM服务稳定运行

set -e

LOG_FILE="./logs/llm_monitor.log"
CHECK_INTERVAL=30  # 检查间隔（秒）
CONSECUTIVE_FAILURES=0  # 连续失败次数
MAX_CONSECUTIVE_FAILURES=2  # 最大连续失败次数（超过则重启）

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_llm_service() {
    local port=$1
    local service_name="LLM服务(端口$port)"
    
    # 多次重试检查HTTP服务健康状态
    for attempt in 1 2 3; do
        if curl -s -f "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            return 0  # 服务正常
        fi
        if [ $attempt -lt 3 ]; then
            sleep 2  # 等待2秒再重试
        fi
    done
    
    log_message "⚠️  $service_name 健康检查失败 (3次重试)"
    return 1  # 服务异常
}

check_ollama_service() {
    local port=$1
    local service_name="Ollama服务(端口$port)"
    
    # 多次重试检查Ollama服务
    for attempt in 1 2 3; do
        if curl -s -f "http://127.0.0.1:$port/api/tags" > /dev/null 2>&1; then
            return 0  # 服务正常
        fi
        if [ $attempt -lt 3 ]; then
            sleep 2  # 等待2秒再重试
        fi
    done
    
    log_message "⚠️  $service_name 健康检查失败 (3次重试)"
    return 1  # 服务异常
}

restart_services() {
    log_message "🔄 检测到服务异常，开始重启..."
    
    # 停止所有服务
    ./stop_multi_llm_services.sh > /dev/null 2>&1 || true
    sleep 5
    
    # 重新启动服务
    ./auto_start_llm_services.sh > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        log_message "✅ LLM服务重启成功"
        return 0
    else
        log_message "❌ LLM服务重启失败"
        return 1
    fi
}

main() {
    log_message "🚀 启动LLM服务监控 (检查间隔: ${CHECK_INTERVAL}秒)"
    
    # 清除可能影响本地连接的proxy设置
    unset http_proxy
    unset https_proxy
    unset HTTP_PROXY
    unset HTTPS_PROXY
    log_message "✅ 已清除proxy设置以确保本地连接"
    
    # 初始等待，让服务完全启动
    log_message "⏳ 等待60秒让服务完全初始化..."
    sleep 60
    
    while true; do
        service_healthy=true
        
        # 检查LLM HTTP服务
        if ! check_llm_service 8000; then
            service_healthy=false
        fi
        
        if ! check_llm_service 8001; then
            service_healthy=false
        fi
        
        # 检查Ollama服务
        if ! check_ollama_service 11434; then
            service_healthy=false
        fi
        
        if ! check_ollama_service 11435; then
            service_healthy=false
        fi
        
        # 处理服务健康状态
        if [ "$service_healthy" = false ]; then
            CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
            log_message "⚠️  检测到服务异常 (连续失败: $CONSECUTIVE_FAILURES/$MAX_CONSECUTIVE_FAILURES)"
            
            if [ $CONSECUTIVE_FAILURES -ge $MAX_CONSECUTIVE_FAILURES ]; then
                log_message "🔄 连续失败次数达到阈值，开始重启服务..."
                restart_services
                CONSECUTIVE_FAILURES=0  # 重置计数器
                sleep 30  # 重启后等待更长时间
            fi
        else
            if [ $CONSECUTIVE_FAILURES -gt 0 ]; then
                log_message "✅ 服务已恢复正常 (重置失败计数器)"
            else
                log_message "✅ 所有LLM服务运行正常"
            fi
            CONSECUTIVE_FAILURES=0  # 重置失败计数器
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# 捕获中断信号
trap 'log_message "📋 LLM监控脚本停止"; exit 0' SIGINT SIGTERM

# 确保日志目录存在
mkdir -p logs

# 启动监控
main 