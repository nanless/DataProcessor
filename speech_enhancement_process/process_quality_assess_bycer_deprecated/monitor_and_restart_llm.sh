#!/bin/bash

# LLM服务监控脚本 - 简化版
# 监控服务健康状态，异常时自动重启

set -e

LOG_FILE="./logs/llm_monitor.log"
CHECK_INTERVAL=30  # 检查间隔（秒）

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_service() {
    local port=$1
    # 检查HTTP服务健康状态，清除proxy设置
    if http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" curl -s -f --connect-timeout 5 "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        return 0  # 服务正常
    else
        return 1  # 服务异常
    fi
}

restart_services() {
    log_message "🔄 检测到服务异常，重启所有服务..."
    
    # 停止所有服务
    ./stop_multi_llm_services.sh > /dev/null 2>&1 || true
    sleep 5
    
    # 重新启动服务
    ./auto_start_llm_services.sh > /dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        log_message "✅ 服务重启成功"
        return 0
    else
        log_message "❌ 服务重启失败"
        return 1
    fi
}

main() {
    log_message "🚀 启动LLM服务监控 (检查间隔: ${CHECK_INTERVAL}秒)"
    
    # 清除proxy设置
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
    
    # 初始等待
    log_message "⏳ 等待60秒让服务完全初始化..."
    sleep 60
    
    consecutive_failures=0
    max_failures=2
    
    while true; do
        service_healthy=true
        
        # 检查所有可能运行的LLM服务端口
        for port in {8000..8010}; do
            if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
                if ! check_service $port; then
                    log_message "⚠️  LLM服务(端口$port) 健康检查失败"
                    service_healthy=false
                fi
            fi
        done
        
        if [ "$service_healthy" = false ]; then
            consecutive_failures=$((consecutive_failures + 1))
            log_message "⚠️  连续失败次数: $consecutive_failures/$max_failures"
            
            if [ $consecutive_failures -ge $max_failures ]; then
                restart_services
                consecutive_failures=0
                sleep 30  # 重启后等待更长时间
            fi
        else
            if [ $consecutive_failures -gt 0 ]; then
                log_message "✅ 服务已恢复正常"
            fi
            consecutive_failures=0
        fi
        
        sleep $CHECK_INTERVAL
    done
}

# 捕获中断信号
trap 'log_message "📋 监控脚本停止"; exit 0' SIGINT SIGTERM

# 确保日志目录存在
mkdir -p logs

# 启动监控
main 