#!/bin/bash

# LLM服务稳定性测试脚本

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}    LLM服务稳定性测试${NC}"
echo -e "${BLUE}========================================${NC}"

# 测试参数
TEST_DURATION=60  # 测试时长（秒）
TEST_INTERVAL=5   # 测试间隔（秒）
MAX_CONCURRENT=10 # 最大并发请求数

log_message() {
    echo -e "[$(date '+%H:%M:%S')] $1"
}

test_llm_service() {
    local port=$1
    local service_name="LLM服务(端口$port)"
    
    # 健康检查
    if ! curl -s -f "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
        echo -e "${RED}✗ $service_name 不可用${NC}"
        return 1
    fi
    
    # 功能测试
    local response=$(curl -s -X POST "http://127.0.0.1:$port/normalize" \
        -H "Content-Type: application/json" \
        -d '{"text1": "Hello, world!", "text2": "hello world"}')
    
    if echo "$response" | grep -q '"success":true'; then
        echo -e "${GREEN}✓ $service_name 工作正常${NC}"
        return 0
    else
        echo -e "${RED}✗ $service_name 功能异常${NC}"
        return 1
    fi
}

# 并发测试函数
stress_test_llm() {
    local port=$1
    local request_count=$2
    
    for i in $(seq 1 $request_count); do
        curl -s -X POST "http://127.0.0.1:$port/normalize" \
            -H "Content-Type: application/json" \
            -d "{\"text1\": \"Test $i\", \"text2\": \"test $i\"}" > /dev/null &
    done
    
    wait  # 等待所有后台请求完成
}

main() {
    log_message "${YELLOW}开始LLM服务稳定性测试 (持续时间: ${TEST_DURATION}秒)${NC}"
    
    start_time=$(date +%s)
    test_count=0
    success_count=0
    
    while true; do
        current_time=$(date +%s)
        elapsed=$((current_time - start_time))
        
        if [ $elapsed -ge $TEST_DURATION ]; then
            break
        fi
        
        test_count=$((test_count + 1))
        log_message "${BLUE}第 $test_count 轮测试${NC}"
        
        # 测试两个服务
        service1_ok=false
        service2_ok=false
        
        if test_llm_service 8000; then
            service1_ok=true
        fi
        
        if test_llm_service 8001; then
            service2_ok=true
        fi
        
        # 如果至少一个服务正常，进行压力测试
        if [ "$service1_ok" = true ] || [ "$service2_ok" = true ]; then
            success_count=$((success_count + 1))
            
            # 随机选择一个可用的服务进行压力测试
            if [ "$service1_ok" = true ]; then
                log_message "${YELLOW}对8000端口进行压力测试...${NC}"
                stress_test_llm 8000 $MAX_CONCURRENT
            else
                log_message "${YELLOW}对8001端口进行压力测试...${NC}"
                stress_test_llm 8001 $MAX_CONCURRENT
            fi
        fi
        
        log_message "${BLUE}剩余时间: $((TEST_DURATION - elapsed))秒${NC}"
        sleep $TEST_INTERVAL
    done
    
    # 计算成功率
    if [ $test_count -gt 0 ]; then
        success_rate=$((success_count * 100 / test_count))
        log_message "${GREEN}测试完成！${NC}"
        log_message "总测试轮数: $test_count"
        log_message "成功轮数: $success_count"
        log_message "成功率: ${success_rate}%"
        
        if [ $success_rate -ge 90 ]; then
            log_message "${GREEN}✅ LLM服务稳定性良好${NC}"
            return 0
        else
            log_message "${YELLOW}⚠️  LLM服务稳定性需要改进${NC}"
            return 1
        fi
    else
        log_message "${RED}❌ 未完成任何测试${NC}"
        return 1
    fi
}

# 确保服务正在运行
if ! pgrep -f "llm_service.py" > /dev/null; then
    echo -e "${RED}错误: LLM服务未运行，请先启动服务${NC}"
    echo "运行: ./auto_start_llm_services.sh"
    exit 1
fi

# 运行测试
main 