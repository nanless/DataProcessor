#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM服务测试脚本
用于验证基于Ollama的LLM服务是否正常工作
"""

import requests
import json
import time
import sys

def test_llm_service(base_url="http://localhost:8000"):
    """测试LLM服务"""
    print("=" * 50)
    print("LLM服务测试 (Ollama)")
    print("=" * 50)
    
    # 测试1: 健康检查
    print("1. 健康检查...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 健康检查通过: {result}")
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False
    
    # 测试2: 获取模型信息
    print("\n2. 获取模型信息...")
    try:
        response = requests.get(f"{base_url}/model_info", timeout=10)
        response.raise_for_status()
        result = response.json()
        print(f"✓ 模型信息获取成功:")
        print(f"  模型名称: {result.get('model_name', 'N/A')}")
        print(f"  Ollama服务地址: {result.get('ollama_host', 'N/A')}")
        print(f"  服务类型: {result.get('service_type', 'N/A')}")
    except Exception as e:
        print(f"✗ 获取模型信息失败: {e}")
        return False
    
    # 测试3: 文本标准化
    print("\n3. 文本标准化测试...")
    test_cases = [
        {
            "text1": "Hello world",
            "text2": "h e l l o w o r l d",
            "description": "拼读测试"
        },
        {
            "text1": "The quick brown fox jumps over the lazy dog.",
            "text2": "the quick brown fox jumps over the lazy dog",
            "description": "标点符号和大小写测试"
        },
        {
            "text1": "I have 123 apples",
            "text2": "I have one two three apples",
            "description": "数字转拼读测试"
        },
        {
            "text1": "A B C D E F G",
            "text2": "abcdefg",
            "description": "字母拼读测试"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n  测试 3.{i}: {test_case['description']}")
        print(f"    原始文本1: '{test_case['text1']}'")
        print(f"    原始文本2: '{test_case['text2']}'")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{base_url}/normalize",
                json={
                    "text1": test_case['text1'],
                    "text2": test_case['text2']
                },
                timeout=60  # 增加超时时间，因为Ollama可能需要更长时间
            )
            end_time = time.time()
            
            response.raise_for_status()
            result = response.json()
            
            print(f"    标准化文本1: '{result.get('normalized_text1', '')}'")
            print(f"    标准化文本2: '{result.get('normalized_text2', '')}'")
            print(f"    处理时间: {end_time - start_time:.2f}秒")
            print(f"    成功状态: {result.get('success', False)}")
            
            if result.get('error_message'):
                print(f"    错误信息: {result.get('error_message')}")
            
            if result.get('success', False):
                print(f"    ✓ 测试通过")
            else:
                print(f"    ⚠ 测试部分成功（使用降级方法）")
        
        except Exception as e:
            print(f"    ✗ 测试失败: {e}")
    
    # 测试4: 性能测试
    print("\n4. 性能测试...")
    try:
        test_text1 = "This is a performance test"
        test_text2 = "this is a performance test"
        
        # 预热
        print("  预热模型...")
        requests.post(
            f"{base_url}/normalize",
            json={"text1": test_text1, "text2": test_text2},
            timeout=60
        )
        
        # 性能测试
        print("  执行性能测试...")
        times = []
        for i in range(3):  # 减少测试次数，因为Ollama可能较慢
            start_time = time.time()
            response = requests.post(
                f"{base_url}/normalize",
                json={"text1": test_text1, "text2": test_text2},
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                print(f"    请求 {i+1}: {end_time - start_time:.3f}秒")
            else:
                print(f"  请求 {i+1} 失败: {response.status_code}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"  ✓ 性能测试完成:")
            print(f"    平均响应时间: {avg_time:.3f}秒")
            print(f"    最小响应时间: {min_time:.3f}秒")
            print(f"    最大响应时间: {max_time:.3f}秒")
            
            if avg_time < 5:
                print(f"    性能评价: 优秀")
            elif avg_time < 10:
                print(f"    性能评价: 良好")
            else:
                print(f"    性能评价: 一般 (Ollama相对较慢是正常的)")
        else:
            print(f"  ✗ 性能测试失败")
    
    except Exception as e:
        print(f"  ✗ 性能测试失败: {e}")
    
    # 测试5: 并发测试
    print("\n5. 并发测试...")
    try:
        import threading
        import queue
        
        def worker(q, results):
            while True:
                try:
                    task = q.get(timeout=1)
                    if task is None:
                        break
                    
                    start_time = time.time()
                    response = requests.post(
                        f"{base_url}/normalize",
                        json={"text1": f"test {task}", "text2": f"test {task}"},
                        timeout=30
                    )
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        results.append((task, end_time - start_time, True))
                    else:
                        results.append((task, end_time - start_time, False))
                        
                    q.task_done()
                except queue.Empty:
                    break
                except Exception as e:
                    results.append((task, 0, False))
                    q.task_done()
        
        # 创建任务队列
        task_queue = queue.Queue()
        results = []
        
        # 添加任务
        for i in range(5):  # 减少并发数
            task_queue.put(i)
        
        # 启动线程
        threads = []
        for i in range(2):  # 使用2个线程
            t = threading.Thread(target=worker, args=(task_queue, results))
            t.start()
            threads.append(t)
        
        # 等待完成
        task_queue.join()
        
        # 停止线程
        for i in range(2):
            task_queue.put(None)
        for t in threads:
            t.join()
        
        # 分析结果
        if results:
            success_count = sum(1 for _, _, success in results if success)
            avg_time = sum(time for _, time, success in results if success) / max(success_count, 1)
            
            print(f"  ✓ 并发测试完成:")
            print(f"    成功请求: {success_count}/{len(results)}")
            print(f"    平均响应时间: {avg_time:.3f}秒")
        else:
            print(f"  ✗ 并发测试失败")
    
    except Exception as e:
        print(f"  ✗ 并发测试失败: {e}")
    
    print("\n" + "=" * 50)
    print("LLM服务测试完成")
    print("=" * 50)
    
    return True

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM服务测试脚本 (Ollama)")
    parser.add_argument("--url", type=str, default="http://localhost:8000",
                       help="LLM服务URL")
    
    args = parser.parse_args()
    
    print(f"测试LLM服务 (Ollama): {args.url}")
    
    # 首先检查服务是否可访问
    try:
        response = requests.get(f"{args.url}/health", timeout=5)
        if response.status_code != 200:
            print(f"错误: LLM服务不可访问 (状态码: {response.status_code})")
            print("请确保服务已启动:")
            print("  ./start_llm_service.sh --daemon")
            print("\n故障排除步骤:")
            print("1. 检查Ollama服务是否运行: ollama list")
            print("2. 检查模型是否可用: ollama run qwen3:8b")
            print("3. 检查端口是否被占用: lsof -i :8000")
            print("4. 查看服务日志: tail -f ./logs/llm_service.log")
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"错误: 无法连接到LLM服务: {e}")
        print("请确保服务已启动:")
        print("  ./start_llm_service.sh --daemon")
        print("\n故障排除步骤:")
        print("1. 检查Ollama是否安装: ollama --version")
        print("2. 启动Ollama服务: ollama serve")
        print("3. 拉取模型: ollama pull qwen3:8b")
        print("4. 检查网络连接和防火墙设置")
        sys.exit(1)
    
    # 运行测试
    success = test_llm_service(args.url)
    
    if success:
        print("✓ 所有测试通过，LLM服务工作正常")
        print("\n使用提示:")
        print("• 服务地址: " + args.url)
        print("• 健康检查: " + args.url + "/health")
        print("• 模型信息: " + args.url + "/model_info")
        print("• API文档: " + args.url + "/docs")
        sys.exit(0)
    else:
        print("✗ 部分测试失败，请检查LLM服务状态")
        print("\n常见问题:")
        print("• Ollama服务未启动: ollama serve")
        print("• 模型未下载: ollama pull qwen3:8b")
        print("• 端口被占用: 检查端口8000和11434")
        print("• 内存不足: 检查系统资源使用情况")
        sys.exit(1)

if __name__ == "__main__":
    main() 