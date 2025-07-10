#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试GPU配置和LLM服务的脚本
"""

import os
import torch
import requests
from typing import List

# 设置代理绕过
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

def test_gpu_availability():
    """测试GPU可用性"""
    print("=" * 50)
    print("测试GPU可用性")
    print("=" * 50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    device_count = torch.cuda.device_count()
    print(f"✓ 检测到 {device_count} 个GPU")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}, 显存: {props.total_memory / 1024**3:.1f}GB")
    
    return True

def test_llm_service():
    """测试LLM服务"""
    print("\n" + "=" * 50)
    print("测试LLM服务")
    print("=" * 50)
    
    # 创建会话，绕过代理
    session = requests.Session()
    session.proxies = {'http': None, 'https': None}
    
    try:
        # 健康检查
        health_url = "http://localhost:8000/health"
        response = session.get(health_url, timeout=10)
        response.raise_for_status()
        print("✓ LLM服务健康检查通过")
        
        # 获取模型信息
        model_info_url = "http://localhost:8000/model_info"
        response = session.get(model_info_url, timeout=10)
        response.raise_for_status()
        info = response.json()
        print(f"✓ LLM模型: {info.get('model_name', 'unknown')}")
        print(f"✓ 服务类型: {info.get('service_type', 'unknown')}")
        print(f"✓ 状态: {info.get('status', 'unknown')}")
        
        # 测试文本标准化
        test_data = {
            "text1": "Hello World Test",
            "text2": "hello world test"
        }
        
        normalize_url = "http://localhost:8000/normalize"
        response = session.post(normalize_url, json=test_data, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"✓ 文本标准化测试通过")
        print(f"  原始文本1: '{test_data['text1']}'")
        print(f"  标准化后1: '{result['normalized_text1']}'")
        print(f"  原始文本2: '{test_data['text2']}'")
        print(f"  标准化后2: '{result['normalized_text2']}'")
        print(f"  成功状态: {result['success']}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLM服务测试失败: {e}")
        return False

def test_gpu_assignment():
    """测试GPU分配"""
    print("\n" + "=" * 50)
    print("测试GPU分配")
    print("=" * 50)
    
    asr_gpus = [1, 2, 3]  # ASR使用的GPU
    llm_gpu = 0  # LLM使用的GPU
    
    print(f"计划配置:")
    print(f"  GPU {llm_gpu}: LLM服务 (qwen2.5:7b)")
    print(f"  GPU {asr_gpus}: ASR服务 (Kimi-Audio)")
    
    # 检查GPU是否可用
    if not torch.cuda.is_available():
        print("❌ CUDA不可用")
        return False
    
    device_count = torch.cuda.device_count()
    max_required = max(max(asr_gpus), llm_gpu)
    
    if device_count <= max_required:
        print(f"❌ 可用GPU数量({device_count})不足，需要GPU {max_required}")
        return False
    
    print(f"✓ GPU分配可行，系统有{device_count}个GPU")
    
    # 模拟设置CUDA_VISIBLE_DEVICES
    print(f"\n推荐启动命令:")
    print(f"1. 启动Ollama服务 (GPU {llm_gpu}):")
    print(f"   CUDA_VISIBLE_DEVICES={llm_gpu} ollama serve &")
    print(f"2. 启动LLM服务 (GPU {llm_gpu}):")
    print(f"   CUDA_VISIBLE_DEVICES={llm_gpu} python llm_service.py --model_name qwen2.5:7b --host 0.0.0.0 --port 8000")
    print(f"3. 启动ASR评估 (GPU {asr_gpus}):")
    print(f"   python enhancement_audio_quality_assessment.py --gpu_ids {' '.join(map(str, asr_gpus))}")
    
    return True

def main():
    """主函数"""
    print("GPU配置和LLM服务测试脚本")
    print("检查系统配置是否满足要求...")
    
    success = True
    
    # 测试GPU
    if not test_gpu_availability():
        success = False
    
    # 测试LLM服务
    if not test_llm_service():
        success = False
    
    # 测试GPU分配
    if not test_gpu_assignment():
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("✅ 所有测试通过，系统配置正确")
        print("可以运行音频质量评估脚本")
    else:
        print("❌ 某些测试失败，请检查配置")
        print("确保:")
        print("1. CUDA和GPU可用")
        print("2. LLM服务在GPU 0上运行")
        print("3. 网络代理设置正确")
    print("=" * 50)

if __name__ == "__main__":
    main() 