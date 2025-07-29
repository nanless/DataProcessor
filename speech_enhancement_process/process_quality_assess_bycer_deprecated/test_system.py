#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音频质量评估系统测试脚本
基本的系统功能验证
"""

import os
import sys
import json
import requests
import torch
from pathlib import Path

# 设置代理绕过，确保可以访问本地LLM服务
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

def test_gpu_availability():
    """测试GPU可用性"""
    print("🔍 检查GPU可用性...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，请检查GPU驱动")
        return False
    
    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 张GPU")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    return True

def test_config_files():
    """测试配置文件"""
    print("\n🔍 检查配置文件...")
    
    config_file = "multi_directory_config_example.json"
    if not os.path.exists(config_file):
        print(f"❌ 配置文件不存在: {config_file}")
        return False
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 检查必要字段
        required_fields = [
            'directory_configs',
            'kimi_model_path', 
            'kimi_audio_dir',
            'llm_model_config'
        ]
        
        for field in required_fields:
            if field not in config:
                print(f"❌ 配置文件缺少必要字段: {field}")
                return False
        
        model_config = config['llm_model_config']
        print(f"✅ 配置文件有效")
        print(f"   模型: {model_config.get('model_name', 'unknown')}")
        print(f"   类型: {model_config.get('model_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件解析失败: {e}")
        return False

def test_llm_services():
    """测试LLM服务连接"""
    print("\n🔍 检查LLM服务...")
    
    # 创建不使用代理的session
    session = requests.Session()
    session.proxies = {'http': None, 'https': None}
    
    # 检查可能运行的服务端口
    active_services = []
    
    for port in range(8000, 8010):
        try:
            response = session.get(f"http://localhost:{port}/health", timeout=3)
            if response.status_code == 200:
                active_services.append(port)
                print(f"✅ LLM服务(端口{port}) 运行正常")
        except:
            continue
    
    if not active_services:
        print("⚠️  未检测到运行中的LLM服务")
        print("   提示: 运行 ./auto_start_llm_services.sh 启动服务")
        return False
    
    # 测试文本标准化功能
    test_port = active_services[0]
    try:
        test_data = {
            "text1": "Hello, world!",
            "text2": "hello world"
        }
        
        response = session.post(
            f"http://localhost:{test_port}/normalize",
            json=test_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ 文本标准化功能正常")
                return True
            else:
                print(f"❌ 文本标准化失败: {result.get('error_message')}")
                return False
        else:
            print(f"❌ API请求失败: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ 文本标准化测试失败: {e}")
        return False

def test_scripts():
    """测试核心脚本文件"""
    print("\n🔍 检查核心脚本...")
    
    required_scripts = [
        "auto_start_llm_services.sh",
        "enhancement_audio_quality_assessment.py", 
        "llm_service.py",
        "run_multi_directory_assessment_auto_gpu.sh",
        "stop_multi_llm_services.sh",
        "monitor_and_restart_llm.sh"
    ]
    
    all_exist = True
    for script in required_scripts:
        if os.path.exists(script):
            print(f"✅ {script}")
        else:
            print(f"❌ {script} 不存在")
            all_exist = False
    
    return all_exist

def test_directories():
    """测试关键目录"""
    print("\n🔍 检查关键目录...")
    
    # 创建日志目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    print(f"✅ 日志目录: {log_dir}")
    
    # 检查配置文件中的路径
    try:
        with open("multi_directory_config_example.json", 'r') as f:
            config = json.load(f)
        
        kimi_model_path = config.get('kimi_model_path', '')
        kimi_audio_dir = config.get('kimi_audio_dir', '')
        
        if os.path.exists(kimi_model_path):
            print(f"✅ Kimi模型路径: {kimi_model_path}")
        else:
            print(f"⚠️  Kimi模型路径不存在: {kimi_model_path}")
        
        if os.path.exists(kimi_audio_dir):
            print(f"✅ Kimi代码目录: {kimi_audio_dir}")
        else:
            print(f"⚠️  Kimi代码目录不存在: {kimi_audio_dir}")
            
    except:
        print("⚠️  无法验证配置文件中的路径")
    
    return True

def main():
    """主测试函数"""
    print("🚀 音频质量评估系统测试")
    print("=" * 50)
    
    tests = [
        ("GPU可用性", test_gpu_availability),
        ("配置文件", test_config_files),
        ("核心脚本", test_scripts),
        ("关键目录", test_directories),
        ("LLM服务", test_llm_services),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name}测试出错: {e}")
    
    print("\n" + "=" * 50)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！系统准备就绪。")
        print("\n下一步: 运行 ./run_multi_directory_assessment_auto_gpu.sh")
        return 0
    else:
        print("⚠️  部分测试失败，请检查上述问题。")
        return 1

if __name__ == "__main__":
    exit(main()) 