#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本标准化Prompt测试脚本
用于验证改进后的prompt在各种场景下的效果
"""

import sys
import os
import requests
import json
from typing import List, Tuple

# 清除代理设置，确保可以访问本地服务
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None) 
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

# 测试用例：包含各种常见的语音识别文本标准化场景
TEST_CASES = [
    # 英语测试用例
    {
        "name": "英语基础测试",
        "text1": "Hello, world! It's 2023.",
        "text2": "hello world its two thousand twenty three",
        "expected_similar": True
    },
    {
        "name": "英语缩写词测试", 
        "text1": "I don't think it's okay.",
        "text2": "i do not think its okay",
        "expected_similar": True
    },
    {
        "name": "英语数字测试",
        "text1": "I have 3 apples and 25 oranges.",
        "text2": "i have three apples and twenty five oranges", 
        "expected_similar": True
    },
    {
        "name": "英语字母拼读测试",
        "text1": "My name is A B C.",
        "text2": "my name is abc",
        "expected_similar": True
    },
    {
        "name": "英语连字符测试",
        "text1": "Twenty-one state-of-the-art devices.",
        "text2": "twenty one state of the art devices",
        "expected_similar": True
    },
    
    # 中文测试用例
    {
        "name": "中文基础测试",
        "text1": "你好，世界！这是测试。",  
        "text2": "你好世界这是测试",
        "expected_similar": True
    },
    {
        "name": "中文数字测试",
        "text1": "我有3个苹果和25个橘子。",
        "text2": "我有三个苹果和二十五个橘子",
        "expected_similar": True
    },
    {
        "name": "中文繁简转换测试",
        "text1": "這個經過測試的產品。",
        "text2": "这个经过测试的产品",
        "expected_similar": True
    },
    {
        "name": "中文年份测试",
        "text1": "今年是2023年。",
        "text2": "今年是二零二三年",
        "expected_similar": True
    },
    
    # 混合语言测试用例
    {
        "name": "中英混合测试",
        "text1": "我有一个iPhone，连接WiFi。",
        "text2": "我有一个iphone连接wifi",
        "expected_similar": True
    },
    
    # 语音识别特殊情况
    {
        "name": "重复词测试",
        "text1": "我我觉得这个不错。",
        "text2": "我觉得这个不错",
        "expected_similar": True
    },
    {
        "name": "语气词测试", 
        "text1": "Um, I think, uh, it's good.",
        "text2": "um i think uh its good",
        "expected_similar": True
    },
    
    # 同音词测试
    {
        "name": "同音词测试",
        "text1": "我在家里。",
        "text2": "我再家里",
        "expected_similar": False  # 这种情况需要根据上下文判断
    }
]

class TextNormalizationTester:
    """文本标准化测试器"""
    
    def __init__(self, llm_service_url="http://127.0.0.1:8000"):
        self.llm_service_url = llm_service_url
        
    def test_service_availability(self) -> bool:
        """测试LLM服务是否可用"""
        try:
            response = requests.get(f"{self.llm_service_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ LLM服务可用")
                return True
            else:
                print("❌ LLM服务响应异常")
                return False
        except Exception as e:
            print(f"❌ LLM服务不可用: {e}")
            return False
    
    def normalize_text_pair(self, text1: str, text2: str) -> Tuple[str, str, bool]:
        """调用LLM服务进行文本标准化"""
        try:
            response = requests.post(
                f"{self.llm_service_url}/normalize",
                json={"text1": text1, "text2": text2},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return (
                    result["normalized_text1"],
                    result["normalized_text2"], 
                    result["success"]
                )
            else:
                print(f"❌ API请求失败: {response.status_code}")
                return "", "", False
                
        except Exception as e:
            print(f"❌ 标准化请求失败: {e}")
            return "", "", False
    
    def run_single_test(self, test_case: dict) -> dict:
        """运行单个测试用例"""
        print(f"\n🧾 测试: {test_case['name']}")
        print(f"   原文1: {test_case['text1']}")
        print(f"   原文2: {test_case['text2']}")
        
        norm1, norm2, success = self.normalize_text_pair(
            test_case['text1'], test_case['text2']
        )
        
        if not success:
            return {
                "name": test_case['name'],
                "success": False,
                "error": "LLM标准化失败"
            }
        
        print(f"   标准化1: {norm1}")
        print(f"   标准化2: {norm2}")
        
        # 判断标准化结果
        texts_similar = norm1.strip() == norm2.strip()
        expected_similar = test_case.get('expected_similar', True)
        
        test_passed = texts_similar == expected_similar
        
        if test_passed:
            print("   ✅ 测试通过")
        else:
            expected_desc = "应该相似" if expected_similar else "应该不同"
            actual_desc = "相似" if texts_similar else "不同"
            print(f"   ❌ 测试失败: {expected_desc}，实际{actual_desc}")
        
        return {
            "name": test_case['name'],
            "success": success,
            "test_passed": test_passed,
            "original_text1": test_case['text1'],
            "original_text2": test_case['text2'],
            "normalized_text1": norm1,
            "normalized_text2": norm2,
            "texts_similar": texts_similar,
            "expected_similar": expected_similar
        }
    
    def run_all_tests(self) -> dict:
        """运行所有测试用例"""
        print("🚀 开始文本标准化测试")
        print("=" * 60)
        
        if not self.test_service_availability():
            return {"error": "LLM服务不可用"}
        
        results = []
        passed_count = 0
        
        for test_case in TEST_CASES:
            result = self.run_single_test(test_case)
            results.append(result)
            
            if result.get('test_passed', False):
                passed_count += 1
        
        # 输出总结
        print("\n" + "=" * 60)
        print(f"📊 测试总结:")
        print(f"   总测试数: {len(TEST_CASES)}")
        print(f"   通过数量: {passed_count}")
        print(f"   失败数量: {len(TEST_CASES) - passed_count}")
        print(f"   通过率: {passed_count/len(TEST_CASES)*100:.1f}%")
        
        return {
            "total_tests": len(TEST_CASES),
            "passed_tests": passed_count,
            "failed_tests": len(TEST_CASES) - passed_count,
            "pass_rate": passed_count/len(TEST_CASES)*100,
            "results": results
        }
    
    def save_results(self, results: dict, filename: str = "text_normalization_test_results.json"):
        """保存测试结果到文件"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"📁 测试结果已保存到: {filename}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="文本标准化Prompt测试")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8001",
                       help="LLM服务URL")
    parser.add_argument("--save", type=str, default="",
                       help="保存结果的文件名")
    
    args = parser.parse_args()
    
    tester = TextNormalizationTester(args.url)
    results = tester.run_all_tests()
    
    if args.save:
        tester.save_results(results, args.save)
    
    # 返回适当的退出码
    if 'error' in results:
        sys.exit(1)
    elif results['failed_tests'] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main() 