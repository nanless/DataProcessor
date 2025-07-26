#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM服务脚本
使用Ollama部署Qwen2.5-32B模型
提供HTTP API接口用于文本标准化
"""

import os
import json
import logging
import argparse
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import requests
import time

# 设置代理绕过，确保可以访问本地Ollama服务
os.environ['no_proxy'] = 'localhost,127.0.0.1,::1'
os.environ['NO_PROXY'] = 'localhost,127.0.0.1,::1'

# 尝试导入ollama库
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextNormalizationRequest(BaseModel):
    """文本标准化请求模型"""
    text1: str
    text2: str

class TextNormalizationResponse(BaseModel):
    """文本标准化响应模型"""
    normalized_text1: str
    normalized_text2: str
    success: bool
    error_message: Optional[str] = None

class OllamaLLMService:
    """Ollama LLM服务类"""
    
    def __init__(self, model_name: str = "qwen2.5:32b", ollama_host: str = "localhost:11434"):
        """
        初始化Ollama LLM服务
        
        Args:
            model_name: Ollama模型名称
            ollama_host: Ollama服务地址
        """
        self.model_name = model_name
        self.ollama_host = ollama_host
        self.ollama_url = f"http://{ollama_host}"
        self.client = None
        
        logger.info(f"Ollama LLM服务配置:")
        logger.info(f"  模型名称: {self.model_name}")
        logger.info(f"  Ollama服务地址: {self.ollama_url}")
        
    def check_ollama_service(self, max_retries=5, retry_delay=2):
        """检查Ollama服务是否运行，支持重试"""
        # 创建一个绕过代理的session
        session = requests.Session()
        session.proxies = {
            'http': None,
            'https': None
        }
        
        for attempt in range(max_retries):
            try:
                response = session.get(f"{self.ollama_url}/api/version", timeout=10)
                response.raise_for_status()
                version_info = response.json()
                logger.info(f"✓ Ollama服务正常运行，版本: {version_info.get('version', 'unknown')}")
                return True
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Ollama服务检查失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                else:
                    logger.error(f"✗ Ollama服务在 {max_retries} 次尝试后仍不可用: {e}")
                    return False
            except Exception as e:
                logger.error(f"✗ Ollama服务检查出现意外错误: {e}")
                return False
    
    def load_model(self):
        """加载/检查模型"""
        try:
            if not OLLAMA_AVAILABLE:
                raise ImportError("ollama库未安装，请运行: pip install ollama")
            
            # 检查Ollama服务（支持重试）
            logger.info("正在检查Ollama服务连接...")
            if not self.check_ollama_service(max_retries=10, retry_delay=3):
                logger.warning("Ollama服务检查失败，但继续尝试初始化客户端...")
            
            # 初始化Ollama客户端
            try:
                self.client = ollama.Client(host=self.ollama_url)
                logger.info("✓ Ollama客户端初始化成功")
            except Exception as e:
                logger.error(f"Ollama客户端初始化失败: {e}")
                raise
            
            # 检查模型是否可用（添加重试逻辑）
            for attempt in range(3):
                try:
                    models = self.client.list()
                    # ollama客户端返回的是ListResponse对象，模型属性是model而不是name
                    available_models = [model.model for model in models.models]
                    
                    if self.model_name not in available_models:
                        logger.warning(f"模型 {self.model_name} 未找到，尝试拉取...")
                        self.client.pull(self.model_name)
                        logger.info(f"✓ 成功拉取模型: {self.model_name}")
                    else:
                        logger.info(f"✓ 找到模型: {self.model_name}")
                    break
                    
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"检查模型失败 (尝试 {attempt + 1}/3): {e}")
                        time.sleep(5)
                    else:
                        logger.error(f"检查/拉取模型失败: {e}")
                        raise
            
            # 测试模型推理（可选，失败不影响启动）
            try:
                logger.info("测试模型推理...")
                test_response = self.client.generate(
                    model=self.model_name,
                    prompt="Test",
                    options={"max_tokens": 5, "temperature": 0}
                )
                logger.info(f"✓ 模型推理测试成功")
                
            except Exception as e:
                logger.warning(f"模型推理测试失败，但服务继续启动: {e}")
                # 不抛出异常，让服务继续启动
            
            logger.info("✓ Ollama模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            logger.warning("服务将以降级模式启动，仅提供基础文本标准化功能")
            # 不抛出异常，让服务继续启动
            self.client = None
    
    def normalize_text_pair(self, text1: str, text2: str) -> Dict[str, str]:
        """
        使用Ollama LLM标准化文本对
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            包含标准化结果的字典
        """
        # 检查Ollama客户端是否可用
        if self.client is None:
            logger.error(f"Ollama客户端不可用，无法进行LLM文本标准化")
            raise RuntimeError("Ollama服务不可用，无法进行LLM文本标准化")
        
        try:
            # 构造提示词
            prompt = self._build_normalization_prompt(text1, text2)
            
            # 调用Ollama API
            response = self.client.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.3,
                    "top_p": 0.8,
                    "max_tokens": 512,
                    "stop": ["<|endoftext|>", "<|im_end|>"]
                }
            )
            
            if not response or 'response' not in response:
                raise RuntimeError("Ollama生成失败")
            
            # 解析响应
            response_text = response['response'].strip()
            
            # 提取标准化后的文本
            normalized_text1, normalized_text2 = self._extract_normalized_texts(response_text, text1, text2)
            
            logger.info(f"Ollama标准化完成")
            logger.info(f"原始文本1: '{text1}' -> 标准化: '{normalized_text1}'")
            logger.info(f"原始文本2: '{text2}' -> 标准化: '{normalized_text2}'")
            
            return {
                "normalized_text1": normalized_text1,
                "normalized_text2": normalized_text2,
                "success": True,
                "error_message": None
            }
            
        except Exception as e:
            logger.error(f"Ollama文本标准化失败: {e}")
            # 只使用LLM，失败时直接返回错误
            raise RuntimeError(f"LLM文本标准化失败: {str(e)}")
    
    def _build_normalization_prompt(self, text1: str, text2: str) -> str:
        """构造标准化提示词"""
        prompt = f"""你是语音识别文本标准化专家。请将两段文本标准化为统一格式，用于WER/CER计算。

# 核心原则
- 只做格式标准化，不改变语义内容
- 保持原文的语言错误（如同音词误用）
- 两个文本必须使用完全相同的标准化规则

# 标准化规则

## 英语标准化：
**基础处理**:
- 转小写: "Hello" → "hello"
- 去标点: "Hello!" → "hello" 
- 去连字符: "twenty-one" → "twenty one"

**缩写展开**:
- "don't" → "do not"
- "I'm" → "i am" 
- "can't" → "cannot"
- "won't" → "will not"
- "it's" → "it is" (永远展开为"it is"，不是"its")

**数字转换**:
- 单独数字: "3" → "three"
- 连续数字: "123" → "one hundred twenty three"
- 年份: "2023" → "two thousand twenty three"

**字母拼读统一规则** (重要):
- 检查两个文本中是否有字母序列
- 如果一个是拼读形式(如"a b c")，另一个是连写(如"abc")
- 统一转换为拼读形式: "abc" → "a b c"
- 示例: "My name is ABC" + "my name is a b c" → 都变成 "my name is a b c"

## 中文标准化：
- 去标点: "你好！" → "你好"
- 去空格: "你 好" → "你好" 
- 繁转简: "這個" → "这个"
- 数字转换: "3个" → "三个", "2023年" → "二零二三年"
- **同音词保持原样**: "在家"保持"在家", "再家"保持"再家" (不纠错)

# 处理步骤示例

输入: "My name is A B C." + "my name is abc"
步骤:
1. 基础标准化: "my name is a b c" + "my name is abc"  
2. 检测拼读模式: 第一个已是拼读，第二个是连写
3. 统一为拼读: "my name is a b c" + "my name is a b c"

输入: "我在家里。" + "我再家里"
步骤:
1. 去标点: "我在家里" + "我再家里"
2. 保持同音词原样: "我在家里" + "我再家里" (不改变"再")

输入: "Hello, it's ok!" + "hello it is okay"  
步骤:
1. 基础处理: "hello its ok" + "hello it is okay"
2. 缩写展开: "hello it is ok" + "hello it is okay"  
3. 注意"it's"必须展开为"it is"

---

待处理文本:
文本1: "{text1}"
文本2: "{text2}"

按规则处理后直接返回，注意格式：

标准化文本1: 处理结果
标准化文本2: 处理结果

重要：
- 只返回处理后的纯文本，不要加引号、方括号等任何标记
- 确保两个文本使用完全相同的处理规则"""
        
        return prompt
    
    def _extract_normalized_texts(self, response: str, original_text1: str, original_text2: str) -> tuple:
        """从Ollama响应中提取标准化后的文本"""
        import re
        
        try:
            logger.info(f"Ollama响应: {response}")
            
            # 查找标准化文本的模式，支持多种格式
            patterns = [
                # 标准格式（主要格式）
                (r'标准化文本1[:：]\s*(.+?)(?=\n标准化文本2|标准化文本2|$)', r'标准化文本2[:：]\s*(.+?)(?=\n|$)'),
                # 方括号格式
                (r'标准化文本1[:：]\s*\[(.+?)\]', r'标准化文本2[:：]\s*\[(.+?)\]'),
                # 简化格式
                (r'文本1[:：]\s*(.+?)(?=\n文本2|文本2|$)', r'文本2[:：]\s*(.+?)(?=\n|$)'),
                # 编号格式
                (r'1\.\s*(.+?)(?=\n2\.|2\.|$)', r'2\.\s*(.+?)(?=\n|$)'),
                # 英文格式
                (r'Normalized Text 1[:：]\s*(.+?)(?=\nNormalized Text 2|Normalized Text 2|$)', 
                 r'Normalized Text 2[:：]\s*(.+?)(?=\n|$)'),
                # 混合格式（处理多行响应）
                (r'(?:标准化文本1|Text 1)[:：]\s*(.+?)(?=\n(?:标准化文本2|Text 2)|(?:标准化文本2|Text 2)|$)', 
                 r'(?:标准化文本2|Text 2)[:：]\s*(.+?)(?=\n|$)'),
            ]
            
            for pattern1, pattern2 in patterns:
                match1 = re.search(pattern1, response, re.DOTALL | re.IGNORECASE)
                match2 = re.search(pattern2, response, re.DOTALL | re.IGNORECASE)
                
                if match1 and match2:
                    norm_text1 = match1.group(1).strip()
                    norm_text2 = match2.group(1).strip()
                    
                    # 清理各种可能的标记和格式
                    norm_text1 = re.sub(r'^\[|]$|^"|"$|^\'|\'$', '', norm_text1).strip()
                    norm_text2 = re.sub(r'^\[|]$|^"|"$|^\'|\'$', '', norm_text2).strip()
                    
                    # 验证提取的文本是否有效
                    if norm_text1 and norm_text2:
                        logger.info(f"成功提取标准化文本")
                        return norm_text1, norm_text2
            
            # 如果没有匹配到，尝试更简单的提取方法
            lines = response.strip().split('\n')
            extracted_texts = []
            
            for line in lines:
                line = line.strip()
                # 寻找包含实际文本内容的行
                if line and not line.startswith(('标准化', '文本', '规则', '请', '以下', '格式')):
                    # 移除可能的编号和标点
                    cleaned_line = re.sub(r'^\d+\.\s*|^[-*]\s*|^[:：]\s*', '', line).strip()
                    if cleaned_line:
                        extracted_texts.append(cleaned_line)
            
            if len(extracted_texts) >= 2:
                logger.info(f"使用简化提取方法成功")
                return extracted_texts[0], extracted_texts[1]
            
            logger.error(f"无法从Ollama响应中提取标准化文本")
            raise RuntimeError("LLM响应格式无法解析")
                
        except Exception as e:
            logger.error(f"提取标准化文本失败: {e}")
            raise RuntimeError(f"LLM响应解析失败: {str(e)}")
    


# 全局LLM服务实例
llm_service = None

async def lifespan(app):
    """应用生命周期管理"""
    global llm_service
    
    # 启动时初始化
    try:
        # 从环境变量获取配置
        model_name = os.getenv("LLM_MODEL_NAME", "qwen2.5:32b")
        ollama_host = os.getenv("OLLAMA_HOST", "localhost:11434")
        
        llm_service = OllamaLLMService(
            model_name=model_name,
            ollama_host=ollama_host
        )
        
        # 加载模型
        llm_service.load_model()
        
        logger.info("LLM服务启动完成")
        
        yield  # 应用运行期间
        
    except Exception as e:
        logger.error(f"LLM服务启动失败: {e}")
        # 即使启动失败也要继续，这样服务至少可以返回错误信息
        yield
    
    # 关闭时清理（如果需要）
    logger.info("LLM服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="LLM文本标准化服务", 
    description="使用Ollama Qwen2.5-32B进行文本标准化",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "healthy", "service": "LLM文本标准化服务 (Ollama)"}

@app.post("/normalize", response_model=TextNormalizationResponse)
async def normalize_text(request: TextNormalizationRequest):
    """文本标准化接口"""
    try:
        if llm_service is None:
            raise HTTPException(status_code=503, detail="LLM服务未初始化")
        
        # 验证输入
        if not request.text1 or not request.text2:
            raise HTTPException(status_code=400, detail="文本不能为空")
        
        # 调用LLM进行标准化
        result = llm_service.normalize_text_pair(request.text1, request.text2)
        
        return TextNormalizationResponse(
            normalized_text1=result["normalized_text1"],
            normalized_text2=result["normalized_text2"],
            success=result["success"],
            error_message=result["error_message"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"文本标准化失败: {e}")
        raise HTTPException(status_code=500, detail=f"内部服务错误: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """获取模型信息"""
    if llm_service is None:
        raise HTTPException(status_code=503, detail="LLM服务未初始化")
    
    # 检查服务状态
    ollama_available = llm_service.client is not None
    
    info = {
        "model_name": llm_service.model_name,
        "ollama_host": llm_service.ollama_host,
        "service_type": "ollama",
        "status": "online" if ollama_available else "degraded",
        "ollama_available": ollama_available
    }
    
    if not ollama_available:
        info["warning"] = "Ollama服务不可用，仅提供基础文本标准化功能"
    
    return info

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM文本标准化服务 (Ollama)")
    parser.add_argument("--model_name", type=str, 
                       default="qwen2.5:32b",
                       help="Ollama模型名称")
    parser.add_argument("--ollama_host", type=str, default="localhost:11434",
                       help="Ollama服务地址")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                       help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000,
                       help="服务端口")
    parser.add_argument("--workers", type=int, default=1,
                       help="工作进程数")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["LLM_MODEL_NAME"] = args.model_name
    os.environ["OLLAMA_HOST"] = args.ollama_host
    
    print("LLM文本标准化服务 (Ollama)")
    print("=" * 50)
    print(f"模型名称: {args.model_name}")
    print(f"Ollama服务地址: {args.ollama_host}")
    print(f"服务地址: {args.host}:{args.port}")
    print(f"工作进程数: {args.workers}")
    print("=" * 50)
    
    # 启动服务
    uvicorn.run(
        "llm_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=False,
        access_log=True
    )

if __name__ == "__main__":
    main() 