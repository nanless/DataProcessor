# LLM文本标准化服务 - Ollama版本

本服务使用 Ollama 部署 Qwen3-8B 模型，提供文本标准化功能。

## 系统要求

- Python 3.8+
- 足够的内存（建议8GB以上）
- 网络连接（用于下载模型）

## 安装步骤

### 1. 安装 Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. 安装 Python 依赖

```bash
pip install ollama fastapi uvicorn pydantic requests
```

### 3. 启动服务

```bash
# 后台启动模式（推荐）
./start_llm_service.sh --daemon

# 前台启动模式
./start_llm_service.sh
```

## 使用方法

### 基本用法

启动服务后，可以通过以下方式使用：

#### 1. 健康检查

```bash
curl http://localhost:8000/health
```

#### 2. 获取模型信息

```bash
curl http://localhost:8000/model_info
```

#### 3. 文本标准化

```bash
curl -X POST http://localhost:8000/normalize \
  -H "Content-Type: application/json" \
  -d '{
    "text1": "Hello World",
    "text2": "h e l l o w o r l d"
  }'
```

### Python 客户端示例

```python
import requests

def normalize_text_pair(text1, text2):
    response = requests.post(
        "http://localhost:8000/normalize",
        json={
            "text1": text1,
            "text2": text2
        }
    )
    return response.json()

# 使用示例
result = normalize_text_pair("Hello World", "h e l l o w o r l d")
print(result)
```

## 测试

### 运行完整测试

```bash
python test_llm_service.py
```

### 运行特定测试

```bash
python test_qwen3_4b.py
```

### 自定义测试

```bash
python test_llm_service.py --url http://localhost:8000
```

## 配置选项

### 环境变量

- `MODEL_NAME`: 模型名称（默认: qwen3:8b）
- `OLLAMA_HOST`: Ollama服务地址（默认: localhost:11434）
- `HOST`: 服务主机地址（默认: 0.0.0.0）
- `PORT`: 服务端口（默认: 8000）

### 启动参数

```bash
python llm_service.py --help
```

## 管理服务

### 启动服务

```bash
./start_llm_service.sh --daemon
```

### 停止服务

```bash
./stop_llm_service.sh
```

### 查看日志

```bash
tail -f ./logs/llm_service.log
tail -f ./logs/ollama.log
```

### 检查服务状态

```bash
# 检查LLM服务
curl http://localhost:8000/health

# 检查Ollama服务
curl http://localhost:11434/api/version
```

## 故障排除

### 常见问题

1. **Ollama服务无法启动**
   ```bash
   # 检查Ollama安装
   ollama --version
   
   # 手动启动Ollama服务
   ollama serve
   ```

2. **模型未找到**
   ```bash
   # 拉取模型
   ollama pull qwen3:8b
   
   # 检查已安装的模型
   ollama list
   ```

3. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :8000
   lsof -i :11434
   
   # 杀死占用进程
   kill -9 <PID>
   ```

4. **内存不足**
   ```bash
   # 检查内存使用
   free -h
   
   # 使用更小的模型
   export MODEL_NAME=qwen3:7b
   ```

### 调试模式

```bash
# 前台运行以查看详细输出
./start_llm_service.sh

# 查看详细日志
tail -f ./logs/llm_service.log
```

## 性能优化

### 1. 模型选择

- `qwen3:8b`: 默认模型，平衡性能和质量
- `qwen3:7b`: 较小模型，更快但质量稍低
- `qwen3:14b`: 更大模型，质量更高但速度较慢

### 2. 系统资源

- 建议至少8GB内存
- SSD存储可提高模型加载速度
- 多核CPU可提高并发处理能力

### 3. 网络配置

- 确保端口8000和11434可访问
- 如需远程访问，修改HOST配置

## API 文档

启动服务后，可访问以下地址查看详细API文档：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 更新日志

### v2.0.0
- 从vLLM迁移到Ollama
- 模型从Qwen3-4B升级到Qwen3-8B
- 简化部署流程
- 增强错误处理和日志记录

## 支持

如遇到问题，请检查：

1. 系统日志: `./logs/llm_service.log`
2. Ollama日志: `./logs/ollama.log`
3. 服务状态: `curl http://localhost:8000/health`
4. 模型状态: `ollama list`

## 许可证

本项目采用MIT许可证。 