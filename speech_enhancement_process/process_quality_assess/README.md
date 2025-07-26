# 音频质量评估系统

一个基于深度学习的音频增强质量评估系统，支持自动GPU配置和多目录批量处理，使用Kimi-Audio进行语音识别，采用Qwen3-32B大语言模型进行文本标准化。

## 核心特性

- **自适应GPU配置**: 自动检测4卡或8卡GPU环境，优化资源分配
- **多目录批量处理**: 支持同时评估多个数据集或算法结果  
- **LLM文本标准化**: 使用Qwen3-32B模型进行智能文本标准化
- **TEN VAD集成**: 先进的语音活动检测技术
- **灵活的音量匹配**: 支持多种音量匹配策略
- **详细的评估报告**: 生成WER/CER指标和处理统计

## 系统架构

### GPU配置方案

#### 4卡配置 (推荐最小配置)
```
GPU 0: Qwen3-32B LLM服务 (端口8000)
GPU 1: Kimi-Audio ASR处理
GPU 2: Kimi-Audio ASR处理  
GPU 3: Kimi-Audio ASR处理
```

#### 8卡配置 (高性能配置)
```
组1:
  GPU 0: Qwen3-32B LLM服务 (端口8000)
  GPU 1,2,3: Kimi-Audio ASR处理

组2:
  GPU 4: Qwen3-32B LLM服务 (端口8001)
  GPU 5,6,7: Kimi-Audio ASR处理
```

### 处理流程
1. **音频预处理**: 重采样、音量匹配、格式转换
2. **语音识别**: 使用Kimi-Audio进行转录
3. **文本标准化**: 使用Qwen3-32B LLM智能标准化
4. **质量评估**: 计算WER/CER指标
5. **结果输出**: 生成详细报告

## 快速开始

### 环境要求

**硬件要求**:
- GPU: NVIDIA A100/H100 (32GB+ VRAM)
- 内存: 64GB+ RAM
- 存储: 500GB+ 可用空间

**软件要求**:
- 操作系统: Linux Ubuntu 20.04+
- Python: 3.8+
- CUDA: 11.8+
- Docker: 可选，用于容器化部署

### 安装依赖

```bash
# 激活conda环境
conda activate kimi-audio

# 安装Python依赖
pip install -r requirements.txt

# 安装Ollama (如果未安装)
curl -fsSL https://ollama.ai/install.sh | sh
```

### 一键运行

```bash
cd speech_enhancement_process/process_quality_assess

# 自动启动服务并运行评估
./run_multi_directory_assessment_auto_gpu.sh
```

### 分步执行

```bash
# 1. 启动LLM服务
./auto_start_llm_services.sh

# 2. 运行音频质量评估
python3 enhancement_audio_quality_assessment.py --config_file config.json

# 3. 停止所有服务
./stop_multi_llm_services.sh
```

## 配置说明

### 基础配置文件

创建配置文件 `config.json`:

```json
{
  "directory_configs": [
    {
      "name": "dataset_name",
      "original_dir": "/path/to/original/audio",
      "enhanced_dir": "/path/to/enhanced/audio", 
      "output_dir": "/path/to/output/results",
      "volume_matching": true,
      "volume_matching_method": "ten_vad_energy"
    }
  ],
  "kimi_model_path": "/path/to/Kimi-Audio-7B-Instruct",
  "kimi_audio_dir": "/path/to/Kimi-Audio",
  "use_llm_normalization": true,
  "llm_service_url": "http://localhost:8000",
  "llm_timeout": 30,
  "llm_max_retries": 3,
  "ten_vad_hop_size": 256,
  "ten_vad_threshold": 0.5,
  "max_audio_length": 600,
  "skip_existing": false
}
```

### 配置参数详解

#### 目录配置 (directory_configs)
- `name`: 配置名称，用于日志和报告标识
- `original_dir`: 原始音频文件目录
- `enhanced_dir`: 增强音频文件目录
- `output_dir`: 评估结果输出目录
- `volume_matching`: 是否启用音量匹配
- `volume_matching_method`: 音量匹配方法 (`ten_vad_energy` 推荐)

#### 模型配置
- `kimi_model_path`: Kimi-Audio模型路径
- `kimi_audio_dir`: Kimi-Audio代码库路径

#### LLM配置
- `use_llm_normalization`: 启用LLM文本标准化
- `llm_service_url`: LLM服务地址
- `llm_timeout`: LLM服务超时时间(秒)
- `llm_max_retries`: LLM服务最大重试次数

#### TEN VAD配置
- `ten_vad_hop_size`: VAD帧跳跃大小(样本数)
- `ten_vad_threshold`: VAD检测阈值(0.0-1.0)

#### 其他配置
- `max_audio_length`: 最大音频长度(秒)
- `skip_existing`: 跳过已存在的结果文件

### 多目录批量处理

支持同时处理多个数据集：

```json
{
  "directory_configs": [
    {
      "name": "dataset_A_zipenhancer",
      "original_dir": "/data/dataset_A/original",
      "enhanced_dir": "/data/dataset_A/zipenhancer_enhanced",
      "output_dir": "/results/dataset_A_zipenhancer"
    },
    {
      "name": "dataset_A_mossformer", 
      "original_dir": "/data/dataset_A/original",
      "enhanced_dir": "/data/dataset_A/mossformer_enhanced",
      "output_dir": "/results/dataset_A_mossformer"
    }
  ]
}
```

## 使用指南

### 命令行参数

```bash
python3 enhancement_audio_quality_assessment.py [选项]

主要参数:
  --config_file CONFIG_FILE    配置文件路径
  --num_gpus NUM_GPUS          GPU数量(自动检测)
  --skip_existing              跳过已存在的结果
  --max_audio_length SECONDS   最大音频长度
  --llm_service_url URL        LLM服务地址
  --llm_timeout TIMEOUT        LLM超时时间

示例:
  python3 enhancement_audio_quality_assessment.py --config_file config.json --skip_existing
```

### 服务管理

#### 启动服务
```bash
# 自动启动 (推荐)
./auto_start_llm_services.sh

# 手动启动LLM服务
python3 llm_service.py --model_name qwen3:32b --port 8000
```

#### 检查服务状态
```bash
# 检查LLM服务健康状态
curl http://localhost:8000/health

# 8卡配置检查第二组服务
curl http://localhost:8001/health

# 获取模型信息
curl http://localhost:8000/model_info
```

#### 停止服务
```bash
# 停止所有服务
./stop_multi_llm_services.sh

# 手动停止
kill $(cat ./logs/ollama_*.pid)
kill $(cat ./logs/llm_service_*.pid)
```

### 监控和调试

#### 查看日志
```bash
# 查看所有日志
tail -f ./logs/*.log

# 查看LLM服务日志
tail -f ./logs/llm_service_*.log

# 查看Ollama日志  
tail -f ./logs/ollama_*.log
```

#### 监控GPU使用
```bash
# 实时监控GPU状态
watch -n 1 nvidia-smi

# 查看GPU进程
nvidia-smi pmon -i 0,1,2,3,4,5,6,7
```

#### 性能监控
```bash
# 系统资源监控
htop

# 网络连接检查
netstat -tulpn | grep -E ':(8000|8001|11434|11435)'
```

## 输出结果

### 目录结构
```
output_dir/
├── audio_results/              # 处理后的音频文件
│   ├── original/              # 标准化后的原始音频
│   └── enhanced/              # 标准化后的增强音频
├── text_results/              # 文本识别结果
│   ├── filename_text_results.json
│   └── ...
├── summary_report.json        # 总体评估报告
└── processing_log.txt         # 处理日志
```

### 评估指标

#### 文本质量指标
- **WER (Word Error Rate)**: 词错误率
- **CER (Character Error Rate)**: 字符错误率

#### 音频处理指标
- **Volume Gain**: 音量增益值
- **Processing Time**: 处理时间
- **Success Rate**: 成功处理率

### 报告格式

```json
{
  "file_path": "/path/to/audio.wav",
  "original_transcription": "原始转录文本",
  "enhanced_transcription": "增强转录文本", 
  "original_transcription_normalized": "标准化原始文本",
  "enhanced_transcription_normalized": "标准化增强文本",
  "wer": 0.15,
  "cer": 0.08,
  "volume_gain": 2.5,
  "is_usable": true,
  "processing_time": 12.5,
  "text_normalization_config": {
    "method": "llm_only",
    "llm_service_url": "http://localhost:8000",
    "llm_enabled": true
  }
}
```

## 故障排除

### 常见问题

#### 1. GPU内存不足
**现象**: CUDA out of memory错误
**解决方案**:
- 确保每个LLM GPU有32GB+ VRAM
- 减少并行处理的音频数量
- 检查其他GPU进程

```bash
# 检查GPU内存使用
nvidia-smi

# 清理GPU缓存
python3 -c "import torch; torch.cuda.empty_cache()"
```

#### 2. LLM服务连接失败
**现象**: HTTP连接超时或拒绝连接
**解决方案**:
- 检查服务是否正常启动
- 验证端口是否被占用
- 检查防火墙设置

```bash
# 检查端口占用
lsof -i :8000 -i :8001

# 检查服务进程
ps aux | grep -E "(ollama|llm_service)"

# 重启服务
./stop_multi_llm_services.sh
./auto_start_llm_services.sh
```

#### 3. 模型下载失败
**现象**: 模型拉取超时或失败
**解决方案**:
- 检查网络连接
- 使用代理或镜像源
- 手动下载模型文件

```bash
# 手动拉取模型
ollama pull qwen3:32b

# 检查已安装模型
ollama list
```

#### 4. 音频处理错误
**现象**: 音频文件读取或处理失败
**解决方案**:
- 检查音频文件格式和完整性
- 验证文件路径和权限
- 检查音频长度是否超限

```bash
# 检查音频文件信息
ffprobe audio_file.wav

# 转换音频格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

### 调试模式

启用详细日志输出:

```bash
# 设置日志级别
export PYTHONPATH=/path/to/project
export LOG_LEVEL=DEBUG

# 运行调试模式
python3 enhancement_audio_quality_assessment.py --config_file config.json --verbose
```

### 性能优化

#### GPU优化
- 确保GPU驱动和CUDA版本兼容
- 使用合适的批处理大小
- 避免GPU内存碎片

#### 网络优化
- 使用本地LLM服务减少网络延迟
- 增加连接超时时间
- 启用HTTP连接池

#### 存储优化
- 使用SSD存储提高I/O性能
- 合理设置临时文件目录
- 定期清理处理缓存

## 高级用法

### 自定义文本标准化

虽然系统只使用LLM进行文本标准化，但可以通过修改LLM提示词来自定义标准化规则:

```python
# 在llm_service.py中修改_build_normalization_prompt方法
def _build_normalization_prompt(self, text1: str, text2: str) -> str:
    # 自定义提示词内容
    return custom_prompt
```

### 批处理脚本

创建批处理脚本 `batch_process.sh`:

```bash
#!/bin/bash
CONFIG_DIR="./configs"
for config in "$CONFIG_DIR"/*.json; do
    echo "处理配置: $config"
    python3 enhancement_audio_quality_assessment.py --config_file "$config"
done
```

### 结果分析

使用Python脚本分析评估结果:

```python
import json
import pandas as pd

def analyze_results(result_dir):
    results = []
    for json_file in glob.glob(f"{result_dir}/**/*.json", recursive=True):
        with open(json_file, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    df = pd.DataFrame(results)
    print(f"平均WER: {df['wer'].mean():.3f}")
    print(f"平均CER: {df['cer'].mean():.3f}")
    return df
```

## API参考

### LLM服务API

#### 健康检查
```http
GET /health
```

#### 模型信息
```http
GET /model_info
```

#### 文本标准化
```http
POST /normalize
Content-Type: application/json

{
  "text1": "原始文本1", 
  "text2": "原始文本2"
}
```

### 配置文件Schema

完整的JSON Schema定义请参考 `config_schema.json`。

## 贡献指南

欢迎提交Issues和Pull Requests来改进项目:

1. Fork项目仓库
2. 创建特性分支: `git checkout -b feature/new-feature`
3. 提交更改: `git commit -am 'Add new feature'`
4. 推送分支: `git push origin feature/new-feature`
5. 创建Pull Request

## 许可证

本项目采用MIT许可证，详情请见 `LICENSE` 文件。

## 更新日志

### v2.0.0 (最新)
- 升级为Qwen3-32B模型
- 简化文本标准化为仅使用LLM
- 优化GPU资源分配
- 改进错误处理和日志记录

### v1.0.0
- 初始版本发布
- 支持多GPU配置
- 集成TEN VAD
- 实现批量处理功能

## 联系方式

如有问题或建议，请通过以下方式联系:

- 项目Issues: [GitHub Issues](https://github.com/your-repo/issues)
- 邮箱: your-email@domain.com
- 文档: [在线文档](https://your-docs.com) 