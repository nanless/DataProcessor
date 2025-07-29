# 音频质量评估系统

基于深度学习的音频增强质量评估系统，采用独立GPU配置，每张GPU卡同时运行ASR和LLM服务。使用Kimi-Audio进行语音识别，使用Qwen大语言模型进行智能文本标准化。

## 系统特性

- **独立GPU配置**: 每张GPU独立运行ASR+LLM服务，无单点故障
- **线性扩展**: N张GPU = N个独立的ASR+LLM服务单元
- **智能文本标准化**: Qwen模型自动处理语音识别结果的格式差异
- **多目录批量处理**: 支持同时评估多个数据集
- **自动服务管理**: 一键启动、监控、停止所有服务

## 快速开始

### 环境要求

- **硬件**: NVIDIA GPU (8GB+ VRAM), 32GB+ RAM
- **软件**: Linux, Python 3.8+, CUDA 11.8+, conda

### 系统检查

```bash
# 1. 进入项目目录
cd speech_enhancement_process/process_quality_assess

# 2. 激活环境
conda activate kimi-audio

# 3. 运行系统测试（推荐）
python test_system.py
```

### 一键运行

```bash
# 启动服务并运行评估
./run_multi_directory_assessment_auto_gpu.sh
```

## GPU配置方案

系统自动检测GPU数量，为每张GPU配置独立服务：

**4张GPU示例**:
```
GPU 0: ASR+LLM (HTTP:8000, Ollama:11434)
GPU 1: ASR+LLM (HTTP:8001, Ollama:11435)
GPU 2: ASR+LLM (HTTP:8002, Ollama:11436)
GPU 3: ASR+LLM (HTTP:8003, Ollama:11437)
```

**优势**:
- 完全独立，任意一张GPU故障不影响其他GPU
- 负载均衡，每张GPU承担相同工作量
- 线性扩展，增加GPU即可线性提升性能

## 完整使用流程

### 1. 准备配置文件

创建或修改 `multi_directory_config_example.json`:

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
  "llm_timeout": 30,
  "llm_max_retries": 3,
  "llm_model_config": {
    "model_name": "qwen3:32b",
    "model_type": "qwen3"
  }
}
```

### 2. 启动LLM服务

```bash
# 自动启动所有GPU的服务
./auto_start_llm_services.sh --model-type qwen3 --model-name qwen3:32b

# 检查服务状态
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### 3. 运行质量评估

```bash
# 方式1: 使用配置文件
python enhancement_audio_quality_assessment.py \
    --config_file multi_directory_config_example.json

# 方式2: 使用命令行参数
python enhancement_audio_quality_assessment.py \
    --original_dir /path/to/original \
    --enhanced_dir /path/to/enhanced \
    --output_dir /path/to/output \
    --llm_model_type qwen3 \
    --llm_model_name qwen3:32b
```

### 4. 监控服务（可选）

```bash
# 启动服务监控
./monitor_and_restart_llm.sh &

# 查看监控日志
tail -f ./logs/llm_monitor.log
```

### 5. 停止服务

```bash
# 停止所有服务
./stop_multi_llm_services.sh
```

## 核心脚本说明

| 脚本 | 功能 | 使用场景 |
|------|------|----------|
| `run_multi_directory_assessment_auto_gpu.sh` | 主运行脚本 | 一键完成所有操作 |
| `auto_start_llm_services.sh` | 启动LLM服务 | 手动管理服务 |
| `enhancement_audio_quality_assessment.py` | 音频质量评估主程序 | 核心处理程序 |
| `llm_service.py` | LLM HTTP服务 | 提供文本标准化API |
| `stop_multi_llm_services.sh` | 停止所有服务 | 清理环境 |
| `monitor_and_restart_llm.sh` | 服务监控 | 长期运行的稳定性保证 |
| `test_system.py` | 系统功能测试 | 验证环境和配置 |

## 配置参数详解

### 目录配置 (directory_configs)

- `name`: 配置名称，用于日志标识
- `original_dir`: 原始音频文件目录
- `enhanced_dir`: 增强音频文件目录  
- `output_dir`: 评估结果输出目录
- `volume_matching`: 是否启用音量匹配 (建议true)
- `volume_matching_method`: 音量匹配方法 (推荐`ten_vad_energy`)

### LLM模型配置 (llm_model_config)

- `model_name`: Ollama模型名称 (如`qwen3:32b`, `qwen2.5:32b`)
- `model_type`: 模型类型，影响prompt优化策略
  - `qwen3`: 会添加"减少思考"指令，提高响应速度
  - `qwen2.5`: 使用标准prompt

### 其他关键参数

- `kimi_model_path`: Kimi-Audio模型路径
- `kimi_audio_dir`: Kimi-Audio代码库路径
- `use_llm_normalization`: 必须为true，系统只支持LLM标准化
- `llm_timeout`: LLM服务超时时间(秒)
- `max_audio_length`: 最大音频长度(秒)，超过会截断

## 输出结果

### 目录结构
```
output_dir/
├── subset_0/                    # GPU 0处理的结果
│   ├── filename_assessment.json
│   └── subset_0_results.json
├── subset_1/                    # GPU 1处理的结果
├── ...
├── quality_assessment_results.json  # 合并后的所有结果
└── assessment_summary.json         # 统计摘要
```

### 关键指标

- **WER (Word Error Rate)**: 词错误率，越低越好
- **CER (Character Error Rate)**: 字符错误率，越低越好
- **is_usable**: CER < 5% 的音频被标记为可用

### 结果示例
```json
{
  "original_transcription": "原始ASR结果",
  "enhanced_transcription": "增强ASR结果",
  "original_transcription_normalized": "LLM标准化后的原始文本",
  "enhanced_transcription_normalized": "LLM标准化后的增强文本",
  "wer": 0.15,
  "cer": 0.08,
  "is_usable": true,
  "volume_scale_factor": 2.3,
  "processing_time": 12.5
}
```

## 故障排除

### 常见问题

#### 1. GPU内存不足
**现象**: CUDA out of memory

**解决方案**:
```bash
# 检查GPU状态
nvidia-smi

# 使用更小的模型
./auto_start_llm_services.sh --model-name qwen3:14b --model-type qwen3

# 减少并发数（修改配置文件中的GPU数量）
```

#### 2. LLM服务启动失败  
**现象**: 连接拒绝或超时

**解决方案**:
```bash
# 检查端口占用
lsof -i :8000-8010

# 强制停止所有服务
./stop_multi_llm_services.sh

# 查看详细日志
tail -f ./logs/llm_service_*.log
tail -f ./logs/ollama_*.log

# 重新启动
./auto_start_llm_services.sh
```

#### 3. 模型下载失败
**现象**: 模型拉取超时

**解决方案**:
```bash  
# 手动拉取模型
ollama pull qwen3:32b

# 检查已安装模型
ollama list

# 使用代理（如果需要）
export https_proxy=http://proxy:port
ollama pull qwen3:32b
```

#### 4. 音频文件读取失败
**现象**: 文件格式不支持或路径错误

**解决方案**:
```bash
# 检查音频文件
ffprobe audio_file.wav

# 转换格式
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# 检查路径权限
ls -la /path/to/audio/files/
```

### 调试模式

启用详细日志:
```bash
export LOG_LEVEL=DEBUG
python enhancement_audio_quality_assessment.py --config_file config.json
```

查看实时日志:
```bash
# 查看所有服务日志
tail -f ./logs/*.log

# 查看特定GPU日志
tail -f ./logs/llm_service_0.log   # GPU 0
tail -f ./logs/ollama_0.log        # GPU 0的Ollama服务
```

## 高级用法

### 多数据集批量处理

配置多个目录同时处理:

```json
{
  "directory_configs": [
    {
      "name": "dataset_A_algorithm1",
      "original_dir": "/data/dataset_A/original",
      "enhanced_dir": "/data/dataset_A/algorithm1_enhanced",
      "output_dir": "/results/dataset_A_algorithm1"
    },
    {
      "name": "dataset_A_algorithm2",
      "original_dir": "/data/dataset_A/original", 
      "enhanced_dir": "/data/dataset_A/algorithm2_enhanced",
      "output_dir": "/results/dataset_A_algorithm2"
    }
  ]
}
```

### 自定义脚本集成

```bash
#!/bin/bash
# 批量评估脚本示例

configs=("config1.json" "config2.json" "config3.json")

for config in "${configs[@]}"; do
    echo "处理配置: $config"
    python enhancement_audio_quality_assessment.py --config_file "$config"
    echo "完成: $config"
done
```

### 结果分析

使用Python分析结果:

```python
import json
import pandas as pd
import glob

# 读取所有结果
results = []
for file in glob.glob("*/quality_assessment_results.json"):
    with open(file) as f:
        data = json.load(f)
        results.extend(data)

# 转换为DataFrame分析
df = pd.DataFrame(results)
print(f"平均WER: {df['wer'].mean():.3f}")
print(f"平均CER: {df['cer'].mean():.3f}")
print(f"可用音频比例: {df['is_usable'].mean():.3f}")
```

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    音频质量评估系统                          │
├─────────────────────────────────────────────────────────────┤
│  GPU 0              GPU 1              GPU 2              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Kimi-Audio  │    │ Kimi-Audio  │    │ Kimi-Audio  │    │
│  │ (ASR)       │    │ (ASR)       │    │ (ASR)       │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Ollama      │    │ Ollama      │    │ Ollama      │    │
│  │ :11434      │    │ :11435      │    │ :11436      │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ LLM HTTP    │    │ LLM HTTP    │    │ LLM HTTP    │    │
│  │ :8000       │    │ :8001       │    │ :8002       │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 更新日志

- **v2.0** (当前): 重构为独立GPU配置，简化脚本，优化文档
- **v1.5**: 支持qwen3模型，添加prompt优化
- **v1.0**: 初始版本，支持多GPU配置

## 完整流程总结

```bash
# 1. 系统检查
python test_system.py

# 2. 编辑配置文件（根据需要）
nano multi_directory_config_example.json

# 3. 一键运行（推荐）
./run_multi_directory_assessment_auto_gpu.sh

# 或者分步运行：
# 3a. 启动服务
./auto_start_llm_services.sh --model-type qwen3 --model-name qwen3:32b

# 3b. 运行评估
python enhancement_audio_quality_assessment.py --config_file multi_directory_config_example.json

# 3c. 停止服务
./stop_multi_llm_services.sh
```

## 技术支持

如有问题请检查：
1. 日志文件 `./logs/*.log`
2. GPU状态 `nvidia-smi`  
3. 服务状态 `curl localhost:8000/health`
4. 磁盘空间和权限

项目地址: [GitHub](https://github.com/your-repo) 