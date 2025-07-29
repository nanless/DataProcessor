# 基于Groundtruth的音频增强集成系统

## 📋 项目简介

本系统基于Groundtruth文本，智能选择最优的音频增强方法，生成高质量的音频数据集。系统支持多种增强算法（zipenhancer、mossformer等），通过ASR识别和CER评估自动选择最佳结果，并可根据配置控制音量匹配和优先级策略。

### 🎯 核心功能
- **智能选择**: 基于CER评估自动选择最优增强方法
- **优先级控制**: CER相等时按配置优先级选择（mossformer > zipenhancer）  
- **音量匹配**: 可配置是否使用TEN-VAD进行音量匹配
- **详细记录**: 为每个音频生成对应的JSON详情文件
- **多GPU并行**: 支持多GPU并行处理，提高处理效率
- **LLM标准化**: 使用大语言模型进行文本标准化，提高CER计算准确性

## 🏗️ 系统架构

```
音频增强集成系统
├── 输入数据
│   ├── 原始音频文件
│   ├── 增强音频文件 (zipenhancer, mossformer等)
│   ├── wav.scp (音频路径映射)
│   └── text (groundtruth文本)
├── 处理流程
│   ├── ASR识别 (Kimi-Audio)
│   ├── LLM标准化 (Qwen3:32B)
│   ├── CER评估与比较
│   ├── 优先级选择
│   └── 音量匹配 (可选TEN-VAD)
└── 输出结果
    ├── 最优音频文件
    ├── 详细JSON记录
    └── 统计报告
```

## 📁 文件结构

```
speech_enhancement_process/intergrate_enhanced_speech_bygtcer/
├── groundtruth_based_integration.py    # 主处理程序
├── audio_processors.py                 # 音频处理器类
├── llm_service.py                      # LLM文本标准化服务
├── config.json                         # 配置文件
├── run_groundtruth_integration.sh      # 主运行脚本
├── auto_start_llm_services.sh          # 启动LLM服务脚本
├── stop_multi_llm_services.sh          # 停止LLM服务脚本
└── README.md                           # 项目文档 (本文件)
```

## ⚙️ 环境要求

### Python环境
- Python 3.8+
- CUDA支持的GPU环境

### 必需依赖
```bash
# 核心依赖
torch>=1.9.0
torchaudio>=0.9.0
numpy>=1.21.0
librosa>=0.8.0
soundfile>=0.10.0

# ASR相关
kimi-audio  # Kimi-Audio模型

# LLM相关  
requests>=2.25.0
fastapi>=0.68.0
uvicorn>=0.15.0

# 评估相关
jiwer>=2.3.0  # CER/WER计算

# 其他工具
pathlib
json
datetime
multiprocessing
logging
```

### 外部服务
- **Ollama**: 本地LLM服务框架，运行Qwen3:32B模型
- **Kimi-Audio**: 语音识别模型

## 🚀 快速开始

### 1. 启动LLM服务
```bash
# 启动所有LLM服务 (自动启动8个端口: 8000-8007)
./auto_start_llm_services.sh

# 检查服务状态
curl http://localhost:8000/health
```

### 2. 配置数据集
编辑 `config.json` 文件，配置数据集信息：

```json
{
    "global_config": {
        "max_degradation": 0.02,
        "num_gpus": 8,
        "output_individual_json": true,
        "ten_vad_config": {
            "hop_size": 256,
            "threshold": 0.5
        }
    },
    "datasets": [
        {
            "name": "your_dataset",
            "base_dir": "/path/to/dataset",
            "groundtruth_file": "/path/to/text",
            "wav_scp_file": "/path/to/wav.scp",
            "language": "zh",
            "enhancement_methods": [
                {
                    "name": "zipenhancer", 
                    "use_ten_vad_volume_matching": true,
                    "description": "ZIP增强器，使用TEN-VAD音量匹配"
                },
                {
                    "name": "mossformer",
                    "use_ten_vad_volume_matching": false, 
                    "description": "MossFormer增强器，不使用音量匹配"
                }
            ],
            "output_dir": "/path/to/output"
        }
    ]
}
```

### 3. 运行处理
```bash
# 处理所有数据集
./run_groundtruth_integration.sh

# 处理指定数据集
./run_groundtruth_integration.sh --datasets dataset1,dataset2

# 使用指定GPU数量
./run_groundtruth_integration.sh --num_gpus 4

# 查看帮助
./run_groundtruth_integration.sh --help
```

### 4. 停止服务
```bash
# 停止所有LLM服务
./stop_multi_llm_services.sh
```

## 📝 配置详解

### 全局配置 (global_config)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `kimi_model_path` | string | 必需 | Kimi-Audio模型路径 |
| `max_degradation` | float | 0.02 | 最大CER退化阈值 |
| `num_gpus` | int | 8 | 使用的GPU数量 |
| `output_individual_json` | bool | true | 是否生成单独JSON文件 |
| `use_llm_normalization` | bool | true | 是否使用LLM文本标准化 |
| `ten_vad_config` | object | - | TEN-VAD配置参数 |

### 数据集配置 (datasets)

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | string | 数据集名称 |
| `base_dir` | string | 数据集根目录 |
| `groundtruth_file` | string | groundtruth文本文件路径 |
| `wav_scp_file` | string | wav.scp音频映射文件路径 |
| `language` | string | 语言类型 (zh/en/mixed) |
| `enhancement_methods` | array | 增强方法配置列表 |
| `output_dir` | string | 输出目录路径 |

### 增强方法配置 (enhancement_methods)

| 参数 | 类型 | 说明 |
|------|------|------|
| `name` | string | 增强方法名称 |
| `use_ten_vad_volume_matching` | bool | 是否使用TEN-VAD音量匹配 |
| `description` | string | 方法描述 |

## 🔄 处理流程详解

### 1. 初始化阶段
- 加载配置文件和数据集信息
- 初始化音频处理器、ASR模型、LLM服务
- 建立utterance ID映射关系

### 2. 音频处理阶段
对每个音频文件执行以下步骤：

#### 2.1 ASR识别
```python
# 原始音频识别
original_transcription = kimi_asr.transcribe(original_audio)

# 增强音频识别
for method in enhancement_methods:
    enhanced_audio = find_enhanced_audio(original_audio, method)
    enhanced_transcription = kimi_asr.transcribe(enhanced_audio)
```

#### 2.2 文本标准化
```python
# 使用LLM进行文本标准化
normalized_transcription = llm_normalizer.normalize(transcription)
normalized_groundtruth = llm_normalizer.normalize(groundtruth)
```

#### 2.3 CER计算与比较
```python
# 计算字符错误率
cer = calculate_cer(normalized_groundtruth, normalized_transcription)

# 比较不同方法的CER
best_method = select_best_method(cer_results, method_priority)
```

#### 2.4 优先级选择
当多个方法CER相等时（差异<1e-8），按优先级选择：
```python
method_priority = {
    'mossformer': 1,    # 最高优先级
    'zipenhancer': 2,   # 第二优先级  
    'resemble': 3       # 第三优先级
}
```

#### 2.5 音量匹配（可选）
根据选择的增强方法配置决定是否进行音量匹配：
```python
if selected_method_config['use_ten_vad_volume_matching']:
    matched_audio = ten_vad_volume_match(enhanced_audio, original_audio)
else:
    matched_audio = enhanced_audio  # 直接使用
```

### 3. 输出生成阶段
- 复制最优音频到输出目录
- 生成详细的JSON记录文件
- 更新处理统计信息

## 📊 输出格式

### 音频文件
处理后的最优音频文件，保持原始文件名和目录结构。

### JSON详情文件
每个音频对应一个同名JSON文件，包含完整的处理信息：

```json
{
  "audio_file_info": {
    "utterance_id": "King-ASR-612_000080001",
    "original_path": "/path/to/original.wav",
    "selected_method": "mossformer", 
    "selected_path": "/path/to/enhanced.wav",
    "output_audio_path": "/path/to/output.wav",
    "language": "en",
    "volume_matching_method": "TEN-VAD",
    "volume_gain_applied": 1.2
  },
  "groundtruth_info": {
    "groundtruth_text": "do you like pink"
  },
  "asr_results": {
    "original_evaluation": {
      "transcription": "Do you like pink?",
      "normalized_transcription": "do you like pink",
      "normalized_groundtruth": "do you like pink", 
      "cer": 0.0,
      "wer": 0.0,
      "success": true
    },
    "enhanced_evaluations": {
      "zipenhancer": {
        "transcription": "Do you like pink?",
        "cer": 0.0,
        "success": true,
        "method_config": {
          "name": "zipenhancer",
          "use_ten_vad_volume_matching": true
        }
      },
      "mossformer": {
        "transcription": "Do you like pink?", 
        "cer": 0.0,
        "success": true,
        "method_config": {
          "name": "mossformer",
          "use_ten_vad_volume_matching": false
        }
      }
    }
  },
  "decision_analysis": {
    "decision_info": {
      "reason": "enhanced_selected_by_priority",
      "original_cer": 0.0,
      "enhanced_cer": 0.0,
      "method": "mossformer",
      "equal_cer_methods": ["zipenhancer", "mossformer"],
      "priority_selected": "mossformer",
      "selection_note": "CER相等时优先选择mossformer"
    }
  },
  "processing_metadata": {
    "processing_time": 2.5,
    "timestamp": "2025-07-29T11:25:51.930982",
    "gpu_id": 0,
    "success": true
  },
  "config_info": {
    "dataset_name": "King-ASR-EN-Kid",
    "max_degradation_threshold": 0.02
  }
}
```

#### JSON字段说明

**audio_file_info**: 音频文件基本信息
- `utterance_id`: 话语唯一标识符
- `selected_method`: 选择的增强方法
- `volume_matching_method`: 音量匹配方法 (TEN-VAD/None)

**asr_results**: ASR识别结果
- `original_evaluation`: 原始音频识别结果
- `enhanced_evaluations`: 各增强方法的识别结果

**decision_analysis**: 决策分析信息
- `reason`: 选择原因代码
  - `original_selected`: 选择原始音频
  - `enhanced_selected_as_default`: 选择增强音频(默认逻辑)
  - `enhanced_selected_by_priority`: 选择增强音频(优先级决定)
- `equal_cer_methods`: CER相等的方法列表
- `priority_selected`: 优先级选择的方法

## 🔧 故障排除

### 常见问题

#### 1. LLM服务连接失败
```bash
# 检查服务状态
curl http://localhost:8000/health

# 重启服务
./stop_multi_llm_services.sh
./auto_start_llm_services.sh
```

#### 2. utterance ID不匹配错误
确保wav.scp文件格式正确：
```
utterance_id1  /path/to/audio1.wav
utterance_id2  /path/to/audio2.wav
```

#### 3. 增强音频找不到
检查增强音频目录结构：
```
/base/dir/dataset_name_method_enhanced/
├── same/structure/as/original/
└── audio_file.wav
```

#### 4. GPU内存不足
```bash
# 减少GPU使用数量
./run_groundtruth_integration.sh --num_gpus 2

# 或在config.json中调整
"num_gpus": 2
```

#### 5. 处理速度慢
- 增加GPU数量
- 检查LLM服务响应时间
- 优化音量匹配配置

### 日志分析
主要日志文件：`run_groundtruth_integration.log`

关键日志信息：
- `INFO`: 正常处理信息
- `WARNING`: 警告信息，如文件未找到
- `ERROR`: 错误信息，需要排查

## 📈 性能优化

### 1. 多GPU并行
```json
{
    "global_config": {
        "num_gpus": 8,
        "gpu_ids": [0, 1, 2, 3, 4, 5, 6, 7]
    }
}
```

### 2. LLM服务优化
- 使用多个LLM服务实例
- 调整timeout和重试参数
- 使用更快的模型

### 3. 音量匹配优化
根据需求选择是否使用TEN-VAD：
- 高质量要求：使用TEN-VAD
- 速度优先：关闭音量匹配

## 🤝 使用示例

### 示例1：处理中文数据集
```bash
# 配置config.json
{
    "datasets": [{
        "name": "chinese_dataset",
        "language": "zh",
        "enhancement_methods": [
            {"name": "mossformer", "use_ten_vad_volume_matching": true}
        ]
    }]
}

# 运行处理
./run_groundtruth_integration.sh --datasets chinese_dataset
```

### 示例2：批量处理多个数据集
```bash
./run_groundtruth_integration.sh --datasets dataset1,dataset2,dataset3 --num_gpus 4
```

### 示例3：调试模式
```bash
# 单GPU处理，便于调试
./run_groundtruth_integration.sh --num_gpus 1 --datasets test_dataset
```

## 📚 技术细节

### utterance ID处理机制
系统支持两种utterance ID格式：
1. **完全匹配**: `utterance_id == filename_stem`
2. **包含关系**: `filename_stem in utterance_id`

通过反向映射机制确保正确的ID匹配。

### 优先级选择算法
```python
# CER比较精度：1e-8
if abs(cer1 - cer2) < 1e-8:
    # CER相等，按优先级选择
    selected = min(methods, key=lambda m: method_priority[m])
else:
    # CER不等，选择更低的
    selected = min(methods, key=lambda m: m.cer)
```

### 文本标准化流程
1. ASR原始输出
2. LLM标准化处理
3. 统一格式转换
4. CER/WER计算

## 📄 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🚧 更新日志

### v2.0.0 (2025-07-29)
- ✅ 新增优先级选择机制
- ✅ 新增TEN-VAD音量匹配控制
- ✅ 新增单独JSON详情文件生成
- ✅ 修复utterance ID提取问题
- ✅ 优化多GPU并行处理
- ✅ 完善错误处理和日志记录

### v1.0.0
- 🎉 初始版本发布
- 基础音频增强集成功能
- ASR识别和CER评估
- 多GPU支持 