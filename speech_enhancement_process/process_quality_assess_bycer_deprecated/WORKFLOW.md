# 音频降噪质量评估与整合工作流程

## 概述

这是一套完整的音频降噪质量评估和智能整合工具链，包含以下主要功能：

1. **质量评估**：使用ASR+LLM对降噪前后音频进行CER/WER评估
2. **智能整合**：基于CER结果自动选择最佳音频版本
3. **多GPU支持**：支持多GPU并行处理提高效率
4. **详细记录**：完整记录评估和决策过程

## 工作流程

### 步骤1：音频降噪处理

首先使用各种降噪模型处理原音频：

```
原音频目录/
├── speaker1/audio1.wav
└── speaker2/audio2.wav

↓ 降噪处理 ↓

ZipEnhancer降噪结果/          MossFormer降噪结果/
├── speaker1/audio1.wav       ├── speaker1/audio1.wav
└── speaker2/audio2.wav       └── speaker2/audio2.wav
```

### 步骤2：质量评估

使用ASR和LLM对降噪效果进行评估：

```bash
# 启动LLM服务
bash auto_start_llm_services.sh

# 运行质量评估
python enhancement_audio_quality_assessment.py \
    --original_dir "/path/to/original" \
    --enhanced_dir "/path/to/enhanced" \
    --output_dir "/path/to/assessment_results"
```

**输出结果：**
- `quality_assessment_results.json`：详细评估结果
- `assessment_summary.json`：统计摘要
- 包含CER、WER、转录文本等信息

### 步骤3：智能整合

基于CER结果选择最佳音频版本：

```bash
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated"
```

**输出结果：**
- 整合后的最佳音频文件
- `audio_integration_records.json`：选择决策记录
- `integration_summary.json`：整合统计摘要

## 详细使用指南

### 1. 环境准备

```bash
# 激活conda环境
conda activate kimi-audio

# 检查GPU状态
nvidia-smi

# 切换到脚本目录
cd /root/code/github_repos/DataProcessor/speech_enhancement_process/process_quality_assess
```

### 2. 启动LLM服务

```bash
# 自动启动LLM服务（独立GPU配置）
bash auto_start_llm_services.sh --model-name qwen3:32b --model-type qwen3

# 检查服务状态
curl http://localhost:8000/health
curl http://localhost:8001/health  # 如果有多GPU
```

### 3. 运行质量评估

#### 单目录评估
```bash
python enhancement_audio_quality_assessment.py \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced_quality_assessment"
```

#### 多目录评估（推荐）
```bash
# 使用配置文件
python enhancement_audio_quality_assessment.py \
    --config_file multi_directory_config_example.json

# 或者直接运行完整脚本
bash run_multi_directory_assessment_auto_gpu.sh
```

### 4. 运行音频整合

```bash
# 基本整合
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated"

# 使用自定义参数
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" \
    --output_dir "/path/to/integrated" \
    --cer_threshold 0.03 \
    --no_prefer_original
```

### 5. 停止服务

```bash
# 停止LLM服务
bash stop_multi_llm_services.sh
```

## 配置参数说明

### 质量评估参数

- `--cer_threshold`: CER阈值，默认5%
- `--llm_model_name`: LLM模型名称，如qwen3:32b
- `--llm_model_type`: 模型类型，影响prompt策略
- `--ten_vad_threshold`: VAD阈值，默认0.5
- `--skip_existing`: 跳过已存在的结果

### 音频整合参数

- `--cer_threshold`: CER阈值，低于此值认为可用
- `--prefer_original`: 质量相近时优先选择原音频
- `--enhanced_dirs`: 支持多个降噪目录

## 输出文件说明

### 质量评估输出

```
assessment_output/
├── quality_assessment_results.json    # 详细评估结果
├── assessment_summary.json           # 统计摘要
├── subset_0/                         # GPU子集结果
│   ├── audio1_assessment.json
│   └── audio1_gain_applied.wav
└── subset_1/
    └── ...
```

### 音频整合输出

```
integrated_output/
├── speaker1/                         # 保持原有结构
│   ├── audio1.wav                   # 最佳版本音频
│   └── audio2.wav
├── speaker2/
│   └── audio3.wav
├── audio_integration_records.json    # 详细选择记录
└── integration_summary.json         # 统计摘要
```

## 决策逻辑

### 质量评估逻辑

1. **ASR转录**：使用Kimi-Audio对原音频和降噪音频进行转录
2. **文本标准化**：使用LLM对转录文本进行标准化处理
3. **指标计算**：计算WER（词错误率）和CER（字符错误率）
4. **可用性判断**：CER < 5%的音频被认为是可用的

### 音频整合逻辑

```
对于每个音频文件：
1. 读取所有降噪版本的CER数据
2. 筛选CER < 阈值的可用版本
3. 如果没有可用版本 → 使用原音频
4. 如果有可用版本 → 选择CER最低的版本
5. 如果启用prefer_original且CER较高 → 可能选择原音频
6. 拷贝选定版本到输出目录
```

## 监控和故障排除

### 服务监控

```bash
# 启动服务监控
bash monitor_and_restart_llm.sh &

# 查看服务状态
curl http://localhost:8000/health
curl http://localhost:8000/model_info
```

### 常见问题

1. **LLM服务连接失败**
   - 检查服务是否启动：`curl http://localhost:8000/health`
   - 查看服务日志：`tail -f logs/llm_service_*.log`
   - 重启服务：`bash auto_start_llm_services.sh`

2. **GPU内存不足**
   - 检查GPU使用情况：`nvidia-smi`
   - 减少并行GPU数量
   - 检查是否有其他进程占用GPU

3. **评估结果为空**
   - 检查音频文件格式和路径
   - 确认ASR模型正常加载
   - 查看错误日志

4. **整合选择异常**
   - 检查CER阈值设置是否合理
   - 确认评估结果文件存在且完整
   - 查看decision_info中的详细原因

## 性能优化建议

1. **GPU配置**：使用多GPU并行处理，每张GPU独立运行ASR+LLM
2. **批处理**：一次处理多个目录配置
3. **缓存机制**：启用skip_existing跳过已处理文件
4. **资源监控**：使用监控脚本自动恢复异常服务

## 示例工作流程

```bash
# 1. 启动服务
bash auto_start_llm_services.sh

# 2. 运行完整评估
bash run_multi_directory_assessment_auto_gpu.sh

# 3. 整合最佳音频
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated"

# 4. 停止服务
bash stop_multi_llm_services.sh
```

## 扩展功能

- 支持更多音频格式
- 自定义评估指标
- 集成更多降噪模型
- 可视化分析工具
- 批量配置管理

---

更多详细信息请参考各脚本的帮助文档：
- `bash run_audio_integration.sh -h`
- `python enhancement_audio_quality_assessment.py -h` 