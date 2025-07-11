# 音频质量评估脚本 - 多目录处理版本

## 功能概述

这个脚本支持对多个原始音频目录和对应的增强音频目录进行质量评估，每个目录可以独立设置是否启用音量匹配(volume_matching)。脚本会按顺序处理每个目录配置，并生成对应的评估结果。

## 主要特性

- **多目录支持**: 可以同时处理多个原始音频目录和对应的增强音频目录
- **独立音量匹配**: 每个目录可以独立设置是否启用音量匹配
- **顺序处理**: 按配置顺序处理每个目录
- **多GPU并行**: 在单个目录内使用多GPU并行处理
- **配置文件支持**: 支持通过JSON配置文件设置复杂的多目录配置
- **灵活的命令行参数**: 支持通过命令行参数快速设置多目录配置
- **TEN VAD集成**: 使用先进的TEN VAD进行语音活动检测

## 使用方法

### 1. 通过配置文件使用（推荐）

```bash
python enhancement_audio_quality_assessment.py --config_file multi_directory_config.json
```

配置文件示例 (`multi_directory_config.json`):
```json
{
  "directory_configs": [
    {
      "name": "King-ASR-EN-Kid_zipenhancer",
      "original_dir": "/path/to/original/audio1",
      "enhanced_dir": "/path/to/enhanced/audio1",
      "output_dir": "/path/to/output1",
      "volume_matching": true,
      "volume_matching_method": "ten_vad_energy"
    },
    {
      "name": "King-ASR-EN-Kid_demucs",
      "original_dir": "/path/to/original/audio2",
      "enhanced_dir": "/path/to/enhanced/audio2",
      "output_dir": "/path/to/output2",
      "volume_matching": false,
      "volume_matching_method": "ten_vad_energy"
    }
  ],
  "kimi_model_path": "/path/to/kimi/model",
  "kimi_audio_dir": "/path/to/kimi/audio",
  "num_gpus": 3,
  "gpu_ids": [1, 2, 3],
  "text_normalization": "llm",
  "use_llm_normalization": true,
  "llm_service_url": "http://localhost:8000",
  "ten_vad_hop_size": 256,
  "ten_vad_threshold": 0.5
}
```

### 2. 通过命令行参数使用

#### 多目录模式
```bash
python enhancement_audio_quality_assessment.py \
  --original_dirs "/path/to/original1" "/path/to/original2" "/path/to/original3" \
  --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" "/path/to/enhanced3" \
  --output_dirs "/path/to/output1" "/path/to/output2" "/path/to/output3" \
  --config_names "zipenhancer" "demucs" "speechbrain" \
  --volume_matching "true" "true" "false" \
  --volume_methods "ten_vad_energy" "ten_vad_energy" "ten_vad_energy" \
  --num_gpus 3 \
  --gpu_ids 1 2 3
```

#### 单目录模式（向后兼容）
```bash
python enhancement_audio_quality_assessment.py \
  --original_dir "/path/to/original" \
  --enhanced_dir "/path/to/enhanced" \
  --output_dir "/path/to/output" \
  --volume_method "ten_vad_energy" \
  --num_gpus 3 \
  --gpu_ids 1 2 3
```

## 配置说明

### 目录配置参数

- `name`: 配置名称，用于识别和日志输出
- `original_dir`: 原始音频目录路径
- `enhanced_dir`: 增强音频目录路径
- `output_dir`: 输出结果目录路径
- `volume_matching`: 是否启用音量匹配 (true/false)
- `volume_matching_method`: 音量匹配方法 ("ten_vad_energy")

### 全局配置参数

- `kimi_model_path`: Kimi-Audio模型路径
- `kimi_audio_dir`: Kimi-Audio代码目录
- `num_gpus`: 用于ASR的GPU数量
- `gpu_ids`: GPU ID列表（建议使用1,2,3，保留GPU 0给LLM服务）
- `text_normalization`: 文本标准化级别 ("basic", "advanced", "llm")
- `use_llm_normalization`: 是否使用LLM进行文本标准化
- `llm_service_url`: LLM服务URL
- `skip_existing`: 是否跳过已存在的结果
- `max_audio_length`: 最大音频长度（秒）
- `ten_vad_hop_size`: TEN VAD帧跳跃大小（样本数，256样本=16ms@16kHz）
- `ten_vad_threshold`: TEN VAD阈值（0.0-1.0，默认0.5）

## 输出结果

### 目录结构
```
output_dir/
├── subset_0/
│   ├── audio1_assessment.json
│   ├── audio1_text_results.json
│   ├── audio1_gain_applied.wav
│   └── subset_0_results.json
├── subset_1/
│   └── ...
├── subset_2/
│   └── ...
├── quality_assessment_results.json
├── quality_assessment_results.csv
└── assessment_summary.json
```

### 总体摘要
处理完所有目录后，会在第一个输出目录的上级目录生成 `multi_directory_summary.json`，包含所有目录的汇总信息。

## 音量匹配说明

- **启用音量匹配** (`volume_matching: true`): 
  - 使用TEN VAD检测语音段
  - 基于语音段能量计算gain值
  - 将gain应用到增强音频，使其音量与原始音频匹配
  - 保存处理后的音频文件

- **禁用音量匹配** (`volume_matching: false`):
  - 直接使用原始增强音频进行评估
  - 不进行音量调整
  - 适用于已经进行过音量处理的增强音频

## TEN VAD配置说明

- **hop_size**: 帧跳跃大小（样本数）
  - 默认值：256（在16kHz下约16毫秒）
  - 较小的值提供更精细的检测，但计算量更大
  
- **threshold**: VAD检测阈值
  - 范围：0.0-1.0
  - 默认值：0.5
  - 较高的值更严格（更少检测为语音），较低的值更宽松

## 使用场景

1. **多种增强算法对比**: 对同一组原始音频使用不同的增强算法，比较效果
2. **不同数据集评估**: 对不同的音频数据集应用相同的增强算法
3. **参数调优**: 对同一增强算法的不同参数设置进行评估
4. **批量处理**: 一次性处理多个音频增强项目

## 注意事项

1. **GPU资源**: 建议使用GPU 0运行LLM服务，GPU 1-3运行ASR推理
2. **内存管理**: 大量目录处理时注意内存使用
3. **存储空间**: 每个目录会生成处理后的音频文件，确保有足够存储空间
4. **网络连接**: 如果使用LLM文本标准化，确保LLM服务正常运行
5. **目录结构**: 确保增强音频目录结构与原始音频目录一致
6. **TEN VAD依赖**: 确保TEN VAD模块在 `../include/ten_vad.py` 路径下可用

## 错误处理

- 如果某个目录不存在或无权访问，脚本会跳过该目录并继续处理下一个
- 如果某个目录中没有找到音频对，会记录警告并跳过
- 每个目录的处理结果独立保存，单个目录失败不会影响其他目录
- 如果TEN VAD初始化失败，会记录错误并可能回退到基础处理

## 性能优化建议

1. **并行处理**: 合理设置GPU数量和ID
2. **跳过已存在**: 使用 `--skip_existing` 避免重复处理
3. **音频长度限制**: 设置合理的 `max_audio_length` 避免处理过长音频
4. **TEN VAD优化**: 根据音频特性调整hop_size和threshold参数
5. **批处理**: 将相关的目录组合在一起处理，提高效率