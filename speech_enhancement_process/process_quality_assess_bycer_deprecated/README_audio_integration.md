# 音频整合脚本 - 基于CER的智能音频选择

## 概述

这个脚本根据ASR和LLM文本标准化后的CER结果，智能地选择最佳音频版本进行整合。

## 主要功能

- 🎯 **智能选择**：基于CER自动选择最佳音频版本
- 📊 **多模型比较**：支持同时比较多个降噪模型
- 📁 **结构保持**：维持原有目录结构
- 📋 **详细记录**：记录每个选择决策的详细信息
- 🔄 **回退机制**：无可用降噪版本时自动回退到原音频

## 使用方法

### 基本使用

```bash
# 使用便捷脚本
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated"
```

### 参数说明

- `--original_dir`: 原音频目录
- `--enhanced_dirs`: 降噪音频目录列表（可多个）  
- `--output_dir`: 输出目录
- `--cer_threshold`: CER阈值（默认5%）
- `--prefer_original`: 优先选择原音频（默认启用）

## 使用示例

```bash
# 比较两种降噪方法
bash run_audio_integration.sh \
    --original_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid" \
    --enhanced_dirs \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_zipenhancer_enhanced" \
        "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_mossformer_enhanced" \
    --output_dir "/root/group-shared/voiceprint/data/speech/speaker_verification/King-ASR-EN-Kid_integrated"
```

## 前置条件

需要先运行质量评估脚本生成CER数据：

```bash
python enhancement_audio_quality_assessment.py \
    --enhanced_dir /path/to/enhanced_audio \
    --original_dir /path/to/original_audio \
    --output_dir /path/to/assessment_results
```

## 输出文件

1. **整合音频**：输出目录中的最佳音频版本
2. **audio_integration_records.json**：详细的选择记录
3. **integration_summary.json**：统计摘要

## 决策逻辑

1. 读取所有版本的CER数据
2. 筛选CER < 阈值的可用版本
3. 选择CER最低的版本
4. 考虑原音频优先策略
5. 拷贝最佳版本到输出目录

## 最佳实践

- 先运行质量评估获取CER数据
- 根据需求调整CER阈值（3%-8%）
- 使用多个降噪模型提高覆盖率
- 检查统计摘要确认效果 