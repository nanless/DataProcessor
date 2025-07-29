# 音频降噪质量评估与智能整合系统

## 项目概述

本项目实现了一套完整的音频降噪质量评估和智能整合系统，能够：

1. **自动评估**降噪效果（使用ASR+LLM）
2. **智能选择**最佳音频版本（基于CER指标）
3. **多GPU并行**处理提高效率
4. **保持结构**维持原有目录层次
5. **详细记录**完整的决策过程

## 新增功能（本次开发）

### 🎯 智能音频整合系统

根据您的需求，新开发了基于CER的智能音频整合功能：

#### 核心文件

1. **`audio_integration_by_cer.py`** (23KB, 554行)
   - 核心整合逻辑实现
   - 支持多个降噪模型比较
   - 智能决策算法
   - 详细记录生成

2. **`run_audio_integration.sh`** (9.3KB, 278行)
   - 便捷的bash脚本界面
   - 参数验证和错误处理
   - 自动环境配置
   - 实时状态显示

3. **`audio_integration_example.sh`** (5.6KB, 138行)
   - 详细使用示例
   - 多种场景演示
   - 交互式运行选项

#### 文档支持

4. **`README_audio_integration.md`** (2.3KB, 77行)
   - 快速开始指南
   - 基本使用说明

5. **`WORKFLOW.md`** (7.1KB, 275行)
   - 完整工作流程
   - 详细操作指南
   - 故障排除方案

## 功能特点

### ✅ 智能决策逻辑

```python
决策流程：
1. 读取所有降噪版本的CER数据
2. 筛选CER < 阈值的可用版本  
3. 无可用版本 → 使用原音频
4. 有可用版本 → 选择CER最低版本
5. 考虑原音频优先策略
6. 拷贝最佳版本到输出目录
```

### ✅ 多模型支持

- 同时比较多个降噪模型（如ZipEnhancer、MossFormer、Resemble等）
- 自动识别降噪方法名称
- 支持任意数量的降噪模型

### ✅ 灵活配置

- 可调CER阈值（默认5%）
- 原音频优先策略开关
- 支持批量处理
- 保持目录结构

### ✅ 详细记录

**选择记录 (`audio_integration_records.json`)**:
```json
{
  "relative_path": "speaker1/audio1.wav",
  "selected_method": "ZipEnhancer", 
  "is_enhanced": true,
  "decision_info": {
    "reason": "enhanced_better",
    "selected_cer": 0.023,
    "improvement_note": "CER=0.023 < 阈值5%"
  },
  "asr_info": {
    "original_transcription": "hello world",
    "enhanced_transcription": "hello world", 
    "cer": 0.023,
    "wer": 0.0,
    "is_usable": true
  }
}
```

**统计摘要 (`integration_summary.json`)**:
```json
{
  "statistics": {
    "total_files": 1000,
    "original_selected": 300,
    "enhanced_selected": 700,
    "enhancement_methods": {
      "ZipEnhancer": 400,
      "MossFormer": 300
    },
    "cer_improvement_stats": {
      "mean": 0.032,
      "median": 0.025
    }
  }
}
```

## 使用方法

### 基本使用

```bash
# 比较两种降噪方法，选择最佳版本
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/zipenhancer" "/path/to/mossformer" \
    --output_dir "/path/to/integrated"
```

### 高级配置

```bash
# 使用严格CER阈值，禁用原音频优先
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated" \
    --cer_threshold 0.03 \
    --no_prefer_original
```

### 示例运行

```bash
# 查看使用示例
bash audio_integration_example.sh

# 直接运行示例
bash audio_integration_example.sh  # 然后选择 y
```

## 输出结果

### 1. 整合音频文件
- 📁 保持原有目录结构
- 🎵 每个位置的最佳音频版本
- 📊 自动选择质量最优版本

### 2. 详细选择记录
- 📄 `audio_integration_records.json`: 每个文件的选择详情
- 📈 `integration_summary.json`: 统计摘要和分析

### 3. 决策透明度
- ✅ 明确记录选择原因
- 📊 CER数据和改善情况
- 🔍 ASR转录和标准化结果

## 与现有系统集成

### 完整工作流程

```bash
# 1. 启动LLM服务
bash auto_start_llm_services.sh

# 2. 运行质量评估（现有功能）
bash run_multi_directory_assessment_auto_gpu.sh

# 3. 智能音频整合（新功能）
bash run_audio_integration.sh \
    --original_dir "/path/to/original" \
    --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
    --output_dir "/path/to/integrated"

# 4. 停止服务
bash stop_multi_llm_services.sh
```

### 配合现有功能

- **依赖评估结果**: 使用现有的质量评估结果作为决策依据
- **兼容多GPU**: 与现有的多GPU评估系统协同工作
- **统一配置**: 使用相同的CER阈值和评估标准

## 技术实现亮点

### 🔧 健壮的错误处理
- 文件不存在时的优雅降级
- 评估结果缺失时的回退机制
- 详细的错误信息记录

### ⚡ 高效的文件操作
- 逐文件处理，内存友好
- 智能路径解析和映射
- 批量文件拷贝优化

### 📊 智能决策算法
- 基于统计学的CER阈值设定
- 考虑原音频质量的保守策略
- 多模型性能综合评估

### 🔍 完整的可追溯性
- 每个决策的详细记录
- 统计数据和改善分析
- 便于后续优化和调试

## 文件结构总览

```
process_quality_assess/
├── 🎯新增核心功能
│   ├── audio_integration_by_cer.py      # 主要整合逻辑
│   ├── run_audio_integration.sh         # 便捷运行脚本
│   └── audio_integration_example.sh     # 使用示例
├── 📖新增文档
│   ├── README_audio_integration.md      # 快速指南
│   ├── WORKFLOW.md                      # 完整工作流程
│   └── PROJECT_SUMMARY.md              # 项目总结
├── 🔧现有功能（已优化）
│   ├── enhancement_audio_quality_assessment.py
│   ├── auto_start_llm_services.sh
│   ├── run_multi_directory_assessment_auto_gpu.sh
│   └── ...
└── 📁输出示例
    ├── integrated_audio/                # 整合后音频
    ├── audio_integration_records.json   # 选择记录
    └── integration_summary.json        # 统计摘要
```

## 开发成果总结

✅ **完全满足需求**: 
- 根据CER判断是否替换原音频
- 支持多个降噪模型比较
- 保持目录结构
- 详细记录选择信息

✅ **超越基本要求**:
- 智能决策算法
- 灵活配置选项  
- 完整的工作流程
- 详细的使用文档

✅ **工程质量保证**:
- 健壮的错误处理
- 完整的日志记录
- 用户友好的界面
- 可扩展的架构

## 后续使用建议

1. **首次使用**: 运行 `audio_integration_example.sh` 查看示例
2. **生产环境**: 使用配置文件批量处理多个数据集
3. **性能优化**: 根据硬件配置调整CER阈值
4. **质量监控**: 定期检查integration_summary.json了解选择效果

---

**开发者**: Claude Assistant  
**完成时间**: 2024年  
**代码行数**: 约800行（新增功能）  
**文档页数**: 约20页  

这套系统现在已经完全实现了您的需求，可以智能地根据CER结果选择最佳音频版本，并提供详细的决策记录。 