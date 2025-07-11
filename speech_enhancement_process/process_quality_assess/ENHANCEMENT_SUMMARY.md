# 音频质量评估脚本扩展总结 - TEN VAD集成版本

## 扩展内容概述

本次扩展主要实现了对音频质量评估脚本的多目录处理功能，并将VAD（语音活动检测）从WebRTC VAD升级到更先进的TEN VAD。允许用户输入多个原始音频目录和对应的增强音频目录，并为每个目录独立设置是否启用音量匹配功能。

## 主要修改内容

### 0. TEN VAD集成（新增）

**替换WebRTC VAD为TEN VAD**:
- 移除webrtcvad依赖
- 集成先进的TEN VAD技术
- 添加TEN VAD配置参数：`ten_vad_hop_size` 和 `ten_vad_threshold`
- 更新AudioVolumeProcessor类以使用TEN VAD API

**TEN VAD优势**:
- 更准确的语音活动检测
- 更灵活的参数配置
- 更好的噪声环境适应性
- 支持更精细的帧级检测控制

### 1. 配置结构重构

**原配置结构**:
```python
CONFIG = {
    'original_dir': "/path/to/original",
    'enhanced_dir': "/path/to/enhanced", 
    'output_dir': "/path/to/output",
    'volume_matching_method': 'vad_energy',
    # 其他全局配置...
}
```

**新配置结构**:
```python
CONFIG = {
    'directory_configs': [
        {
            'name': 'config_name',
            'original_dir': "/path/to/original",
            'enhanced_dir': "/path/to/enhanced",
            'output_dir': "/path/to/output",
            'volume_matching': True,  # 新增：独立的音量匹配开关
            'volume_matching_method': 'vad_energy',
        },
        # 可以有多个目录配置...
    ],
    # 全局配置...
}
```

### 2. 核心类修改

#### QualityAssessmentProcessor 类
- **构造函数**: 添加了 `dir_config` 参数，接收当前目录配置
- **音量匹配逻辑**: 根据 `volume_matching` 配置决定是否进行音量匹配
- **结果记录**: 在结果中记录音量匹配状态和目录配置信息

#### 关键修改点：
```python
# 原来的构造函数
def __init__(self, config: Dict, gpu_id: int = 0):

# 新的构造函数  
def __init__(self, config: Dict, dir_config: Dict, gpu_id: int = 0):
    self.volume_matching = dir_config.get('volume_matching', True)
    self.volume_matching_method = dir_config.get('volume_matching_method', 'vad_energy')
```

### 3. 音频处理逻辑优化

**条件性音量匹配**:
```python
# 根据配置决定是否进行音量匹配
if self.volume_matching:
    if self.volume_matching_method == 'ten_vad_energy':
        gain = self.volume_processor.calculate_gain_vad_energy(original_audio, enhanced_audio)
        matched_audio, actual_gain = self.volume_processor.apply_gain(enhanced_audio, gain)
    else:
        matched_audio = enhanced_audio
        actual_gain = 1.0
else:
    # 不进行音量匹配
    matched_audio = enhanced_audio
    actual_gain = 1.0
```

**TEN VAD音频处理流程**:
```python
# TEN VAD初始化
self.vad = TenVad(self.hop_size, self.threshold)

# 逐帧VAD检测
for i in range(num_frames):
    audio_frame = audio_int16[start_idx:end_idx]
    out_probability, out_flag = self.vad.process(audio_frame)
    vad_results.append(bool(out_flag))
```

### 4. 命令行参数扩展

**新增多目录参数**:
- `--config_file`: 配置文件路径
- `--original_dirs`: 原始音频目录列表
- `--enhanced_dirs`: 增强音频目录列表  
- `--output_dirs`: 输出目录列表
- `--config_names`: 配置名称列表
- `--volume_matching`: 音量匹配开关列表
- `--volume_methods`: 音量匹配方法列表

**向后兼容性**:
- 保留原有的单目录参数（`--original_dir`, `--enhanced_dir`, `--output_dir`）
- 自动检测参数类型，选择相应的处理模式

### 5. 主函数重构

**处理流程**:
1. 解析命令行参数或配置文件
2. 构建目录配置列表
3. 按顺序处理每个目录配置
4. 为每个目录独立运行多GPU并行处理
5. 生成每个目录的独立结果
6. 生成总体摘要

**关键特性**:
- **顺序处理**: 按配置顺序处理每个目录
- **错误隔离**: 单个目录失败不影响其他目录
- **独立结果**: 每个目录生成独立的评估结果
- **总体摘要**: 生成所有目录的汇总信息

### 6. 结果输出增强

**每个目录的输出**:
- 原有的所有输出文件
- 增强的评估摘要（包含目录配置信息）
- 音量匹配状态记录

**总体输出**:
- `multi_directory_summary.json`: 所有目录的汇总信息
- 包含处理时间、目录数量、音频对总数等统计信息

## 使用方式

### 1. 配置文件方式（推荐）
```bash
python enhancement_audio_quality_assessment.py --config_file multi_directory_config.json
```

### 2. 命令行参数方式
```bash
python enhancement_audio_quality_assessment.py \
  --original_dirs "/path/to/original1" "/path/to/original2" \
  --enhanced_dirs "/path/to/enhanced1" "/path/to/enhanced2" \
  --output_dirs "/path/to/output1" "/path/to/output2" \
  --volume_matching "true" "false" \
  --num_gpus 3 --gpu_ids 1 2 3
```

### 3. 单目录方式（向后兼容）
```bash
python enhancement_audio_quality_assessment.py \
  --original_dir "/path/to/original" \
  --enhanced_dir "/path/to/enhanced" \
  --output_dir "/path/to/output"
```

## 新增文件

1. **multi_directory_config_example.json**: 示例配置文件
2. **README_multi_directory.md**: 详细使用说明
3. **run_multi_directory_assessment.sh**: 启动脚本示例
4. **ENHANCEMENT_SUMMARY.md**: 本总结文档

## 技术优势

1. **灵活性**: 支持多种配置方式，适应不同使用场景
2. **扩展性**: 易于添加新的音量匹配方法
3. **稳定性**: 错误隔离，单个目录失败不影响整体
4. **兼容性**: 保持与原有脚本的向后兼容
5. **可维护性**: 清晰的代码结构，易于维护和扩展

## 适用场景

1. **算法对比**: 同一数据集使用多种增强算法
2. **数据集评估**: 同一算法在多个数据集上的表现
3. **参数调优**: 同一算法不同参数设置的比较
4. **批量处理**: 大规模音频增强项目的质量评估

## 性能考虑

- **内存管理**: 每个目录独立处理，避免内存累积
- **GPU利用率**: 在单个目录内保持多GPU并行
- **存储优化**: 支持跳过已存在结果，避免重复处理
- **错误恢复**: 支持从中断点继续处理

这次扩展大大提升了脚本的实用性和灵活性，特别适合需要批量处理多个音频增强项目的场景。 