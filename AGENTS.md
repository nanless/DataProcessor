# AGENTS.md

## 环境

- **唯一环境**: `conda activate kimi-audio`
- **无包管理文件**: 没有 `requirements.txt`、`pyproject.toml`、`setup.py`。依赖需手动安装到 conda 环境中。
- **关键模型路径**（硬编码于源码）:
  - Kimi-Audio-7B-Instruct: `/root/data/pretrained_models/Kimi-Audio-7B-Instruct`
  - Kimi-Audio 代码: `/root/code/github_repos/Kimi-Audio`
  - Resemble-Enhance: `/root/data/pretrained_ models/resemble-enhance/ enhancer_stage2`
  - TEN-VAD: `../include/ten_vad`（相对于脚本目录）
- **Ollama 服务**: 本地运行 Qwen3:32B，端口 11434-11437（每 GPU 一个），HTTP 服务端口 8000-8007。

## 项目结构

单仓库，非 monorepo，无 setup 安装。脚本直接 `python <script>` 运行。

```
DataProcessor/
├── cut_noise_segments.py                    # 音频切分工具
├── parse_vibravox_dataset.py                # Parquet → WAV 提取
├── speech_enhancement_process/
│   ├── zipenhancer_batch_inference.py       # ModelScope ZipEnhancer (16kHz)
│   ├── resemble_enhance_batch_inference.py   # Resemble-Enhance (44.1kHz)
│   ├── mossformergan_batch_inference.py      # ClearVoice MossFormerGAN (16kHz)
│   ├── process_quality_assess_bycer_deprecated/  # [已废弃] 旧版评估管线
│   └── intergrate_enhanced_speech_bygtcer/       # [当前] 基于Groundtruth的集成管线
```

## 两条管线的区别

| | `process_quality_assess_bycer_deprecated/` | `intergrate_enhanced_speech_bygtcer/` |
|---|---|---|
| **状态** | 已废弃 | 当前主用 |
| **评估方式** | 比较原始音频 vs 增强音频的 CER | 使用 groundtruth 文本计算 CER |
| **配置格式** | JSON config file (`directory_configs` 数组) | `config.json`（`global_config` + `datasets` 嵌套结构） |
| **LLM 服务** | 同一文件 | 独立版本（新增 qwen3 的"减少思考" prompt 优化） |

勿混淆两条管线的 `llm_service.py` 或 shell 脚本。

## 命令行速查

### 当前管线 (bygtcer) — 主要使用

```bash
cd speech_enhancement_process/intergrate_enhanced_speech_bygtcer

# 启动 LLM 服务（必须先运行）
./auto_start_llm_services.sh

# 运行主管线
./run_groundtruth_integration.sh
./run_groundtruth_integration.sh --datasets dataset1,dataset2 --num_gpus 4

# 后处理（按顺序）：
python extract_cer_stats_from_log.py [可选的日志路径]
python create_onlyenhanced_outputs.py
python create_kaldi_files_for_onlyenhanced.py [--with-features]

# 停止服务
./stop_multi_llm_services.sh
```

### 旧管线 (deprecated) — 仅供参考

```bash
cd speech_enhancement_process/process_quality_assess_bycer_deprecated

python test_system.py                                    # 唯一的系统检查脚本
./auto_start_llm_services.sh --model-type qwen3 --model-name qwen3:32b
python enhancement_audio_quality_assessment.py --config_file config.json
./stop_multi_llm_services.sh
```

### 批量增强（独立脚本，无需服务）

```bash
python speech_enhancement_process/zipenhancer_batch_inference.py
python speech_enhancement_process/resemble_enhance_batch_inference.py
python speech_enhancement_process/mossformergan_batch_inference.py
```

### 音频切分

```bash
python cut_noise_segments.py --input_dir /path/to/input --output_dir /path/to/output --duration 10
```

## 重要注意事项

### 硬编码路径

所有三个批量增强脚本的输入/输出路径都硬编码在 `@dataclass` 配置类中。必须编辑源码才能更改目录 —— 不支持命令行参数。同样，`parse_vibravox_dataset.py` 的路径也是硬编码的。

### CUDA 多进程必须用 spawn

所有批量增强脚本在模块级别执行 `mp.set_start_method('spawn', force=True)`。这在 `import` 时立即运行，如果忽略会导致 CUDA 错误。

### GPU 隔离模式

多 GPU 脚本使用以下模式：将 `CUDA_VISIBLE_DEVICES=<单_gpu_id>` 传入子进程，然后在子进程内部使用 `cuda:0`。这是 PyTorch 多进程的常见变通方案，但如果设备映射被篡改则容易出问题。

### 模型采样率差异

- **16kHz**: ZipEnhancer、MossFormerGAN (ClearVoice)
- **44.1kHz**: Resemble-Enhance

每个批量增强脚本内的 `fft_downsample` 方法会处理输入重采样（使用 librosa，fallback 到线性插值）。输出总是模型的原生采样率。

### ClearVoice 的文件输入/输出模式

MossFormerGAN 使用 ClearVoice 的基于文件的 API：`clearvoice_instance(temp_input_path, online_write=False)`。它写入临时文件然后读回，而非内存中处理。需要在 `TMPDIR` 中有足够空间。

### 本地连接代理绕过

两个管线的 `llm_service.py` 在发出本地 HTTP 请求前，都会取消 `http_proxy`/`https_proxy` 并将 `no_proxy` 设为 `localhost,127.0.0.1,::1`。运行脚本（`run_groundtruth_integration.sh`）在执行健康检查前也会临时取消代理。如果忘记这个步骤，对 Ollama 的 HTTP 请求可能通过代理路由，导致连接失败。

### CER 相等时的优先级

当多个增强方法的 CER 在 `1e-8` 误差范围内相等时，按以下优先级选择：
1. mossformer（最高）
2. zipenhancer / mtfaa
3. resemble（最低）

此规则在 `groundtruth_based_integration.py` 和 `audio_integration_by_cer.py` 中均有编码。

### 文件名清理

所有脚本都包含相同的 `_sanitize_file_path()` 逻辑：将空格/特殊字符替换为下划线，对超过 200 字符的长文件名进行 MD5 哈希截断。长度限制为 200 个字符。切勿移除 —— 写入路径时会被触发。

### 音频数组维度处理

ModelScope 和 ClearVoice 返回不同维度的数组（2D、>2D、torch vs numpy）。每个批量增强脚本都有冗长（且重复）的 `squeeze`/`flatten` 逻辑来处理此问题。修改音频 pipeline 时需保留此逻辑。

### 8-GPU 假设

默认 GPU 列表为 `[0,1,2,3,4,5,6,7]`。LLM 服务脚本将 8000-8007 端口映射到 GPU 0-7。如果硬件 GPU 数量不同，脚本会进行自适应，但默认值假定为 8 GPU。

### 无构建/检查/测试

- 无测试框架（无 pytest、unittest）。仅存在一个程序化的系统状态检查脚本 `test_system.py`。
- 无 lint、formatter、typechecker、CI/CD 配置。
- 所有脚本均直接运行，无需构建步骤。

### config.json 约定

bygtcer 管线的 `config.json` 将所有内容嵌套在 `global_config` 和 `datasets` 键下。数据集使用 Kaldi 风格的 `text` 和 `wav.scp` 文件。直接修改顶层键会破坏管线。
