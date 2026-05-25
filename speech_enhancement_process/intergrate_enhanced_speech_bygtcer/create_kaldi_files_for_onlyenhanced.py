#!/usr/bin/env python3
"""为 *_integrated_by_groundtruth_onlyenhanced 目录生成 kaldi_files。

生成内容：
  - wav.scp：路径指向 onlyenhanced 音频
  - text / text.tn / text.tn.checked / utt2spk / spk2utt / *_single：从 integrated kaldi_files 复制

可选 --with-features：调用 CosyVoice 工具重新提取 embedding / speech token 并生成 parquet。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"

COPY_FROM_INTEGRATED = [
    "text",
    "text.tn",
    "text.tn.checked",
    "utt2spk",
    "spk2utt",
    "spk2utt_single",
    "utt2spk_single",
]


def load_config() -> list[dict]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)["datasets"]


def load_wav_scp(path: Path) -> list[tuple[str, str]]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"{path}:{line_num} 无效 wav.scp 行: {line}")
            entries.append((parts[0], parts[1]))
    return entries


def remap_wav_path(original_path: str, base_dir: Path, onlyenhanced_dir: Path) -> Path:
    rel = Path(original_path).relative_to(base_dir)
    target = onlyenhanced_dir / rel
    if not target.exists():
        alt = onlyenhanced_dir / rel.with_suffix(".wav" if rel.suffix == ".WAV" else ".WAV")
        if alt.exists():
            return alt
        raise FileNotFoundError(f"onlyenhanced 音频不存在: {target}")
    return target


def write_wav_scp(entries: list[tuple[str, str]], out_path: Path) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for utt, wav in entries:
            f.write(f"{utt}\t{wav}\n")


def copy_or_link(src: Path, dst: Path, use_hardlink: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_hardlink:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def generate_basic_kaldi(dataset: dict, use_hardlink: bool = True) -> dict:
    name = dataset["name"]
    base_dir = Path(dataset["base_dir"])
    wav_scp_file = Path(dataset["wav_scp_file"])
    integrated_dir = Path(dataset["output_dir"])
    onlyenhanced_dir = Path(str(integrated_dir) + "_onlyenhanced")
    integrated_kaldi = integrated_dir / "kaldi_files"
    onlyenhanced_kaldi = onlyenhanced_dir / "kaldi_files"

    stats = {
        "dataset": name,
        "onlyenhanced_kaldi_dir": str(onlyenhanced_kaldi),
        "wav_scp_count": 0,
        "missing_audio": 0,
        "copied_files": [],
        "errors": [],
    }

    if not onlyenhanced_dir.exists():
        stats["errors"].append(f"onlyenhanced 目录不存在: {onlyenhanced_dir}")
        return stats
    if not integrated_kaldi.exists():
        stats["errors"].append(f"integrated kaldi_files 不存在: {integrated_kaldi}")
        return stats

    onlyenhanced_kaldi.mkdir(parents=True, exist_ok=True)

    remapped = []
    for utt, orig_path in load_wav_scp(wav_scp_file):
        try:
            target = remap_wav_path(orig_path, base_dir, onlyenhanced_dir)
            remapped.append((utt, str(target)))
        except FileNotFoundError as e:
            stats["missing_audio"] += 1
            stats["errors"].append(str(e))

    write_wav_scp(remapped, onlyenhanced_kaldi / "wav.scp")
    stats["wav_scp_count"] = len(remapped)

    for fname in COPY_FROM_INTEGRATED:
        src = integrated_kaldi / fname
        dst = onlyenhanced_kaldi / fname
        if not src.exists():
            stats["errors"].append(f"源文件缺失: {src}")
            continue
        copy_or_link(src, dst, use_hardlink)
        stats["copied_files"].append(fname)

    return stats


def run_cosyvoice_features(kaldi_dir: Path, cosyvoice_dir: Path, model_dir: Path, num_thread: int) -> None:
    campplus = model_dir / "campplus.onnx"
    speech_tokenizer_candidates = [
        model_dir / "speech_tokenizer_v3.onnx",
        model_dir / "speech_tokenizer_v1.onnx",
        model_dir / "speech_tokenizer_v2.onnx",
    ]
    speech_tokenizer = next((p for p in speech_tokenizer_candidates if p.exists()), None)

    if not campplus.exists():
        raise FileNotFoundError(f"campplus.onnx 不存在: {campplus}")
    if speech_tokenizer is None:
        raise FileNotFoundError(f"speech_tokenizer onnx 不存在于: {model_dir}")

    extract_embedding = cosyvoice_dir / "tools" / "extract_embedding.py"
    extract_speech_token = cosyvoice_dir / "tools" / "extract_speech_token.py"
    make_parquet = cosyvoice_dir / "tools" / "make_parquet_list.py"

    for tool in (extract_embedding, extract_speech_token, make_parquet):
        if not tool.exists():
            raise FileNotFoundError(f"CosyVoice 工具不存在: {tool}")

    subprocess.run(
        [
            sys.executable,
            str(extract_embedding),
            "--dir",
            str(kaldi_dir),
            "--onnx_path",
            str(campplus),
            "--num_thread",
            str(num_thread),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(extract_speech_token),
            "--dir",
            str(kaldi_dir),
            "--onnx_path",
            str(speech_tokenizer),
            "--num_thread",
            str(num_thread),
        ],
        check=True,
    )
    parquet_dir = kaldi_dir / "parquet_step2"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(make_parquet),
            "--src_dir",
            str(kaldi_dir),
            "--des_dir",
            str(parquet_dir),
            "--num_utts_per_parquet",
            "1000",
            "--num_processes",
            str(max(1, num_thread // 2)),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="为 onlyenhanced 数据集生成 kaldi_files")
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="逗号分隔的数据集名，默认全部",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制而非硬链接 metadata 文件",
    )
    parser.add_argument(
        "--with-features",
        action="store_true",
        help="重新提取 embedding / speech token / parquet（耗时较长）",
    )
    parser.add_argument(
        "--cosyvoice-dir",
        type=str,
        default="/root/code/github_repos/CosyVoice",
        help="CosyVoice 仓库路径",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="CosyVoice 预训练模型目录（含 campplus.onnx）",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=16,
        help="特征提取线程数",
    )
    args = parser.parse_args()

    selected = {x.strip() for x in args.datasets.split(",") if x.strip()}
    all_stats = []

    for ds in load_config():
        if selected and ds["name"] not in selected:
            continue
        print(f"\n=== {ds['name']} ===")
        stats = generate_basic_kaldi(ds, use_hardlink=not args.copy)
        all_stats.append(stats)

        if stats["errors"] and stats["wav_scp_count"] == 0:
            print(f"  失败: {stats['errors'][0]}")
            continue

        print(f"  kaldi_files: {stats['onlyenhanced_kaldi_dir']}")
        print(f"  wav.scp: {stats['wav_scp_count']} 条")
        print(f"  复制 metadata: {', '.join(stats['copied_files'])}")
        if stats["missing_audio"]:
            print(f"  缺失音频: {stats['missing_audio']}")
        if stats["errors"]:
            print(f"  警告 {len(stats['errors'])} 条（前3条）:")
            for err in stats["errors"][:3]:
                print(f"    - {err}")

        if args.with_features:
            model_dir = Path(args.model_dir) if args.model_dir else None
            if not model_dir or not model_dir.exists():
                print("  跳过特征提取: 未指定有效 --model-dir")
                continue
            kaldi_dir = Path(stats["onlyenhanced_kaldi_dir"])
            print(f"  开始特征提取 (model={model_dir}) ...")
            run_cosyvoice_features(
                kaldi_dir,
                Path(args.cosyvoice_dir),
                model_dir,
                args.num_thread,
            )
            print("  特征提取完成")

    summary_path = SCRIPT_DIR / "onlyenhanced_kaldi_summary.json"
    summary_path.write_text(json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n汇总: {summary_path}")


if __name__ == "__main__":
    main()
