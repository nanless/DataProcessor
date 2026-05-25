#!/usr/bin/env python3
"""从 run_groundtruth_integration.log 提取每条音频的原音频/mtfaa/mossformer CER 统计。"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_FILE = Path(__file__).parent / "run_groundtruth_integration.log"
OUTPUT_CSV = Path(__file__).parent / "cer_stats_all_audio.csv"
OUTPUT_SUMMARY = Path(__file__).parent / "cer_stats_summary.txt"

JSON_LINE_RE = re.compile(r"生成单独JSON文件:\s*(.+\.json)")


def parse_log_json_paths(log_path: Path) -> list[str]:
    paths = []
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = JSON_LINE_RE.search(line)
            if m:
                paths.append(m.group(1).strip())
    return paths


def load_cer_from_json(json_path: str) -> dict | None:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    info = data.get("audio_file_info", {})
    asr = data.get("asr_results", {})
    original = asr.get("original_evaluation", {})
    enhanced = asr.get("enhanced_evaluations", {})

    def get_cer(method: str):
        ev = enhanced.get(method, {})
        if not ev.get("success", False):
            return None
        return ev.get("cer")

    return {
        "utterance_id": info.get("utterance_id", ""),
        "dataset": data.get("config_info", {}).get("dataset_name", ""),
        "original_path": info.get("original_path", ""),
        "selected_method": info.get("selected_method", ""),
        "original_cer": original.get("cer") if original.get("success") else None,
        "mtfaa_cer": get_cer("mtfaa"),
        "mossformer_cer": get_cer("mossformer"),
    }


def summarize(rows: list[dict]) -> str:
    lines = []
    lines.append("=" * 80)
    lines.append("CER 统计汇总 (基于 run_groundtruth_integration.log 中的 JSON 结果)")
    lines.append("=" * 80)
    lines.append(f"总音频数: {len(rows)}")

    def avg(vals):
        return sum(vals) / len(vals) if vals else float("nan")

    def stats_block(name: str, key: str):
        vals = [r[key] for r in rows if r[key] is not None]
        missing = len(rows) - len(vals)
        lines.append(f"\n--- {name} ---")
        lines.append(f"  有效样本: {len(vals)}, 缺失: {missing}")
        if vals:
            lines.append(f"  平均 CER: {avg(vals):.6f}")
            lines.append(f"  中位数 CER: {sorted(vals)[len(vals)//2]:.6f}")
            lines.append(f"  最小 CER: {min(vals):.6f}")
            lines.append(f"  最大 CER: {max(vals):.6f}")
            zero = sum(1 for v in vals if v == 0.0)
            lines.append(f"  CER=0 数量: {zero} ({100*zero/len(vals):.1f}%)")

    stats_block("原音频", "original_cer")
    stats_block("MTFAA 降噪", "mtfaa_cer")
    stats_block("MossFormer 降噪", "mossformer_cer")

    # 两两比较
    both_enh = [
        (r["mtfaa_cer"], r["mossformer_cer"])
        for r in rows
        if r["mtfaa_cer"] is not None and r["mossformer_cer"] is not None
    ]
    if both_enh:
        mtfaa_better = sum(1 for m, s in both_enh if m < s - 1e-9)
        moss_better = sum(1 for m, s in both_enh if s < m - 1e-9)
        equal = len(both_enh) - mtfaa_better - moss_better
        lines.append("\n--- MTFAA vs MossFormer ---")
        lines.append(f"  MTFAA 更低: {mtfaa_better} ({100*mtfaa_better/len(both_enh):.1f}%)")
        lines.append(f"  MossFormer 更低: {moss_better} ({100*moss_better/len(both_enh):.1f}%)")
        lines.append(f"  CER 相等: {equal} ({100*equal/len(both_enh):.1f}%)")

    orig_mtfaa = [
        (r["original_cer"], r["mtfaa_cer"])
        for r in rows
        if r["original_cer"] is not None and r["mtfaa_cer"] is not None
    ]
    if orig_mtfaa:
        improved = sum(1 for o, m in orig_mtfaa if m < o - 1e-9)
        degraded = sum(1 for o, m in orig_mtfaa if m > o + 1e-9)
        same = len(orig_mtfaa) - improved - degraded
        lines.append("\n--- 原音频 vs MTFAA ---")
        lines.append(f"  MTFAA 改善: {improved} ({100*improved/len(orig_mtfaa):.1f}%)")
        lines.append(f"  MTFAA 退化: {degraded} ({100*degraded/len(orig_mtfaa):.1f}%)")
        lines.append(f"  无变化: {same} ({100*same/len(orig_mtfaa):.1f}%)")

    orig_moss = [
        (r["original_cer"], r["mossformer_cer"])
        for r in rows
        if r["original_cer"] is not None and r["mossformer_cer"] is not None
    ]
    if orig_moss:
        improved = sum(1 for o, m in orig_moss if m < o - 1e-9)
        degraded = sum(1 for o, m in orig_moss if m > o + 1e-9)
        same = len(orig_moss) - improved - degraded
        lines.append("\n--- 原音频 vs MossFormer ---")
        lines.append(f"  MossFormer 改善: {improved} ({100*improved/len(orig_moss):.1f}%)")
        lines.append(f"  MossFormer 退化: {degraded} ({100*degraded/len(orig_moss):.1f}%)")
        lines.append(f"  无变化: {same} ({100*same/len(orig_moss):.1f}%)")

    # 按数据集
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)

    lines.append("\n--- 按数据集 ---")
    for ds in sorted(by_ds):
        ds_rows = by_ds[ds]
        lines.append(f"\n  [{ds}] 样本数: {len(ds_rows)}")
        for key, label in [
            ("original_cer", "原音频"),
            ("mtfaa_cer", "MTFAA"),
            ("mossformer_cer", "MossFormer"),
        ]:
            vals = [r[key] for r in ds_rows if r[key] is not None]
            if vals:
                lines.append(f"    {label} 平均 CER: {avg(vals):.6f}")

    lines.append("")
    return "\n".join(lines)


def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else LOG_FILE
    print(f"解析日志: {log_path}")
    json_paths = parse_log_json_paths(log_path)
    print(f"找到 {len(json_paths)} 条 JSON 记录")

    rows = []
    missing_json = 0
    for i, jp in enumerate(json_paths):
        row = load_cer_from_json(jp)
        if row is None:
            missing_json += 1
            continue
        rows.append(row)
        if (i + 1) % 10000 == 0:
            print(f"  已处理 {i + 1}/{len(json_paths)} ...")

    print(f"成功读取 {len(rows)} 条, JSON 缺失/损坏: {missing_json}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "dataset",
                "original_path",
                "selected_method",
                "original_cer",
                "mtfaa_cer",
                "mossformer_cer",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"逐条 CER 已写入: {OUTPUT_CSV}")

    summary = summarize(rows)
    OUTPUT_SUMMARY.write_text(summary, encoding="utf-8")
    print(f"汇总统计已写入: {OUTPUT_SUMMARY}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
