#!/usr/bin/env python3
"""从 integrated_by_groundtruth 的 JSON 结果生成 onlyenhanced 输出目录。

仅在两路降噪音频 (mtfaa / mossformer) 中选择 CER 更低者，不回退原音频。
CER 相等时优先 mtfaa（与 groundtruth_based_integration.py 一致）。
"""

import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import json as json_module

SCRIPT_DIR = Path(__file__).parent
CONFIG_FILE = SCRIPT_DIR / "config.json"

METHOD_PRIORITY = {
    "mtfaa": 0,
    "mossformer": 1,
}


def load_config(config_path: Path) -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json_module.load(f)["datasets"]


def find_enhanced_audio(original_path: str, method_name: str, base_dir: Path) -> Path | None:
    original_rel_path = Path(original_path).relative_to(base_dir)
    enhanced_dir = base_dir.parent / f"{base_dir.name}_{method_name}_enhanced"
    for suffix in (".wav", ".WAV"):
        candidate = enhanced_dir / original_rel_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def pick_best_enhanced_method(enhanced_evaluations: dict) -> tuple[str | None, float | None]:
    best_method = None
    best_cer = float("inf")
    best_priority = 999

    for method_name in ("mtfaa", "mossformer"):
        ev = enhanced_evaluations.get(method_name)
        if not ev or not ev.get("success"):
            continue
        current_cer = ev["cer"]
        current_priority = METHOD_PRIORITY.get(method_name, 999)
        if current_cer < best_cer - 1e-9 or (
            abs(current_cer - best_cer) < 1e-8 and current_priority < best_priority
        ):
            best_method = method_name
            best_cer = current_cer
            best_priority = current_priority

    return best_method, (best_cer if best_method else None)


def process_one_json(args: tuple) -> dict:
    json_path_str, base_dir_str, integrated_dir_str, onlyenhanced_dir_str = args
    json_path = Path(json_path_str)
    base_dir = Path(base_dir_str)
    onlyenhanced_dir = Path(onlyenhanced_dir_str)

    result = {
        "json_path": json_path_str,
        "status": "ok",
        "selected_method": None,
        "target_path": None,
    }

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        original_path = data["audio_file_info"]["original_path"]
        enhanced_evaluations = data["asr_results"]["enhanced_evaluations"]
        best_method, best_cer = pick_best_enhanced_method(enhanced_evaluations)

        if not best_method:
            result["status"] = "no_enhanced_available"
            return result

        source_path = find_enhanced_audio(original_path, best_method, base_dir)
        if not source_path:
            result["status"] = "enhanced_file_missing"
            result["selected_method"] = best_method
            return result

        relative_path = Path(original_path).relative_to(base_dir)
        target_path = onlyenhanced_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        result["selected_method"] = best_method
        result["selected_cer"] = best_cer
        result["source_path"] = str(source_path)
        result["target_path"] = str(target_path)
        return result

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        return result


def collect_json_files(integrated_dir: Path) -> list[Path]:
    skip_names = {
        "groundtruth_based_integration_results.json",
        "integration_summary.json",
        "onlyenhanced_summary.json",
    }
    json_files = []
    for jp in integrated_dir.rglob("*.json"):
        if jp.name in skip_names:
            continue
        if "kaldi_files" in jp.parts:
            continue
        json_files.append(jp)
    return sorted(json_files)


def run_dataset(dataset: dict, workers: int) -> dict:
    name = dataset["name"]
    base_dir = Path(dataset["base_dir"])
    integrated_dir = Path(dataset["output_dir"])
    onlyenhanced_dir = Path(str(integrated_dir) + "_onlyenhanced")

    if not integrated_dir.exists():
        return {"dataset": name, "error": f"integrated 目录不存在: {integrated_dir}"}

    onlyenhanced_dir.mkdir(parents=True, exist_ok=True)
    json_files = collect_json_files(integrated_dir)

    stats = {
        "dataset": name,
        "onlyenhanced_dir": str(onlyenhanced_dir),
        "total_json": len(json_files),
        "copied": 0,
        "no_enhanced_available": 0,
        "enhanced_file_missing": 0,
        "error": 0,
        "method_counts": {"mtfaa": 0, "mossformer": 0},
    }

    task_args = [
        (str(jp), str(base_dir), str(integrated_dir), str(onlyenhanced_dir))
        for jp in json_files
    ]

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_one_json, arg) for arg in task_args]
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            done += 1
            status = res["status"]
            if status == "ok":
                stats["copied"] += 1
                method = res.get("selected_method")
                if method in stats["method_counts"]:
                    stats["method_counts"][method] += 1
            elif status == "no_enhanced_available":
                stats["no_enhanced_available"] += 1
            elif status == "enhanced_file_missing":
                stats["enhanced_file_missing"] += 1
            else:
                stats["error"] += 1

            if done % 5000 == 0:
                print(f"  [{name}] 进度 {done}/{len(json_files)}")

    summary_path = onlyenhanced_dir / "onlyenhanced_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    return stats


def main():
    workers = 8
    if len(sys.argv) > 1:
        workers = int(sys.argv[1])

    datasets = load_config(CONFIG_FILE)
    print(f"共 {len(datasets)} 个数据集, workers={workers}\n")

    all_stats = []
    for ds in datasets:
        print(f"处理数据集: {ds['name']}")
        stats = run_dataset(ds, workers)
        all_stats.append(stats)
        if "error" in stats and stats.get("total_json", 0) == 0:
            print(f"  跳过: {stats['error']}")
            continue
        print(
            f"  输出: {stats['onlyenhanced_dir']}\n"
            f"  复制成功: {stats['copied']}/{stats['total_json']}\n"
            f"  mtfaa: {stats['method_counts']['mtfaa']}, "
            f"mossformer: {stats['method_counts']['mossformer']}\n"
            f"  无可用降噪: {stats['no_enhanced_available']}, "
            f"源文件缺失: {stats['enhanced_file_missing']}, "
            f"错误: {stats['error']}\n"
        )

    summary_file = SCRIPT_DIR / "onlyenhanced_all_datasets_summary.json"
    summary_file.write_text(
        json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"总汇总: {summary_file}")


if __name__ == "__main__":
    main()
