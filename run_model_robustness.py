#!/usr/bin/env python3
"""
Model robustness experiment runner.

Runs the main translation pipeline under multiple translation/reflow model pairs
and saves a comparable result table for framework robustness analysis.
"""

import copy
import json
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from run_translation import load_config, validate_config, TranslationPipeline  # noqa: E402


def load_matrix(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Matrix file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except Exception as exc:
        raise RuntimeError("PyYAML is required for YAML model matrix files.") from exc
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def apply_case(base_config, case):
    config = copy.deepcopy(base_config)
    def expand(value):
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            return os.environ.get(value[2:-1], "")
        return value
    if "translation" in case:
        for key, value in case["translation"].items():
            config["translation"]["api"][key] = expand(value)
    if "reflow" in case:
        for key, value in case["reflow"].items():
            config["reflow"]["api"][key] = expand(value)
    if "paths" in case:
        config["paths"].update(case["paths"])
    if "planner" in case:
        config.setdefault("planner", {}).update(case["planner"])
    if "ablation" in case:
        config.setdefault("ablation", {}).update(case["ablation"])
    return config


def main():
    parser = argparse.ArgumentParser(description="Run model robustness experiments")
    parser.add_argument("--config", default="config.yaml", help="Base config path")
    parser.add_argument("--matrix", required=True, help="JSON/YAML file describing model cases")
    parser.add_argument("--category", nargs="+", help="Optional categories")
    parser.add_argument("--layout", nargs="+", help="Optional layouts")
    parser.add_argument("--workers", type=int, default=1, help="Pipeline workers per case")
    args = parser.parse_args()

    base_config = load_config(str(BASE_DIR / args.config))
    matrix = load_matrix(BASE_DIR / args.matrix if not Path(args.matrix).is_absolute() else Path(args.matrix))
    cases = matrix.get("cases", matrix if isinstance(matrix, list) else [])
    if not cases:
        raise RuntimeError("No experiment cases found in matrix file.")

    robustness_dir = BASE_DIR / "runs" / "output" / "robustness"
    robustness_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    for idx, case in enumerate(cases, start=1):
        case_name = case.get("name", f"case_{idx}")
        config = apply_case(base_config, case)
        config["paths"]["output_dir"] = f"runs/output/robustness/{case_name}"
        config.setdefault("processing", {})["skip_existing"] = False
        if not validate_config(config):
            raise RuntimeError(f"Config validation failed for case: {case_name}")

        print(f"\n{'=' * 80}")
        print(f"🧪 Robustness Case {idx}: {case_name}")
        print(f"{'=' * 80}")

        pipeline = TranslationPipeline(config)
        results = pipeline.run(
            categories=args.category,
            layouts=args.layout,
            max_workers=args.workers,
        )
        case_result = {
            "name": case_name,
            "translation_model": config["translation"]["api"]["model"],
            "reflow_model": config["reflow"]["api"]["model"],
            "result_count": len(results),
            "success_count": sum(1 for item in results if item.get("success")),
            "results": results,
        }
        summary["cases"].append(case_result)

    summary_path = robustness_dir / "model_robustness_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Robustness summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
