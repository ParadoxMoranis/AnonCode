#!/usr/bin/env python3
"""
Operation ablation runner.

Evaluates how different layout actions contribute to overall pipeline behavior by
toggling rewriting / line spacing / char spacing / font size independently.
"""

import copy
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from run_translation import load_config, validate_config, TranslationPipeline  # noqa: E402


DEFAULT_CASES = [
    {
        "name": "all_enabled",
        "policy": {
            "allow_rewriting": True,
            "allow_line_spacing": True,
            "allow_char_spacing": True,
            "allow_font_size": True,
        },
    },
    {
        "name": "no_rewriting",
        "policy": {
            "allow_rewriting": False,
            "allow_line_spacing": True,
            "allow_char_spacing": True,
            "allow_font_size": True,
        },
    },
    {
        "name": "no_char_spacing",
        "policy": {
            "allow_rewriting": True,
            "allow_line_spacing": True,
            "allow_char_spacing": False,
            "allow_font_size": True,
        },
    },
    {
        "name": "no_font_size",
        "policy": {
            "allow_rewriting": True,
            "allow_line_spacing": True,
            "allow_char_spacing": True,
            "allow_font_size": False,
        },
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run operation ablation experiments")
    parser.add_argument("--config", default="config.yaml", help="Base config path")
    parser.add_argument("--cases-json", help="Optional JSON file overriding default ablation cases")
    parser.add_argument("--category", nargs="+", help="Optional categories")
    parser.add_argument("--layout", nargs="+", help="Optional layouts")
    parser.add_argument("--workers", type=int, default=1, help="Pipeline workers per case")
    args = parser.parse_args()

    config = load_config(str(BASE_DIR / args.config))
    if not validate_config(config):
        raise RuntimeError("Base config validation failed.")

    cases = DEFAULT_CASES
    if args.cases_json:
        case_path = BASE_DIR / args.cases_json if not Path(args.cases_json).is_absolute() else Path(args.cases_json)
        cases = json.loads(case_path.read_text(encoding="utf-8"))

    output_dir = BASE_DIR / "runs" / "output" / "operation_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "cases": [],
    }

    for case in cases:
        case_name = case["name"]
        case_config = copy.deepcopy(config)
        case_config["paths"]["output_dir"] = f"runs/output/operation_ablation/{case_name}"
        case_config.setdefault("processing", {})["skip_existing"] = False
        case_config.setdefault("ablation", {})["operation_policy"] = case["policy"]

        print(f"\n{'=' * 80}")
        print(f"🧪 Operation Ablation: {case_name}")
        print(f"{'=' * 80}")

        pipeline = TranslationPipeline(case_config)
        results = pipeline.run(
            categories=args.category,
            layouts=args.layout,
            max_workers=args.workers,
        )

        summary["cases"].append(
            {
                "name": case_name,
                "policy": case["policy"],
                "success_count": sum(1 for item in results if item.get("success")),
                "result_count": len(results),
                "results": results,
            }
        )

    summary_path = output_dir / "operation_ablation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Operation ablation summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
