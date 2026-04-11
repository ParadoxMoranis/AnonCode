#!/usr/bin/env python3
"""
Aggregate planner attribution logs into action contribution statistics.
"""

import json
import sys
import argparse
from collections import Counter
from pathlib import Path


def iter_planner_files(root: Path):
    if root.is_file() and root.name.endswith("_planner.json"):
        yield root
        return
    for path in root.rglob("*_planner.json"):
        yield path


def main():
    parser = argparse.ArgumentParser(description="Analyze planner attribution logs")
    parser.add_argument("input", help="Planner JSON file or directory")
    args = parser.parse_args()

    root = Path(args.input)
    action_counter = Counter()
    stage_counter = Counter()
    issue_counter = Counter()
    files = 0

    for planner_path in iter_planner_files(root):
        payload = json.loads(planner_path.read_text(encoding="utf-8"))
        files += 1
        for event in payload.get("action_log", []):
            stage_counter[event.get("stage", "unknown")] += 1
            for action in event.get("applied_actions", []):
                action_counter[action] += 1
            issue = event.get("metadata", {}).get("issue")
            if issue:
                issue_counter[issue] += 1

    summary = {
        "planner_files": files,
        "action_counts": dict(action_counter),
        "stage_counts": dict(stage_counter),
        "issue_counts": dict(issue_counter),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
