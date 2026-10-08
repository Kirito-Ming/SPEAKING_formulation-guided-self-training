#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from fgst.metrics import paired_metrics
from fgst.verifiers import verify


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score paired base and adapted D00 responses.")
    parser.add_argument("--tasks", type=Path, default=Path("data/transfer_tasks.parquet"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    tasks = pd.read_parquet(args.tasks).set_index("task_id")
    base = {row["task_id"]: row for row in read_jsonl(args.base)}
    adapted = {row["task_id"]: row for row in read_jsonl(args.adapted)}
    if set(base) != set(adapted):
        raise ValueError("base and adapted task IDs differ")
    if not set(base).issubset(tasks.index):
        raise ValueError("predictions contain unknown task IDs")

    paired = []
    for task_id in sorted(base):
        task = tasks.loc[task_id]
        base_success, base_parseable = verify(task.source_id, base[task_id]["output_text"], task.gold_verifier_payload)
        adapted_success, adapted_parseable = verify(task.source_id, adapted[task_id]["output_text"], task.gold_verifier_payload)
        permissive_base_success, permissive_base_parseable = verify(task.source_id, base[task_id]["output_text"], task.gold_verifier_payload, permissive=True)
        permissive_adapted_success, permissive_adapted_parseable = verify(task.source_id, adapted[task_id]["output_text"], task.gold_verifier_payload, permissive=True)
        transition = "rescue" if not base_success and adapted_success else "harm" if base_success and not adapted_success else "success_preserved" if base_success else "failure_persisted"
        paired.append({
            "task_id": task_id,
            "source_id": task.source_id,
            "model_id": base[task_id]["model_id"],
            "base_output": base[task_id]["output_text"],
            "adapted_output": adapted[task_id]["output_text"],
            "base_success": base_success,
            "adapted_success": adapted_success,
            "base_parseable": base_parseable,
            "adapted_parseable": adapted_parseable,
            "permissive_base_success": permissive_base_success,
            "permissive_adapted_success": permissive_adapted_success,
            "permissive_base_parseable": permissive_base_parseable,
            "permissive_adapted_parseable": permissive_adapted_parseable,
            "transition": transition,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(paired).to_parquet(args.output_dir / "paired_predictions.parquet", index=False)
    overall = paired_metrics(paired)
    by_source = {source: paired_metrics([row for row in paired if row["source_id"] == source]) for source in sorted({row["source_id"] for row in paired})}
    summary = {"paper_comparable": len(paired) == 900, "overall": overall, "by_source": by_source}
    (args.output_dir / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
