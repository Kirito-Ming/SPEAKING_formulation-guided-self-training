#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "data/transfer_tasks.parquet": "efd61b564dd80c330367f0db97383845494d92341ccf6702a6cbe67049ec70a1",
    "weights/qwen3_4b/adapter_state.safetensors": "f31ab7538184cbbc834acb62c6d8e5202f19b03b79905f6d34028673206deacb",
    "weights/llama_3_2_3b/adapter_state.safetensors": "604c0ee20f141842687c7a134fc069952171554292c8aa9055ccb1fb652f47f2",
    "reference/paper_transfer_results.csv": "fab0b13958e154bc4b24fccfc67495de92f69d8ca7be7d5f47ecbde096a35366",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    for relative, expected in EXPECTED.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise SystemExit(f"SHA-256 mismatch for {relative}: {actual}")
    tasks = pd.read_parquet(ROOT / "data/transfer_tasks.parquet")
    counts = tasks.groupby("source_id").size().to_dict()
    if len(tasks) != 900 or counts != {"drop": 300, "openbookqa": 300, "sciq": 300}:
        raise SystemExit(f"unexpected transfer partition: rows={len(tasks)}, sources={counts}")
    if tasks.task_id.duplicated().any():
        raise SystemExit("duplicate task_id in transfer partition")
    print(json.dumps({"status": "ok", "rows": len(tasks), "source_counts": counts, "assets": EXPECTED}, indent=2))


if __name__ == "__main__":
    main()
