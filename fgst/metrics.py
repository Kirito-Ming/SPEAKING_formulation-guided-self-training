from __future__ import annotations

from collections import Counter
from typing import Iterable


def paired_metrics(rows: Iterable[dict]) -> dict:
    records = list(rows)
    n = len(records)
    base_success = sum(bool(row["base_success"]) for row in records)
    adapted_success = sum(bool(row["adapted_success"]) for row in records)
    rescue = sum((not row["base_success"]) and row["adapted_success"] for row in records)
    harm = sum(row["base_success"] and (not row["adapted_success"]) for row in records)
    if adapted_success - base_success != rescue - harm:
        raise ValueError("paired transition identity failed")
    base_failures = n - base_success
    return {
        "evaluations": n,
        "base_success": base_success,
        "adapted_success": adapted_success,
        "rescue": rescue,
        "harm": harm,
        "net_gain": rescue - harm,
        "base_success_rate": base_success / n if n else None,
        "adapted_success_rate": adapted_success / n if n else None,
        "unconditional_rescue_rate": rescue / n if n else None,
        "unconditional_harm_rate": harm / n if n else None,
        "conditional_rescue": rescue / base_failures if base_failures else None,
        "conditional_harm": harm / base_success if base_success else None,
        "transition_counts": dict(Counter(row["transition"] for row in records)),
    }
