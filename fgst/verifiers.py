from __future__ import annotations

import json
import re
from typing import Any


def strict_final(text: str) -> tuple[str, bool]:
    matches = re.findall(r"final answer\s*:\s*([^\n]+)", str(text), flags=re.I)
    return (matches[-1].strip(), True) if matches else ("", False)


def permissive_final(text: str) -> tuple[str, bool]:
    final, parsed = strict_final(text)
    if parsed:
        return final, True
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    return (lines[-1], True) if lines else ("", False)


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def verify(source_id: str, output_text: str, gold_payload: str | dict[str, Any], permissive: bool = False) -> tuple[bool, bool]:
    gold = json.loads(gold_payload) if isinstance(gold_payload, str) else gold_payload
    final, parsed = permissive_final(output_text) if permissive else strict_final(output_text)
    if not parsed:
        return False, False
    expected = str(gold["answer"])
    if source_id in {"openbookqa", "sciq"}:
        match = re.search(r"\b([A-D])\b", final.upper())
        actual = match.group(1) if match else ""
        return actual == expected.upper(), bool(actual)
    if source_id == "drop":
        actual = _norm(final)
        targets = [expected, *gold.get("aliases", [])]
        exact = any(actual == _norm(target) for target in targets)
        expected_tokens, actual_tokens = set(_norm(expected).split()), set(actual.split())
        f1 = 0.0 if not expected_tokens or not actual_tokens else 2 * len(expected_tokens & actual_tokens) / (len(expected_tokens) + len(actual_tokens))
        return exact or f1 >= 0.8, bool(actual)
    raise KeyError(f"unsupported transfer source: {source_id}")
