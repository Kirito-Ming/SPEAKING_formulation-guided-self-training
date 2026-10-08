from __future__ import annotations

import json
from typing import Any


def visible_payload(task: dict[str, Any]) -> dict[str, Any]:
    raw = task["visible_input_json"]
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


def compile_d00(task: dict[str, Any]) -> str:
    """Compile the unchanged standard request (D00), with no rule wrapper."""
    payload = visible_payload(task)
    parts = []
    for key in ("passage", "context", "claim", "question", "prompt", "choices"):
        value = payload.get(key)
        if value not in (None, "", [], {}):
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            parts.append(f"{key.title()}: {rendered}")
    return "\n\n".join(parts)


def render_d00(tokenizer: Any, task: dict[str, Any], model_id: str) -> str:
    kwargs = {"enable_thinking": False} if model_id == "qwen3_4b" else {}
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": compile_d00(task)}],
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
