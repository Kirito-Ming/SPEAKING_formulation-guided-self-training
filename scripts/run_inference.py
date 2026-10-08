#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fgst.lora import load_trajectory_layers
from fgst.tasks import render_d00


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic D00 responses.")
    parser.add_argument("--model-id", choices=["qwen3_4b", "llama_3_2_3b"], required=True)
    parser.add_argument("--condition", choices=["base", "adapted"], required=True)
    parser.add_argument("--base-model", required=True, help="Hugging Face ID or local model directory")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--tasks", type=Path, default=Path("data/transfer_tasks.parquet"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Smoke-test only; omit for paper reproduction")
    parser.add_argument("--device-map", default="auto")
    args = parser.parse_args()
    if args.condition == "adapted" and not args.adapter:
        parser.error("--adapter is required for --condition adapted")

    torch.manual_seed(20260825)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    )
    adapter_manifest = None
    if args.condition == "adapted":
        adapter_manifest = load_trajectory_layers(model, args.adapter)
    model.eval()

    tasks = pd.read_parquet(args.tasks).sort_values("task_id")
    if args.limit:
        tasks = tasks.head(args.limit)
    task_rows = tasks.to_dict("records")
    results: list[dict] = []
    started = time.time()
    for offset in range(0, len(task_rows), args.batch_size):
        batch = task_rows[offset : offset + args.batch_size]
        prompts = [render_d00(tokenizer, task, args.model_id) for task in batch]
        encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
        prompt_width = encoded.input_ids.shape[1]
        target_device = next(model.parameters()).device
        encoded = {key: value.to(target_device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        texts = tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        for task, text in zip(batch, texts, strict=True):
            results.append({
                "task_id": task["task_id"],
                "source_id": task["source_id"],
                "model_id": args.model_id,
                "condition": args.condition,
                "view_id": "D00",
                "output_text": text,
            })
        print(f"{args.condition}: {len(results)}/{len(task_rows)}", flush=True)

    write_jsonl(args.output, results)
    manifest = {
        "model_id": args.model_id,
        "base_model": args.base_model,
        "condition": args.condition,
        "view_id": "D00",
        "decoding": {"do_sample": False, "temperature": 0, "top_p": 1, "max_new_tokens": args.max_new_tokens},
        "rows": len(results),
        "seed": 20260825,
        "adapter": adapter_manifest,
        "elapsed_seconds": time.time() - started,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
