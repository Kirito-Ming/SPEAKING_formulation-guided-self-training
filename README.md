# Formulation-guided self-training: minimal inference reproduction

This directory reproduces the paper's transfer evaluation with the two frozen
listeners, their trajectory-layer weights, and the 900-task transfer partition.
It compares deterministic responses to the unchanged standard request (`D00`)
before and after adaptation, then reports paired success, rescue, harm, and net
gain. No Human-speaker rule or routing wrapper is used at deployment.

## What is included

| Paper term | Release artifact |
|---|---|
| Qwen3-4B | `Qwen/Qwen3-4B-Instruct-2507` + `weights/qwen3_4b/` |
| Llama-3.2-3B | `meta-llama/Llama-3.2-3B-Instruct` + `weights/llama_3_2_3b/` |
| trajectory layers | rank-16 LoRA on attention and feed-forward projections, alpha 32 |
| Transfer | `data/transfer_tasks.parquet`, 300 OpenBookQA + 300 SciQ + 300 DROP |
| D00 | unchanged standard request compiled by `fgst.tasks.compile_d00` |
| strict task success | strict `Final answer:` extraction plus source verifier |
| rescue | base failure and adapted success |
| harm | base success and adapted failure |
| net gain | rescue minus harm |

Both listeners use the Pareto-filtered runs reported in the paper. Development
selection retained 2,232 of 4,464 task-level trajectories for Qwen3-4B (50%)
and 1,011 of 4,043 for Llama-3.2-3B (approximately 25%). `C6` and `C5` are the
respective run identifiers; they are not names of additional learning
objectives. Both selected listeners use response-token cross-entropy alone.

## Requirements

- Linux x86-64, Python 3.10 or 3.11
- NVIDIA driver compatible with CUDA 12.x
- one GPU with at least 16 GB memory per listener; 24 GB is recommended
- about 35 GB free disk for environment, Hugging Face cache, and outputs
- accepted access to the gated Llama repository for the Llama run

The base weights are intentionally not duplicated. The scripts download them
from Hugging Face or accept local paths through environment variables.

## Clean Linux setup

```bash
cd release/formulation-guided-self-training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
pytest -q
python scripts/verify_assets.py
```

For Llama, authenticate after accepting its upstream license:

```bash
export HF_TOKEN=your_read_token
```

For an offline machine, point each variable to an already downloaded model:

```bash
export QWEN3_4B_MODEL=/models/Qwen3-4B-Instruct-2507
export LLAMA_3_2_3B_MODEL=/models/Llama-3.2-3B-Instruct
```

## Run the transfer evaluation

Run one listener at a time. Each command produces base responses, adapted
responses, row-level transitions, and `metrics.json`.

```bash
scripts/run_transfer.sh qwen3_4b outputs/qwen3_4b
scripts/run_transfer.sh llama_3_2_3b outputs/llama_3_2_3b
```

Generation is greedy (`temperature=0`, `top_p=1`) with one D00 response per
task. `--batch-size 1` is the conservative default. To verify installation
quickly, add `--limit 3`; limited runs set `paper_comparable=false` and cannot
be compared with paper results.

Expected full-partition results from the frozen paper run:

| Listener | Evaluations | Base success | Adapted success | Rescue | Harm | Net gain |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-4B | 900 | 43 | 741 | 700 | 2 | +698 |
| Llama-3.2-3B | 900 | 0 | 533 | 533 | 0 | +533 |
| Pooled | 1,800 | 43 | 1,274 | 1,233 | 2 | +1,231 |

Exact generation can depend on GPU kernels and software revisions. The pinned
environment, deterministic decoding, asset hashes, and row-level outputs make
such differences auditable. A run is structurally valid only when it evaluates
all 900 tasks and satisfies:

```text
adapted_success - base_success = rescue - harm
```

## Inspect success and failure cases

`scored/paired_predictions.parquet` contains `transition` with four values:
`rescue`, `harm`, `success_preserved`, and `failure_persisted`. For example:

```bash
python - <<'PY'
import pandas as pd
p = pd.read_parquet('outputs/qwen3_4b/scored/paired_predictions.parquet')
print(p.groupby(['source_id', 'transition']).size().unstack(fill_value=0))
print(p[p.transition.isin(['rescue', 'harm'])][
    ['task_id', 'source_id', 'transition', 'base_output', 'adapted_output']
].head(10).to_string(index=False))
PY
```

`conditional_rescue` divides rescue by base failures. `conditional_harm`
divides harm by base successes and is `null` when there are no base successes,
as in the Llama transfer result. Unconditional rescue and harm rates divide by
all evaluations. Parseability is recorded separately from correctness.

## Reproducibility boundary

This is an inference reproduction, not the full discovery/training pipeline.
It includes the selected weights and test set needed to reproduce Figure 6's
transfer comparison. It does not rerun Human-speaker-rule discovery, candidate
trajectory generation, Pareto selection, or the 60 optimization steps.

The code is Apache-2.0. Base models, adapters, and dataset-derived rows have
additional upstream terms. In particular, SciQ is non-commercial and the
OpenBookQA dataset card does not currently declare a license. Read
`THIRD_PARTY_NOTICES.md` before publishing this directory unchanged.
