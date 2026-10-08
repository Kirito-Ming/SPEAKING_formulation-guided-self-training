# Third-party notices

The repository's Apache-2.0 license does not replace upstream terms.

| Asset | Upstream | Terms / status |
|---|---|---|
| Qwen3-4B-Instruct-2507 | `Qwen/Qwen3-4B-Instruct-2507` | Apache-2.0 |
| Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Meta Llama 3.2 Community License; gated access may require acceptance |
| OpenBookQA rows | `allenai/openbookqa`, `main` | Hugging Face dataset card currently reports the license as unknown; verify redistribution rights before a public release |
| SciQ rows | `allenai/sciq` | CC BY-NC 3.0 |
| DROP rows | `ucinlp/drop` | CC BY-SA 4.0 |

`data/transfer_tasks.parquet` is a frozen research evaluation subset derived
from those three datasets. It must not be relicensed as Apache-2.0. The LoRA
trajectory layers are derived for use with their named base models; users and
distributors must comply with the applicable base-model terms.

The links and license labels above were checked on 2026-09-23. Recheck them
before redistribution.
