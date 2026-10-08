# Transfer partition

`transfer_tasks.parquet` is the paper's frozen 900-task transfer partition:

| Source | Tasks | Training/model-selection use |
|---|---:|---|
| OpenBookQA | 300 | None |
| SciQ | 300 | None |
| DROP | 300 | None |

The file stores the standard request fields (`visible_input_json`) and a
source-specific verifier payload. Evaluation sends only the standard request
to the listener. Gold verifier payloads are read only after generation.

The split was sampled with seed `20260825`, after excluding 50 compatibility
pilot rows per source and historical overlaps. Its provenance SHA-256 is
`efd61b564dd80c330367f0db97383845494d92341ccf6702a6cbe67049ec70a1`.

See `THIRD_PARTY_NOTICES.md` before redistributing this file.
