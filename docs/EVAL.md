# Evaluation protocol

[中文](zh/EVAL.md)

## Units

Each frozen item is one image and one overall MOS label. Models are called stateless: current image + current prompt only.

Predictions are a single finite number inside the source scale, or a recorded parse error.

## Metrics (per source)

| Name | Use |
| --- | --- |
| coverage | fraction of items with an in-range numeric prediction |
| schema_valid | fraction whose payload was valid JSON with a `value` key (logits/grammar: always 1) |
| Spearman ρ | primary ranking metric; tie-aware midranks |
| Pearson r | linear agreement |
| MAE | absolute error on the source scale |
| median latency_ms | wall time of the model call, including image preprocess |
| grouped bootstrap 95% CI | resample `group_id` clusters, 2000 draws, seed 42 |

Never average SPAQ (0–100) with KonIQ (1–5).

## Pairing

`compare_week0.py` requires the same `item_id` set on both runs. Delta is right − left on Spearman, with a grouped bootstrap CI. Latency reports the median ratio.

## Binding

A formal run writes `<stem>.run.json` next to the JSONL (model id, mode, release hash, item count). `score_week0.py` refuses to score if that metadata is missing unless `--allow-untracked` (diagnostics only).
