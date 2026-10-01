# week0-iqa-smoke technical IQA report

## Run

- model: `Qwen3-VL-2B-Instruct` (`qwen3vl`)
- mode: `generate`
- checkpoint: `Qwen/Qwen3-VL-2B-Instruct`
- items: 2 expected / 2 responses / 2 in-range
- errors: {}

## Per-source metrics

SPAQ is 0–100 MOS; KonIQ-10k is 1–5 MOS. Do not average them.

### koniq10k

- n=2 (groups=2, valid=2)
- coverage: 1.0000
- schema_valid: 1.0000
- Spearman ρ: 1.0000  CI95 n/a
- Pearson r: 1.0000
- MAE: 0.2358
- median latency: 1466.8 ms

## Note

Do not mix SPAQ and KonIQ MOS. Compare paired runs on identical items; report coverage and schema_valid beside Spearman/Pearson/MAE.
