# week0-iqa-smoke technical IQA report

## Run

- model: `Qwen3-VL-2B-Instruct` (`qwen3vl`)
- mode: `logits`
- checkpoint: `Qwen/Qwen3-VL-2B-Instruct`
- items: 8 expected / 2 responses / 2 in-range
- errors: {"missing_response": 6}

## Per-source metrics

SPAQ is 0–100 MOS; KonIQ-10k is 1–5 MOS. Do not average them.

### koniq10k

- n=4 (groups=4, valid=2)
- coverage: 0.5000
- schema_valid: 0.5000
- Spearman ρ: 1.0000  CI95 n/a
- Pearson r: 1.0000
- MAE: 0.4737
- median latency: 1143.8 ms

### spaq

- n=4 (groups=4, valid=0)
- coverage: 0.0000
- schema_valid: 0.0000
- Spearman ρ: n/a  CI95 n/a
- Pearson r: n/a
- MAE: n/a
- median latency: n/a ms

## Note

Do not mix SPAQ and KonIQ MOS. Compare paired runs on identical items; report coverage and schema_valid beside Spearman/Pearson/MAE.
