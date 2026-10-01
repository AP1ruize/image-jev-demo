# ImageJev

Typed, schema-valid **technical image quality** decisions from small vision-language models.

Week 0 does not train anything. It compares three frozen-model inference modes on the same IQA items:

| Mode | What the network does | JSON |
| --- | --- | --- |
| `generate` | Autoregressive tokens for `{"value": …}` | Parsed after the fact; can fail |
| `grammar` | Same generation, but only number/`}` tokens are allowed after a `{"value":` prefix | Valid by construction |
| `logits` | One prefill, softmax over score bins / option letters | Assembled in code; always valid |

The hypothesis: on closed numeric scales, reading candidate logits should match JSON-generation rank correlation, raise schema coverage to 100%, and cut latency by skipping the decode loop. If scores are unchanged, a wrapper API alone is not an accuracy upgrade.

## What this is / is not

- **Is:** an open research harness for photographic **technical quality** (sharpness, noise, exposure, smartphone MOS — not beauty, not composition taxonomy).
- **Is not:** a text generator, a photo editor, or a reproduction of any closed commercial decision model. Independent interface experiment.

Base models for Week 0: `Qwen/Qwen3-VL-2B-Instruct` and `Qwen/Qwen3.5-2B`.

## Benchmark

Frozen sources, native metrics, no mixed-scale overall score:

| Source | Construct | Scale | Primary metric |
| --- | --- | --- | --- |
| [SPAQ](https://github.com/h4nwei/SPAQ) | Smartphone perceptual quality MOS | 0–100 | Spearman ρ, Pearson r, MAE, coverage, latency |
| [KonIQ-10k](https://database.mmsp-kn.de/) | In-the-wild perceptual quality MOS | 1–5 | same |

Test images never become training examples. Group-level bootstrap 95% CIs are reported per source.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GPU eval extra (install a CUDA torch wheel first if needed):

```powershell
pip install -r requirements-eval.txt
```

Copy `configs/local.example.json` to `configs/local.json` and point it at your SPAQ / KonIQ files.

```powershell
python scripts/build_source_catalog.py
python scripts/build_week0_release.py
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode generate --dry-run
```

Real run (writes JSONL + scores + markdown):

```powershell
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode generate
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode grammar
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode logits
python scripts/compare_week0.py `
  --left outputs/runs/<generate>.jsonl `
  --right outputs/runs/<logits>.jsonl `
  --output outputs/reports/week0-qwen3vl2b-generate-vs-logits.json
```

`--max-items N` is for smokes. `--resume` appends to an existing JSONL.

## Repository

```text
configs/     versioned Week 0 + model profiles
docs/        datasets, eval protocol, Week 0 notes
registry/    source facts and license status
schemas/     JSON Schema for items and predictions
scripts/     catalog, freeze, eval, score, compare
src/image_jev/   stdlib scoring + optional HF inference
tests/       no GPU required
```

Code is Apache-2.0. Dataset files keep their own terms; see `docs/DATASETS.md`.
