# ImageJev

[中文](README.zh-CN.md)

Typed technical image-quality scores from a small vision-language model, read out in one forward pass.

The useful result is a **language LoRA plus a pointer head** on frozen 4-bit `Qwen/Qwen3.5-2B`. On the SPAQ smartphone MOS test split it stays within 0.007 Spearman of the same model trained to generate a JSON score, at about one third of the latency. Details and confidence intervals are in [docs/RESULTS.md](docs/RESULTS.md).

## SPAQ test, 1,658 photos

Same Week 0 split, same 65,536 pixel cap, vision encoder frozen. Spearman ρ is the ranking metric. Latency is the median model call on an RTX A2000 Laptop (4 GB).

| Readout | Spearman ρ | MAE | Median latency |
| --- | ---: | ---: | ---: |
| Pointer head only | 0.824 | 9.27 | 209 ms |
| Language LoRA + pointer, last step | 0.896 | 7.04 | 406 ms |
| Language LoRA + pointer, validation-selected step 900 | 0.895 | 7.27 | 410 ms |
| 4-bit QLoRA, generated JSON | 0.902 | 6.78 | 1,374 ms |

The last-step pointer is 0.006 Spearman below JSON generation. That gap is small, and its 95% interval does not include zero. Picking the checkpoint with the best validation loss (step 900) did not beat the last step. Coverage is 1.0 for every row: the pointer always emits a legal distribution.

## What this is

ImageJev scores **technical / perceptual quality** (sharpness, noise, exposure, overall MOS). It does not score beauty or composition, and it does not write a critique.

The network does not decode an answer. A frozen vision encoder and a 4-bit language model, with a small language LoRA, produce one hidden state. A pointer head scores that state against fixed bins (0, 10, …, 100 on SPAQ). The reported score is the expected bin. JSON, when present, is assembled in Python.

Week 0, before any training, compared three frozen readouts on the same items: free JSON generation, a JSON-number grammar, and a logit readout. The trained pointer is the follow-up. See [docs/WEEK0.md](docs/WEEK0.md) and [docs/EVAL.md](docs/EVAL.md).

The next experiment trains six SPAQ scores together, and four KonIQ scores together, on both Qwen3.5-2B and Qwen3-VL-2B. It is specified in [docs/PLAN.md](docs/PLAN.md) and has not been run.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

GPU training and evaluation need a CUDA torch wheel, then:

```powershell
pip install -r requirements-eval.txt
```

Point `configs/local.json` at your own SPAQ and KonIQ files (`configs/local.example.json` is the template). Dataset files are not in this repository. Their copyrights stay with the authors; see [docs/DATASETS.md](docs/DATASETS.md).

```powershell
python scripts/build_source_catalog.py
python scripts/build_week0_release.py
python scripts/eval_week0.py --config configs/models.qwen3vl2b.json --mode generate --dry-run
```

SPAQ pointer training entry points, after a frozen release exists:

```powershell
python scripts/train_warmup.py --config configs/warmup.spaq.json
python scripts/run_spaq_heldout.py --config configs/warmup.spaq.heldout.json
python scripts/train_spaq_pointer_lora.py --config configs/pointer-lora.spaq.qwen35-2b.json
```

## Layout

```text
configs/          model profiles and the Week 0 split
docs/             protocol, datasets, SPAQ result, and the next plan
registry/         source links and upstream license notes
schemas/          JSON Schema for items and predictions
scripts/          catalog, freeze, train, eval, score
src/image_jev/    scoring, frozen inference, pointer head
tests/            no GPU required
```

## License

Code in this repository is Apache-2.0. You may use the code commercially.

Datasets keep their own copyrights and terms. This repository does not grant rights to those images or labels. The training and evaluation reported here are a personal research experiment on those datasets.
