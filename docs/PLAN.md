# Next experiment

[中文](zh/PLAN.md)

Agreed scope for the next run. It is not started. Published SPAQ overall-MOS numbers stay in [RESULTS.md](RESULTS.md). This plan does not replace them; the new SPAQ run learns six scores together, so it is a different training run from that single-score model.

## Constraints

- Bases: `Qwen/Qwen3.5-2B` and `Qwen/Qwen3-VL-2B-Instruct`. Report them separately. Do not average them.
- Weights: about 2B parameters, NF4 4-bit only. No bf16 and no other quantization.
- Training pixels: `max_pixels=65536`. Do not train at the Week 0 inference cap of 262144.
- Vision encoder stays frozen on the 4 GB GPU.
- One training procedure, two datasets, separate splits. Do not mix SPAQ and KonIQ images in one optimizer run.
- Every label of one image stays in the same split. Test images are not used for training or for picking a checkpoint.
- Picking the checkpoint with the best validation loss is part of training, not a separate variant.
- Metrics stay on each source's own scale. Do not add SPAQ 0–100 scores to KonIQ 1–5 scores or to 0–1 defect rates. Spearman ρ, MAE, coverage, and median latency are reported per target.

The machine on hand is a 4 GB RTX A2000 Laptop. An external NVIDIA P100 (16 GB) is for the vision-merger row only, and only when that GPU is available.

## Datasets

### SPAQ, one joint run

Six human scores, each 0–100:

| Target | Meaning |
| --- | --- |
| MOS | overall perceptual quality |
| Brightness | brightness |
| Colorfulness | colorfulness |
| Contrast | contrast |
| Noisiness | noise |
| Sharpness | sharpness |

Split seed remains `image-jev-week0-v0.1`: 7,807 train, 1,660 validation, 1,658 test.

### KonIQ, one joint run

KonIQ-10k and KonIQ++ are the same photographs, joined on filename. KonIQ++ adds defect rates. It is not a second image collection.

| Target | Source column | Scale | SPAQ axis |
| --- | --- | --- | --- |
| overall MOS | KonIQ-10k `MOS` | about 1–5 | MOS |
| contrast | KonIQ++ `contrast` | about 0–1, share of raters who marked a contrast problem | Contrast |
| colors | KonIQ++ `colors` | about 0–1, share of raters who marked a color problem | Colorfulness |
| blur | KonIQ++ `blur` | about 0–1, share of raters who marked blur | Sharpness, opposite direction |

`blur` is trained as that rate. It is not rewritten into a 0–100 sharpness score. A higher blur rate means lower sharpness; the reports say so and do not pool the two numbers.

Not used from these files: KonIQ++ `artifacts`, `other`, `qmos`, and `degraded.amount`; KonIQ-10k `SD`, `MOS_zscore`, and vote counts `c1`–`c5`; the computed columns in `koniq10k_indicators.csv`. SPAQ brightness and noisiness have no human counterpart here, so they are SPAQ-only.

The KonIQ split uses the same seed function as SPAQ. Counts are fixed when the split is built.

## Variants

Each row runs four times: SPAQ six-way and KonIQ four-way, on each base.

| Variant | What changes | Where it runs |
| --- | --- | --- |
| Frozen `generate` | no training; the model writes the score | 4 GB |
| Frozen `grammar` | no training; only tokens that complete a JSON number | 4 GB |
| Frozen `logits` | no training; one forward pass over score bins | 4 GB |
| Pointer head only | frozen backbone, trained pointer | 4 GB |
| Language LoRA + pointer | language LoRA r=8 plus the pointer; vision frozen | 4 GB |
| 4-bit QLoRA, JSON generation | language LoRA r=8; the model writes JSON | 4 GB |
| Language LoRA + pointer, vision merger unfrozen | the row above, plus the vision merger | P100 16 GB, not the 4 GB laptop |

Already finished, and only for Qwen3.5-2B overall SPAQ MOS: pointer head, language LoRA + pointer, and JSON QLoRA. Those runs are single-target. They do not fill the new six-way SPAQ cells or any KonIQ cell. Qwen3-VL-2B has no training run; its 4-bit inference was only a two-image smoke. Frozen `grammar` has no run.

If a joint backward of all targets does not fit in 4 GB, split that variant into one run per target. Each single-target run uses the recipe that already fit.

The vision-merger probe on Qwen3.5-2B peaked at 4,046 MB allocated on the 4 GB card, so that row waits. The same probe has not been run for Qwen3-VL-2B. On the P100, start with SPAQ overall MOS for Qwen3.5-2B. Unfreezing the entire vision encoder was not measured; probe it before training it. Qwen3-VL-2B needs its own probe.
