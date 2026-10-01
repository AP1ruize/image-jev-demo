# Week 0 — frozen inference, no new weights

Goal: measure whether **typed readout** (candidate logits, optional JSON grammar) is worth it for technical IQA on 2B VLMs **before** any fine-tune.

## Modes

1. **generate** — greedy `{"value": number}` with `max_new_tokens=32`. Same task the IQA literature usually asks a VLM to do. Parse failures lower coverage.
2. **grammar** — assistant prefix `{"value":`, then only tokens that continue a JSON number and `}`. Schema-valid, still autoregressive.
3. **logits** — lettered (or single-token) bins, one forward pass, softmax, expected MOS. JSON is built in Python. Coverage is 100% by construction.

Shared across modes: same frozen items, same images, greedy decoding, thinking disabled, processor `min_pixels=3136` / `max_pixels=262144`.

## What would count as “worth it”

- Spearman ρ on SPAQ and KonIQ close to `generate` (no large quality drop).
- `schema_valid` = 1.0 for `grammar` and `logits`.
- Median latency: `logits` << `generate`. Grammar should sit in between (fewer illegal tokens, still a decode loop).

If ρ is essentially unchanged, the API shape is a serving win, not an intelligence win. Training the distribution (later weeks) is a separate question.

## Parallel fields (optional)

SPAQ also has five attribute MOS columns. `--parallel-attributes` on `logits` prefills the image once, then scores MOS + attributes as isolated suffixes. That path is a latency demo; official Week 0 tables still use **overall MOS only** so they stay paired with `generate`.

## Not in Week 0

- LoRA / RL / calibration training
- Aesthetic or composition labels
- Pixel-space retouching
