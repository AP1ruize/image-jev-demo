# SPAQ results

[中文](zh/RESULTS.md)

Personal research experiment. One source, one scale: SPAQ smartphone perceptual quality, MOS 0–100. Do not average this with KonIQ or any other scale.

## Split and model

Week 0 split, seed `image-jev-week0-v0.1`: 7,807 train, 1,660 validation, 1,658 test. Test images are not used for training or for choosing a checkpoint.

Base model: `Qwen/Qwen3.5-2B`, NF4 4-bit, `max_pixels=65536`, vision encoder frozen. The pointer uses bins 0, 10, …, 100. The reported score is the expected bin after softmax. Machine: NVIDIA RTX A2000 Laptop GPU, 4 GB.

| Run | What is trained | Checkpoint used on test |
| --- | --- | --- |
| Pointer only | Pointer head, 40 epochs, best validation loss (epoch 39) | that epoch |
| Language LoRA + pointer | LoRA r=8 on the language model, plus the pointer, 1 epoch, 976 steps | last step |
| Validation-selected LoRA + pointer | same training run, scored every 100 steps on 256 validation images | step 900, then confirmed on all 1,660 validation images |
| JSON QLoRA | LoRA r=8, 1 epoch, generates `{"value": number}` | last step |

## Test metrics

| Readout | Spearman ρ (95% CI) | Pearson r | MAE | Median latency | Peak GPU allocated |
| --- | --- | ---: | ---: | ---: | ---: |
| Pointer only | 0.824 [0.808, 0.837] | 0.841 | 9.27 | 209 ms | 1,973 MB |
| Language LoRA + pointer, last step | 0.896 [0.886, 0.905] | 0.905 | 7.04 | 406 ms | 2,902 MB |
| Language LoRA + pointer, step 900 | 0.895 [0.885, 0.904] | 0.905 | 7.27 | 410 ms | 2,008 MB |
| 4-bit QLoRA, generated JSON | 0.902 [0.892, 0.910] | 0.913 | 6.78 | 1,374 ms | 1,909 MB |

Coverage and schema validity are 1.0 on all 1,658 test images. Intervals are cluster bootstrap over `group_id`, 2,000 draws, seed 42. On this split each image is its own group.

Paired gaps on Spearman (right minus left is not the convention here; the number is the pointer readout minus the named baseline):

- Last-step LoRA + pointer minus JSON QLoRA: **−0.006** [−0.010, −0.003]
- Step 900 minus last step: **−0.001** [−0.003, +0.001]
- Last-step LoRA + pointer minus pointer-only: **+0.072** [+0.062, +0.083]

Median latency of the last-step pointer is 406 / 1,374 ≈ 0.30 of JSON generation, about 3.4× faster. Step 900’s lower peak memory is the eval process after training, not a smaller model.

## How to read this

The pointer head alone, on a frozen backbone, already beats an untrained logit readout by a wide margin (validation ρ 0.851 versus 0.549 on an earlier 120-image check). Almost all of the remaining gain to JSON QLoRA comes from the language LoRA, not from decoding text.

Training loss of the LoRA-plus-pointer run does not fall smoothly (about 1.30 at step 1, about 1.47 at step 976). The validation monitor is lowest at step 900 (cross-entropy 1.361 on 256 images; 1.344 on all 1,660). That checkpoint does not improve the test ranking over the last step. The interval on that difference includes zero.

Unfreezing the vision merger was probed with one image and one backward. Peak allocated memory was 4,046 MB on a 4,096 MB card, so that run was not trained.

## Scope

Overall SPAQ MOS only. No KonIQ test number is claimed for the trained pointer. Code is Apache-2.0. Dataset copyrights stay with their authors.
