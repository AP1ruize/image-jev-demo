# 下一步实验

[English](../PLAN.md)

下一步要跑的范围已经定了，实验还没开始。已经公开的 SPAQ 总体 MOS 数字仍在 [RESULTS.md](RESULTS.md)。这次 SPAQ 是六项一起学，和那个只学总分的模型不是同一次训练。

## 约束

- 基座：`Qwen/Qwen3.5-2B` 和 `Qwen/Qwen3-VL-2B-Instruct`。分开报，不平均。
- 权重：约 2B 参数，只用 NF4 4-bit。不用 bf16，也不用其他量化。
- 训练像素上限：`max_pixels=65536`。不要用 Week 0 推理用的 262144 来训练。
- 在 4 GB 显卡上，视觉编码器保持冻结。
- 同一套训练流程，两个数据集，切分分开。不要把 SPAQ 和 KonIQ 的图放进同一次优化。
- 一张图的所有标注留在同一个切分里。测试图不参与训练，也不参与选 checkpoint。
- 按验证损失选 checkpoint 是训练步骤，不是单独一种模型。
- 指标留在各自的尺度上。不要把 SPAQ 的 0–100、KonIQ 的 1–5 和 0–1 的缺陷比例加在一起。每个目标分开报 Spearman ρ、MAE、覆盖率和中位延迟。

手头的机器是 4 GB 的 RTX A2000 Laptop。外接的 NVIDIA P100（16 GB）只用于视觉 merger 那一行，而且要等那张卡在手边。

## 数据集

### SPAQ，一次联合训练

六项都是人工分，尺度都是 0–100：

| 目标 | 含义 |
| --- | --- |
| MOS | 总体感知质量 |
| Brightness | 亮度 |
| Colorfulness | 色彩 |
| Contrast | 对比度 |
| Noisiness | 噪声 |
| Sharpness | 锐度 |

切分种子仍是 `image-jev-week0-v0.1`：训练 7,807，验证 1,660，测试 1,658。

### KonIQ，一次联合训练

KonIQ-10k 和 KonIQ++ 是同一批照片，按文件名对齐。KonIQ++ 补的是缺陷比例，不是另一套图。

| 目标 | 来源列 | 尺度 | 对应的 SPAQ 轴 |
| --- | --- | --- | --- |
| 总体 MOS | KonIQ-10k 的 `MOS` | 大约 1–5 | MOS |
| contrast | KonIQ++ 的 `contrast` | 大约 0–1，认为对比度有问题的人数比例 | Contrast |
| colors | KonIQ++ 的 `colors` | 大约 0–1，认为色彩有问题的人数比例 | Colorfulness |
| blur | KonIQ++ 的 `blur` | 大约 0–1，认为发糊的人数比例 | Sharpness，方向相反 |

`blur` 按这个比例来训，不改写成 0–100 的锐度分。发糊的人越多，锐度越低。报告里说明这一点，两套数字不加在一起。

这些列不用：KonIQ++ 的 `artifacts`、`other`、`qmos`、`degraded.amount`；KonIQ-10k 的 `SD`、`MOS_zscore` 和投票人数 `c1`–`c5`；`koniq10k_indicators.csv` 里由程序算出的列。SPAQ 的亮度和噪声在这里没有人工对应项，所以只在 SPAQ 上训。

KonIQ 的切分用和 SPAQ 相同的种子函数。张数在切分建出来时定死。

## 变体

每一行都跑四次：SPAQ 六项一次，KonIQ 四项一次，两个基座各跑这两次。

| 变体 | 更新什么 | 在哪跑 |
| --- | --- | --- |
| 冻结 `generate` | 不训练，模型把分数写出来 | 4 GB |
| 冻结 `grammar` | 不训练，只允许能写成 JSON 数字的 token | 4 GB |
| 冻结 `logits` | 不训练，一次前向在分数档上读概率 | 4 GB |
| 只训指针头 | 骨干冻结，只训指针头 | 4 GB |
| 语言 LoRA + 指针头 | 语言 LoRA r=8 加指针头，视觉层冻结 | 4 GB |
| 4-bit QLoRA，生成 JSON | 语言 LoRA r=8，模型把 JSON 写出来 | 4 GB |
| 语言 LoRA + 指针头，再解冻视觉 merger | 上一行之外再更新视觉 merger | P100 16 GB，不是这台 4 GB 笔记本 |

已经完成的，只有 Qwen3.5-2B 的 SPAQ 总体 MOS：指针头、语言 LoRA + 指针头、生成 JSON 的 QLoRA。那三次都是单项训练，不能填这次的 SPAQ 六项，也不能填任何 KonIQ 格子。Qwen3-VL-2B 没有训练记录；4-bit 推理只做过两张图的冒烟。冻结的 `grammar` 没有运行记录。

如果所有目标一次反向在 4 GB 里放不下，就把该变体拆成每个目标各训一次。单项训练用的是已经放得下的配方。

Qwen3.5-2B 的视觉 merger 探针峰值占用 4,046 MB，所以这一行在 4 GB 上不做。Qwen3-VL-2B 还没有同样的探针。P100 在手边时，先对 Qwen3.5-2B 的 SPAQ 总分做这一行。整个视觉编码器全部解冻还没测过，训练前先探针。Qwen3-VL-2B 需要自己的探针。
