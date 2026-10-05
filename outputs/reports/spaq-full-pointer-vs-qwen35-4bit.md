# spaq-pointer-full-v0.1 与 spaq-qwen35-2b-4bit-full-v0.1 对比报告

用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集和完整测试集。
指针头是 ImageJev 的决策读出。对照模型是原版 Qwen3.5-2B 的 4-bit QLoRA，生成 JSON 分数。
validation 只用于给指针头选 epoch。正式分数、延迟和资源都来自 test。

## 模型与运行元数据

- 指针头：`qwen3.5-2b-4bit-pointer-full`，训练 7807 张，最佳 epoch 39
- 原版 QLoRA：`qwen3.5-2b-4bit-qlora-full`，基座 `Qwen/Qwen3.5-2B`，量化 `nf4`，epoch 1
- 测试张数：1658
- 像素上限：65536
- GPU：NVIDIA RTX A2000 Laptop GPU

## 分来源原生指标

只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。

### spaq / image-jev pointer

- n_total / n_valid / coverage：1658 / 1658 / 1.0000
- schema_valid：1.0000
- Spearman ρ：0.8240  CI95 [0.8082, 0.8370]
- Pearson r：0.8408
- MAE：9.2654
- 中位延迟：209.31 ms

### spaq / qwen3.5-2b 4-bit QLoRA

- n_total / n_valid / coverage：1658 / 1658 / 1.0000
- schema_valid：1.0000
- Spearman ρ：0.9021  CI95 [0.8925, 0.9102]
- Pearson r：0.9133
- MAE：6.7830
- 中位延迟：1374.03 ms

## 配对比较

Delta 是 pointer − QLoRA 的 Spearman ρ。延迟比和显存比都是 pointer / QLoRA。

- paired valid：1658 / 1658
- Spearman QLoRA / pointer：0.9021 / 0.8240
- Δρ：-0.0781  CI95 [-0.0898, -0.0672]

## 推理时间与资源

- 指针头中位 / p95 延迟：209.31 / 249.30 ms
- QLoRA 中位 / p95 延迟：1374.03 / 1765.89 ms
- 延迟比（pointer/QLoRA）：0.1523
- 指针头峰值 GPU allocated / reserved：1973.14 / 2040.00 MB
- QLoRA 峰值 GPU allocated / reserved：1908.85 / 1942.00 MB
- 显存比（pointer/QLoRA，allocated）：1.0337
- 指针头进程 RSS：5316.28 MB
- QLoRA 进程 RSS：520.53 MB

## 覆盖边界与局限

- 只覆盖 SPAQ 总体感知质量，不覆盖 AlphaHalf 的主题、主体、情绪或叙事。
- 指针头不更新骨干。QLoRA 用 rank 8 适配器更新语言模型，视觉编码器保持冻结。
- 两者都是 4-bit NF4，像素上限相同。延迟是模型前向或生成时间，不含读图。
- 内部实验用途不授权商业发布。
