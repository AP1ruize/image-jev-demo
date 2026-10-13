# spaq-pointer-lora-v0.1 对比报告

用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集和完整测试集。
这一版在冻结视觉编码器的 4-bit Qwen3.5-2B 上同时更新语言 LoRA 和指针头，推理仍是一次前向。
对照是只训指针头的 warmup，以及生成 JSON 的 4-bit QLoRA。

## 模型与运行元数据

- 语言 LoRA + 指针头：`qwen3.5-2b-4bit-pointer-lora`，epoch 1，LoRA r=8
- 头初始化：`outputs/warmup/spaq-pointer-full-v0.1/pointer-head.pt`
- 冻结指针头：`qwen3.5-2b-4bit-pointer-full`，最佳 epoch 39
- JSON QLoRA：`qwen3.5-2b-4bit-qlora-full`，epoch 1，LoRA r=8
- 训练 / 测试张数：7807 / 1658
- 像素上限：65536
- 训练损失 起点 / 终点：1.3014 / 1.5232

## 分来源原生指标

只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。

### spaq / language LoRA + pointer

- n_total / n_valid / coverage：1658 / 1658 / 1.0000
- schema_valid：1.0000
- Spearman ρ：0.8959  CI95 [0.8858, 0.9046]
- Pearson r：0.9054
- MAE：7.0382
- 中位延迟：405.89 ms

### spaq / frozen pointer

- Spearman ρ：0.8240  CI95 [0.8082, 0.8370]
- Pearson r：0.8408
- MAE：9.2654
- 中位延迟：209.31 ms

### spaq / JSON QLoRA

- Spearman ρ：0.9021  CI95 [0.8925, 0.9102]
- Pearson r：0.9133
- MAE：6.7830
- 中位延迟：1374.03 ms

## 配对比较

Delta 是（语言 LoRA + 指针头）减对照的 Spearman ρ。正值表示这一版排序更好。

- 相对冻结指针头：Δρ 0.0719  CI95 [0.0619, 0.0830]
- 相对 JSON QLoRA：Δρ -0.0062  CI95 [-0.0096, -0.0027]
- paired valid：1658 / 1658

## 推理时间与资源

- 语言 LoRA + 指针头中位 / p95 延迟：405.89 / 532.29 ms
- 峰值 GPU allocated：2901.59 MB

## 覆盖边界与局限

- 只覆盖 SPAQ 总体感知质量。
- 视觉编码器保持冻结。语言 LoRA 与指针头一起用指针交叉熵训练一个 epoch，正式分数用最后一个 checkpoint。
- 内部实验用途不授权商业发布。
