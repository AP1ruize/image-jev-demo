# spaq-pointer-warmup-heldout-v0.1 评测报告

用途：非商业内部模型实验。这是 SPAQ 手机照片感知质量代理，不是完整 AlphaHalf 产品 benchmark。
评分切分是 Week 0 的 validation。冻结 test 没有进入训练或本报告。

## 模型与运行元数据

- **当前模型**：Qwen3.5-2B 4-bit pointer warmup（`qwen3.5-2b-4bit-pointer-heldout`）
- 架构：`qwen35`；格式：`safetensors`；量化：`nf4`
- Vision：`True`
- 阶段：Valen warmup，骨干冻结，只训练指针决策头
- 基座：`Qwen/Qwen3.5-2B`
- 图像像素上限：65536
- 训练 / 验证张数：256 / 120
- 头学习率：0.0002；投影维：256；最多 epoch：40
- 按验证损失选出的 epoch：39（从 0 计）
- 训练损失 起点 / 最佳 epoch：2.3446 / 1.5036
- 验证损失 起点 / 最佳 epoch：2.3412 / 1.5464
- 切分种子：`image-jev-week0-v0.1`
- 对话策略：stateless_per_item（每张图一次前向，不携带上一张图）
- 指针响应：`outputs\runs\spaq-pointer-warmup-heldout-v0.1-pointer.jsonl`
- logits 响应：`outputs\runs\spaq-pointer-warmup-heldout-v0.1-logits.jsonl`
- 指针评分：`outputs\runs\spaq-pointer-warmup-heldout-v0.1-pointer.score.json`
- logits 评分：`outputs\runs\spaq-pointer-warmup-heldout-v0.1-logits.score.json`
- 配对比较：`outputs\runs\spaq-pointer-warmup-heldout-v0.1.compare.json`

## 总体完成情况

- 验证期望题数：120
- 指针响应 / 有效：120 / 120
- logits 响应 / 有效：120 / 120
- 指针错误切片：{}
- logits 错误切片：{}

## 分来源原生指标

本报告只有 SPAQ（0–100 MOS）。不要和 KonIQ 或其他量表混成一个总分。
主指标 95% 区间为按 `group_id` 的 cluster bootstrap。

### spaq / pointer

- 任务类型：`continuous`
- n_total / n_valid / coverage：120 / 120 / 1.0000
- n_groups：120
- schema_valid：1.0000
- 中位延迟：196.23 ms
- 主指标 95% 组级 CI：[0.7851, 0.8883]
- Spearman ρ：0.8508
- Pearson r：0.8788
- MAE：8.4017

### spaq / logits

- 任务类型：`continuous`
- n_total / n_valid / coverage：120 / 120 / 1.0000
- n_groups：120
- schema_valid：1.0000
- 中位延迟：243.81 ms
- 主指标 95% 组级 CI：[0.3895, 0.6775]
- Spearman ρ：0.5485
- Pearson r：0.5258
- MAE：28.5023

## 配对比较

同一批验证图。Delta 是 pointer − logits 的 Spearman ρ。延迟比是 pointer / logits 的中位数。

- paired valid：120 / 120
- Spearman logits / pointer：0.5485 / 0.8508
- Δρ：0.3023  CI95 [0.1709, 0.4569]
- schema_valid logits / pointer：1.0000 / 1.0000
- 中位延迟 logits / pointer：243.81 / 196.23 ms
- 延迟比（pointer/logits）：0.8048

## 训练诊断

训练集上的 Spearman 只说明头是否拟合了训练特征，不能当作评测成绩。

- 训练 Spearman ρ：0.8506
- 训练 MAE：8.6795

## 覆盖边界与局限

- 只覆盖 SPAQ 总体感知质量 MOS，不覆盖 AlphaHalf 主题、主体、人物状态、Moment、情绪、故事或证据化点评。
- 骨干保持 4-bit 冻结。这一步没有做 Valen 的 joint / vision_top / RLCD。
- 验证集按 Week 0 种子切出。不要把这些数字和尚未运行的 Week 0 test leaderboard 混在一起。
- 内部实验用途不授权商业发布；不要分发原图、标注或模型。
