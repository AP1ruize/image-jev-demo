# spaq-pointer-full-v0.1 评测报告

用途：非商业内部模型实验。这是 SPAQ 手机照片感知质量代理，不是完整 AlphaHalf 产品 benchmark。
评分切分是 Week 0 的 test。训练不用这个切分。

## 模型与运行元数据

- **当前模型**：Qwen3.5-2B 4-bit pointer warmup, full SPAQ train/test（`qwen3.5-2b-4bit-pointer-full`）
- 架构：`qwen35`；格式：`safetensors`；量化：`nf4`
- Vision：`True`
- 阶段：Valen warmup，骨干冻结，只训练指针决策头
- 基座：`Qwen/Qwen3.5-2B`
- 图像像素上限：65536
- 训练 / 验证 / 测试张数：7807 / 1660 / 1658
- 正式评分切分：`test`
- 头学习率：0.0002；投影维：256；最多 epoch：40
- 按验证损失选出的 epoch：39（从 0 计）
- 训练损失 起点 / 最佳 epoch：2.3454 / 1.6251
- 验证损失 起点 / 最佳 epoch：2.3411 / 1.6241
- 切分种子：`image-jev-week0-v0.1`
- 对话策略：stateless_per_item（每张图一次前向，不携带上一张图）
- 指针响应：`outputs\runs\spaq-pointer-full-v0.1-pointer.jsonl`
- logits 响应：`outputs\runs\spaq-pointer-full-v0.1-logits.jsonl`
- 指针评分：`outputs\runs\spaq-pointer-full-v0.1-pointer.score.json`
- logits 评分：`outputs\runs\spaq-pointer-full-v0.1-logits.score.json`
- 配对比较：`outputs\runs\spaq-pointer-full-v0.1.compare.json`

## 总体完成情况

- 验证期望题数：1658
- 指针响应 / 有效：1658 / 1658
- logits 响应 / 有效：0 / None
- 指针错误切片：{}
- logits 错误切片：{}

## 分来源原生指标

本报告只有 SPAQ（0–100 MOS）。不要和 KonIQ 或其他量表混成一个总分。
主指标 95% 区间为按 `group_id` 的 cluster bootstrap。

### spaq / pointer

- 任务类型：`continuous`
- n_total / n_valid / coverage：1658 / 1658 / 1.0000
- n_groups：1658
- schema_valid：1.0000
- 中位延迟：209.31 ms
- 主指标 95% 组级 CI：[0.8082, 0.8370]
- Spearman ρ：0.8240
- Pearson r：0.8408
- MAE：9.2654

### spaq / logits

- 任务类型：`continuous`
- n_total / n_valid / coverage：None / None / n/a
- n_groups：None
- schema_valid：n/a
- 中位延迟：n/a ms
- 主指标 95% 组级 CI：n/a
- Spearman ρ：n/a
- Pearson r：n/a
- MAE：n/a

## 配对比较

同一批验证图。Delta 是 pointer − logits 的 Spearman ρ。延迟比是 pointer / logits 的中位数。

- paired valid：None / None
- Spearman logits / pointer：n/a / n/a
- Δρ：n/a  CI95 n/a
- schema_valid logits / pointer：n/a / n/a
- 中位延迟 logits / pointer：n/a / n/a ms
- 延迟比（pointer/logits）：n/a

## 训练诊断

训练集上的 Spearman 只说明头是否拟合了训练特征，不能当作评测成绩。

- 训练 Spearman ρ：0.8269
- 训练 MAE：9.2783

## 推理时间与资源

- 加载后常驻：GPU allocated 1828.07 MB，reserved 1872.00 MB，进程 RSS 1092.02 MB
- 评分切分前向次数：1658 / 1658
- 中位延迟 / p95：209.31 / 249.30 ms
- 推理峰值 GPU allocated / reserved：1973.14 / 2040.00 MB
- 推理时进程 RSS：5316.28 MB
- GPU：NVIDIA RTX A2000 Laptop GPU

## 覆盖边界与局限

- 只覆盖 SPAQ 总体感知质量 MOS，不覆盖 AlphaHalf 主题、主体、人物状态、Moment、情绪、故事或证据化点评。
- 骨干保持 4-bit 冻结。这一步没有做 Valen 的 joint / vision_top / RLCD。
- 正式分数来自 `test`。validation 只用来给指针头选 epoch。
- 内部实验用途不授权商业发布；不要分发原图、标注或模型。
