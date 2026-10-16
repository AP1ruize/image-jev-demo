# Week 0 — 冻结推理，不训练新权重

[English](../WEEK0.md)

目的：在微调之前，看**类型化读出**（候选 logits，可选 JSON 语法约束）对 2B 视觉语言模型的技术图像质量评测值不值得。

## 模式

1. **generate** — 贪心生成 `{"value": number}`，`max_new_tokens=32`。图像质量文献里常见的问法。解析失败会拉低覆盖率。
2. **grammar** — 先写上 `{"value":`，之后只允许能接成 JSON 数字和 `}` 的 token。结构上合法，仍然要自回归解码。
3. **logits** — 按字母或单 token 分档，一次前向，softmax，期望 MOS。JSON 由 Python 组装。覆盖率按构造就是 100%。

三种模式共用冻结题目、同一批图、贪心解码、关闭 thinking，以及 processor 的 `min_pixels=3136` / `max_pixels=262144`。

## 怎样算值得

- SPAQ 和 KonIQ 上的 Spearman ρ 接近 `generate`（没有明显掉点）。
- `grammar` 和 `logits` 的 `schema_valid` 为 1.0。
- 中位延迟：`logits` 明显短于 `generate`。grammar 应落在中间（非法 token 更少，但仍有解码循环）。

如果 ρ 基本不变，接口形状是推理上的收益，不是智力上的收益。训过的指针头是另一项实验，数字在 [RESULTS.md](RESULTS.md)。

## 属性

SPAQ 还有五列属性 MOS。Week 0 的正式表只用**总体 MOS**，这样不同读出可以配对。

## Week 0 不做的事

- LoRA、强化学习、校准训练
- 审美或构图标签
- 像素级修图
