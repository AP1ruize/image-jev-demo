# 评测协议

[English](../EVAL.md)

## 单位

每一道冻结题目是一张图、一个总体 MOS。模型无状态：只有当前图像和当前提示。

预测是源量表内的一个有限数字，或者一条解析错误。

## 指标（按数据源）

| 名称 | 用途 |
| --- | --- |
| coverage | 预测落在量表内的比例 |
| schema_valid | 载荷是带 `value` 的合法 JSON 的比例（logits / grammar 按构造为 1） |
| Spearman ρ | 主排序指标；平局用中间秩 |
| Pearson r | 线性一致 |
| MAE | 源量表上的绝对误差 |
| median latency_ms | 模型调用的墙钟时间，含图像预处理 |
| grouped bootstrap 95% CI | 按 `group_id` 重抽样，2000 次，种子 42 |

不要把 SPAQ（0–100）和 KonIQ（1–5）平均成一个总分。

## 配对

`compare_week0.py` 要求两次运行的 `item_id` 集合相同。Delta 是右减左的 Spearman，并给出分组 bootstrap 区间。延迟报中位数之比。

## 绑定

正式运行会在 JSONL 旁边写 `<stem>.run.json`（模型 id、模式、release 哈希、题目数）。缺少这份元数据时，`score_week0.py` 拒绝打分，除非加上 `--allow-untracked`（只用于诊断）。
