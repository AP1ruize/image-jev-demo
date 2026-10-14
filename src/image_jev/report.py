from __future__ import annotations

import json
from typing import Any


def fmt_num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def fmt_ci(ci: list[float] | None) -> str:
    if not ci or len(ci) != 2:
        return "n/a"
    return f"[{fmt_num(ci[0])}, {fmt_num(ci[1])}]"


def render_score(score: dict, run_meta: dict) -> str:
    by_source = score.get("by_source") or {}
    n_valid = sum(source.get("n_valid") or 0 for source in by_source.values())
    lines = [
        f"# {run_meta.get('release_id', 'week0')} technical IQA report",
        "",
        "## Run",
        "",
        f"- model: `{run_meta.get('display_name') or run_meta.get('model_id')}` (`{run_meta.get('family')}`)",
        f"- mode: `{run_meta.get('mode')}`",
        f"- checkpoint: `{run_meta.get('base_model')}`",
        f"- items: {score.get('n_expected')} expected / {score.get('n_responses')} responses / {n_valid} in-range",
        f"- errors: {json.dumps(score.get('errors') or {}, ensure_ascii=False)}",
        "",
        "## Per-source metrics",
        "",
        "SPAQ is 0–100 MOS; KonIQ-10k is 1–5 MOS. Do not average them.",
        "",
    ]
    for source, row in sorted(by_source.items()):
        lines.extend(
            [
                f"### {source}",
                "",
                f"- n={row.get('n_total')} (groups={row.get('n_groups')}, valid={row.get('n_valid')})",
                f"- coverage: {fmt_num(row.get('coverage'))}",
                f"- schema_valid: {fmt_num(row.get('schema_valid'))}",
                f"- Spearman ρ: {fmt_num(row.get('spearman_rho'))}  CI95 {fmt_ci(row.get('primary_metric_ci95_group_bootstrap'))}",
                f"- Pearson r: {fmt_num(row.get('pearson_r'))}",
                f"- MAE: {fmt_num(row.get('mae'))}",
                f"- median latency: {fmt_num(row.get('median_latency_ms'), 1)} ms",
                "",
            ]
        )
    lines.extend(["## Note", "", score.get("interpretation") or "", ""])
    return "\n".join(lines).rstrip() + "\n"


def _resource_lines(resources: dict) -> list[str]:
    forward = resources.get("scored_forward") or {}
    resident = resources.get("resident_after_load") or {}
    if not forward and not resident:
        return ["- 本次没有重新测量推理资源。"]
    lines = []
    if resident:
        lines.append(
            f"- 加载后常驻：GPU allocated {fmt_num(resident.get('gpu_allocated_mb'), 2)} MB，"
            f"reserved {fmt_num(resident.get('gpu_reserved_mb'), 2)} MB，"
            f"进程 RSS {fmt_num(resident.get('process_rss_mb'), 2)} MB"
        )
    if forward:
        lines.extend(
            [
                f"- 评分切分前向次数：{forward.get('measured_forwards')} / {forward.get('n')}",
                f"- 中位延迟 / p95：{fmt_num(forward.get('median_latency_ms'), 2)} / {fmt_num(forward.get('p95_latency_ms'), 2)} ms",
                f"- 推理峰值 GPU allocated / reserved：{fmt_num(forward.get('gpu_max_allocated_mb'), 2)} / {fmt_num(forward.get('gpu_max_reserved_mb'), 2)} MB",
                f"- 推理时进程 RSS：{fmt_num(forward.get('process_rss_mb'), 2)} MB",
                f"- GPU：{forward.get('gpu_name') or 'n/a'}",
            ]
        )
    return lines


def render_heldout(score_pointer: dict, score_logits: dict, compare_result: dict, run_meta: dict) -> str:
    """Chinese report laid out like AlphaHalfBench public-proxy score reports."""
    pointer_row = (score_pointer.get("by_source") or {}).get("spaq") or {}
    logits_row = (score_logits.get("by_source") or {}).get("spaq") or {}
    paired = (compare_result.get("by_source") or {}).get("spaq") or {}
    lines = [
        f"# {run_meta.get('release_id')} 评测报告",
        "",
        "用途：非商业内部模型实验。这是 SPAQ 手机照片感知质量代理，不是完整 AlphaHalf 产品 benchmark。",
        f"评分切分是 Week 0 的 {run_meta.get('scored_split') or 'validation'}。训练不用这个切分。",
        "",
        "## 模型与运行元数据",
        "",
        f"- **当前模型**：{run_meta.get('display_name')}（`{run_meta.get('model_id')}`）",
        f"- 架构：`{run_meta.get('family')}`；格式：`safetensors`；量化：`{run_meta.get('quantization')}`",
        "- Vision：`True`",
        "- 阶段：Valen warmup，骨干冻结，只训练指针决策头",
        f"- 基座：`{run_meta.get('base_model')}`",
        f"- 图像像素上限：{run_meta.get('max_pixels')}",
        f"- 训练 / 验证 / 测试张数：{run_meta.get('n_train')} / {run_meta.get('n_validation')} / {run_meta.get('n_test')}",
        f"- 正式评分切分：`{run_meta.get('scored_split') or 'validation'}`",
        f"- 头学习率：{run_meta.get('head_lr')}；投影维：{run_meta.get('projection_dim')}；最多 epoch：{run_meta.get('epochs')}",
        f"- 按验证损失选出的 epoch：{run_meta.get('best_epoch')}（从 0 计）",
        f"- 训练损失 起点 / 最佳 epoch：{fmt_num(run_meta.get('train_loss_start'))} / {fmt_num(run_meta.get('train_loss_best'))}",
        f"- 验证损失 起点 / 最佳 epoch：{fmt_num(run_meta.get('val_loss_start'))} / {fmt_num(run_meta.get('val_loss_best'))}",
        f"- 切分种子：`{run_meta.get('split_seed')}`",
        f"- 对话策略：stateless_per_item（每张图一次前向，不携带上一张图）",
        f"- 指针响应：`{run_meta.get('pointer_responses')}`",
        f"- logits 响应：`{run_meta.get('logits_responses')}`",
        f"- 指针评分：`{run_meta.get('pointer_score')}`",
        f"- logits 评分：`{run_meta.get('logits_score')}`",
        f"- 配对比较：`{run_meta.get('compare_path')}`",
        "",
        "## 总体完成情况",
        "",
        f"- 验证期望题数：{score_pointer.get('n_expected')}",
        f"- 指针响应 / 有效：{score_pointer.get('n_responses')} / {pointer_row.get('n_valid')}",
        f"- logits 响应 / 有效：{score_logits.get('n_responses')} / {logits_row.get('n_valid')}",
        f"- 指针错误切片：{json.dumps(score_pointer.get('errors') or {}, ensure_ascii=False)}",
        f"- logits 错误切片：{json.dumps(score_logits.get('errors') or {}, ensure_ascii=False)}",
        "",
        "## 分来源原生指标",
        "",
        "本报告只有 SPAQ（0–100 MOS）。不要和 KonIQ 或其他量表混成一个总分。",
        "主指标 95% 区间为按 `group_id` 的 cluster bootstrap。",
        "",
        "### spaq / pointer",
        "",
        f"- 任务类型：`continuous`",
        f"- n_total / n_valid / coverage：{pointer_row.get('n_total')} / {pointer_row.get('n_valid')} / {fmt_num(pointer_row.get('coverage'))}",
        f"- n_groups：{pointer_row.get('n_groups')}",
        f"- schema_valid：{fmt_num(pointer_row.get('schema_valid'))}",
        f"- 中位延迟：{fmt_num(pointer_row.get('median_latency_ms'), 2)} ms",
        f"- 主指标 95% 组级 CI：{fmt_ci(pointer_row.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Spearman ρ：{fmt_num(pointer_row.get('spearman_rho'))}",
        f"- Pearson r：{fmt_num(pointer_row.get('pearson_r'))}",
        f"- MAE：{fmt_num(pointer_row.get('mae'))}",
        "",
        "### spaq / logits",
        "",
        f"- 任务类型：`continuous`",
        f"- n_total / n_valid / coverage：{logits_row.get('n_total')} / {logits_row.get('n_valid')} / {fmt_num(logits_row.get('coverage'))}",
        f"- n_groups：{logits_row.get('n_groups')}",
        f"- schema_valid：{fmt_num(logits_row.get('schema_valid'))}",
        f"- 中位延迟：{fmt_num(logits_row.get('median_latency_ms'), 2)} ms",
        f"- 主指标 95% 组级 CI：{fmt_ci(logits_row.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Spearman ρ：{fmt_num(logits_row.get('spearman_rho'))}",
        f"- Pearson r：{fmt_num(logits_row.get('pearson_r'))}",
        f"- MAE：{fmt_num(logits_row.get('mae'))}",
        "",
        "## 配对比较",
        "",
        "同一批验证图。Delta 是 pointer − logits 的 Spearman ρ。延迟比是 pointer / logits 的中位数。",
        "",
        f"- paired valid：{paired.get('n_paired_valid')} / {paired.get('n_total')}",
        f"- Spearman logits / pointer：{fmt_num(paired.get('left_spearman_rho'))} / {fmt_num(paired.get('right_spearman_rho'))}",
        f"- Δρ：{fmt_num(paired.get('delta_right_minus_left'))}  CI95 {fmt_ci(paired.get('ci95'))}",
        f"- schema_valid logits / pointer：{fmt_num(paired.get('left_schema_valid'))} / {fmt_num(paired.get('right_schema_valid'))}",
        f"- 中位延迟 logits / pointer：{fmt_num(paired.get('left_median_latency_ms'), 2)} / {fmt_num(paired.get('right_median_latency_ms'), 2)} ms",
        f"- 延迟比（pointer/logits）：{fmt_num(paired.get('latency_ratio_right_over_left'))}",
        "",
        "## 训练诊断",
        "",
        "训练集上的 Spearman 只说明头是否拟合了训练特征，不能当作评测成绩。",
        "",
        f"- 训练 Spearman ρ：{fmt_num(run_meta.get('train_spearman_rho'))}",
        f"- 训练 MAE：{fmt_num(run_meta.get('train_mae'))}",
        "",
        "## 推理时间与资源",
        "",
        *_resource_lines(run_meta.get("inference_resources") or {}),
        "",
        "## 覆盖边界与局限",
        "",
        "- 只覆盖 SPAQ 总体感知质量 MOS，不覆盖 AlphaHalf 主题、主体、人物状态、Moment、情绪、故事或证据化点评。",
        "- 骨干保持 4-bit 冻结。这一步没有做 Valen 的 joint / vision_top / RLCD。",
        f"- 正式分数来自 `{run_meta.get('scored_split') or 'validation'}`。validation 只用来给指针头选 epoch。",
        "- 内部实验用途不授权商业发布；不要分发原图、标注或模型。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_full_pair(pointer_meta: dict, pointer_score: dict, qwen_meta: dict, qwen_score: dict, paired: dict) -> str:
    """Compare the pointer head with a QLoRA-tuned Qwen3.5-2B on the same SPAQ test split."""
    pointer = (pointer_score.get("by_source") or {}).get("spaq") or {}
    qwen = (qwen_score.get("by_source") or {}).get("spaq") or {}
    row = (paired.get("by_source") or {}).get("spaq") or {}
    pointer_res = ((pointer_meta.get("inference_resources") or {}).get("scored_forward")) or {}
    qwen_res = qwen_meta.get("inference_resources") or {}
    lines = [
        f"# {pointer_meta.get('release_id')} 与 {qwen_meta.get('release_id')} 对比报告",
        "",
        "用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集和完整测试集。",
        "指针头是 ImageJev 的决策读出。对照模型是原版 Qwen3.5-2B 的 4-bit QLoRA，生成 JSON 分数。",
        "validation 只用于给指针头选 epoch。正式分数、延迟和资源都来自 test。",
        "",
        "## 模型与运行元数据",
        "",
        f"- 指针头：`{pointer_meta.get('model_id')}`，训练 {pointer_meta.get('n_train')} 张，最佳 epoch {pointer_meta.get('best_epoch')}",
        f"- 原版 QLoRA：`{qwen_meta.get('model_id')}`，基座 `{qwen_meta.get('base_model')}`，量化 `{qwen_meta.get('quantization')}`，epoch {qwen_meta.get('epochs')}",
        f"- 测试张数：{pointer_meta.get('n_test')}",
        f"- 像素上限：{pointer_meta.get('max_pixels')}",
        f"- GPU：{pointer_res.get('gpu_name') or qwen_res.get('gpu_name') or 'n/a'}",
        "",
        "## 分来源原生指标",
        "",
        "只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。",
        "",
        "### spaq / image-jev pointer",
        "",
        f"- n_total / n_valid / coverage：{pointer.get('n_total')} / {pointer.get('n_valid')} / {fmt_num(pointer.get('coverage'))}",
        f"- schema_valid：{fmt_num(pointer.get('schema_valid'))}",
        f"- Spearman ρ：{fmt_num(pointer.get('spearman_rho'))}  CI95 {fmt_ci(pointer.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(pointer.get('pearson_r'))}",
        f"- MAE：{fmt_num(pointer.get('mae'))}",
        f"- 中位延迟：{fmt_num(pointer.get('median_latency_ms'), 2)} ms",
        "",
        "### spaq / qwen3.5-2b 4-bit QLoRA",
        "",
        f"- n_total / n_valid / coverage：{qwen.get('n_total')} / {qwen.get('n_valid')} / {fmt_num(qwen.get('coverage'))}",
        f"- schema_valid：{fmt_num(qwen.get('schema_valid'))}",
        f"- Spearman ρ：{fmt_num(qwen.get('spearman_rho'))}  CI95 {fmt_ci(qwen.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(qwen.get('pearson_r'))}",
        f"- MAE：{fmt_num(qwen.get('mae'))}",
        f"- 中位延迟：{fmt_num(qwen.get('median_latency_ms'), 2)} ms",
        "",
        "## 配对比较",
        "",
        "Delta 是 pointer − QLoRA 的 Spearman ρ。延迟比和显存比都是 pointer / QLoRA。",
        "",
        f"- paired valid：{row.get('n_paired_valid')} / {row.get('n_total')}",
        f"- Spearman QLoRA / pointer：{fmt_num(row.get('left_spearman_rho'))} / {fmt_num(row.get('right_spearman_rho'))}",
        f"- Δρ：{fmt_num(row.get('delta_right_minus_left'))}  CI95 {fmt_ci(row.get('ci95'))}",
        "",
        "## 推理时间与资源",
        "",
        f"- 指针头中位 / p95 延迟：{fmt_num(pointer_res.get('median_latency_ms'), 2)} / {fmt_num(pointer_res.get('p95_latency_ms'), 2)} ms",
        f"- QLoRA 中位 / p95 延迟：{fmt_num(qwen_res.get('median_latency_ms'), 2)} / {fmt_num(qwen_res.get('p95_latency_ms'), 2)} ms",
        f"- 延迟比（pointer/QLoRA）：{fmt_num(_ratio(pointer_res.get('median_latency_ms'), qwen_res.get('median_latency_ms')))}",
        f"- 指针头峰值 GPU allocated / reserved：{fmt_num(pointer_res.get('gpu_max_allocated_mb'), 2)} / {fmt_num(pointer_res.get('gpu_max_reserved_mb'), 2)} MB",
        f"- QLoRA 峰值 GPU allocated / reserved：{fmt_num(qwen_res.get('gpu_max_allocated_mb'), 2)} / {fmt_num(qwen_res.get('gpu_max_reserved_mb'), 2)} MB",
        f"- 显存比（pointer/QLoRA，allocated）：{fmt_num(_ratio(pointer_res.get('gpu_max_allocated_mb'), qwen_res.get('gpu_max_allocated_mb')))}",
        f"- 指针头进程 RSS：{fmt_num(pointer_res.get('process_rss_mb'), 2)} MB",
        f"- QLoRA 进程 RSS：{fmt_num(qwen_res.get('process_rss_mb'), 2)} MB",
        "",
        "## 覆盖边界与局限",
        "",
        "- 只覆盖 SPAQ 总体感知质量，不覆盖 AlphaHalf 的主题、主体、情绪或叙事。",
        "- 指针头不更新骨干。QLoRA 用 rank 8 适配器更新语言模型，视觉编码器保持冻结。",
        "- 两者都是 4-bit NF4，像素上限相同。延迟是模型前向或生成时间，不含读图。",
        "- 内部实验用途不授权商业发布。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def render_pointer_lora(
    frozen_meta: dict,
    frozen_score: dict,
    qlora_meta: dict,
    qlora_score: dict,
    lora_meta: dict,
    lora_score: dict,
    versus_frozen: dict,
    versus_qlora: dict,
) -> str:
    """Compare a language-LoRA pointer with the frozen pointer and JSON QLoRA on one SPAQ test split."""
    frozen = (frozen_score.get("by_source") or {}).get("spaq") or {}
    qlora = (qlora_score.get("by_source") or {}).get("spaq") or {}
    lora = (lora_score.get("by_source") or {}).get("spaq") or {}
    frozen_row = (versus_frozen.get("by_source") or {}).get("spaq") or {}
    qlora_row = (versus_qlora.get("by_source") or {}).get("spaq") or {}
    lines = [
        f"# {lora_meta.get('release_id')} 对比报告",
        "",
        "用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集和完整测试集。",
        "这一版在冻结视觉编码器的 4-bit Qwen3.5-2B 上同时更新语言 LoRA 和指针头，推理仍是一次前向。",
        "对照是只训指针头的 warmup，以及生成 JSON 的 4-bit QLoRA。",
        "",
        "## 模型与运行元数据",
        "",
        f"- 语言 LoRA + 指针头：`{lora_meta.get('model_id')}`，epoch {lora_meta.get('epochs')}，LoRA r={lora_meta.get('lora_r')}",
        f"- 头初始化：`{lora_meta.get('head_init')}`",
        f"- 冻结指针头：`{frozen_meta.get('model_id')}`，最佳 epoch {frozen_meta.get('best_epoch')}",
        f"- JSON QLoRA：`{qlora_meta.get('model_id')}`，epoch {qlora_meta.get('epochs')}，LoRA r={qlora_meta.get('lora_r')}",
        f"- 训练 / 测试张数：{lora_meta.get('n_train')} / {lora_meta.get('n_test')}",
        f"- 像素上限：{lora_meta.get('max_pixels')}",
        f"- 训练损失 起点 / 终点：{fmt_num(lora_meta.get('loss_start'))} / {fmt_num(lora_meta.get('loss_end'))}",
        "",
        "## 分来源原生指标",
        "",
        "只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。",
        "",
        "### spaq / language LoRA + pointer",
        "",
        f"- n_total / n_valid / coverage：{lora.get('n_total')} / {lora.get('n_valid')} / {fmt_num(lora.get('coverage'))}",
        f"- schema_valid：{fmt_num(lora.get('schema_valid'))}",
        f"- Spearman ρ：{fmt_num(lora.get('spearman_rho'))}  CI95 {fmt_ci(lora.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(lora.get('pearson_r'))}",
        f"- MAE：{fmt_num(lora.get('mae'))}",
        f"- 中位延迟：{fmt_num(lora.get('median_latency_ms'), 2)} ms",
        "",
        "### spaq / frozen pointer",
        "",
        f"- Spearman ρ：{fmt_num(frozen.get('spearman_rho'))}  CI95 {fmt_ci(frozen.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(frozen.get('pearson_r'))}",
        f"- MAE：{fmt_num(frozen.get('mae'))}",
        f"- 中位延迟：{fmt_num(frozen.get('median_latency_ms'), 2)} ms",
        "",
        "### spaq / JSON QLoRA",
        "",
        f"- Spearman ρ：{fmt_num(qlora.get('spearman_rho'))}  CI95 {fmt_ci(qlora.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(qlora.get('pearson_r'))}",
        f"- MAE：{fmt_num(qlora.get('mae'))}",
        f"- 中位延迟：{fmt_num(qlora.get('median_latency_ms'), 2)} ms",
        "",
        "## 配对比较",
        "",
        "Delta 是（语言 LoRA + 指针头）减对照的 Spearman ρ。正值表示这一版排序更好。",
        "",
        f"- 相对冻结指针头：Δρ {fmt_num(frozen_row.get('delta_right_minus_left'))}  CI95 {fmt_ci(frozen_row.get('ci95'))}",
        f"- 相对 JSON QLoRA：Δρ {fmt_num(qlora_row.get('delta_right_minus_left'))}  CI95 {fmt_ci(qlora_row.get('ci95'))}",
        f"- paired valid：{frozen_row.get('n_paired_valid')} / {frozen_row.get('n_total')}",
        "",
        "## 推理时间与资源",
        "",
        f"- 语言 LoRA + 指针头中位 / p95 延迟：{fmt_num((lora_meta.get('inference_resources') or {}).get('median_latency_ms'), 2)} / {fmt_num((lora_meta.get('inference_resources') or {}).get('p95_latency_ms'), 2)} ms",
        f"- 峰值 GPU allocated：{fmt_num((lora_meta.get('inference_resources') or {}).get('gpu_max_allocated_mb'), 2)} MB",
        "",
        "## 覆盖边界与局限",
        "",
        "- 只覆盖 SPAQ 总体感知质量。",
        "- 视觉编码器保持冻结。语言 LoRA 与指针头一起用指针交叉熵训练一个 epoch，正式分数用最后一个 checkpoint。",
        "- 内部实验用途不授权商业发布。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _ci_entirely_below_zero(ci) -> bool:
    return isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[1] is not None and float(ci[1]) < 0


def merger_decision(probe: dict | None, qlora_ci) -> str:
    """Say whether a 4GB card should unfreeze the visual merger. Never starts that training."""
    if not probe:
        return "视觉 merger 探针还没有跑，所以不启动 merger 训练。"
    if probe.get("oom"):
        return "不建议解冻视觉 merger。4GB A2000 上 joint 前向加反向发生 OOM。没有启动 merger 训练。"
    if not probe.get("grad_ok"):
        return "不建议解冻视觉 merger。探针没有让 merger 拿到梯度，说明这次并没有真正反传到视觉 merger。"
    if not probe.get("fits"):
        return "不建议解冻视觉 merger。峰值显存离 4GB 总量不到 512MB，没有余量。没有启动 merger 训练。"
    if _ci_entirely_below_zero(qlora_ci):
        return (
            "探针能稳定跑完，而且相对 JSON QLoRA 的 Spearman 差值置信区间整段低于零。"
            "按约定只可以再考虑短程 merger 训练。本次不启动那个训练。"
        )
    return "探针能跑完，但相对 JSON QLoRA 的 Spearman 差值置信区间没有整段低于零。不建议解冻视觉 merger，也不启动 merger 训练。"


def render_pointer_lora_valselect(
    last_meta: dict,
    last_score: dict,
    qlora_meta: dict,
    qlora_score: dict,
    selected_meta: dict,
    selected_score: dict,
    versus_last: dict,
    versus_qlora: dict,
) -> str:
    """Compare a validation-selected language-LoRA pointer with the last-step run and JSON QLoRA."""
    last = (last_score.get("by_source") or {}).get("spaq") or {}
    qlora = (qlora_score.get("by_source") or {}).get("spaq") or {}
    chosen = (selected_score.get("by_source") or {}).get("spaq") or {}
    last_row = (versus_last.get("by_source") or {}).get("spaq") or {}
    qlora_row = (versus_qlora.get("by_source") or {}).get("spaq") or {}
    resources = selected_meta.get("inference_resources") or {}
    probe = selected_meta.get("merger_probe") or {}
    monitor = selected_meta.get("val_monitor") or []
    monitor_lines = [
        f"  - step {item.get('step')}：训练损失 {fmt_num(item.get('train_loss'))}，验证交叉熵 {fmt_num(item.get('loss'))}"
        for item in monitor
    ]
    lines = [
        f"# {selected_meta.get('release_id')} 对比报告",
        "",
        "用途：非商业内部模型实验。同一份 SPAQ Week 0 完整训练集、验证集和测试集。",
        "这一版在冻结视觉编码器的 4-bit Qwen3.5-2B 上同时更新语言 LoRA 和指针头，推理仍是一次前向。",
        "正式权重按验证集指针交叉熵选取，不是最后一个 optimizer step。测试集不参与选模。",
        "对照是同一配置的最后一步语言 LoRA + 指针头，以及生成 JSON 的 4-bit QLoRA。",
        "",
        "## 模型与运行元数据",
        "",
        f"- 验证集选出的语言 LoRA + 指针头：`{selected_meta.get('model_id')}`，epoch {selected_meta.get('epochs')}，LoRA r={selected_meta.get('lora_r')}",
        f"- 头初始化：`{selected_meta.get('head_init')}`",
        f"- 权重目录：`{selected_meta.get('checkpoint_dir')}`",
        f"- 最后一步对照：`{last_meta.get('model_id')}`，optimizer step {last_meta.get('optimizer_step')}",
        f"- JSON QLoRA：`{qlora_meta.get('model_id')}`，epoch {qlora_meta.get('epochs')}，LoRA r={qlora_meta.get('lora_r')}",
        f"- 训练 / 验证 / 测试张数：{selected_meta.get('n_train')} / {selected_meta.get('n_validation')} / {selected_meta.get('n_test')}",
        f"- 像素上限：{selected_meta.get('max_pixels')}",
        f"- 训练顺序种子：{selected_meta.get('seed')}。前 5 个训练下标：{selected_meta.get('order_head')}",
        f"- 选模监控：每 {selected_meta.get('eval_steps')} 个 optimizer step 以及最后一步，在固定种子子集上算指针交叉熵",
        f"- 监控子集：{selected_meta.get('monitor_size')} / {selected_meta.get('n_validation')}，种子 {selected_meta.get('monitor_seed')}",
        f"- 选中 step：{selected_meta.get('selected_step')}",
        f"- 监控验证交叉熵：{fmt_num(selected_meta.get('monitor_loss'))}",
        f"- 全量验证交叉熵：{fmt_num(selected_meta.get('val_full_loss'))}（n={selected_meta.get('val_full_n')}）",
        f"- 训练损失 起点 / 选中 step / 终点：{fmt_num(selected_meta.get('loss_start'))} / {fmt_num(selected_meta.get('train_loss_selected'))} / {fmt_num(selected_meta.get('loss_end'))}",
        "",
        "监控记录：",
        *monitor_lines,
        "",
        "## 分来源原生指标",
        "",
        "只报告 SPAQ 0–100 MOS。主指标 95% 区间按 `group_id` 做 cluster bootstrap。",
        "",
        "### spaq / validation-selected language LoRA + pointer",
        "",
        f"- n_total / n_valid / coverage：{chosen.get('n_total')} / {chosen.get('n_valid')} / {fmt_num(chosen.get('coverage'))}",
        f"- schema_valid：{fmt_num(chosen.get('schema_valid'))}",
        f"- Spearman ρ：{fmt_num(chosen.get('spearman_rho'))}  CI95 {fmt_ci(chosen.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(chosen.get('pearson_r'))}",
        f"- MAE：{fmt_num(chosen.get('mae'))}",
        f"- 中位延迟：{fmt_num(chosen.get('median_latency_ms'), 2)} ms",
        "",
        "### spaq / last-step language LoRA + pointer",
        "",
        f"- Spearman ρ：{fmt_num(last.get('spearman_rho'))}  CI95 {fmt_ci(last.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(last.get('pearson_r'))}",
        f"- MAE：{fmt_num(last.get('mae'))}",
        f"- 中位延迟：{fmt_num(last.get('median_latency_ms'), 2)} ms",
        "",
        "### spaq / JSON QLoRA",
        "",
        f"- Spearman ρ：{fmt_num(qlora.get('spearman_rho'))}  CI95 {fmt_ci(qlora.get('primary_metric_ci95_group_bootstrap'))}",
        f"- Pearson r：{fmt_num(qlora.get('pearson_r'))}",
        f"- MAE：{fmt_num(qlora.get('mae'))}",
        f"- 中位延迟：{fmt_num(qlora.get('median_latency_ms'), 2)} ms",
        "",
        "## 配对比较",
        "",
        "Delta 是（验证集选出的语言 LoRA + 指针头）减对照的 Spearman ρ。正值表示这一版排序更好。",
        "",
        f"- 相对最后一步语言 LoRA + 指针头：Δρ {fmt_num(last_row.get('delta_right_minus_left'))}  CI95 {fmt_ci(last_row.get('ci95'))}",
        f"- 相对 JSON QLoRA：Δρ {fmt_num(qlora_row.get('delta_right_minus_left'))}  CI95 {fmt_ci(qlora_row.get('ci95'))}",
        f"- paired valid：{qlora_row.get('n_paired_valid')} / {qlora_row.get('n_total')}",
        "",
        "## 推理时间与资源",
        "",
        f"- 验证集选出的模型中位 / p95 延迟：{fmt_num(resources.get('median_latency_ms'), 2)} / {fmt_num(resources.get('p95_latency_ms'), 2)} ms",
        f"- 峰值 GPU allocated / reserved：{fmt_num(resources.get('gpu_max_allocated_mb'), 2)} / {fmt_num(resources.get('gpu_max_reserved_mb'), 2)} MB",
        "",
        "## 视觉 merger 探针",
        "",
        "探针在同一 4-bit NF4 加载、语言 LoRA 和 `use_reentrant=False` 梯度检查点下，只把视觉 merger 设为可训练，用一张图做一次前向和反向。",
        "这次加载里 merger 的线性层本身是 NF4，不能直接打开梯度，所以探针只把这两层反量化成 fp16。其余模块不改精度。",
        "视觉编码器的其余参数保持冻结。探针不更新正式权重。",
        "",
        f"- merger 模块：{probe.get('modules') or 'n/a'}",
        f"- merger 参数量：{probe.get('n_parameters') if probe else 'n/a'}",
        f"- merger 权重：{probe.get('merger_storage') or 'n/a'}",
        f"- OOM：{probe.get('oom') if probe else 'n/a'}",
        f"- 梯度已回传：{probe.get('grad_ok') if probe else 'n/a'}",
        f"- 峰值 allocated / reserved：{fmt_num(probe.get('gpu_max_allocated_mb'), 2)} / {fmt_num(probe.get('gpu_max_reserved_mb'), 2)} MB",
        f"- 显存总量：{fmt_num(probe.get('gpu_total_mb'), 2)} MB",
        f"- nvidia-smi 已用 / 总量：{fmt_num(probe.get('nvidia_used_mb'), 2)} / {fmt_num(probe.get('nvidia_total_mb'), 2)} MB",
        f"- 余量：{fmt_num(probe.get('headroom_mb'), 2)} MB（门槛 512 MB）",
        f"- {merger_decision(probe, qlora_row.get('ci95'))}",
        "",
        "## 覆盖边界与局限",
        "",
        "- 只覆盖 SPAQ 总体感知质量。",
        "- 视觉编码器在这次训练里保持冻结。语言 LoRA 与指针头一起用指针交叉熵训练一个 epoch。",
        "- 选模用固定种子的验证子集做监控，选中的 checkpoint 再在全部验证集上确认交叉熵。正式分数用这份权重，不用最后一步。",
        "- 内部实验用途不授权商业发布。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _ratio(left, right) -> float | None:
    if not left or not right:
        return None
    return float(left) / float(right)


def render_compare(result: dict, left_meta: dict, right_meta: dict) -> str:
    lines = [
        "# Week 0 paired comparison",
        "",
        f"- left: `{left_meta.get('mode')}` / `{left_meta.get('model_id')}`",
        f"- right: `{right_meta.get('mode')}` / `{right_meta.get('model_id')}`",
        "",
        "Delta is right − left on Spearman ρ. Latency ratio is right / left medians.",
        "",
    ]
    for source, row in sorted((result.get("by_source") or {}).items()):
        lines.extend(
            [
                f"### {source}",
                "",
                f"- paired valid: {row.get('n_paired_valid')} / {row.get('n_total')}",
                f"- Spearman left/right: {fmt_num(row.get('left_spearman_rho'))} / {fmt_num(row.get('right_spearman_rho'))}",
                f"- Δρ: {fmt_num(row.get('delta_right_minus_left'))}  CI95 {fmt_ci(row.get('ci95'))}",
                f"- schema_valid left/right: {fmt_num(row.get('left_schema_valid'))} / {fmt_num(row.get('right_schema_valid'))}",
                f"- median latency left/right: {fmt_num(row.get('left_median_latency_ms'), 1)} / {fmt_num(row.get('right_median_latency_ms'), 1)} ms",
                f"- latency ratio (right/left): {fmt_num(row.get('latency_ratio_right_over_left'))}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"
