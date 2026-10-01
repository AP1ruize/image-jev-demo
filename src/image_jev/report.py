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


def render_heldout(score_pointer: dict, score_logits: dict, compare_result: dict, run_meta: dict) -> str:
    """Chinese report laid out like AlphaHalfBench public-proxy score reports."""
    pointer_row = (score_pointer.get("by_source") or {}).get("spaq") or {}
    logits_row = (score_logits.get("by_source") or {}).get("spaq") or {}
    paired = (compare_result.get("by_source") or {}).get("spaq") or {}
    lines = [
        f"# {run_meta.get('release_id')} 评测报告",
        "",
        "用途：非商业内部模型实验。这是 SPAQ 手机照片感知质量代理，不是完整 AlphaHalf 产品 benchmark。",
        "评分切分是 Week 0 的 validation。冻结 test 没有进入训练或本报告。",
        "",
        "## 模型与运行元数据",
        "",
        f"- **当前模型**：{run_meta.get('display_name')}（`{run_meta.get('model_id')}`）",
        f"- 架构：`{run_meta.get('family')}`；格式：`safetensors`；量化：`{run_meta.get('quantization')}`",
        "- Vision：`True`",
        "- 阶段：Valen warmup，骨干冻结，只训练指针决策头",
        f"- 基座：`{run_meta.get('base_model')}`",
        f"- 图像像素上限：{run_meta.get('max_pixels')}",
        f"- 训练 / 验证张数：{run_meta.get('n_train')} / {run_meta.get('n_validation')}",
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
        "## 覆盖边界与局限",
        "",
        "- 只覆盖 SPAQ 总体感知质量 MOS，不覆盖 AlphaHalf 主题、主体、人物状态、Moment、情绪、故事或证据化点评。",
        "- 骨干保持 4-bit 冻结。这一步没有做 Valen 的 joint / vision_top / RLCD。",
        "- 验证集按 Week 0 种子切出。不要把这些数字和尚未运行的 Week 0 test leaderboard 混在一起。",
        "- 内部实验用途不授权商业发布；不要分发原图、标注或模型。",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


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
