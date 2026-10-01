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
