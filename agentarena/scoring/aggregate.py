"""Turn sample records into leaderboard rows, battles and per-template breakdowns."""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from typing import Any

import numpy as np

from agentarena.schemas import SampleRecord
from agentarena.scoring.rating import Battle, bootstrap_ratings
from agentarena.scoring.stats import cluster_bootstrap_ci, pass_at_k, pass_hat_k


def _nanmean(xs: list[float | None]) -> float:
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def make_battles(
    records: list[SampleRecord], judged: dict[tuple[str, int, str, str], float] | None = None
) -> list[Battle]:
    """Pair models on the same (task instance, repeat). Success decides; if both
    succeed, partial-credit score decides; a pairwise judge verdict (if present)
    breaks remaining ties."""
    judged = judged or {}
    by_slot: dict[tuple[str, int], dict[str, SampleRecord]] = defaultdict(dict)
    for r in records:
        by_slot[(r.task_id, r.repeat)][r.model] = r
    battles = []
    for (task_id, rep), slot in by_slot.items():
        for a, b in itertools.combinations(sorted(slot), 2):
            ra, rb = slot[a], slot[b]
            if ra.outcome.success != rb.outcome.success:
                out = 1.0 if ra.outcome.success else 0.0
            elif abs(ra.outcome.score - rb.outcome.score) > 1e-9:
                out = 1.0 if ra.outcome.score > rb.outcome.score else 0.0
            else:
                out = judged.get((task_id, rep, a, b), 0.5)
            battles.append(Battle(a, b, out))
    return battles


def leaderboard(
    records: list[SampleRecord], judged: dict | None = None, rubric: dict[str, float] | None = None
) -> dict[str, Any]:
    # Infrastructure errors (quota, auth, outages) say nothing about the model: exclude them
    # from every score instead of counting them as failures, and report them separately.
    excluded: dict[str, int] = defaultdict(int)
    for r in records:
        if r.trajectory.stop_reason == "error":
            excluded[r.model] += 1
    records = [r for r in records if r.trajectory.stop_reason != "error"]
    models = sorted({r.model for r in records})
    by_model: dict[str, list[SampleRecord]] = defaultdict(list)
    for r in records:
        by_model[r.model].append(r)
    ratings = bootstrap_ratings(make_battles(records, judged), models)
    max_rep = max((r.repeat for r in records), default=0) + 1

    rows = []
    for m in models:
        recs = by_model[m]
        row: dict[str, Any] = {"model": m, **ratings[m], "samples": len(recs), "excluded_errors": excluded.get(m, 0)}
        for domain in ("all", "tool_use", "research"):
            sub = [r for r in recs if domain == "all" or r.domain == domain]
            per_task: dict[str, list[float]] = defaultdict(list)
            for r in sub:
                per_task[r.task_id].append(float(r.outcome.success))
            mean, lo, hi = cluster_bootstrap_ci(list(per_task.values()))
            row[f"pass1_{domain}"] = {"mean": mean, "lo": lo, "hi": hi, "tasks": len(per_task)}
        per_task = defaultdict(list)
        for r in recs:
            per_task[r.task_id].append(r.outcome.success)
        row["pass_hat_k"] = {
            k: _nanmean([pass_hat_k(sum(v), len(v), k) for v in per_task.values()]) for k in range(1, max_rep + 1)
        }
        row["pass_at_k"] = {
            k: _nanmean([pass_at_k(sum(v), len(v), k) for v in per_task.values()]) for k in range(1, max_rep + 1)
        }
        solved = sum(r.outcome.success for r in recs)
        cost = sum(r.cost_usd for r in recs)
        row.update(
            mean_score=_nanmean([r.outcome.score for r in recs]),
            cost_usd=cost,
            cost_per_solved=cost / solved if solved else float("nan"),
            **{f"mean_{k}": _nanmean([r.metrics.get(k, float("nan")) for r in recs]) for k in (
                "steps", "tool_calls", "invalid_call_rate", "redundant_calls", "looped", "recovery_rate",
                "input_tokens", "output_tokens", "wall_time_s", "mean_turn_latency_s", "budget_exhausted", "errored",
            )},
            citation_precision=_nanmean(
                [r.outcome.details.get("citation_precision", float("nan")) for r in recs if r.domain == "research"]
            ),
            gold_recall=_nanmean(
                [r.outcome.details.get("gold_recall", float("nan")) for r in recs if r.domain == "research"]
            ),
        )
        if rubric:
            row["judge_rubric"] = _nanmean([rubric[r.sample_id] for r in recs if r.sample_id in rubric])
        rows.append(row)
    rows.sort(key=lambda r: -r["elo"])

    templates = sorted({(r.domain, r.template) for r in records})
    heat = {
        m: {t: _nanmean([float(r.outcome.success) for r in by_model[m] if r.template == t]) for _, t in templates}
        for m in models
    }
    return {"rows": rows, "templates": [{"domain": d, "template": t} for d, t in templates], "heatmap": heat,
            "repeats": max_rep, "unscored": {m: n for m, n in excluded.items() if m not in models}}
