"""Process metrics computed from a trajectory: how the agent got there, not just whether."""

from __future__ import annotations

import json
from collections import Counter

from agentarena.schemas import Trajectory

PROTOCOL_ERRORS = {"unknown_tool", "bad_json", "schema"}


def trajectory_metrics(traj: Trajectory) -> dict[str, float | None]:
    calls = traj.tool_calls
    results = {r.call_id: r for r in traj.tool_results}
    sigs = [(c.name, json.dumps(c.arguments, sort_keys=True)) for c in calls]
    counts = Counter(sigs)

    invalid = sum(1 for r in results.values() if r.error_kind in PROTOCOL_ERRORS)
    domain_err = sum(1 for r in results.values() if r.error_kind == "domain")

    # Recovery: after a failed call, does a later call to the same tool succeed?
    failed_at = [(i, c.name) for i, c in enumerate(calls) if results.get(c.id) and results[c.id].is_error]
    recovered = sum(
        any(c.name == name and not results[c.id].is_error for c in calls[i + 1:] if c.id in results)
        for i, name in failed_at
    )

    live = [s for s in traj.steps if not s.cached]
    return {
        "steps": len(traj.steps),
        "tool_calls": len(calls),
        "invalid_calls": invalid,
        "domain_errors": domain_err,
        "invalid_call_rate": invalid / len(calls) if calls else 0.0,
        "redundant_calls": len(sigs) - len(counts),
        "looped": float(any(n >= 3 for n in counts.values()) or traj.stop_reason == "max_steps"),
        "recovery_rate": recovered / len(failed_at) if failed_at else None,
        "input_tokens": traj.usage.input_tokens,
        "output_tokens": traj.usage.output_tokens,
        "wall_time_s": traj.wall_time_s,
        "mean_turn_latency_s": sum(s.latency_s for s in live) / len(live) if live else None,
        "budget_exhausted": float(traj.stop_reason in ("max_steps", "max_tokens", "timeout")),
        "errored": float(traj.stop_reason == "error"),
    }
