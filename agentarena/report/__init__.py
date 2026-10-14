"""Static, self-contained HTML report (no server, data embedded as JSON)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from agentarena.schemas import SampleRecord
from agentarena.scoring.aggregate import leaderboard
from agentarena.store import RunStore

TEMPLATE = Path(__file__).with_name("template.html")
MAX_RESULT_CHARS = 1500


def _clean(x: Any) -> Any:
    """JSON has no NaN; map it to null recursively."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    return x


def _sample_view(r: SampleRecord) -> dict[str, Any]:
    steps = []
    for s in r.trajectory.steps:
        results = {res.call_id: res for res in s.tool_results}
        steps.append({
            "index": s.index, "text": s.text, "in": s.usage.input_tokens, "out": s.usage.output_tokens,
            "latency": s.latency_s, "cached": s.cached,
            "calls": [{
                "name": c.name, "arguments": c.arguments,
                "result": {"content": results[c.id].content[:MAX_RESULT_CHARS], "is_error": results[c.id].is_error}
                if c.id in results else None,
            } for c in s.tool_calls],
        })
    return {
        "model": r.model, "task_id": r.task_id, "template": r.template, "repeat": r.repeat, "prompt": r.prompt,
        "success": r.outcome.success, "score": r.outcome.score, "details": r.outcome.details,
        "stop_reason": r.trajectory.stop_reason, "error": r.trajectory.error, "cost": r.cost_usd, "steps": steps,
    }


def judge_inputs(store: RunStore) -> tuple[dict, dict]:
    pairwise, rubric = {}, {}
    for j in store.load_judgments():
        if j["kind"] == "pairwise":
            pairwise[(j["task_id"], j["repeat"], j["a"], j["b"])] = j["outcome"]
        elif j["kind"] == "rubric":
            rubric[j["sample_id"]] = j["score"]
    return pairwise, rubric


def build_report(store: RunStore, out: Path | None = None) -> Path:
    records = store.load()
    if not records:
        raise ValueError(f"no samples in {store.samples}")
    pairwise, rubric = judge_inputs(store)
    board = leaderboard(records, pairwise, rubric)
    data = {
        "manifest": store.manifest(),
        "models": sorted(r["model"] for r in board["rows"]),
        "board": board,
        "samples": [_sample_view(r) for r in sorted(records, key=lambda r: (r.model, r.task_id, r.repeat))],
    }
    payload = json.dumps(_clean(data), default=str).replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/", payload)
    out = out or store.dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
