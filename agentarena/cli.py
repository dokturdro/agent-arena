"""`arena` command line."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from dotenv import load_dotenv

from agentarena.agents import ProviderError, make_model
from agentarena.config import ROOT, load_models, load_suite
from agentarena.report import build_report
from agentarena.schemas import SampleRecord
from agentarena.scoring.aggregate import is_tie, leaderboard, paired_samples
from agentarena.scoring.judge import cohen_kappa, format_answer, pairwise_judge, rubric_judge
from agentarena.store import RunStore
from agentarena.tasks import REGISTRY

app = typer.Typer(add_completion=False, no_args_is_help=True, help="Grade agents on tool use and research.")
RUNS = ROOT / "runs"
CACHE = ROOT / ".arena_cache"


@app.callback()
def _env() -> None:
    load_dotenv(ROOT / ".env")


def _run_dir(run_id: str) -> Path:
    d = RUNS / run_id
    if not (d / "manifest.json").exists():
        raise typer.BadParameter(f"no run '{run_id}' in {RUNS}")
    return d


@app.command()
def run(
    models: Annotated[str, typer.Option("--models", "-m", help="Comma-separated names from config/models.yaml")],
    suite: Annotated[str, typer.Option("--suite", "-s", help="Suite name in config/suites or a YAML path")] = "smoke",
    run_id: Annotated[str | None, typer.Option(help="Reuse an id to resume an interrupted run")] = None,
    no_cache: Annotated[bool, typer.Option("--no-cache", help="Bypass the response cache")] = False,
    report: Annotated[bool, typer.Option(help="Build the HTML report when done")] = True,
) -> None:
    """Run models on a suite. Re-running with the same --run-id resumes and skips finished samples."""
    from agentarena.runner import run_suite

    registry = load_models()
    names = [n.strip() for n in models.split(",") if n.strip()]
    missing = [n for n in names if n not in registry]
    if missing:
        raise typer.BadParameter(f"unknown models {missing}; known: {', '.join(registry)}")
    s = load_suite(suite)
    run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{s.name}"
    try:
        store = asyncio.run(run_suite(s, [registry[n] for n in names], RUNS / run_id,
                                      None if no_cache else CACHE, print_progress))
    except ProviderError as e:
        typer.echo(f"error: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\nrun: {run_id}  ({store.samples})")
    if report:
        print_board(store)
        typer.echo(f"report: {build_report(store)}")


def print_progress(rec: SampleRecord, done: int, total: int) -> None:
    if rec.trajectory.stop_reason == "error":
        mark = typer.style("ERR ", fg="yellow")
    else:
        mark = typer.style("PASS", fg="green") if rec.outcome.success else typer.style("FAIL", fg="red")
    extra = f"  [{rec.trajectory.stop_reason}: {rec.trajectory.error}]" if rec.trajectory.error else ""
    typer.echo(f"[{done:>4}/{total}] {mark} {rec.model:<20} {rec.task_id:<32} r{rec.repeat} "
               f"{int(rec.metrics['steps']):>2} steps ${rec.cost_usd:.4f}{extra}")


def print_board(store: RunStore) -> None:
    board = leaderboard(store.load())
    typer.echo(f"\n{'rank':<5}{'model':<22}{'elo':>6}{'95% CI':>14}{'pass@1':>9}{'tool':>7}{'resch':>7}"
               f"{'pass^k':>8}{'$/solved':>10}{'n':>6}{'errors':>8}")
    k = board["repeats"]
    for r in board["rows"]:
        cps = r["cost_per_solved"]
        ci = "{:.0f}-{:.0f}".format(r["lo"], r["hi"])
        cost = "-" if cps != cps else "${:.4f}".format(cps)
        typer.echo(
            f"{r['rank']:<5}{r['model']:<22}{r['elo']:>6.0f}{ci:>14}"
            f"{r['pass1_all']['mean']:>9.0%}{r['pass1_tool_use']['mean']:>7.0%}{r['pass1_research']['mean']:>7.0%}"
            f"{r['pass_hat_k'][k]:>8.0%}{cost:>10}{r['samples']:>6}{r['excluded_errors']:>8}"
        )
    for m, n in board["unscored"].items():
        typer.echo(f"-    {m:<22} not scored: all {n} samples hit API errors (see trace viewer / samples.jsonl)")
    if any(r["excluded_errors"] for r in board["rows"]) or board["unscored"]:
        typer.echo("API-error samples are excluded from scores; re-run with the same --run-id to retry them.")


@app.command()
def report(run_id: str, open_browser: Annotated[bool, typer.Option("--open")] = False) -> None:
    """(Re)build the HTML report for a run (offline, no API calls)."""
    store = RunStore(_run_dir(run_id))
    print_board(store)
    out = build_report(store)
    typer.echo(f"report: {out}")
    if open_browser:
        webbrowser.open(out.as_uri())


@app.command("rescore")
def rescore_cmd(run_id: str) -> None:
    """Re-grade a finished run with the current graders by replaying recorded tool calls (no API calls).
    Don't run this while the same run is still executing."""
    from agentarena.runner import rescore

    store = RunStore(_run_dir(run_id))
    old = store.load()
    new = [rescore(r) for r in old]
    changed = [(a, b) for a, b in zip(old, new) if a.outcome.success != b.outcome.success]
    for a, b in changed:
        typer.echo(f"  {a.sample_id}: {'PASS' if a.outcome.success else 'FAIL'} -> {'PASS' if b.outcome.success else 'FAIL'}")
    store.samples.write_text("".join(r.model_dump_json() + "\n" for r in new), encoding="utf-8")
    typer.echo(f"rescored {len(new)} samples, {len(changed)} changed")


@app.command()
def runs() -> None:
    """List runs."""
    for d in sorted(RUNS.glob("*/manifest.json")):
        m = json.loads(d.read_text(encoding="utf-8"))
        n = sum(1 for _ in (d.parent / "samples.jsonl").open(encoding="utf-8")) if (d.parent / "samples.jsonl").exists() else 0
        typer.echo(f"{m['run_id']:<36} {n:>5} samples  models: {', '.join(x['name'] for x in m['models'])}")


@app.command("list")
def list_things() -> None:
    """List task templates and configured models."""
    typer.echo("templates:")
    for name, t in REGISTRY.items():
        typer.echo(f"  {t.domain:<9} {name}")
    typer.echo("models:")
    for name, c in load_models().items():
        typer.echo(f"  {name:<22} {c.provider:<14} ${c.price.input}/{c.price.output} per Mtok")


def _submission(r: SampleRecord) -> tuple[str, list[dict[str, Any]]]:
    for c in reversed(r.trajectory.tool_calls):
        if c.name == "submit_answer":
            return str(c.arguments.get("answer", "")), list(c.arguments.get("citations") or [])
    return r.trajectory.final_answer, []


@app.command()
def judge(
    run_id: str,
    judge_model: Annotated[str, typer.Option("--judge-model", "-j")],
    pairwise: Annotated[bool, typer.Option(help="Also break tied battles with position-swapped pairwise judging")] = False,
    concurrency: int = 4,
) -> None:
    """LLM-judge the research answers of a run (rubric score, optional pairwise tie-breaks)."""
    store = RunStore(_run_dir(run_id))
    registry = load_models()
    jcfg = registry[judge_model]
    jm = make_model(jcfg)
    records = [r for r in store.load() if r.domain == "research"]
    family = {n: c.model_family for n, c in registry.items()}
    skipped = sorted({r.model for r in records if family.get(r.model) == jcfg.model_family})
    if skipped:
        typer.echo(f"skipping same-family contestants (self-preference bias): {', '.join(skipped)}")
    records = [r for r in records if r.model not in skipped]
    seen = {(j["kind"], j.get("sample_id") or (j["task_id"], j["repeat"], j["a"], j["b"])) for j in store.load_judgments()}
    sem = asyncio.Semaphore(concurrency)

    async def do_rubric(r: SampleRecord) -> None:
        if ("rubric", r.sample_id) in seen:
            return
        answer, cites = _submission(r)
        async with sem:
            out = await rubric_judge(jm, r.prompt, str(r.outcome.details.get("expected")), answer, cites)
        store.append_judgment({"kind": "rubric", "sample_id": r.sample_id, "judge": judge_model, **out})

    async def do_pair(a: SampleRecord, b: SampleRecord) -> None:
        if ("pairwise", (a.task_id, a.repeat, a.model, b.model)) in seen:
            return
        async with sem:
            out = await pairwise_judge(jm, a.prompt, str(a.outcome.details.get("expected")),
                                       format_answer(*_submission(a)), format_answer(*_submission(b)))
        store.append_judgment({"kind": "pairwise", "task_id": a.task_id, "repeat": a.repeat, "a": a.model,
                               "b": b.model, "outcome": out, "judge": judge_model})

    async def main() -> None:
        jobs = [do_rubric(r) for r in records]
        if pairwise:
            jobs += [do_pair(a, b) for a, b in paired_samples(records) if is_tie(a, b)]
        await asyncio.gather(*jobs)

    asyncio.run(main())
    typer.echo(f"judgments: {store.judgments}")


@app.command("validate-judge")
def validate_judge(
    judge_model: Annotated[str, typer.Option("--judge-model", "-j")],
    labels: Path = ROOT / "data" / "judge_labels.jsonl",
) -> None:
    """Measure judge agreement with human labels (Cohen's kappa) before trusting it."""
    jm = make_model(load_models()[judge_model])
    items = [json.loads(x) for x in labels.read_text(encoding="utf-8").splitlines() if x.strip()]

    async def main() -> list[dict[str, Any]]:
        sem = asyncio.Semaphore(4)

        async def one(it: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                return await rubric_judge(jm, it["question"], it["reference"], it["answer"], it.get("citations", []))

        return await asyncio.gather(*(one(it) for it in items))

    outs = asyncio.run(main())
    human = [bool(it["human_correct"]) for it in items]
    pred = [o["correct"] for o in outs]
    acc = sum(h == p for h, p in zip(human, pred)) / len(items)
    typer.echo(f"items: {len(items)}  accuracy: {acc:.0%}  cohen's kappa: {cohen_kappa(human, pred):.2f}")
    typer.echo("(kappa > 0.8 is strong agreement; < 0.6 means do not trust this judge)")
    for it, o, h in zip(items, outs, human):
        if o["correct"] != h:
            typer.echo(f"  disagree: human={h} judge={o['correct']}  {it['question']} -> {it['answer']!r}  ({o['rationale']})")


if __name__ == "__main__":
    app()
