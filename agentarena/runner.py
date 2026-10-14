"""Executes model x task x repeat concurrently, with per-provider rate limits,
hard budgets, a disk cache and resumable JSONL output."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from agentarena.agents import Model, ModelConfig, ProviderError, make_model
from agentarena.agents.cache import CachedModel, DiskCache
from agentarena.agents.loop import Budget, run_agent
from agentarena.config import Suite
from agentarena.schemas import SampleRecord, Score, Trajectory
from agentarena.scoring.trajectory import trajectory_metrics
from agentarena.store import RunStore
from agentarena.tasks import Task, build_tasks


def sample_id(model: str, task: Task, repeat: int) -> str:
    return f"{model}|{task.task_id}|r{repeat}"


async def run_sample(
    model: Model, task: Task, repeat: int, budget: Budget, run_id: str, cache: DiskCache | None
) -> SampleRecord:
    env = task.build_env()
    salt = f"{task.task_id}|r{repeat}"
    agent = model.for_sample(task, env, salt)
    if cache is not None and model.cfg.provider != "fake":
        agent = CachedModel(agent, cache, salt)

    traj = Trajectory()
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(run_agent(agent, task, env, budget, traj), timeout=budget.timeout_s)
    except TimeoutError:
        traj.stop_reason = "timeout"
    except ProviderError as e:
        traj.stop_reason, traj.error = "error", str(e)
    except Exception as e:  # noqa: BLE001 - one bad sample must not kill the run
        traj.stop_reason, traj.error = "error", f"{type(e).__name__}: {e}"
    traj.wall_time_s = time.perf_counter() - t0

    try:
        outcome = task.score(env, traj)
    except Exception as e:  # noqa: BLE001
        outcome = Score(success=False, score=0.0, details={"grader_error": f"{type(e).__name__}: {e}"})
    return SampleRecord(
        run_id=run_id,
        sample_id=sample_id(model.name, task, repeat),
        model=model.name,
        task_id=task.task_id,
        template=task.template,
        domain=task.domain,
        seed=task.seed,
        repeat=repeat,
        prompt=task.prompt,
        trajectory=traj,
        outcome=outcome,
        metrics=trajectory_metrics(traj),
        cost_usd=model.cfg.cost(traj.usage.input_tokens, traj.usage.output_tokens),
    )


def rescore(rec: SampleRecord) -> SampleRecord:
    """Re-grade a recorded sample with the current grader, without calling any model.

    Tools are deterministic, so replaying the recorded tool calls on a fresh copy of
    the task's world reproduces the final state exactly.
    """
    from agentarena.tasks import REGISTRY

    task = REGISTRY[rec.template](rec.seed)
    env = task.build_env()
    for call in rec.trajectory.tool_calls:
        env.execute(call)
    try:
        outcome = task.score(env, rec.trajectory)
    except Exception as e:  # noqa: BLE001
        outcome = Score(success=False, score=0.0, details={"grader_error": f"{type(e).__name__}: {e}"})
    return rec.model_copy(update={"outcome": outcome, "prompt": task.prompt})


async def run_suite(
    suite: Suite,
    model_cfgs: list[ModelConfig],
    run_dir: Path,
    cache_dir: Path | None,
    on_done: Callable[[SampleRecord, int, int], None] | None = None,
) -> RunStore:
    models = [make_model(c) for c in model_cfgs]  # fail on missing keys before creating the run
    tasks = build_tasks(suite.templates, suite.seeds)
    store = RunStore(run_dir)
    store.init(suite.model_dump(mode="json"),
               [c.model_dump(mode="json", exclude={"api_key_env"}) for c in model_cfgs])
    store.repair()
    done = store.done_ids()
    cache = DiskCache(cache_dir) if cache_dir else None

    # One semaphore per endpoint, so two models on the same provider share its rate limit.
    sems: dict[str, asyncio.Semaphore] = {}
    for c in model_cfgs:
        sems.setdefault(f"{c.provider}|{c.base_url}", asyncio.Semaphore(c.concurrency))

    jobs = [
        (m, t, r)
        for r in range(suite.repeats)
        for t in tasks
        for m in models
        if sample_id(m.name, t, r) not in done
    ]
    total, finished = len(jobs), 0

    async def one(m: Model, t: Task, r: int) -> None:
        nonlocal finished
        async with sems[f"{m.cfg.provider}|{m.cfg.base_url}"]:
            rec = await run_sample(m, t, r, suite.budget, run_dir.name, cache)
        store.append(rec)
        finished += 1
        if on_done:
            on_done(rec, finished, total)

    await asyncio.gather(*(one(*j) for j in jobs))
    return store
