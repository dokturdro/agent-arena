import asyncio

from agentarena.agents import ModelConfig
from agentarena.agents.base import Model
from agentarena.agents.cache import CachedModel, DiskCache
from agentarena.agents.loop import Budget
from agentarena.config import Suite
from agentarena.report import build_report
from agentarena.runner import run_suite
from agentarena.schemas import Completion, Message, Usage
from agentarena.scoring.aggregate import leaderboard

FAKES = [
    ModelConfig(name="fake-oracle", provider="fake", model="oracle", price={"input": 1, "output": 5}),
    ModelConfig(name="fake-noisy", provider="fake", model="noisy", params={"p_fail": 0.5}),
    ModelConfig(name="fake-lazy", provider="fake", model="lazy"),
]
SUITE = Suite(name="t", templates=["buy_cheapest_in_stock", "founding_year", "infeasible_purchase"],
              seeds=[1, 2], repeats=2, budget=Budget(max_steps=8))


def test_end_to_end_run_report_and_resume(tmp_path):
    run_dir = tmp_path / "run"
    store = asyncio.run(run_suite(SUITE, FAKES, run_dir, None))
    recs = store.load()
    assert len(recs) == 3 * 3 * 2 * 2
    board = leaderboard(recs)
    order = [r["model"] for r in board["rows"]]
    assert order[0] == "fake-oracle" and order[-1] == "fake-lazy"
    assert board["rows"][0]["pass1_all"]["mean"] == 1.0

    # Resume: nothing left to do, no duplicates.
    asyncio.run(run_suite(SUITE, FAKES, run_dir, None))
    assert len(store.load()) == len(recs)

    # Drop half the samples and resume: only the missing ones are re-run.
    lines = store.samples.read_text(encoding="utf-8").splitlines()
    store.samples.write_text("\n".join(lines[:10]) + "\n" + lines[10][:40], encoding="utf-8")  # torn last line
    asyncio.run(run_suite(SUITE, FAKES, run_dir, None))
    assert len({r.sample_id for r in store.load()}) == len(recs) == len(store.load())

    html = build_report(store).read_text(encoding="utf-8")
    assert "fake-oracle" in html and "/*__DATA__*/" not in html


def test_api_errors_are_excluded_and_retried(tmp_path):
    store = asyncio.run(run_suite(SUITE, FAKES[:2], tmp_path / "run", None))
    recs = store.load()
    broken = recs[0].model_copy(deep=True)
    broken.trajectory.stop_reason, broken.trajectory.error = "error", "RateLimitError: quota"
    broken.outcome.success = False
    store.append(broken)  # supersedes the good record for that sample id

    assert broken.sample_id not in store.done_ids()
    board = leaderboard(store.load())
    row = next(r for r in board["rows"] if r["model"] == broken.model)
    assert row["excluded_errors"] == 1 and row["samples"] == len(recs) // 2 - 1

    asyncio.run(run_suite(SUITE, FAKES[:2], tmp_path / "run", None))  # resume retries it
    assert all(r.trajectory.stop_reason != "error" for r in store.load())
    assert len(store.load()) == len(recs)


def test_rescore_reproduces_outcomes(tmp_path):
    from agentarena.runner import rescore

    store = asyncio.run(run_suite(SUITE, FAKES, tmp_path / "run", None))
    for rec in store.load():
        again = rescore(rec)
        assert again.outcome.success == rec.outcome.success, rec.sample_id
        assert again.outcome.score == rec.outcome.score


def test_parenthetical_justification_is_not_hedging():
    from agentarena.tasks import build_tasks

    task = build_tasks(["compare_revenue"], [1])[0]
    win, lose = task.answer, task.forbidden[0]
    assert task.answer_correct(f"{win} ($659.2 million vs. {lose}'s $57.0 million)")
    assert not task.answer_correct(f"{win} or {lose}")
    assert not task.answer_correct(lose)


class CountingModel(Model):
    calls = 0

    async def complete(self, messages, tools, *, max_tokens):
        CountingModel.calls += 1
        return Completion(text="hi", usage=Usage(input_tokens=1, output_tokens=1))


def test_cache_hits_and_salt_isolation(tmp_path):
    inner = CountingModel(ModelConfig(name="x", provider="test", model="x"))
    cache = DiskCache(tmp_path)
    msgs = [Message(role="user", content="q")]
    a = CachedModel(inner, cache, "task#1|r0")
    asyncio.run(a.complete(msgs, [], max_tokens=10))
    second = asyncio.run(a.complete(msgs, [], max_tokens=10))
    assert CountingModel.calls == 1 and second.cached
    asyncio.run(CachedModel(inner, cache, "task#1|r1").complete(msgs, [], max_tokens=10))
    assert CountingModel.calls == 2  # a different repeat is an independent sample
