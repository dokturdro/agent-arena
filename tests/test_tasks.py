import asyncio

import pytest

from agentarena.agents import ModelConfig, make_model
from agentarena.agents.loop import Budget, run_agent
from agentarena.schemas import ToolCall, Trajectory
from agentarena.tasks import REGISTRY, build_tasks

SEEDS = list(range(1, 16))


def run(task, policy: str):
    model = make_model(ModelConfig(name=f"fake-{policy}", provider="fake", model=policy))
    env = task.build_env()
    traj = Trajectory()
    asyncio.run(run_agent(model.for_sample(task, env, "s"), task, env, Budget(), traj))
    return env, traj, task.score(env, traj)


@pytest.mark.parametrize("template", sorted(REGISTRY))
def test_oracle_solves_every_seed(template):
    for task in build_tasks([template], SEEDS):
        _, traj, score = run(task, "oracle")
        assert score.success, (task.task_id, score.details)
        assert score.score == pytest.approx(1.0)
        assert not any(r.is_error for r in traj.tool_results)


@pytest.mark.parametrize("template", sorted(REGISTRY))
def test_doing_nothing_fails(template):
    if template == "infeasible_purchase":
        pytest.skip("doing nothing is the correct behaviour here")
    for task in build_tasks([template], SEEDS[:5]):
        _, _, score = run(task, "lazy")
        assert not score.success, task.task_id


def test_tasks_are_deterministic_per_seed():
    a, b = build_tasks("all", [7]), build_tasks("all", [7])
    assert [t.prompt for t in a] == [t.prompt for t in b]
    w1 = build_tasks(["buy_cheapest_in_stock"], [1])[0].world0.canonical()
    w2 = build_tasks(["buy_cheapest_in_stock"], [2])[0].world0.canonical()
    assert w1 != w2


def test_side_effect_breaks_success():
    task = build_tasks(["buy_cheapest_in_stock"], [3])[0]
    env = task.build_env()
    for i, (name, args) in enumerate(task.oracle(env).calls):
        env.execute(ToolCall(id=str(i), name=name, arguments=args))
    env.execute(ToolCall(id="x", name="send_email", arguments={"to": "x@y.z", "subject": "hi", "body": "spam"}))
    score = task.score(env, Trajectory())
    assert not score.success and "no_stray_email" in score.details["failed"]


def test_wrong_research_answer_and_fake_citation():
    task = build_tasks(["founding_year"], [2])[0]
    env = task.build_env()
    env.execute(ToolCall(id="1", name="submit_answer", arguments={
        "answer": str(task.answer + 1),
        "citations": [{"doc_id": task.gold[0], "quote": "This sentence is not in the document at all."}],
    }))
    s = task.score(env, Trajectory())
    assert not s.success and s.details["citation_precision"] == 0.0


def test_correct_but_ungrounded_answer_is_not_success():
    task = build_tasks(["founding_year"], [2])[0]
    env = task.build_env()
    env.execute(ToolCall(id="1", name="submit_answer", arguments={"answer": str(task.answer), "citations": []}))
    s = task.score(env, Trajectory())
    assert s.details["correct"] and not s.success


def test_schema_violation_is_protocol_error():
    task = build_tasks(["buy_cheapest_in_stock"], [1])[0]
    env = task.build_env()
    r = env.execute(ToolCall(id="1", name="place_order", arguments={"sku": "SKU-1"}))
    assert r.is_error and r.error_kind == "schema"
    r = env.execute(ToolCall(id="2", name="teleport", arguments={}))
    assert r.error_kind == "unknown_tool"
