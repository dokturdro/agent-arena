import math

import pytest

from agentarena.schemas import ToolCall, ToolResult, Step, Trajectory
from agentarena.scoring.judge import cohen_kappa
from agentarena.scoring.rating import Battle, bootstrap_ratings, fit_bt, win_prob
from agentarena.scoring.stats import cluster_bootstrap_ci, pass_at_k, pass_hat_k, wilson
from agentarena.scoring.trajectory import trajectory_metrics


def test_pass_hat_k_and_pass_at_k():
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(2, 3, 3) == 0.0
    assert pass_hat_k(2, 4, 2) == pytest.approx(1 / 6)
    assert pass_at_k(1, 4, 2) == pytest.approx(0.5)
    assert pass_at_k(0, 3, 3) == 0.0
    assert math.isnan(pass_hat_k(1, 1, 2))


def test_wilson_bounds():
    lo, hi = wilson(5, 10)
    assert 0.2 < lo < 0.5 < hi < 0.8
    assert wilson(0, 10)[0] == 0.0


def test_cluster_bootstrap_contains_mean():
    mean, lo, hi = cluster_bootstrap_ci([[1, 1, 1], [0, 0, 0], [1, 0, 1], [1, 1, 0]])
    assert lo <= mean <= hi and mean == pytest.approx(7 / 12)


def test_bt_recovers_ordering_and_win_probability():
    # a beats b 75% of the time, b beats c 75% of the time.
    battles = [Battle("a", "b", 1)] * 75 + [Battle("a", "b", 0)] * 25 + [Battle("b", "c", 1)] * 75 + [Battle("b", "c", 0)] * 25
    r = fit_bt(battles, ["a", "b", "c"], prior=0)
    assert r["a"] > r["b"] > r["c"]
    assert win_prob(r["a"], r["b"]) == pytest.approx(0.75, abs=0.02)
    assert sum(r.values()) / 3 == pytest.approx(1000)


def test_bt_undefeated_stays_finite_and_ranks():
    battles = [Battle("a", "b", 1)] * 20 + [Battle("a", "b", 0.5)] * 2
    out = bootstrap_ratings(battles, ["a", "b"], rounds=100)
    assert math.isfinite(out["a"]["elo"]) and out["a"]["rank"] == 1 and out["b"]["rank"] == 2


def test_ties_give_equal_ratings():
    out = fit_bt([Battle("a", "b", 0.5)] * 10, ["a", "b"])
    assert out["a"] == pytest.approx(out["b"])


def test_trajectory_metrics_counts_errors_redundancy_recovery():
    c = lambda i, n, a: ToolCall(id=str(i), name=n, arguments=a)  # noqa: E731
    r = lambda i, n, err=None: ToolResult(call_id=str(i), name=n, content="x", is_error=err is not None, error_kind=err)  # noqa: E731
    traj = Trajectory(steps=[
        Step(index=0, tool_calls=[c(1, "place_order", {"sku": "A"})], tool_results=[r(1, "place_order", "schema")]),
        Step(index=1, tool_calls=[c(2, "place_order", {"sku": "A", "quantity": 1})], tool_results=[r(2, "place_order")]),
        Step(index=2, tool_calls=[c(3, "list_orders", {})], tool_results=[r(3, "list_orders")]),
        Step(index=3, tool_calls=[c(4, "list_orders", {})], tool_results=[r(4, "list_orders")]),
    ])
    m = trajectory_metrics(traj)
    assert m["invalid_calls"] == 1 and m["invalid_call_rate"] == 0.25
    assert m["redundant_calls"] == 1
    assert m["recovery_rate"] == 1.0
    assert m["looped"] == 0.0


def test_cohen_kappa():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, True, False, False], [True, False, True, False]) == pytest.approx(0.0)
