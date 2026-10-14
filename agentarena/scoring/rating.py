"""Bradley-Terry ratings from pairwise "battles", reported on the Elo scale.

Like LMSYS Chatbot Arena, we fit BT by maximum likelihood rather than running
online Elo updates, which depend on match order. Here a battle is two models
attempting the same task instance on the same repeat index. CIs come from
bootstrapping battles, and ranks follow the Arena convention: a model's rank is
1 + the number of models that are *statistically* better (their lower bound
is above its upper bound).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np


@dataclass
class Battle:
    a: str
    b: str
    outcome: float  # 1 = a wins, 0 = b wins, 0.5 = tie


def fit_bt(battles: list[Battle], models: list[str], iters: int = 200, prior: float = 0.5) -> dict[str, float]:
    """MM algorithm (Hunter 2004). Ties count as half a win each. ``prior`` adds a
    virtual tie between every pair so undefeated models stay finite."""
    if len(models) < 2:  # a rating is relative; one model has nothing to be compared with
        return {m: 1000.0 for m in models}
    idx = {m: i for i, m in enumerate(models)}
    n = len(models)
    wins = np.full((n, n), prior / 2)
    np.fill_diagonal(wins, 0)
    for bt in battles:
        i, j = idx[bt.a], idx[bt.b]
        wins[i, j] += bt.outcome
        wins[j, i] += 1 - bt.outcome
    games = wins + wins.T
    p = np.ones(n)
    for _ in range(iters):
        denom = (games / (p[:, None] + p[None, :])).sum(axis=1)
        p_new = wins.sum(axis=1) / denom
        p_new /= np.exp(np.log(p_new).mean())
        if np.allclose(p, p_new, rtol=1e-9):
            break
        p = p_new
    elo = 400 * np.log10(p)
    elo += 1000 - elo.mean()
    return {m: float(elo[idx[m]]) for m in models}


def bootstrap_ratings(
    battles: list[Battle], models: list[str], rounds: int = 500, seed: int = 0
) -> dict[str, dict[str, float]]:
    point = fit_bt(battles, models)
    rng = np.random.default_rng(seed)
    samples = defaultdict(list)
    if battles:
        for _ in range(rounds):
            pick = rng.integers(0, len(battles), size=len(battles))
            for m, r in fit_bt([battles[i] for i in pick], models).items():
                samples[m].append(r)
    out = {}
    for m in models:
        s = samples[m] or [point[m]]
        lo, hi = np.quantile(s, [0.025, 0.975])
        out[m] = {"elo": point[m], "lo": float(lo), "hi": float(hi)}
    for m in models:
        out[m]["rank"] = 1 + sum(1 for o in models if o != m and out[o]["lo"] > out[m]["hi"])
    return out


def win_prob(elo_a: float, elo_b: float) -> float:
    return 1 / (1 + math.pow(10, (elo_b - elo_a) / 400))
