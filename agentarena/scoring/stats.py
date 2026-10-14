"""Success-rate statistics with honest uncertainty.

* pass@1 with a *clustered* bootstrap CI: repeats of the same task instance are
  correlated, so we resample task instances, not individual runs.
* pass^k (tau-bench): probability that k independent attempts at the same task
  ALL succeed. It drops fast for flaky agents even when pass@1 looks fine.
* pass@k: probability that at least one of k attempts succeeds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pass_hat_k(c: int, n: int, k: int) -> float:
    """Unbiased estimate of P(all k of k attempts succeed) from c successes in n tries."""
    return math.comb(c, k) / math.comb(n, k) if n >= k else float("nan")


def pass_at_k(c: int, n: int, k: int) -> float:
    """Unbiased estimate of P(at least one of k attempts succeeds) (Chen et al., 2021)."""
    if n < k:
        return float("nan")
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def cluster_bootstrap_ci(
    per_task: Sequence[Sequence[float]], iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean over tasks of per-task mean; CI by resampling tasks with replacement."""
    if not per_task:
        return (float("nan"),) * 3
    means = np.array([np.mean(v) for v in per_task])
    rng = np.random.default_rng(seed)
    boots = means[rng.integers(0, len(means), size=(iters, len(means)))].mean(axis=1)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(means.mean()), float(lo), float(hi)
