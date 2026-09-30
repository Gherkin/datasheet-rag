"""Bootstrap confidence intervals for eval means and paired differences (GH #94).

A category holds a handful of items, so one query moving from rank 1 to 2
shifts its MRR a lot. A point mean hides that; an interval shows it.

Rephrasings of one information need share a ``need_id`` and succeed or fail
together, so they are not independent draws. The bootstrap resamples whole
needs (a cluster bootstrap), not single queries; resampling queries would
count one need's four rephrasings as four pieces of evidence and give an
interval that is too narrow.

The interval is the percentile bootstrap: resample clusters with
replacement, take the item mean of each resample, and read the 2.5th and
97.5th percentiles. The seed is fixed so a report reproduces exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel

#: Resamples per interval. Enough that the 2.5% tail rests on 50 draws.
N_RESAMPLES = 2000
LEVEL = 0.95
SEED = 0

Verdict = Literal["better", "worse", "no change", "n too small"]


class Interval(BaseModel):
    """A two-sided bootstrap interval around a mean."""

    lo: float
    hi: float

    def spans_zero(self) -> bool:
        return self.lo <= 0.0 <= self.hi


def _cluster_indices(clusters: Sequence[str]) -> list[np.ndarray]:
    order: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        order.setdefault(c, []).append(i)
    return [np.asarray(ix) for ix in order.values()]


def bootstrap_ci(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    n_resamples: int = N_RESAMPLES,
    level: float = LEVEL,
    seed: int = SEED,
) -> Interval | None:
    """Percentile cluster-bootstrap interval for the item mean of ``values``.

    ``clusters[i]`` names the cluster item ``i`` belongs to. None when there
    are fewer than two clusters: one cluster resamples to itself and would
    give a zero-width interval that claims certainty it does not have.
    """
    if len(values) != len(clusters):
        raise ValueError(f"{len(values)} values but {len(clusters)} cluster labels")
    groups = _cluster_indices(clusters)
    if len(groups) < 2:
        return None
    vals = np.asarray(values, dtype=float)
    sums = np.array([vals[g].sum() for g in groups])
    sizes = np.array([len(g) for g in groups], dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(n_resamples, len(groups)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(means, [tail, 100.0 - tail])
    return Interval(lo=float(lo), hi=float(hi))


class PairedDiff(BaseModel):
    """Mean of ``b - a`` over the same items, with its interval."""

    diff: float
    ci: Interval | None
    verdict: Verdict


def paired_diff(
    a: Sequence[float],
    b: Sequence[float],
    clusters: Sequence[str],
    **kw: object,
) -> PairedDiff:
    """Compare two runs scored on the same items, in the same order.

    The pairing is what makes a small set usable: both runs meet the same
    hard and easy items, so the per-item difference cancels item difficulty.
    A direction is claimed only when the interval excludes zero.
    """
    if len(a) != len(b):
        raise ValueError(f"paired runs differ in length: {len(a)} vs {len(b)}")
    deltas = [y - x for x, y in zip(a, b)]
    diff = sum(deltas) / len(deltas) if deltas else 0.0
    if deltas and all(d == 0.0 for d in deltas):
        return PairedDiff(diff=0.0, ci=Interval(lo=0.0, hi=0.0), verdict="no change")
    ci = bootstrap_ci(deltas, clusters, **kw)  # type: ignore[arg-type]
    if ci is None or ci.spans_zero():
        verdict: Verdict = "n too small"
    else:
        verdict = "better" if diff > 0 else "worse"
    return PairedDiff(diff=diff, ci=ci, verdict=verdict)
