"""Numeric helpers shared by saved-evaluation reporting scripts."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping


REWARD_ABS_TOLERANCE = 1e-9


def reward_is(value: float, target: float) -> bool:
    """Compare rubric rewards without assuming decimal tenths are exact floats."""
    return math.isclose(
        float(value), float(target), rel_tol=0.0, abs_tol=REWARD_ABS_TOLERANCE
    )


def reward_matches_any(value: float, targets: Iterable[float]) -> bool:
    return any(reward_is(value, target) for target in targets)


def count_matching_rewards(rows: Iterable[Mapping[str, float]], targets: Iterable[float]) -> int:
    targets = tuple(targets)
    return sum(reward_matches_any(row["reward"], targets) for row in rows)


def reward_distribution(rows: Iterable[Mapping[str, float]]) -> Counter[float]:
    """Group floating representations of the same rubric tenth together."""
    return Counter(round(float(row["reward"]), 9) for row in rows)


def format_distribution(distribution: Mapping[float, int]) -> str:
    return " ".join(
        f"{score:.3f}:{distribution[score]}" for score in sorted(distribution)
    )
