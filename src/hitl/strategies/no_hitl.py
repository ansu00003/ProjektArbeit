"""Baseline: no human in the loop — scores never change."""

from __future__ import annotations

import numpy as np

from .base import BaseHITLStrategy


class NoHITLStrategy(BaseHITLStrategy):
    """Returns the unsupervised base scores forever. Used as ablation baseline."""

    def __init__(self) -> None:
        super().__init__(name="no_hitl")

    def score(self) -> np.ndarray:
        return self.base_scores.copy()

    def update(self, indices, labels) -> None:  # noqa: D401 — record but ignore
        super().update(indices, labels)  # still track so review counts work
