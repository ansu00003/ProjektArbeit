"""Simulated auditor for HITL experiments.

Real auditors are not available, so feedback is simulated from ground-truth
labels. This module supports a configurable noise rate so we can study how
robust each HITL strategy is to imperfect human input — a key dimension of
the evaluation that previous versions of the project did not cover.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class SimulatedAuditor:
    """Provides noisy ground-truth labels.

    Args:
        noise_rate: probability that the auditor flips the true label.
                    0.0 = perfect auditor, 0.3 = 30% mistakes.
        seed: RNG seed for reproducible noise.
        label_col: ground-truth column name in the input frame.
    """

    noise_rate: float = 0.0
    seed: int = 0
    label_col: str = "label"

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def label(self, entries: pd.DataFrame) -> np.ndarray:
        """Return labels (0/1) the simulated auditor assigns to `entries`.

        Each entry's true label is independently flipped with probability
        `noise_rate`.
        """
        if self.label_col not in entries.columns:
            raise KeyError(f"Auditor needs ground-truth column '{self.label_col}'.")
        truth = entries[self.label_col].astype(int).to_numpy()
        if self.noise_rate <= 0:
            return truth
        flip = self._rng.random(size=len(truth)) < self.noise_rate
        noisy = np.where(flip, 1 - truth, truth)
        return noisy
