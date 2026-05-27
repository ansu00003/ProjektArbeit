"""Common interface for HITL strategies.

A strategy is a small object that:
  1. Receives an initial unsupervised anomaly score for every entry.
  2. Selects which entries the auditor should review next (`select_batch`).
  3. Updates its internal state with the auditor's labels (`update`).
  4. Produces a refined anomaly score for every entry (`score`).

The experiment runner calls these in a loop to produce learning curves.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Set

import numpy as np
import pandas as pd


@dataclass
class ReviewBatch:
    """A batch of entries the strategy wants the auditor to label."""
    indices: np.ndarray  # integer positions into the working frame
    reason: str = "selected"


@dataclass
class BaseHITLStrategy(ABC):
    """Abstract base class for all HITL strategies.

    Subclasses MUST implement `score`. They MAY override `select_batch`
    (default: highest unreviewed scores) and `update` (default: no-op).
    """

    name: str = "base"
    reviewed_indices: Set[int] = field(default_factory=set)
    feedback_labels: dict = field(default_factory=dict)  # idx -> 0/1

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def initialize(
        self,
        X: pd.DataFrame,
        base_scores: np.ndarray,
        original_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Called once before the HITL loop starts.

        Args:
            X: feature matrix used by the base detector.
            base_scores: initial anomaly scores (higher = more anomalous).
            original_df: full original frame (for column-aware strategies).
        """
        self.X = X
        self.base_scores = np.asarray(base_scores, dtype=float)
        self.original_df = original_df
        self.reviewed_indices = set()
        self.feedback_labels = {}

    # ------------------------------------------------------------------ #
    # core API
    # ------------------------------------------------------------------ #
    @abstractmethod
    def score(self) -> np.ndarray:
        """Return current per-entry anomaly score (higher = more anomalous)."""

    def select_batch(self, n: int) -> ReviewBatch:
        """Default: pick the top-n unreviewed entries by current score."""
        scores = self.score()
        order = np.argsort(-scores)
        picked = [int(i) for i in order if int(i) not in self.reviewed_indices][:n]
        return ReviewBatch(indices=np.array(picked, dtype=int), reason="top_score")

    def update(self, indices: np.ndarray, labels: np.ndarray) -> None:
        """Record auditor feedback. Override to retrain internal models."""
        for idx, lbl in zip(indices.tolist(), labels.tolist()):
            self.reviewed_indices.add(int(idx))
            self.feedback_labels[int(idx)] = int(lbl)

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    @property
    def n_reviewed(self) -> int:
        return len(self.reviewed_indices)

    def feedback_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, labels) of all feedback collected so far."""
        if not self.feedback_labels:
            return np.array([], dtype=int), np.array([], dtype=int)
        items = sorted(self.feedback_labels.items())
        idx = np.array([i for i, _ in items], dtype=int)
        lbl = np.array([l for _, l in items], dtype=int)
        return idx, lbl
