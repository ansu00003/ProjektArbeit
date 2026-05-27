"""Threshold-Adjustment HITL strategy.

If the auditor's recent feedback shows too many false positives, the
decision threshold is shifted upward (fewer entries flagged); if too many
false negatives, it is shifted downward. Scores themselves are unchanged —
only the implicit cut-off the dashboard uses for "flag vs. not".

This is the classic, lightest-touch HITL approach.
"""

from __future__ import annotations

import numpy as np

from .base import BaseHITLStrategy


class ThresholdAdjustmentStrategy(BaseHITLStrategy):
    """Shifts a virtual threshold on the base score using feedback ratios."""

    def __init__(self, target_precision: float = 0.7, step: float = 0.1) -> None:
        super().__init__(name="threshold_adjustment")
        self.target_precision = target_precision
        self.step = step
        self._score_offset = 0.0

    def update(self, indices, labels) -> None:
        super().update(indices, labels)
        # If we have any feedback at all, adjust the offset toward target precision.
        _, lbls = self.feedback_arrays()
        if len(lbls) == 0:
            return
        precision = float(lbls.mean()) if len(lbls) else 0.0
        if precision < self.target_precision:
            # Too many FPs: raise required score → subtract from output
            self._score_offset -= self.step
        else:
            self._score_offset += self.step * 0.5

    def score(self) -> np.ndarray:
        return self.base_scores + self._score_offset
