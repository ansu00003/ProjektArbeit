"""Active-Learning HITL strategy (uncertainty sampling).

The auditor's time is the bottleneck, so we should spend it on the entries
whose label is *most informative*: the ones the current model is uncertain
about, not the ones it is already confident are anomalies.

This implementation:
  - keeps a supervised RandomForest as the working model (like
    PreferenceModelStrategy)
  - selects the next review batch by *uncertainty* (P close to 0.5),
    optionally mixed with diversity (random tie-break)
  - falls back to top-base-score selection until enough labels exist
    to fit the working model

Uncertainty sampling is the most-cited active-learning baseline (Lewis &
Gale 1994; Settles 2009).
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseHITLStrategy, ReviewBatch


class ActiveLearningStrategy(BaseHITLStrategy):
    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        min_train_size: int = 10,
        exploration_ratio: float = 0.2,
    ) -> None:
        super().__init__(name="active_learning")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_train_size = min_train_size
        self.exploration_ratio = exploration_ratio
        self._model = None
        self._scaler = None
        self._cached_scores = None
        self._rng = np.random.default_rng(random_state)

    # ---- training ----
    def update(self, indices, labels) -> None:
        super().update(indices, labels)
        idx, lbl = self.feedback_arrays()
        if len(idx) < self.min_train_size or len(np.unique(lbl)) < 2:
            self._model = None
            return
        X_train = self.X.iloc[idx].to_numpy()
        self._scaler = StandardScaler().fit(X_train)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        ).fit(self._scaler.transform(X_train), lbl)
        self._cached_scores = self._model.predict_proba(
            self._scaler.transform(self.X.to_numpy())
        )[:, 1]

    # ---- scoring ----
    def score(self) -> np.ndarray:
        if self._cached_scores is None:
            return self.base_scores.copy()
        return self._cached_scores.copy()

    # ---- batch selection: uncertainty + a little exploration ----
    def select_batch(self, n: int) -> ReviewBatch:
        unreviewed_mask = np.ones(len(self.X), dtype=bool)
        for i in self.reviewed_indices:
            unreviewed_mask[i] = False
        candidate_pos = np.where(unreviewed_mask)[0]
        if len(candidate_pos) == 0:
            return ReviewBatch(indices=np.array([], dtype=int), reason="exhausted")

        # Cold start: until we have a working model, fall back to base scores
        if self._cached_scores is None:
            order = np.argsort(-self.base_scores[candidate_pos])
            picked = candidate_pos[order][:n]
            return ReviewBatch(indices=picked, reason="cold_start_top_score")

        n_explore = int(round(n * self.exploration_ratio))
        n_uncert = max(0, n - n_explore)

        # Uncertainty: distance of P(anomaly) from 0.5
        probs = self._cached_scores[candidate_pos]
        uncertainty = -np.abs(probs - 0.5)  # higher = more uncertain
        order = np.argsort(-uncertainty)
        uncertain_pick = candidate_pos[order][:n_uncert]

        # Exploration: random uniform from remaining
        remaining = np.setdiff1d(candidate_pos, uncertain_pick, assume_unique=False)
        if n_explore > 0 and len(remaining) > 0:
            explore_pick = self._rng.choice(
                remaining, size=min(n_explore, len(remaining)), replace=False
            )
        else:
            explore_pick = np.array([], dtype=int)

        picked = np.concatenate([uncertain_pick, explore_pick])
        return ReviewBatch(indices=picked.astype(int), reason="uncertainty_sampling")
