"""Hybrid-Scoring HITL strategy.

Combines the unsupervised base score with a supervised preference model
trained on auditor feedback:

    score = α · base_score_normalized + (1 − α) · P_anomaly

α is the trust we place in the unsupervised detector. We adapt α from
feedback: when the supervised model's recent precision exceeds the base
detector's, α decreases (trust the auditor-trained model more) and
vice-versa. This is a simple feedback-driven ensemble that combines the
strengths of both detectors instead of replacing one with the other —
useful when the auditor budget is small.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseHITLStrategy


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class HybridScoringStrategy(BaseHITLStrategy):
    def __init__(
        self,
        n_estimators: int = 200,
        random_state: int = 42,
        min_train_size: int = 10,
        initial_alpha: float = 0.7,
        alpha_step: float = 0.05,
        alpha_min: float = 0.1,
        alpha_max: float = 0.9,
    ) -> None:
        super().__init__(name="hybrid_scoring")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_train_size = min_train_size
        self.alpha = initial_alpha
        self.alpha_step = alpha_step
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._model = None
        self._scaler = None
        self._pref_scores = None
        self._base_norm = None

    def initialize(self, X, base_scores, original_df=None) -> None:
        super().initialize(X, base_scores, original_df)
        self._base_norm = _minmax(self.base_scores)

    def update(self, indices, labels) -> None:
        super().update(indices, labels)
        idx, lbl = self.feedback_arrays()
        if len(idx) < self.min_train_size or len(np.unique(lbl)) < 2:
            self._pref_scores = None
            return
        X_train = self.X.iloc[idx].to_numpy()
        self._scaler = StandardScaler().fit(X_train)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=1,
        ).fit(self._scaler.transform(X_train), lbl)
        self._pref_scores = self._model.predict_proba(
            self._scaler.transform(self.X.to_numpy())
        )[:, 1]

        # ---- adapt α: compare on the labelled subset which detector ranks
        #     anomalies higher (precision-at-k = 5).
        k = min(5, len(idx))
        base_top = np.argsort(-self._base_norm[idx])[:k]
        pref_top = np.argsort(-self._pref_scores[idx])[:k]
        base_prec = float(lbl[base_top].mean())
        pref_prec = float(lbl[pref_top].mean())
        if pref_prec > base_prec:
            self.alpha = max(self.alpha_min, self.alpha - self.alpha_step)
        elif base_prec > pref_prec:
            self.alpha = min(self.alpha_max, self.alpha + self.alpha_step)

    def score(self) -> np.ndarray:
        if self._pref_scores is None:
            return self._base_norm.copy()
        return self.alpha * self._base_norm + (1 - self.alpha) * self._pref_scores
