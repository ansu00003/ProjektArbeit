"""Preference-Model HITL strategy.

Trains a supervised RandomForest on auditor-labelled entries, then uses the
predicted P(anomaly) as the refined score for *all* entries. This is the
"learn what the auditor confirms" approach already present in earlier
versions of the project, refactored to live behind the new strategy API.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from .base import BaseHITLStrategy


class PreferenceModelStrategy(BaseHITLStrategy):
    def __init__(self, n_estimators: int = 200, random_state: int = 42,
                 min_train_size: int = 10) -> None:
        super().__init__(name="preference_model")
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.min_train_size = min_train_size
        self._model = None
        self._scaler = None
        self._cached_scores = None

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

    def score(self) -> np.ndarray:
        if self._cached_scores is None:
            return self.base_scores.copy()
        return self._cached_scores.copy()
