"""Rule-Mining HITL strategy.

Auditor feedback is mined for *whitelist rules* of the form
"all entries matching this pattern are FP" and *blacklist rules*
"all entries matching this pattern are TP". Patterns are simple AND-clauses
over a small set of categorical / discretised features so the resulting
rules are human-readable and could be reviewed by the auditor afterwards.

This represents the "constraints / explicit knowledge" branch of HITL:
instead of fitting a black-box supervised model, the strategy persists
explicit, auditable rules.

Algorithm:
  - Each time feedback arrives, scan all 1- and 2-feature value clauses
    over the candidate columns.
  - A clause becomes a rule if it covers at least `min_support` labelled
    entries AND its empirical purity (FP-share for whitelist, TP-share for
    blacklist) is at least `min_purity`.
  - Score = base_score, with rule overrides:
      whitelist match → score := score - large_penalty
      blacklist match → score := score + large_bonus
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import BaseHITLStrategy


@dataclass
class Rule:
    clause: Tuple[Tuple[str, object], ...]  # (column, value) AND-conjunction
    kind: str  # "whitelist" or "blacklist"
    support: int
    purity: float

    def matches(self, df: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(df), dtype=bool)
        for col, val in self.clause:
            if col not in df.columns:
                return np.zeros(len(df), dtype=bool)
            m &= (df[col].to_numpy() == val)
        return m

    def describe(self) -> str:
        body = " AND ".join(f"{c}={v}" for c, v in self.clause)
        return f"{self.kind.upper()}: IF {body} (support={self.support}, purity={self.purity:.2f})"


class RuleMiningStrategy(BaseHITLStrategy):
    def __init__(
        self,
        candidate_columns: tuple = (
            "weekend", "nwh", "promptly", "high_cash", "marking",
            "top_n", "user", "gl_account",
        ),
        min_support: int = 5,
        min_purity: float = 0.85,
        whitelist_penalty: float = 5.0,
        blacklist_bonus: float = 5.0,
        max_clause_size: int = 2,
    ) -> None:
        super().__init__(name="rule_mining")
        self.candidate_columns = tuple(candidate_columns)
        self.min_support = min_support
        self.min_purity = min_purity
        self.whitelist_penalty = whitelist_penalty
        self.blacklist_bonus = blacklist_bonus
        self.max_clause_size = max_clause_size
        self.rules: List[Rule] = []

    def initialize(self, X, base_scores, original_df=None) -> None:
        super().initialize(X, base_scores, original_df)
        # Use the rich original frame for rule mining if available
        self._rule_frame = original_df if original_df is not None else X
        self._cols = [c for c in self.candidate_columns if c in self._rule_frame.columns]

    # ------------------------------------------------------------------ #
    def _mine(self) -> None:
        idx, lbl = self.feedback_arrays()
        if len(idx) < self.min_support:
            self.rules = []
            return
        sub = self._rule_frame.iloc[idx].copy()
        sub["__label__"] = lbl
        rules: List[Rule] = []
        for size in range(1, self.max_clause_size + 1):
            for cols in combinations(self._cols, size):
                grouped = sub.groupby(list(cols))["__label__"]
                for key, lbls in grouped:
                    if len(lbls) < self.min_support:
                        continue
                    purity_tp = float(lbls.mean())
                    if not isinstance(key, tuple):
                        key = (key,)
                    clause = tuple(zip(cols, key))
                    if purity_tp >= self.min_purity:
                        rules.append(Rule(clause, "blacklist", int(len(lbls)), purity_tp))
                    elif (1 - purity_tp) >= self.min_purity:
                        rules.append(Rule(clause, "whitelist", int(len(lbls)), 1 - purity_tp))
        # Keep only the strongest rules (high purity, high support)
        rules.sort(key=lambda r: (r.purity, r.support), reverse=True)
        self.rules = rules[:50]

    def update(self, indices, labels) -> None:
        super().update(indices, labels)
        self._mine()

    def score(self) -> np.ndarray:
        s = self.base_scores.copy()
        if not self.rules:
            return s
        for rule in self.rules:
            mask = rule.matches(self._rule_frame)
            if rule.kind == "whitelist":
                s[mask] -= self.whitelist_penalty
            else:
                s[mask] += self.blacklist_bonus
        return s

    def explain_rules(self) -> List[str]:
        return [r.describe() for r in self.rules]
