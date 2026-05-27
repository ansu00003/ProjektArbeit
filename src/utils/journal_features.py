"""Domain-specific feature engineering for journal-entry testing.

Adds features grounded in audit literature (SAS 99, IDW PS 210, Benford):

  - leading_digit / second_digit  : for Benford-style analysis
  - benford_deviation             : |observed - expected| for first digit
  - is_round_amount               : amount is multiple of 1.000 / 10.000 etc.
  - just_below_threshold          : amount within 1% below common limits
  - weekend_or_late               : weekend OR non-working hours
  - reversal_candidate            : matched +X / -X pair on same user/account
  - novel_user_account_combo      : (user, gl_account) pair never seen before
                                    in the *training* window

These are deliberately interpretable — they double as SHAP-friendly reasons.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Iterable, Optional


COMMON_THRESHOLDS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000)


def _benford_expected(digit: int) -> float:
    """Expected probability for first digit `d` under Benford's law."""
    if digit < 1 or digit > 9:
        return 0.0
    return math.log10(1 + 1 / digit)


def leading_digit(amount: float) -> int:
    a = abs(float(amount))
    if a <= 0 or not math.isfinite(a):
        return 0
    while a < 1:
        a *= 10
    while a >= 10:
        a /= 10
    return int(a)


def second_digit(amount: float) -> int:
    a = abs(float(amount))
    if a < 10 or not math.isfinite(a):
        return 0
    while a >= 100:
        a /= 10
    return int(a) % 10


def is_round_amount(amount: float, bases: Iterable[int] = (1_000, 5_000, 10_000)) -> int:
    a = abs(float(amount))
    if a <= 0:
        return 0
    return int(any(abs(a % b) < 1e-6 for b in bases))


def just_below_threshold(
    amount: float,
    thresholds: Iterable[int] = COMMON_THRESHOLDS,
    tolerance_pct: float = 0.01,
) -> int:
    """1 if amount is within `tolerance_pct` *below* any approval threshold."""
    a = abs(float(amount))
    for t in thresholds:
        diff = t - a
        if 0 < diff <= t * tolerance_pct:
            return 1
    return 0


def benford_deviation(amount: float, observed_dist: Optional[dict] = None) -> float:
    """Absolute deviation of this entry's first digit from Benford expectation.

    If `observed_dist` (digit -> empirical probability) is given, it is used
    as the comparison baseline; otherwise the deviation is computed against
    the theoretical distribution (per-entry indicator).
    """
    d = leading_digit(amount)
    if d == 0:
        return 0.0
    expected = _benford_expected(d)
    if observed_dist is not None and d in observed_dist:
        return abs(observed_dist[d] - expected)
    # Per-entry: how "non-Benford" this digit is (high digits ⇒ higher dev)
    return abs(1.0 / 9 - expected)


def add_journal_features(
    df: pd.DataFrame,
    amount_col: str = "amount",
    user_col: str = "user",
    account_col: str = "gl_account",
    weekend_col: str = "weekend",
    nwh_col: str = "nwh",
    train_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Return a copy of `df` enriched with domain features.

    `train_mask` (optional) marks the training window; novelty features
    (e.g. unseen user-account combos) use only that window as the reference,
    avoiding leakage.
    """
    out = df.copy()

    if amount_col in out.columns:
        amt = out[amount_col].astype(float)
        out["leading_digit"] = amt.map(leading_digit)
        out["second_digit"] = amt.map(second_digit)
        out["is_round_amount"] = amt.map(is_round_amount)
        out["just_below_threshold"] = amt.map(just_below_threshold)

        # Empirical leading-digit distribution from the training window
        ref = out.loc[train_mask, "leading_digit"] if train_mask is not None else out["leading_digit"]
        ref = ref[ref > 0]
        if len(ref) > 0:
            counts = ref.value_counts(normalize=True).to_dict()
        else:
            counts = {}
        out["benford_deviation"] = out["leading_digit"].map(
            lambda d: abs(counts.get(d, 0.0) - _benford_expected(d)) if d > 0 else 0.0
        )

    if weekend_col in out.columns and nwh_col in out.columns:
        out["weekend_or_late"] = ((out[weekend_col] > 0) | (out[nwh_col] == 1)).astype(int)

    # Reversal candidate: same user+account+|amount| with opposite sign
    if amount_col in out.columns and user_col in out.columns and account_col in out.columns:
        key = (
            out[user_col].astype(str)
            + "|" + out[account_col].astype(str)
            + "|" + out[amount_col].abs().round(2).astype(str)
        )
        # An entry is a reversal candidate if the same key appears with
        # both a positive and a negative amount somewhere in the dataset.
        sign = np.sign(out[amount_col].astype(float))
        sign_per_key = pd.DataFrame({"key": key, "sign": sign}).groupby("key")["sign"]
        has_both = (sign_per_key.min() < 0) & (sign_per_key.max() > 0)
        out["reversal_candidate"] = key.map(has_both.to_dict()).fillna(False).astype(int)

    # Novel user-account combo (relative to training window)
    if user_col in out.columns and account_col in out.columns:
        ref_df = out.loc[train_mask] if train_mask is not None else out
        seen = set(zip(ref_df[user_col].astype(str), ref_df[account_col].astype(str)))
        combo = list(zip(out[user_col].astype(str), out[account_col].astype(str)))
        out["novel_user_account_combo"] = [int(c not in seen) for c in combo]

    return out


JOURNAL_FEATURE_COLS = [
    "leading_digit",
    "second_digit",
    "is_round_amount",
    "just_below_threshold",
    "benford_deviation",
    "weekend_or_late",
    "reversal_candidate",
    "novel_user_account_combo",
]
