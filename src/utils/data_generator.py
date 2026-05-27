"""Synthetic journal entry generator with labeled anomaly types.

Produces a dataset compatible with the existing pipeline (columns: amount,
weekend, nwh, promptly, top_n, high_cash, marking, user, gl_account, label)
and adds an `anomaly_type` column so evaluation can break results down by
type. Multiple realistic audit anomaly types are injected:

  - cash_with_pattern       : original ground-truth rule (high_cash AND marking>0)
  - round_number            : amounts on round multiples of 1.000 / 10.000
  - threshold_avoidance     : amounts just below approval thresholds (e.g. 9.999)
  - weekend_latenight       : posted on weekend AND outside working hours
  - reversal_pattern        : booking immediately reversed by counter-booking
  - novel_user_account      : user posts on a GL account they never used before
  - benford_violation       : leading-digit distribution far from Benford's law
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


ANOMALY_TYPES = [
    "cash_with_pattern",
    "round_number",
    "threshold_avoidance",
    "weekend_latenight",
    "reversal_pattern",
    "novel_user_account",
    "benford_violation",
]


def _normal_amount(rng: np.random.Generator, n: int) -> np.ndarray:
    """Long-tailed normal-business amounts."""
    base = rng.lognormal(mean=6.5, sigma=0.9, size=n)
    return np.round(base, 2)


def _users(rng: np.random.Generator, n: int) -> np.ndarray:
    pool = ["Alice", "Bob", "Charlie", "David", "Fred", "Max"]
    return rng.choice(pool, size=n, p=[0.25, 0.25, 0.2, 0.15, 0.1, 0.05])


def _gl_accounts(rng: np.random.Generator, n: int) -> np.ndarray:
    pool = [f"GL{1000 + i:04d}" for i in range(20)]
    return rng.choice(pool, size=n)


def generate_journal_entries(
    n_samples: int = 2000,
    anomaly_rate: float = 0.08,
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a labeled journal-entry dataset with several anomaly types.

    Args:
        n_samples: total number of entries
        anomaly_rate: fraction of entries that should be anomalies
        start_date / end_date: posting-date range (used for temporal splits)
        seed: RNG seed for reproducibility

    Returns:
        DataFrame with `label` (0/1) and `anomaly_type` columns
    """
    rng = np.random.default_rng(seed)

    n_anom = int(round(n_samples * anomaly_rate))
    n_norm = n_samples - n_anom

    # ---------- Normal entries ----------
    normal = pd.DataFrame({
        "amount": _normal_amount(rng, n_norm),
        "user": _users(rng, n_norm),
        "gl_account": _gl_accounts(rng, n_norm),
        "weekend": rng.choice([0, 1, 2], size=n_norm, p=[0.88, 0.06, 0.06]),
        "nwh": rng.choice([0, 1], size=n_norm, p=[0.95, 0.05]),
        "promptly": rng.choice([1, 2, 3], size=n_norm, p=[0.93, 0.05, 0.02]),
        "top_n": rng.choice([0, 1], size=n_norm, p=[0.97, 0.03]),
        "high_cash": rng.choice([0, 1], size=n_norm, p=[0.96, 0.04]),
        "marking": rng.choice(range(7), size=n_norm,
                              p=[0.95, 0.01, 0.01, 0.01, 0.01, 0.005, 0.005]),
        "label": 0,
        "anomaly_type": "normal",
    })

    # ---------- Anomaly entries: split equally across types ----------
    anomalies: list[pd.DataFrame] = []
    per_type = max(1, n_anom // len(ANOMALY_TYPES))

    # 1) cash_with_pattern (preserves original ground-truth rule)
    df = pd.DataFrame({
        "amount": _normal_amount(rng, per_type) * rng.uniform(2, 6, size=per_type),
        "user": _users(rng, per_type),
        "gl_account": _gl_accounts(rng, per_type),
        "weekend": rng.choice([0, 1, 2], size=per_type, p=[0.7, 0.15, 0.15]),
        "nwh": rng.choice([0, 1], size=per_type, p=[0.6, 0.4]),
        "promptly": rng.choice([1, 2, 3], size=per_type, p=[0.5, 0.3, 0.2]),
        "top_n": 1,
        "high_cash": 1,
        "marking": rng.integers(1, 7, size=per_type),
        "label": 1,
        "anomaly_type": "cash_with_pattern",
    })
    anomalies.append(df)

    # 2) round_number — amounts on suspicious round values
    rounds = rng.choice([1000, 5000, 10000, 25000, 50000, 100000], size=per_type)
    df = pd.DataFrame({
        "amount": rounds.astype(float),
        "user": _users(rng, per_type),
        "gl_account": _gl_accounts(rng, per_type),
        "weekend": rng.choice([0, 1, 2], size=per_type, p=[0.85, 0.075, 0.075]),
        "nwh": rng.choice([0, 1], size=per_type, p=[0.9, 0.1]),
        "promptly": 1,
        "top_n": 1,
        "high_cash": rng.choice([0, 1], size=per_type, p=[0.7, 0.3]),
        "marking": 2,
        "label": 1,
        "anomaly_type": "round_number",
    })
    anomalies.append(df)

    # 3) threshold_avoidance — amounts just below approval limits
    thresholds = rng.choice([1000, 5000, 10000, 25000], size=per_type)
    offsets = rng.uniform(0.5, 5.0, size=per_type)
    df = pd.DataFrame({
        "amount": (thresholds - offsets).round(2),
        "user": _users(rng, per_type),
        "gl_account": _gl_accounts(rng, per_type),
        "weekend": rng.choice([0, 1, 2], size=per_type, p=[0.85, 0.075, 0.075]),
        "nwh": rng.choice([0, 1], size=per_type, p=[0.85, 0.15]),
        "promptly": rng.choice([1, 2], size=per_type, p=[0.7, 0.3]),
        "top_n": 0,
        "high_cash": 0,
        "marking": 3,
        "label": 1,
        "anomaly_type": "threshold_avoidance",
    })
    anomalies.append(df)

    # 4) weekend_latenight
    df = pd.DataFrame({
        "amount": _normal_amount(rng, per_type),
        "user": _users(rng, per_type),
        "gl_account": _gl_accounts(rng, per_type),
        "weekend": rng.choice([1, 2], size=per_type),
        "nwh": 1,
        "promptly": rng.choice([1, 2], size=per_type, p=[0.6, 0.4]),
        "top_n": rng.choice([0, 1], size=per_type, p=[0.7, 0.3]),
        "high_cash": rng.choice([0, 1], size=per_type, p=[0.8, 0.2]),
        "marking": 4,
        "label": 1,
        "anomaly_type": "weekend_latenight",
    })
    anomalies.append(df)

    # 5) reversal_pattern — duplicate entries with sign flip
    half = per_type // 2 or 1
    base_amounts = _normal_amount(rng, half) * rng.uniform(2, 5, size=half)
    base_users = _users(rng, half)
    base_accts = _gl_accounts(rng, half)
    a = pd.DataFrame({
        "amount": base_amounts,
        "user": base_users,
        "gl_account": base_accts,
        "weekend": 0, "nwh": 0, "promptly": 1,
        "top_n": 1, "high_cash": 0, "marking": 6,
        "label": 1, "anomaly_type": "reversal_pattern",
    })
    b = pd.DataFrame({
        "amount": -base_amounts,
        "user": base_users,
        "gl_account": base_accts,
        "weekend": 0, "nwh": 0, "promptly": 1,
        "top_n": 1, "high_cash": 0, "marking": 6,
        "label": 1, "anomaly_type": "reversal_pattern",
    })
    anomalies.append(pd.concat([a, b], ignore_index=True))

    # 6) novel_user_account — rare user on rare GL account
    df = pd.DataFrame({
        "amount": _normal_amount(rng, per_type) * rng.uniform(1, 3, size=per_type),
        "user": rng.choice(["Max"], size=per_type),  # rare user
        "gl_account": rng.choice([f"GL{1100 + i:04d}" for i in range(5)], size=per_type),
        "weekend": rng.choice([0, 1, 2], size=per_type, p=[0.8, 0.1, 0.1]),
        "nwh": rng.choice([0, 1], size=per_type, p=[0.7, 0.3]),
        "promptly": 1,
        "top_n": 1,
        "high_cash": rng.choice([0, 1], size=per_type, p=[0.6, 0.4]),
        "marking": 5,
        "label": 1,
        "anomaly_type": "novel_user_account",
    })
    anomalies.append(df)

    # 7) benford_violation — first digit clustered at 5 (Benford expects skew to 1)
    bench = (5 * 10 ** rng.integers(1, 5, size=per_type)
             + rng.integers(0, 99, size=per_type))
    df = pd.DataFrame({
        "amount": bench.astype(float),
        "user": _users(rng, per_type),
        "gl_account": _gl_accounts(rng, per_type),
        "weekend": rng.choice([0, 1, 2], size=per_type, p=[0.85, 0.075, 0.075]),
        "nwh": rng.choice([0, 1], size=per_type, p=[0.9, 0.1]),
        "promptly": 1,
        "top_n": 0,
        "high_cash": rng.choice([0, 1], size=per_type, p=[0.85, 0.15]),
        "marking": 2,
        "label": 1,
        "anomaly_type": "benford_violation",
    })
    anomalies.append(df)

    df_all = pd.concat([normal, *anomalies], ignore_index=True)

    # Posting dates (uniform in window) — used for temporal split
    dates = pd.to_datetime(start_date) + pd.to_timedelta(
        rng.integers(0, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days,
                     size=len(df_all)),
        unit="D",
    )
    df_all["posting_date"] = dates

    # Stable entry id, then shuffle
    df_all = df_all.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_all["entry_id"] = [f"E{idx:06d}" for idx in range(len(df_all))]

    # Trim/extend to exact size
    if len(df_all) > n_samples:
        df_all = df_all.iloc[:n_samples].reset_index(drop=True)

    return df_all


def save_dataset(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
