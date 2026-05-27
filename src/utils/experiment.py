"""HITL experiment framework.

Contains:
  - simulate_hitl_loop : run one HITL strategy on one dataset for one seed,
                         producing a learning curve (metrics vs. # of reviews).
  - run_strategy_grid  : run several strategies × several seeds, aggregate
                         mean ± std at every checkpoint.
  - per_anomaly_type_breakdown : for the final state, compute recall by
                                 anomaly type.
  - flag_at_top_k      : standard top-k decision rule used to convert a
                         continuous score into binary flags.

The framework deliberately separates *score* (continuous) from *flag*
(binary at top-k) so the same strategy can be evaluated with different
review budgets without retraining.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from src.hitl.auditor import SimulatedAuditor
from src.hitl.strategies import build_strategy


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #
def flag_at_top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Binary flag: 1 for the top-k highest scores, 0 otherwise."""
    if k <= 0:
        return np.zeros_like(scores, dtype=int)
    k = min(k, len(scores))
    threshold_idx = np.argpartition(-scores, k - 1)[:k]
    out = np.zeros(len(scores), dtype=int)
    out[threshold_idx] = 1
    return out


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict:
    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    out = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fpr": float(((y_pred == 1) & (y_true == 0)).sum() / max(neg, 1)),
        "n_flagged": int(y_pred.sum()),
        "n_positive": pos,
    }
    if pos > 0 and neg > 0 and len(np.unique(scores)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        out["roc_auc"] = float("nan")
        out["average_precision"] = float("nan")
    return out


def fit_base_detector(
    X: pd.DataFrame, contamination: float = 0.1, random_state: int = 42
) -> tuple[np.ndarray, IsolationForest, StandardScaler]:
    """Fit an Isolation Forest and return per-row anomaly scores.

    Higher returned score = more anomalous (we negate sklearn's convention).
    """
    scaler = StandardScaler().fit(X.to_numpy())
    Xs = scaler.transform(X.to_numpy())
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=1,
    ).fit(Xs)
    raw = model.score_samples(Xs)  # higher = more normal
    return -raw, model, scaler


# ---------------------------------------------------------------------------- #
# HITL loop (one strategy, one seed)
# ---------------------------------------------------------------------------- #
@dataclass
class HITLRunResult:
    strategy: str
    seed: int
    noise_rate: float
    checkpoints: List[int]                    # cumulative #reviews at each step
    metrics_per_checkpoint: List[Dict]        # one dict per checkpoint
    final_metrics: Dict
    per_type_recall: Dict[str, float]
    runtime_seconds: float
    extra: Dict = field(default_factory=dict)


def simulate_hitl_loop(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    strategy_name: str,
    *,
    seed: int = 0,
    review_batch_size: int = 10,
    max_reviews: int = 200,
    noise_rate: float = 0.0,
    contamination: float = 0.1,
    flag_top_k: Optional[int] = None,
    label_col: str = "label",
    type_col: str = "anomaly_type",
    strategy_kwargs: Optional[dict] = None,
) -> HITLRunResult:
    """Run one HITL strategy on one dataset for one seed.

    Args:
        df: full dataset, must contain `label_col` and feature columns.
        feature_cols: features to feed the base detector and supervised model.
        strategy_name: key in src.hitl.strategies.STRATEGY_REGISTRY.
        seed: RNG seed (controls IF + auditor noise).
        review_batch_size: # entries the auditor labels per round.
        max_reviews: stop after this many reviews.
        noise_rate: auditor mistake probability.
        contamination: IsolationForest contamination.
        flag_top_k: how many entries to flag for metric computation.
                    Defaults to the number of true anomalies in the data
                    (so precision and recall are directly comparable).
        label_col / type_col: ground-truth + anomaly-type columns.
        strategy_kwargs: forwarded to the strategy constructor.

    Returns:
        HITLRunResult with the learning curve and final per-type breakdown.
    """
    t0 = time.time()
    rng_seed = seed
    X = df[list(feature_cols)].copy().reset_index(drop=True)
    y = df[label_col].astype(int).to_numpy()
    if flag_top_k is None:
        flag_top_k = max(1, int(y.sum()))

    base_scores, _, _ = fit_base_detector(X, contamination=contamination, random_state=rng_seed)

    strategy = build_strategy(strategy_name, **(strategy_kwargs or {}))
    strategy.initialize(X, base_scores, original_df=df.reset_index(drop=True))

    auditor = SimulatedAuditor(noise_rate=noise_rate, seed=rng_seed, label_col=label_col)

    checkpoints: List[int] = []
    metrics_per_checkpoint: List[Dict] = []

    # checkpoint at 0 (no feedback yet)
    initial_pred = flag_at_top_k(strategy.score(), flag_top_k)
    metrics_per_checkpoint.append(compute_metrics(y, initial_pred, strategy.score()))
    checkpoints.append(0)

    df_reset = df.reset_index(drop=True)
    while strategy.n_reviewed < max_reviews:
        budget = min(review_batch_size, max_reviews - strategy.n_reviewed)
        batch = strategy.select_batch(budget)
        if len(batch.indices) == 0:
            break
        labels = auditor.label(df_reset.iloc[batch.indices])
        strategy.update(batch.indices, labels)

        scores = strategy.score()
        pred = flag_at_top_k(scores, flag_top_k)
        metrics_per_checkpoint.append(compute_metrics(y, pred, scores))
        checkpoints.append(strategy.n_reviewed)

    final_scores = strategy.score()
    final_pred = flag_at_top_k(final_scores, flag_top_k)
    final_metrics = compute_metrics(y, final_pred, final_scores)

    # Per-anomaly-type recall (final state)
    per_type: Dict[str, float] = {}
    if type_col in df_reset.columns:
        for t, sub in df_reset.groupby(type_col):
            if t == "normal":
                continue
            mask = sub.index.to_numpy()
            if len(mask) == 0:
                continue
            per_type[t] = float(final_pred[mask].mean())

    extra = {}
    if hasattr(strategy, "explain_rules"):
        extra["rules"] = strategy.explain_rules()[:10]
    if hasattr(strategy, "alpha"):
        extra["final_alpha"] = float(strategy.alpha)

    return HITLRunResult(
        strategy=strategy_name,
        seed=seed,
        noise_rate=noise_rate,
        checkpoints=checkpoints,
        metrics_per_checkpoint=metrics_per_checkpoint,
        final_metrics=final_metrics,
        per_type_recall=per_type,
        runtime_seconds=time.time() - t0,
        extra=extra,
    )


# ---------------------------------------------------------------------------- #
# Multi-seed grid
# ---------------------------------------------------------------------------- #
def run_strategy_grid(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    strategies: Sequence[str],
    seeds: Sequence[int],
    *,
    review_batch_size: int = 10,
    max_reviews: int = 200,
    noise_rate: float = 0.0,
    contamination: float = 0.1,
    flag_top_k: Optional[int] = None,
    progress: Optional[Callable[[str], None]] = None,
    strategy_kwargs: Optional[Dict[str, dict]] = None,
) -> List[HITLRunResult]:
    """Run every (strategy, seed) combination and return all results."""
    results: List[HITLRunResult] = []
    sk = strategy_kwargs or {}
    for strategy in strategies:
        for seed in seeds:
            if progress:
                progress(f"  → strategy={strategy} seed={seed}")
            results.append(
                simulate_hitl_loop(
                    df=df,
                    feature_cols=feature_cols,
                    strategy_name=strategy,
                    seed=seed,
                    review_batch_size=review_batch_size,
                    max_reviews=max_reviews,
                    noise_rate=noise_rate,
                    contamination=contamination,
                    flag_top_k=flag_top_k,
                    strategy_kwargs=sk.get(strategy, {}),
                )
            )
    return results


# ---------------------------------------------------------------------------- #
# Aggregation
# ---------------------------------------------------------------------------- #
def aggregate_learning_curves(
    results: Sequence[HITLRunResult], metric: str = "f1"
) -> pd.DataFrame:
    """Aggregate (strategy, checkpoint) → mean and std across seeds."""
    rows = []
    for r in results:
        for cp, m in zip(r.checkpoints, r.metrics_per_checkpoint):
            rows.append({
                "strategy": r.strategy,
                "seed": r.seed,
                "noise_rate": r.noise_rate,
                "n_reviews": cp,
                "metric": m.get(metric, float("nan")),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    agg = (
        df.groupby(["strategy", "noise_rate", "n_reviews"])["metric"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
    )
    return agg


def aggregate_final_metrics(results: Sequence[HITLRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "strategy": r.strategy,
            "seed": r.seed,
            "noise_rate": r.noise_rate,
            **r.final_metrics,
            "runtime_seconds": r.runtime_seconds,
        })
    return pd.DataFrame(rows)


def aggregate_per_type(results: Sequence[HITLRunResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for t, recall in r.per_type_recall.items():
            rows.append({
                "strategy": r.strategy,
                "seed": r.seed,
                "noise_rate": r.noise_rate,
                "anomaly_type": t,
                "recall": recall,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby(["strategy", "noise_rate", "anomaly_type"])["recall"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
