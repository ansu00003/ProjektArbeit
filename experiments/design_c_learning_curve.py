"""Design C — Learning curve.

The goal here is to show, on the same data, how much a model improves when
the auditor gives it more and more labels. To make the comparison fair I
keep the algorithm and the test set fixed and only change the number of
labels the model is trained on.

Why this is the right design:
  - Same model (RandomForest) at every point on the curve, so a jump in F1
    can only come from more feedback, not from swapping IF for RF.
  - One held-out test set per seed, so every budget is judged on the same
    yardstick — and we evaluate against the full label column, not just
    the reviewed entries.
  - Budgets [10, 25, 50, 100, 200, 500], 5 seeds, mean +/- std.

Run:
    python -m experiments.design_c_learning_curve

Outputs go to results/design_c/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_processor import DataProcessor   # noqa: E402


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
CSV_PATH = PROJECT_ROOT / "journal_entries_final7_1.csv"
OUT_DIR = PROJECT_ROOT / "results" / "design_c"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEEDBACK_BUDGETS = [10, 25, 50, 100, 200, 500]
SEEDS = [0, 1, 2, 3, 4]
TEST_SIZE = 0.30           # 30% held-out, nothing trains on this
N_ESTIMATORS = 200


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_features_and_labels() -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(CSV_PATH, sep=";", on_bad_lines="skip")
    processor = DataProcessor()
    processor.identify_features(df)
    df = processor.handle_missing_values(df)
    df = processor.encode_categorical(df)
    df = processor.create_features(df)
    X = processor.get_features_for_training(df)
    y = processor.get_label_column(df)
    if y is None:
        raise RuntimeError(
            "No ground-truth label column found — this script needs a "
            "'label' (or 'is_anomaly') column to work."
        )
    return X.reset_index(drop=True), y.astype(int).to_numpy()


# --------------------------------------------------------------------------- #
# One run: train on `budget` labels, score on the held-out test set
# --------------------------------------------------------------------------- #
def run_one(
    X: pd.DataFrame,
    y: np.ndarray,
    budget: int,
    seed: int,
) -> dict:
    # Same split for every budget at this seed so all curves are comparable.
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
        stratify=y if y.sum() >= 2 else None,
    )

    rng = np.random.default_rng(seed)
    budget = min(budget, len(y_pool))

    # Anomalies are only ~1% of the data. A random sample of 10 entries
    # would almost never contain a positive, and then RF can't learn
    # anything. So we sample stratified: keep the positive share.
    pos_idx = np.where(y_pool == 1)[0]
    neg_idx = np.where(y_pool == 0)[0]
    pos_share = len(pos_idx) / len(y_pool)
    n_pos = max(1, int(round(budget * pos_share)))
    n_pos = min(n_pos, len(pos_idx))
    n_neg = budget - n_pos
    n_neg = min(n_neg, len(neg_idx))

    chosen = np.concatenate([
        rng.choice(pos_idx, size=n_pos, replace=False),
        rng.choice(neg_idx, size=n_neg, replace=False),
    ])
    X_train = X_pool.iloc[chosen]
    y_train = y_pool[chosen]

    scaler = StandardScaler().fit(X_train)
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )
    model.fit(scaler.transform(X_train), y_train)

    y_pred = model.predict(scaler.transform(X_test))
    y_score = model.predict_proba(scaler.transform(X_test))[:, 1]

    return {
        "budget": budget,
        "seed": seed,
        "n_train_pos": int(y_train.sum()),
        "n_train_neg": int(len(y_train) - y_train.sum()),
        "n_test": len(y_test),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_score) if y_test.sum() > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# "No human at all" baseline: just Isolation Forest, no labels involved.
# We flag the top-K entries by anomaly score (K = number of real anomalies
# in the test set). This is the picture you'd get without HITL.
# --------------------------------------------------------------------------- #
def baseline_no_human(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
) -> dict:
    X_pool, X_test, y_pool, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=seed,
        stratify=y if y.sum() >= 2 else None,
    )
    scaler = StandardScaler().fit(X_pool)
    iforest = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    ).fit(scaler.transform(X_pool))

    # sklearn's decision_function returns higher values for normal points.
    # Flip the sign so "higher = more suspicious", which is easier to read.
    test_scores = -iforest.decision_function(scaler.transform(X_test))
    k = int(y_test.sum())
    # Top-K cutoff: same number of flags as there are real anomalies.
    threshold = np.partition(test_scores, -k)[-k] if k > 0 else np.inf
    y_pred = (test_scores >= threshold).astype(int)

    return {
        "budget": 0,
        "seed": seed,
        "n_train_pos": 0,
        "n_train_neg": 0,
        "n_test": len(y_test),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, test_scores) if y_test.sum() > 0 else float("nan"),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def main() -> None:
    print("Loading data ...")
    X, y = load_features_and_labels()
    print(f"  rows={len(X):,}  features={X.shape[1]}  positives={int(y.sum()):,} "
          f"({y.mean() * 100:.2f}%)")

    rows = []

    # First: baseline with no auditor at all (pure Isolation Forest).
    print("\n[Baseline — no human feedback]")
    for seed in SEEDS:
        result = baseline_no_human(X, y, seed)
        rows.append(result)
        print(f"  budget=   0  seed={seed}  "
              f"F1={result['f1']:.3f}  P={result['precision']:.3f}  "
              f"R={result['recall']:.3f}  Acc={result['accuracy']:.3f}")

    # Then: same RandomForest at growing label budgets.
    print("\n[With human feedback — RandomForest at increasing budgets]")
    for budget in FEEDBACK_BUDGETS:
        for seed in SEEDS:
            result = run_one(X, y, budget, seed)
            rows.append(result)
            print(f"  budget={budget:>4}  seed={seed}  "
                  f"F1={result['f1']:.3f}  P={result['precision']:.3f}  "
                  f"R={result['recall']:.3f}  Acc={result['accuracy']:.3f}")

    raw = pd.DataFrame(rows)
    raw_path = OUT_DIR / "learning_curve.csv"
    raw.to_csv(raw_path, index=False)

    summary = (
        raw.groupby("budget")[["precision", "recall", "f1", "accuracy", "roc_auc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_flat = summary.copy()
    summary_flat.columns = [f"{m}_{stat}" for m, stat in summary_flat.columns]
    summary_flat.reset_index().to_csv(OUT_DIR / "learning_curve_summary.csv", index=False)

    # ---- Plot ----
    means = raw.groupby("budget").mean(numeric_only=True)
    stds = raw.groupby("budget").std(numeric_only=True)

    # Pull the budget-0 (no-human) row out so the curve itself only contains
    # the with-feedback budgets. The baseline becomes a horizontal line.
    baseline_means = means.loc[0] if 0 in means.index else None
    feedback_means = means.drop(index=0, errors="ignore")
    feedback_stds = stds.drop(index=0, errors="ignore")

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Dashed horizontal lines = how the system performs with no auditor.
    if baseline_means is not None:
        for col, color in [("f1", "#2563eb"),
                           ("precision", "#16a34a"),
                           ("recall", "#dc2626")]:
            ax.axhline(
                baseline_means[col], linestyle="--", linewidth=1.4,
                color=color, alpha=0.55,
                label=f"{col.upper()} — no human (baseline)",
            )

    # Solid lines = how the system performs with growing human feedback.
    for col, color in [("f1", "#2563eb"),
                       ("precision", "#16a34a"),
                       ("recall", "#dc2626")]:
        ax.errorbar(
            feedback_means.index, feedback_means[col], yerr=feedback_stds[col],
            marker="o", capsize=3, linewidth=2,
            label=f"{col.upper()} — with human feedback",
            color=color,
        )

    ax.set_xlabel("Auditor feedback budget (# labelled entries)")
    ax.set_ylabel("Test-set score")
    ax.set_title("Design C — Effect of human feedback on detection quality\n"
                 "Dashed = no human (Isolation Forest); solid = with human (RandomForest)")
    ax.set_xscale("log")
    ax.set_xticks(FEEDBACK_BUDGETS)
    ax.set_xticklabels(FEEDBACK_BUDGETS)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve.png", dpi=150)
    plt.close(fig)

    # ---- One-line report ----
    f1_low = means["f1"].iloc[0]
    f1_high = means["f1"].iloc[-1]
    delta = f1_high - f1_low
    report = f"""# Design C — Learning Curve

Same algorithm (RandomForest), same held-out test set ({int((1 - TEST_SIZE) * 100)} / {int(TEST_SIZE * 100)} split,
stratified), evaluated against the full ground-truth label column.

| Budget | F1 (mean) | F1 (std) | Precision | Recall | Accuracy |
|---:|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {b} | {means.loc[b, 'f1']:.3f} | {stds.loc[b, 'f1']:.3f} | "
        f"{means.loc[b, 'precision']:.3f} | {means.loc[b, 'recall']:.3f} | "
        f"{means.loc[b, 'accuracy']:.3f} |"
        for b in FEEDBACK_BUDGETS
    ) + f"""

**Headline:** F1 climbs from **{f1_low:.3f}** at {FEEDBACK_BUDGETS[0]} reviews
to **{f1_high:.3f}** at {FEEDBACK_BUDGETS[-1]} reviews — a gain of **+{delta:.3f}**
attributable purely to additional auditor feedback, since the algorithm and
test set are held constant.

![Learning curve](learning_curve.png)
"""
    (OUT_DIR / "REPORT.md").write_text(report)

    print(f"\nWrote:")
    print(f"  {raw_path}")
    print(f"  {OUT_DIR / 'learning_curve_summary.csv'}")
    print(f"  {OUT_DIR / 'learning_curve.png'}")
    print(f"  {OUT_DIR / 'REPORT.md'}")


if __name__ == "__main__":
    main()
