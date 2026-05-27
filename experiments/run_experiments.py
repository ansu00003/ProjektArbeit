"""End-to-end experiment runner.

Reads a JSON config, generates labeled synthetic data, runs every
(strategy × seed × noise_rate) combination, aggregates the results, and
writes:

  results/experiments/
      raw_runs.csv               every checkpoint of every run
      final_metrics.csv          per-run final metrics
      learning_curves.csv        mean ± std at each #reviews (no noise)
      noise_robustness.csv       final F1 vs. noise_rate
      per_type_recall.csv        recall by anomaly type, by strategy
      figures/*.png              plots (matplotlib)
      REPORT.md                  human-readable summary

Run:
    python -m experiments.run_experiments \\
        --config experiments/configs/default.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make `src.*` importable when run as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.utils.data_generator import generate_journal_entries  # noqa: E402
from src.utils.journal_features import (  # noqa: E402
    JOURNAL_FEATURE_COLS,
    add_journal_features,
)
from src.utils.experiment import (  # noqa: E402
    aggregate_final_metrics,
    aggregate_learning_curves,
    aggregate_per_type,
    run_strategy_grid,
)


BASE_FEATURES = ["amount", "weekend", "nwh", "promptly", "top_n", "high_cash", "marking"]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _load_real_csv(path: str, sep: str, n_samples: int | None,
                   data_seed: int) -> pd.DataFrame:
    """Load a real journal-entry CSV and subsample preserving label balance."""
    df = pd.read_csv(path, sep=sep)
    if n_samples is not None and n_samples < len(df):
        rng = np.random.default_rng(data_seed)
        pos = df[df["label"] == 1]
        neg = df[df["label"] == 0]
        n_pos = min(len(pos), int(round(n_samples * pos["label"].mean()
                                         if False else len(pos) / len(df) * n_samples)))
        n_pos = max(n_pos, min(50, len(pos)))  # keep enough positives
        n_neg = n_samples - n_pos
        n_neg = min(n_neg, len(neg))
        pos_idx = rng.choice(pos.index.to_numpy(), size=n_pos, replace=False)
        neg_idx = rng.choice(neg.index.to_numpy(), size=n_neg, replace=False)
        df = df.loc[np.concatenate([pos_idx, neg_idx])].reset_index(drop=True)
        df = df.sample(frac=1, random_state=data_seed).reset_index(drop=True)

    # Provide the columns downstream code expects.
    if "anomaly_type" not in df.columns:
        # Map the `marking` field to a coarse "anomaly type" so per-type
        # recall still works on the real data.
        marking_map = {
            0: "normal",
            1: "duplicate_entry",
            2: "round_number",
            3: "split_transaction",
            4: "unusual_timing",
            5: "account_anomaly",
            6: "payment_reversal",
        }
        df["anomaly_type"] = df["marking"].map(marking_map).fillna("other")
        df.loc[df["label"] == 0, "anomaly_type"] = "normal"
    if "entry_id" not in df.columns:
        if "posting_id" in df.columns:
            df["entry_id"] = df["posting_id"].astype(str) + "_" + df.index.astype(str)
        else:
            df["entry_id"] = [f"E{i:06d}" for i in range(len(df))]
    return df


def prepare_data(cfg: dict) -> tuple[pd.DataFrame, list[str]]:
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    if source == "csv":
        df = _load_real_csv(
            path=data_cfg["csv_path"],
            sep=data_cfg.get("csv_sep", ","),
            n_samples=data_cfg.get("n_samples"),
            data_seed=data_cfg.get("data_seed", 42),
        )
    else:
        df = generate_journal_entries(
            n_samples=data_cfg["n_samples"],
            anomaly_rate=data_cfg["anomaly_rate"],
            seed=data_cfg["data_seed"],
        )

    # User encoding (model needs numeric)
    df["user_encoded"] = pd.factorize(df["user"].astype(str))[0]
    df["gl_account_encoded"] = pd.factorize(df["gl_account"].astype(str))[0]

    use_domain = cfg["features"].get("use_domain_features", True)
    feature_cols = list(BASE_FEATURES) + ["user_encoded", "gl_account_encoded"]
    if use_domain:
        df = add_journal_features(df)
        feature_cols += JOURNAL_FEATURE_COLS
    # Drop any feature columns that are missing in the real CSV
    feature_cols = [c for c in feature_cols if c in df.columns]
    return df, feature_cols


def main(config_path: str) -> None:
    cfg = json.loads(Path(config_path).read_text())

    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    _print("=" * 70)
    _print("HITL ANOMALY-DETECTION EXPERIMENTS")
    _print("=" * 70)
    _print(f"Config: {config_path}")

    # ---- 1. Data --------------------------------------------------------- #
    t0 = time.time()
    df, feature_cols = prepare_data(cfg)
    _print(f"\n[1] Generated dataset: {len(df)} rows, {df['label'].sum()} anomalies "
           f"across {df['anomaly_type'].nunique()} types.")
    _print(f"    Anomaly types: {sorted(df['anomaly_type'].unique())}")
    _print(f"    Features used by model: {len(feature_cols)} columns")
    df.to_csv(out_dir / "dataset.csv", index=False)

    # ---- 2. Strategy comparison + learning curves (noise=0) -------------- #
    exp_cfg = cfg["experiment"]
    _print("\n[2] Strategy comparison (no auditor noise)")
    results_no_noise = run_strategy_grid(
        df=df,
        feature_cols=feature_cols,
        strategies=exp_cfg["strategies"],
        seeds=exp_cfg["seeds"],
        review_batch_size=exp_cfg["review_batch_size"],
        max_reviews=exp_cfg["max_reviews"],
        noise_rate=0.0,
        contamination=exp_cfg["contamination"],
        progress=_print,
    )

    # ---- 3. Noise-robustness study (one strategy comparison per noise) --- #
    _print("\n[3] Noise-robustness study")
    results_noisy = []
    for noise in exp_cfg["noise_rates"]:
        if noise == 0.0:
            continue
        _print(f"  Noise rate = {noise:.2f}")
        results_noisy += run_strategy_grid(
            df=df,
            feature_cols=feature_cols,
            strategies=exp_cfg["strategies"],
            seeds=exp_cfg["seeds"],
            review_batch_size=exp_cfg["review_batch_size"],
            max_reviews=exp_cfg["max_reviews"],
            noise_rate=noise,
            contamination=exp_cfg["contamination"],
            progress=_print,
        )

    all_results = results_no_noise + results_noisy

    # ---- 4. Aggregate ---------------------------------------------------- #
    _print("\n[4] Aggregating results")
    final = aggregate_final_metrics(all_results)
    learning = aggregate_learning_curves(results_no_noise, metric="f1")
    learning_prec = aggregate_learning_curves(results_no_noise, metric="precision")
    learning_rec = aggregate_learning_curves(results_no_noise, metric="recall")
    per_type = aggregate_per_type(results_no_noise)

    final.to_csv(out_dir / "final_metrics.csv", index=False)
    learning.to_csv(out_dir / "learning_curves_f1.csv", index=False)
    learning_prec.to_csv(out_dir / "learning_curves_precision.csv", index=False)
    learning_rec.to_csv(out_dir / "learning_curves_recall.csv", index=False)
    per_type.to_csv(out_dir / "per_type_recall.csv", index=False)

    # raw_runs: flatten per-checkpoint metrics
    flat_rows = []
    for r in all_results:
        for cp, m in zip(r.checkpoints, r.metrics_per_checkpoint):
            flat_rows.append({
                "strategy": r.strategy, "seed": r.seed, "noise_rate": r.noise_rate,
                "n_reviews": cp, **m,
            })
    pd.DataFrame(flat_rows).to_csv(out_dir / "raw_runs.csv", index=False)

    # ---- 5. Plots -------------------------------------------------------- #
    _print("[5] Generating plots")
    _make_plots(final, learning, per_type, fig_dir, exp_cfg)

    # ---- 6. REPORT.md ---------------------------------------------------- #
    _print("[6] Writing REPORT.md")
    _write_report(
        cfg=cfg,
        df=df,
        feature_cols=feature_cols,
        final=final,
        learning=learning,
        per_type=per_type,
        all_results=all_results,
        out_path=Path(cfg["output"]["report_path"]),
    )

    _print(f"\nDone in {time.time() - t0:.1f}s.")
    _print(f"Results: {out_dir}")
    _print(f"Report:  {cfg['output']['report_path']}")


# ---------------------------------------------------------------------------- #
# Plots
# ---------------------------------------------------------------------------- #
def _make_plots(final, learning, per_type, fig_dir, exp_cfg) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 110, "font.size": 10})

    # Plot 1: F1 vs # reviews, one line per strategy.
    # I'm pinning a style per strategy because no_hitl and threshold_adjustment
    # produce identical F1 curves at noise=0 — without different line styles
    # one line gets painted right on top of the other and disappears.
    STYLE = {
        "no_hitl":              {"color": "#16a34a", "linestyle": "-",  "linewidth": 3.0, "marker": "s"},
        "threshold_adjustment": {"color": "#92400e", "linestyle": ":",  "linewidth": 2.0, "marker": "x"},
        "preference_model":     {"color": "#dc2626", "linestyle": "-",  "linewidth": 2.0, "marker": "o"},
        "active_learning":      {"color": "#2563eb", "linestyle": "-",  "linewidth": 2.0, "marker": "^"},
        "hybrid_scoring":       {"color": "#f59e0b", "linestyle": "-",  "linewidth": 2.0, "marker": "D"},
        "rule_mining":          {"color": "#7c3aed", "linestyle": "-",  "linewidth": 2.0, "marker": "v"},
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for strategy, sub in learning.groupby("strategy"):
        sub = sub.sort_values("n_reviews")
        style = STYLE.get(strategy, {"linewidth": 2})
        ax.plot(sub["n_reviews"], sub["f1_mean"], label=strategy,
                markersize=4, markevery=max(1, len(sub) // 10), **style)
        ax.fill_between(
            sub["n_reviews"],
            sub["f1_mean"] - sub["f1_std"].fillna(0),
            sub["f1_mean"] + sub["f1_std"].fillna(0),
            alpha=0.10,
            color=style.get("color"),
        )
    ax.set_xlabel("# auditor reviews")
    ax.set_ylabel("F1 (mean ± std across seeds)")
    ax.set_title("Learning curves by HITL strategy (noise = 0)\n"
                 "Left edge = no human input; movement to the right = effect of HITL")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "learning_curves_f1.png")
    plt.close(fig)

    # Plot 2: final F1 vs. noise_rate per strategy
    if "noise_rate" in final.columns:
        agg = (
            final.groupby(["strategy", "noise_rate"])["f1"]
            .agg(["mean", "std"])
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy, sub in agg.groupby("strategy"):
            sub = sub.sort_values("noise_rate")
            ax.errorbar(
                sub["noise_rate"], sub["mean"], yerr=sub["std"].fillna(0),
                marker="o", capsize=3, label=strategy, linewidth=1.5,
            )
        ax.set_xlabel("Auditor noise rate")
        ax.set_ylabel("Final F1 (mean ± std)")
        ax.set_title("Robustness to auditor noise")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "noise_robustness.png")
        plt.close(fig)

    # Plot 3: per-anomaly-type recall heatmap
    if not per_type.empty:
        pivot = per_type.pivot_table(
            index="anomaly_type", columns="strategy", values="mean", aggfunc="first"
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title("Recall per anomaly type (final, noise = 0)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="black" if v < 0.6 else "white", fontsize=8)
        fig.colorbar(im, ax=ax, label="Recall")
        fig.tight_layout()
        fig.savefig(fig_dir / "per_type_recall.png")
        plt.close(fig)

    # Plot 4: final-metric bars (precision / recall / f1) at noise=0
    no_noise_final = final[final["noise_rate"] == 0.0]
    if not no_noise_final.empty:
        agg = no_noise_final.groupby("strategy")[["precision", "recall", "f1"]].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        agg.plot(kind="bar", ax=ax, width=0.8)
        ax.set_ylabel("Score")
        ax.set_title("Final metrics (noise = 0, mean across seeds)")
        ax.set_ylim(0, 1)
        ax.set_xticklabels(agg.index, rotation=30, ha="right")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "final_metrics_bars.png")
        plt.close(fig)


# ---------------------------------------------------------------------------- #
# Report
# ---------------------------------------------------------------------------- #
def _write_report(cfg, df, feature_cols, final, learning, per_type, all_results, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    no_noise = final[final["noise_rate"] == 0.0]
    final_summary = (
        no_noise.groupby("strategy")[["precision", "recall", "f1", "fpr"]]
        .agg(["mean", "std"])
    )

    # Sample efficiency: # reviews to reach 95% of the strategy's max F1,
    # plus initial F1 (cp=0) and max F1 — captures *speed* of learning.
    eff_rows = []
    for strategy, sub in learning.groupby("strategy"):
        sub = sub.sort_values("n_reviews").reset_index(drop=True)
        max_f1 = float(sub["f1_mean"].max())
        target = 0.95 * max_f1
        hit = sub[sub["f1_mean"] >= target]
        eff_rows.append({
            "strategy": strategy,
            "F1@0_reviews": float(sub["f1_mean"].iloc[0]),
            "F1@max": max_f1,
            "reviews_to_95%_of_max": int(hit["n_reviews"].iloc[0]) if not hit.empty else None,
        })
    eff_df = pd.DataFrame(eff_rows).sort_values("F1@max", ascending=False)

    # Noise robustness
    if "noise_rate" in final.columns:
        noise_table = (
            final.groupby(["strategy", "noise_rate"])["f1"]
            .mean().unstack("noise_rate")
        )
    else:
        noise_table = pd.DataFrame()

    # Best per anomaly type
    if not per_type.empty:
        per_type_pivot = per_type.pivot_table(
            index="anomaly_type", columns="strategy", values="mean"
        )
        best_per_type = per_type_pivot.idxmax(axis=1)
    else:
        per_type_pivot = pd.DataFrame()
        best_per_type = pd.Series(dtype=object)

    # Sample rules from rule-mining strategy (first seed if available)
    sample_rules: list[str] = []
    for r in all_results:
        if r.strategy == "rule_mining" and r.extra.get("rules"):
            sample_rules = r.extra["rules"][:8]
            break

    md = []
    md.append("# HITL Anomaly Detection — Experimental Evaluation\n")
    md.append("_Auto-generated by `experiments/run_experiments.py`._\n")
    md.append("## 1. Setup\n")
    src = cfg["data"].get("source", "synthetic")
    if src == "csv":
        src_desc = f"real CSV ({cfg['data'].get('csv_path')}, subsample of {len(df)})"
    else:
        src_desc = f"{len(df)} synthetic"
    md.append(f"- **Dataset:** {src_desc} journal entries, "
              f"{int(df['label'].sum())} anomalies "
              f"({df['label'].mean() * 100:.1f}%)")
    types_count = df['anomaly_type'].value_counts().drop('normal', errors='ignore')
    md.append(f"- **Anomaly types injected:** {len(types_count)} ({', '.join(types_count.index)})")
    md.append(f"- **Features (n={len(feature_cols)}):** "
              f"{', '.join(feature_cols)}")
    md.append(f"- **Strategies compared:** {', '.join(cfg['experiment']['strategies'])}")
    md.append(f"- **Seeds:** {cfg['experiment']['seeds']}")
    md.append(f"- **Review batch size:** {cfg['experiment']['review_batch_size']}, "
              f"max reviews: {cfg['experiment']['max_reviews']}")
    md.append(f"- **Auditor noise rates studied:** {cfg['experiment']['noise_rates']}")
    md.append("")

    md.append("## 2. Final performance (noise = 0)\n")
    md.append("Mean ± std over seeds; flagging top-K entries where K = # true anomalies.\n")
    md.append("```\n" + final_summary.round(3).to_string() + "\n```\n")

    md.append("![Final metrics](figures/final_metrics_bars.png)\n")

    md.append("## 3. Sample efficiency (learning curves)\n")
    md.append("How many auditor reviews each strategy needs to reach 95 % of its own peak F1, "
              "plus the F1 at 0 reviews (purely unsupervised baseline) and the peak.\n")
    md.append("```\n" + eff_df.round(3).to_string(index=False) + "\n```\n")
    md.append("![Learning curves](figures/learning_curves_f1.png)\n")

    md.append("## 4. Robustness to auditor noise\n")
    if not noise_table.empty:
        md.append("Final F1 (mean across seeds) at each auditor noise rate:\n")
        md.append("```\n" + noise_table.round(3).to_string() + "\n```\n")
    md.append("![Noise robustness](figures/noise_robustness.png)\n")

    md.append("## 5. Per-anomaly-type recall\n")
    if not per_type_pivot.empty:
        md.append("Which strategy catches which type best (final state, noise = 0):\n")
        md.append("```\n" + per_type_pivot.round(2).to_string() + "\n```\n")
        md.append("Best strategy per type:\n")
        md.append("```\n" + best_per_type.to_string() + "\n```\n")
    md.append("![Per-type recall](figures/per_type_recall.png)\n")

    md.append("## 6. Sample rules learned (rule_mining strategy)\n")
    if sample_rules:
        md.append("```\n" + "\n".join(sample_rules) + "\n```\n")
    else:
        md.append("_No rules met the support/purity thresholds with the given budget._\n")

    md.append("## 7. Reproducibility\n")
    if src == "csv":
        md.append(f"- Data source: real journal-entry CSV `{cfg['data'].get('csv_path')}` "
                  f"(subsampled to {len(df)} rows preserving label balance, seed "
                  f"{cfg['data'].get('data_seed', 42)}).")
    else:
        md.append("- Synthetic data is regenerated from a fixed seed via "
                  "`src/utils/data_generator.py`. ")
    md.append("- All randomness flows from the seeds listed in §1; results above are mean ± std.")
    md.append("- Re-run: `python -m experiments.run_experiments --config "
              f"{cfg.get('_config_path', 'experiments/configs/default.json')}`.")
    md.append("")

    md.append("## 8. How to read the results\n")
    md.append("- **no_hitl** is the unsupervised baseline; HITL strategies must beat it.")
    md.append("- **threshold_adjustment** is cheap but only slides the cut-off — caps quickly.")
    md.append("- **preference_model** trains a fresh supervised classifier on each round of feedback.")
    md.append("- **active_learning** picks the *most-uncertain* entries to label, so it should reach high F1 fastest.")
    md.append("- **hybrid_scoring** blends the unsupervised and supervised signal with a feedback-driven α.")
    md.append("- **rule_mining** persists explicit human-readable rules — auditable but coarser.")

    out_path.write_text("\n".join(md))


# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="experiments/configs/default.json")
    args = parser.parse_args()
    main(args.config)
