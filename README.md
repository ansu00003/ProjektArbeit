# Human-in-the-Loop Anomaly Detection for Journal Entry Testing

A Projektarbeit implementing and **comparing several methodological approaches** to involving auditors in the anomaly-detection loop, with a documented experimental evaluation.

## What this project covers

| Topic from the brief | Where it lives |
|---|---|
| Multiple HITL strategies, analyzed and compared | `src/hitl/strategies/` (6 strategies, common API) |
| Domain-specific features for journal-entry testing | `src/utils/journal_features.py` (Benford, round-numbers, threshold-avoidance, …) |
| Realistic data with multiple labeled anomaly types | `src/utils/data_generator.py` (7 anomaly types) |
| Documented experimental evaluation | `experiments/run_experiments.py` → `results/experiments/REPORT.md` |
| Robustness to imperfect auditors | `src/hitl/auditor.py` + noise sweep in the runner |
| Interactive prototype | `src/dashboard/app.py` (Streamlit) |
| API integration | `api_server.py` (FastAPI) |

---

## HITL strategies implemented

All implement the same `BaseHITLStrategy` interface (`select_batch`, `update`, `score`):

| Strategy | Idea |
|---|---|
| `no_hitl` | Unsupervised baseline (Isolation Forest only) — ablation reference |
| `threshold_adjustment` | Auditor feedback shifts the decision cut-off |
| `preference_model` | Supervised RandomForest re-trained on every batch of feedback |
| `active_learning` | Uncertainty sampling: review entries the model is *least sure* about |
| `hybrid_scoring` | `α · IF + (1−α) · PrefModel`; α adapted from feedback-time precision |
| `rule_mining` | Mines auditable AND-clauses ("IF marking=5 AND user=Max → TP") |

---

## Domain features (audit-specific)

Added in `src/utils/journal_features.py`:

- **Benford's Law**: `leading_digit`, `second_digit`, `benford_deviation`
- **Round-number detection**: `is_round_amount` for multiples of 1.000 / 5.000 / 10.000
- **Threshold avoidance**: amounts within 1 % below approval limits (e.g. 9.999)
- **Weekend / late-night posting**: `weekend_or_late`
- **Reversal detection**: matched +X / −X pairs on the same user/account
- **Novel user-account combo**: pair never seen in training window

These are interpretable so they double as SHAP-friendly explanations.

---

## Synthetic data

`src/utils/data_generator.py` produces labeled journal entries with 7 distinct
anomaly types so the evaluation can break results down per type:

`cash_with_pattern · round_number · threshold_avoidance · weekend_latenight · reversal_pattern · novel_user_account · benford_violation`

---

## Running the evaluation

### Design C — Learning curve (fair before/after comparison)

This is the headline experiment: same algorithm (RandomForest), same held-out
test set, only the **number of auditor labels changes**. It isolates the
value of HITL from the choice of model.

```bash
python -m experiments.design_c_learning_curve
```

Outputs (in `results/design_c/`):
- `REPORT.md` — F1/precision/recall per feedback budget, headline gain
- `learning_curve.csv` — per-(budget, seed) raw metrics
- `learning_curve_summary.csv` — mean ± std per budget
- `learning_curve.png` — error-bar plot

### Strategy comparison

The end-to-end experiment runs every (strategy × seed × auditor-noise) combination
and writes a complete report.

```bash
# Full run (5 seeds, 4 noise rates, 6 strategies — a few minutes)
python -m experiments.run_experiments --config experiments/configs/default.json

# Quick smoke run (~30 s)
python -m experiments.run_experiments --config experiments/configs/quick.json
```

Outputs (in `results/experiments/`):
- `REPORT.md` — auto-generated summary with tables and figures
- `learning_curves_*.csv` — F1 / Precision / Recall vs. # reviews
- `final_metrics.csv` — per-run final metrics
- `noise_robustness` plot, `per_type_recall` heatmap, `learning_curves_f1` plot

---

## Other entry points

```bash
# Streamlit dashboard
streamlit run src/dashboard/app.py

# FastAPI backend
python api_server.py

# Original end-to-end demo (single-strategy)
python demo_complete_system.py
```

---

## Repository layout

```
.
├── src/
│   ├── models/                       # Isolation Forest + supervised preference model
│   ├── hitl/
│   │   ├── auditor.py                # Simulated auditor with configurable noise
│   │   ├── feedback_system.py        # SQLite feedback persistence
│   │   └── strategies/               # 6 HITL strategies behind one interface
│   ├── utils/
│   │   ├── data_generator.py         # Labeled synthetic journal entries
│   │   ├── journal_features.py       # Audit-domain features (Benford, …)
│   │   ├── experiment.py             # HITL simulation loop, multi-seed grid
│   │   ├── evaluation.py             # ROC / PR / temporal split / CV
│   │   ├── data_processor.py         # Data preprocessing
│   │   └── metrics.py                # Standard metrics
│   └── dashboard/                    # Streamlit UI
├── experiments/
│   ├── configs/                      # JSON experiment configs
│   └── run_experiments.py            # End-to-end runner
├── api_server.py                     # FastAPI backend
├── demo_complete_system.py           # Single-strategy demo
└── results/                          # Generated outputs (figures, REPORT.md, CSVs)
```

---

## Setup

```bash
pip install -r requirements.txt
```
