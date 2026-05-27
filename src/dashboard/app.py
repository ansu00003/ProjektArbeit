"""HITL Anomaly Detection — Dashboard.

A clean, submission-ready Streamlit interface for the prototype:
upload data → train detector → review flagged entries → submit feedback →
inspect experiment-suite results.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make `src.*` importable when launched via `streamlit run src/dashboard/app.py`
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from src.models.anomaly_detector import AnomalyDetector
from src.models.preference_model import PreferenceModel, FeedbackSimulator
from src.hitl.feedback_system import FeedbackSystem
from src.utils.data_processor import DataProcessor


# --------------------------------------------------------------------------- #
# Page setup + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="HITL Anomaly Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
/* Typography */
html, body, [class*="css"]  {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
h1 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
h2 { font-weight: 600 !important; letter-spacing: -0.01em !important; }
h3 { font-weight: 500 !important; color: #374151 !important; }

/* Hide Streamlit branding / chrome */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #fafafa;
    border-right: 1px solid #e5e7eb;
}
section[data-testid="stSidebar"] h1 { font-size: 1.1rem !important; }

/* Cards / containers */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 16px 20px;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
}
div[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 600 !important;
    color: #111827 !important;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    color: #6b7280 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

/* Buttons */
button[kind="primary"] {
    background-color: #1f2937 !important;
    border: none !important;
}
button[kind="primary"]:hover {
    background-color: #111827 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    font-weight: 500 !important;
}

/* Headline / hero */
.hero-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    color: white;
    padding: 32px 36px;
    border-radius: 12px;
    margin-bottom: 24px;
}
.hero-card h1 { color: white !important; margin-bottom: 8px !important; }
.hero-card p { color: #cbd5e1; margin: 0; }

/* Status pills */
.pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.pill-ok    { background:#ecfdf5; color:#065f46; border:1px solid #6ee7b7; }
.pill-warn  { background:#fffbeb; color:#92400e; border:1px solid #fcd34d; }
.pill-err   { background:#fef2f2; color:#991b1b; border:1px solid #fca5a5; }
.pill-info  { background:#eff6ff; color:#1e40af; border:1px solid #93c5fd; }

/* Subtle dividers */
hr { border: none; border-top: 1px solid #e5e7eb; margin: 24px 0; }

/* DataFrames */
[data-testid="stDataFrameResizable"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}

/* Reduce default top padding */
.block-container { padding-top: 2rem !important; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state() -> None:
    defaults = {
        "detector": None,
        "preference_model": PreferenceModel(),
        "feedback_system": FeedbackSystem(),
        "data_processor": DataProcessor(),
        "current_data": None,
        "processed_data": None,
        "predictions": None,
        "confirmed_indices": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def pill(label: str, kind: str = "info") -> str:
    return f'<span class="pill pill-{kind}">{label}</span>'


def status_pill(condition: bool, ok: str, pending: str) -> str:
    return pill(ok, "ok") if condition else pill(pending, "warn")


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def has_data() -> bool:
    return st.session_state.current_data is not None


def has_model() -> bool:
    return st.session_state.detector is not None


def has_predictions() -> bool:
    return st.session_state.predictions is not None


def n_feedback() -> int:
    try:
        return len(st.session_state.feedback_system.get_all_feedback())
    except Exception:
        return 0


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## HITL Anomaly Detection")
    st.caption("Journal Entry Testing prototype")

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Data",
            "Detection",
            "Review",
            "Simulate Feedback",
            "Metrics",
            "Experiments",
            "Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Pipeline status**")
    st.markdown(
        "- Data uploaded: " + status_pill(has_data(), "Yes", "No"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "- Model trained: " + status_pill(has_model(), "Yes", "No"),
        unsafe_allow_html=True,
    )
    st.markdown(
        "- Anomalies scored: " + status_pill(has_predictions(), "Yes", "No"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"- Feedback collected: **{n_feedback()}**",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page: Overview
# --------------------------------------------------------------------------- #
def page_overview() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <h1>Human-in-the-Loop Anomaly Detection</h1>
            <p>Prototype combining unsupervised anomaly detection (Isolation Forest)
               with auditor feedback loops to reduce false positives in
               Journal Entry Testing.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_rows = len(st.session_state.current_data) if has_data() else 0
    n_anom = int((st.session_state.predictions == -1).sum()) if has_predictions() else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entries loaded", f"{n_rows:,}")
    c2.metric("Anomalies flagged", f"{n_anom:,}")
    c3.metric("Reviews submitted", f"{n_feedback():,}")
    pref_status = "Trained" if st.session_state.preference_model.is_fitted else "Not trained"
    c4.metric("Preference model", pref_status)

    st.markdown("---")

    section_header("Pipeline", "Five steps from raw data to feedback-improved detection.")

    cols = st.columns(5)
    steps = [
        ("1. Data", "Upload CSV; preview; basic stats."),
        ("2. Detection", "Train Isolation Forest, score every entry."),
        ("3. Review", "Human auditor labels flagged entries."),
        ("4. Feedback", "Preference model learns from labels."),
        ("5. Metrics", "Track precision, recall, F1 over time."),
    ]
    for col, (title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"**{title}**")
            st.caption(desc)

    st.markdown("---")

    section_header(
        "What this prototype demonstrates",
        "The Projektarbeit goes beyond a single HITL approach.",
    )
    a, b = st.columns(2)
    with a:
        st.markdown(
            """
            **Six HITL strategies, one common interface**
            - `no_hitl` (baseline)
            - `threshold_adjustment`
            - `preference_model`
            - `active_learning` (uncertainty sampling)
            - `hybrid_scoring` (α-weighted ensemble)
            - `rule_mining` (auditable AND-clauses)
            """
        )
    with b:
        st.markdown(
            """
            **Documented experimental evaluation**
            - Multi-seed runs with mean ± std
            - Learning curves (F1 vs # reviews)
            - Auditor-noise robustness sweep
            - Per-anomaly-type recall breakdown
            - Auto-generated `REPORT.md` + figures
            """
        )

    st.caption(
        "Open the **Experiments** tab to see the latest results, or **Methodology** "
        "for a description of each strategy."
    )


# --------------------------------------------------------------------------- #
# Page: Data
# --------------------------------------------------------------------------- #
def page_data() -> None:
    st.markdown("## Data")
    st.caption("Upload a CSV of journal entries. Semicolon and comma separators are auto-detected.")

    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    use_existing = st.checkbox(
        "Use existing CSV at project root (`journal_entries_final7_1.csv`)",
        value=(uploaded is None),
    )

    df = None
    source_label = ""
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, sep=";", on_bad_lines="skip")
            if df.shape[1] == 1:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, sep=",", on_bad_lines="skip")
            source_label = uploaded.name
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif use_existing:
        existing = ROOT / "journal_entries_final7_1.csv"
        if existing.exists():
            try:
                df = pd.read_csv(existing, sep=";", on_bad_lines="skip")
                source_label = existing.name
            except Exception as e:
                st.error(f"Could not read {existing.name}: {e}")
        else:
            st.info("No CSV found at project root.")

    if df is not None:
        st.session_state.current_data = df
        st.session_state.predictions = None  # reset

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Columns", f"{df.shape[1]}")
        if "label" in df.columns:
            n_pos = int(df["label"].sum())
            c3.metric("Anomalies (label=1)", f"{n_pos:,}")
            c4.metric("Anomaly rate", f"{n_pos / len(df) * 100:.2f}%")
        else:
            c3.metric("Anomalies", "—")
            c4.metric("Ground truth", "absent")

        st.caption(f"Source: `{source_label}`")
        st.markdown("---")

        tabs = st.tabs(["Preview", "Schema", "Missing values"])
        with tabs[0]:
            st.dataframe(df.head(50), use_container_width=True, height=320)
        with tabs[1]:
            schema = pd.DataFrame({
                "column": df.columns,
                "dtype": df.dtypes.astype(str).values,
                "non_null": df.notna().sum().values,
                "null": df.isna().sum().values,
                "unique": [df[c].nunique() for c in df.columns],
            })
            st.dataframe(schema, use_container_width=True, height=320)
        with tabs[2]:
            missing = df.isna().sum().sort_values(ascending=False)
            missing = missing[missing > 0]
            if len(missing) == 0:
                st.success("No missing values.")
            else:
                st.dataframe(
                    missing.rename("missing").to_frame(),
                    use_container_width=True,
                )


# --------------------------------------------------------------------------- #
# Page: Detection
# --------------------------------------------------------------------------- #
def page_detection() -> None:
    st.markdown("## Detection")
    st.caption("Train an Isolation Forest on the loaded data and score every entry.")

    if not has_data():
        st.info("Upload data first on the **Data** tab.")
        return

    c1, c2 = st.columns([2, 3])
    with c1:
        contamination = st.slider(
            "Expected anomaly rate",
            min_value=0.005,
            max_value=0.30,
            value=0.05,
            step=0.005,
            format="%.1f%%",
            help="The fraction of entries Isolation Forest will flag as anomalous.",
        )
        st.caption(
            "Tip: roughly match the real anomaly rate. For the supplied CSV "
            "the ground-truth rate is ~0.94 %."
        )
        train = st.button("Train model", type="primary")
    with c2:
        st.markdown("**What happens when you train**")
        st.markdown(
            "- Numerical features are standardised.\n"
            "- An Isolation Forest with 100 trees is fit on the data.\n"
            "- Every row gets an anomaly score; the lowest scores are flagged."
        )

    if train:
        with st.spinner("Training Isolation Forest…"):
            try:
                df = st.session_state.current_data
                processor = st.session_state.data_processor
                processor.identify_features(df)
                processed = processor.handle_missing_values(df.copy())
                processed = processor.encode_categorical(processed)
                processed = processor.create_features(processed)
                features = processor.get_features_for_training(processed)

                if features.shape[1] == 0:
                    st.error("No usable numeric features in this CSV.")
                    return

                detector = AnomalyDetector(contamination=contamination)
                detector.fit(features)

                preds, scores = detector.predict_with_scores(features)
                processed["prediction"] = preds
                processed["anomaly_score"] = scores
                processed["is_anomaly"] = (preds == -1).astype(int)

                st.session_state.detector = detector
                st.session_state.processed_data = processed
                st.session_state.current_data = processed
                st.session_state.predictions = preds

                n = (preds == -1).sum()
                st.success(
                    f"Training complete — flagged **{n:,}** of **{len(processed):,}** "
                    f"entries ({n / len(processed) * 100:.2f}%)."
                )
            except Exception as e:
                st.error(f"Training failed: {e}")
                return

    if has_predictions():
        st.markdown("---")
        section_header("Score distribution")
        df = st.session_state.current_data
        fig = px.histogram(
            df,
            x="anomaly_score",
            color="is_anomaly",
            nbins=60,
            color_discrete_map={0: "#94a3b8", 1: "#ef4444"},
            labels={"anomaly_score": "Isolation Forest score (lower = more anomalous)",
                    "is_anomaly": "Flagged"},
        )
        fig.update_layout(
            height=380,
            margin=dict(l=8, r=8, t=8, b=8),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        section_header("Top flagged entries")
        anomalies = df[df["prediction"] == -1].sort_values("anomaly_score").head(50)
        st.dataframe(anomalies, use_container_width=True, height=320)


# --------------------------------------------------------------------------- #
# Page: Review
# --------------------------------------------------------------------------- #
def page_review() -> None:
    st.markdown("## Review")
    st.caption(
        "Inspect each flagged entry, see why the model flagged it, and submit a True/False "
        "Positive verdict. Verdicts are persisted to a local SQLite database."
    )

    if not has_predictions():
        st.info("Run detection first on the **Detection** tab.")
        return

    df = st.session_state.current_data
    anomalies = df[df["prediction"] == -1].copy().reset_index(drop=False)
    if len(anomalies) == 0:
        st.success("No anomalies flagged.")
        return

    reviewed = n_feedback()
    pct = reviewed / len(anomalies) if len(anomalies) else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Flagged", f"{len(anomalies):,}")
    c2.metric("Reviewed", f"{reviewed:,}")
    c3.metric("Coverage", f"{pct * 100:.1f}%")
    st.progress(min(pct, 1.0))

    st.markdown("---")
    entry_idx = st.selectbox(
        "Select entry",
        range(len(anomalies)),
        format_func=lambda i: f"#{i + 1} — score {anomalies.iloc[i]['anomaly_score']:.3f}",
    )
    entry = anomalies.iloc[entry_idx]
    actual_idx = int(entry["index"])

    left, right = st.columns([3, 2])
    with left:
        section_header("Entry")
        record = []
        for col, val in entry.drop(["index", "prediction", "is_anomaly"]).items():
            record.append({"Field": col, "Value": str(val)})
        st.dataframe(pd.DataFrame(record), use_container_width=True, height=380, hide_index=True)
    with right:
        section_header("Why this was flagged")
        if has_model():
            try:
                processor = st.session_state.data_processor
                features = processor.get_features_for_training(
                    st.session_state.processed_data
                )
                risk = st.session_state.detector.get_risk_level(
                    features, actual_idx, st.session_state.current_data
                )
                kind = {"HIGH": "err", "MEDIUM": "warn", "LOW": "ok"}.get(risk, "info")
                st.markdown(pill(f"Risk: {risk}", kind), unsafe_allow_html=True)

                reasons = st.session_state.detector.get_top_reasons(
                    features, actual_idx, top_n=5,
                    original_df=st.session_state.current_data,
                )
                st.markdown("\n".join(f"- {r}" for r in reasons))
            except Exception as e:
                st.caption(f"Explanation unavailable: {e}")

    st.markdown("---")
    section_header("Submit verdict")
    with st.form("verdict_form", clear_on_submit=True):
        verdict = st.radio(
            "Verdict",
            ["True Positive — confirmed anomaly", "False Positive — false alarm"],
            horizontal=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            auditor = st.text_input("Auditor ID (optional)")
        with c2:
            comment = st.text_input("Note (optional)")
        submitted = st.form_submit_button("Save verdict", type="primary")

        if submitted:
            label = 1 if verdict.startswith("True") else 0
            entry_id = str(entry.get("entry_id", entry.get("posting_id", actual_idx)))
            try:
                st.session_state.feedback_system.add_feedback(
                    entry_id=entry_id,
                    prediction=-1,
                    auditor_label=label,
                    confidence=float(entry["anomaly_score"]),
                    auditor_id=auditor or None,
                    comments=comment or None,
                )
                if label == 1:
                    st.session_state.confirmed_indices.append(actual_idx)
                st.success("Verdict saved. Pick the next entry to continue.")
            except Exception as e:
                st.error(f"Could not save verdict: {e}")


# --------------------------------------------------------------------------- #
# Page: Simulate feedback
# --------------------------------------------------------------------------- #
def page_simulate() -> None:
    st.markdown("## Simulate auditor feedback")
    st.caption(
        "Use the ground-truth `label` column to simulate auditor verdicts on a "
        "configurable number of flagged entries. The preference model is "
        "(re-)trained on the resulting labels."
    )

    if not has_predictions():
        st.info("Run detection first on the **Detection** tab.")
        return

    df = st.session_state.current_data
    if "label" not in df.columns:
        st.warning("This dataset has no `label` column — simulation needs ground truth.")
        return

    anomalies = df[df["prediction"] == -1]
    if len(anomalies) < 2:
        st.warning("Not enough anomalies for simulation. Lower the contamination on the Detection tab.")
        return

    n = st.slider("Entries to simulate", 1, len(anomalies), value=min(50, len(anomalies)))
    if st.button("Run simulation", type="primary"):
        simulator = FeedbackSimulator(label_column="label")
        sample = anomalies.head(n)
        simulated = simulator.simulate_feedback(sample)

        for idx, row in simulated.iterrows():
            label = int(row["simulated_label"])
            st.session_state.feedback_system.add_feedback(
                entry_id=str(idx),
                prediction=-1,
                auditor_label=label,
                confidence=float(row.get("anomaly_score", 0.0)),
                auditor_id="ground_truth_simulator",
                comments=f"simulated label={label}",
            )

        # Train preference model on simulated labels
        if len(simulated) >= 10:
            processor = st.session_state.data_processor
            feats = processor.get_features_for_training(simulated)
            y = simulated["simulated_label"].values
            if y.sum() > 0 and (y == 0).sum() > 0:
                st.session_state.preference_model.fit_with_split(feats, y)

        summary = simulator.get_feedback_summary(simulated)
        st.success(f"Simulated {n} verdicts.")
        c1, c2, c3 = st.columns(3)
        c1.metric("True positives", summary["true_positives"])
        c2.metric("False positives", summary["false_positives"])
        c3.metric("Precision", f"{summary['precision']:.0%}")

    if st.session_state.preference_model.is_fitted:
        st.markdown("---")
        section_header("Preference model — held-out test metrics")
        m = st.session_state.preference_model.train_metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision", f"{m.get('precision', 0):.2f}")
        c2.metric("Recall", f"{m.get('recall', 0):.2f}")
        c3.metric("F1", f"{m.get('f1_score', 0):.2f}")
        c4.metric("ROC-AUC", f"{m.get('roc_auc', 0):.2f}")


# --------------------------------------------------------------------------- #
# Page: Metrics
# --------------------------------------------------------------------------- #
def page_metrics() -> None:
    st.markdown("## Metrics")
    st.caption("Performance computed from auditor verdicts in the local feedback database.")

    fs = st.session_state.feedback_system
    feedback = fs.get_all_feedback()
    if len(feedback) == 0:
        st.info("No feedback yet. Submit verdicts on the **Review** tab or run a simulation.")
        return

    # If the dataset has a `label` column, use it as ground truth and get
    # real recall / accuracy / F1. Otherwise we only have feedback rows,
    # and those numbers can't be computed honestly — show "—" then.
    gt = None
    if has_data() and "label" in st.session_state.current_data.columns:
        gt = st.session_state.current_data
        scores = (-st.session_state.current_data["anomaly_score"].to_numpy()
                  if "anomaly_score" in st.session_state.current_data.columns else None)
        metrics = fs.calculate_feedback_metrics(scores=scores, ground_truth=gt)
    else:
        metrics = fs.calculate_feedback_metrics()

    mode = metrics.get("mode", "feedback_only")
    if mode == "feedback_only":
        st.markdown(
            pill(
                "Feedback-only mode — recall / accuracy / F1 are undefined "
                "without the full ground truth",
                "warn",
            ),
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            pill("Evaluated against full ground-truth label column", "ok"),
            unsafe_allow_html=True,
        )

    def _fmt(v):
        # NaN != NaN, that's how we detect it.
        try:
            return "—" if v != v else f"{v:.2f}"
        except Exception:
            return "—"

    c1, c2, c3 = st.columns(3)
    c1.metric("Precision", _fmt(metrics['precision']))
    c2.metric("Recall", _fmt(metrics['recall']))
    c3.metric("F1", _fmt(metrics['f1_score']))

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", _fmt(metrics.get('accuracy', float('nan'))))
    c2.metric("ROC-AUC", _fmt(metrics.get('roc_auc', float('nan'))))
    c3.metric("False-positive rate", _fmt(metrics['fpr']))

    st.markdown("---")

    tp = len(fs.get_true_positives())
    fp = len(fs.get_false_positives())
    a, b = st.columns(2)
    with a:
        section_header("Verdict mix")
        if tp + fp > 0:
            fig = go.Figure(data=[go.Pie(
                labels=["True positives", "False positives"],
                values=[tp, fp],
                hole=0.5,
                marker_colors=["#10b981", "#ef4444"],
            )])
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                              showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("No verdicts yet.")
    with b:
        section_header("Verdicts over time")
        if "timestamp" in feedback.columns:
            ts = pd.to_datetime(feedback["timestamp"])
            counts = ts.dt.date.value_counts().sort_index()
            fig = px.line(x=counts.index, y=counts.values,
                          labels={"x": "Date", "y": "Verdicts"})
            fig.update_traces(line_color="#2563eb", line_width=2)
            fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: Experiments
# --------------------------------------------------------------------------- #
def page_experiments() -> None:
    st.markdown("## Experiment Suite")
    st.caption(
        "Auto-generated results from the experiment scripts. "
        "Run them via the commands shown in each tab."
    )

    real = ROOT / "results" / "experiments_real"
    syn = ROOT / "results" / "experiments"
    design_c = ROOT / "results" / "design_c"

    available = []
    if (design_c / "REPORT.md").exists():
        available.append(("Design C — Learning curve", design_c, "design_c"))
    if (real / "REPORT.md").exists():
        available.append(("Real data", real, "default"))
    if (syn / "REPORT.md").exists():
        available.append(("Synthetic data", syn, "default"))
    if not available:
        st.info(
            "No experiment results yet. Run e.g.:\n\n"
            "```bash\n"
            "python -m experiments.design_c_learning_curve\n"
            "python -m experiments.run_experiments --config experiments/configs/default.json\n"
            "```"
        )
        return

    labels = [name for name, _, _ in available]
    tabs = st.tabs(labels)
    for tab, (name, folder, kind) in zip(tabs, available):
        with tab:
            if kind == "design_c":
                _render_design_c(folder)
            else:
                _render_experiment_folder(name, folder)


def _render_design_c(folder: Path) -> None:
    st.markdown(
        "**Goal:** show how much the auditor's labels actually help. To make "
        "the comparison fair I keep the model (RandomForest) and the test "
        "set fixed and only change the **number of labelled entries** the "
        "model is trained on."
    )
    summary_csv = folder / "learning_curve_summary.csv"
    if summary_csv.exists():
        summary = pd.read_csv(summary_csv)
        f1_first = summary["f1_mean"].iloc[0]
        f1_last = summary["f1_mean"].iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("F1 at smallest budget", f"{f1_first:.3f}")
        c2.metric("F1 at largest budget", f"{f1_last:.3f}")
        c3.metric("Δ from feedback", f"+{f1_last - f1_first:.3f}")
        st.markdown("---")
        st.dataframe(summary, use_container_width=True, hide_index=True)
    plot = folder / "learning_curve.png"
    if plot.exists():
        st.image(str(plot),
                 caption="F1 / Precision / Recall vs. feedback budget "
                         "(mean ± std over 5 seeds)",
                 use_container_width=True)
    report = folder / "REPORT.md"
    if report.exists():
        with st.expander("Full auto-generated report (REPORT.md)"):
            st.markdown(report.read_text())


def _render_experiment_folder(name: str, folder: Path) -> None:
    final_csv = folder / "final_metrics.csv"
    if final_csv.exists():
        df = pd.read_csv(final_csv)
        no_noise = df[df["noise_rate"] == 0.0]
        if not no_noise.empty:
            section_header("Final F1 (no auditor noise)")
            agg = (no_noise.groupby("strategy")[["precision", "recall", "f1"]]
                   .mean().sort_values("f1", ascending=False))
            cols = st.columns(min(6, len(agg)))
            for col, (strategy, row) in zip(cols, agg.iterrows()):
                col.metric(strategy, f"{row['f1']:.3f}")

    figs = folder / "figures"
    if figs.exists():
        st.markdown("---")
        section_header("Plots")
        ordered = [
            ("learning_curves_f1.png", "Learning curves (F1 vs # reviews)"),
            ("noise_robustness.png", "Robustness to auditor noise"),
            ("per_type_recall.png", "Per-anomaly-type recall"),
            ("final_metrics_bars.png", "Final metrics by strategy"),
        ]
        for fname, caption in ordered:
            p = figs / fname
            if p.exists():
                st.image(str(p), caption=caption, use_container_width=True)

    report = folder / "REPORT.md"
    if report.exists():
        with st.expander("Full auto-generated report (REPORT.md)"):
            st.markdown(report.read_text())


# --------------------------------------------------------------------------- #
# Page: Methodology
# --------------------------------------------------------------------------- #
def page_methodology() -> None:
    st.markdown("## Methodology")
    st.caption(
        "Six HITL strategies are implemented behind a common interface "
        "(`src/hitl/strategies/`). Each represents a distinct philosophy of "
        "involving the auditor."
    )

    st.markdown("---")
    rows = [
        ("no_hitl",
         "Unsupervised baseline.",
         "Reference point: every other strategy is measured against it."),
        ("threshold_adjustment",
         "Auditor feedback shifts the global decision cut-off.",
         "Cheapest to implement; cannot reorder entries — caps quickly."),
        ("preference_model",
         "Supervised RandomForest re-trained on every batch of feedback.",
         "Replaces the unsupervised score; very accurate but data-hungry."),
        ("active_learning",
         "Uncertainty sampling: review the entries the model is least sure about.",
         "Most sample-efficient at low auditor noise."),
        ("hybrid_scoring",
         "α · Isolation-Forest + (1 − α) · Preference-Model. α adapts from feedback.",
         "Combines unsupervised and supervised signal — best when budget is small."),
        ("rule_mining",
         "Mines auditable AND-clauses (e.g. IF marking=5 AND user=Max → TP).",
         "Interpretable; most robust to noisy auditor verdicts."),
    ]
    for name, what, why in rows:
        with st.container():
            c1, c2 = st.columns([1, 4])
            c1.markdown(f"**`{name}`**")
            c2.markdown(f"{what}\n\n*Strength:* {why}")
            st.markdown(
                "<hr style='margin: 12px 0; border-top:1px solid #f0f0f0;'>",
                unsafe_allow_html=True,
            )

    section_header("Audit-domain features")
    st.markdown(
        "- **Benford's law** — leading-digit deviation from theoretical distribution\n"
        "- **Round-number indicator** — exact multiples of 1.000 / 5.000 / 10.000\n"
        "- **Just-below-threshold** — amounts within 1 % below approval limits\n"
        "- **Weekend / late-night posting** — `weekend > 0` OR `nwh = 1`\n"
        "- **Reversal candidate** — matched +X / −X pair on same user / account\n"
        "- **Novel user-account combo** — pair never seen in the training window"
    )

    section_header("Evaluation protocol")
    st.markdown(
        "- **Multi-seed runs** (default 5) with mean ± std across seeds.\n"
        "- **Learning curves**: precision / recall / F1 measured every 10 reviews.\n"
        "- **Robustness sweep** at auditor noise rates 0 / 10 / 20 / 30 %.\n"
        "- **Per-anomaly-type recall** for each marking class.\n"
        "- **Reproducible**: every result is regenerated by "
        "`python -m experiments.run_experiments --config <yaml>`."
    )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
PAGES = {
    "Overview": page_overview,
    "Data": page_data,
    "Detection": page_detection,
    "Review": page_review,
    "Simulate Feedback": page_simulate,
    "Metrics": page_metrics,
    "Experiments": page_experiments,
    "Methodology": page_methodology,
}

PAGES[page]()
