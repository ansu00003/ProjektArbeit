"""Streamlit dashboard for Human-in-the-Loop anomaly detection."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.anomaly_detector import AnomalyDetector
from models.preference_model import PreferenceModel, FeedbackSimulator
from hitl.feedback_system import FeedbackSystem
from utils.data_processor import DataProcessor


st.set_page_config(page_title="HITL Anomaly Detection System", layout="wide")

# Initialize session state
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'preference_model' not in st.session_state:
    st.session_state.preference_model = PreferenceModel()
if 'feedback_system' not in st.session_state:
    st.session_state.feedback_system = FeedbackSystem()
if 'data_processor' not in st.session_state:
    st.session_state.data_processor = DataProcessor()
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'confirmed_indices' not in st.session_state:
    st.session_state.confirmed_indices = []


def main():
    st.title("Human-in-the-Loop Anomaly Detection System")
    st.markdown("*Automated detection of suspicious financial entries with human verification*")
    
    # Sidebar with progress tracker
    st.sidebar.header("Navigation")
    
    # Show progress with clean formatting
    st.sidebar.subheader("Progress")
    
    step1_done = st.session_state.current_data is not None
    step2_done = st.session_state.detector is not None
    step3_done = st.session_state.predictions is not None
    
    st.sidebar.checkbox("Step 1: Upload Data", value=step1_done, disabled=True)
    st.sidebar.checkbox("Step 2: Train AI Model", value=step2_done, disabled=True)
    st.sidebar.checkbox("Step 3: Find Anomalies", value=step3_done, disabled=True)
    
    st.sidebar.markdown("---")
    
    # Menu
    menu = st.sidebar.selectbox(
        "Go to:",
        ["Overview", "Step 1: Upload Data", "Step 2: Train Model", "Step 3: Detect Anomalies", 
         "Step 4: Human Review", "Simulate Feedback", "Results & Metrics"]
    )
    
    if menu == "Overview":
        show_overview()
    elif menu == "Step 1: Upload Data":
        show_data_upload()
    elif menu == "Step 2: Train Model":
        show_model_training()
    elif menu == "Step 3: Detect Anomalies":
        show_anomaly_detection()
    elif menu == "Step 4: Human Review":
        show_feedback_review()
    elif menu == "Simulate Feedback":
        show_simulation()
    elif menu == "Results & Metrics":
        show_metrics()


def show_overview():
    st.header("System Overview")
    
    # Simple explanation
    st.info("""
    **Overview:**
    
    This tool helps auditors find suspicious financial transactions (anomalies) 
    in a large dataset. Instead of checking every single entry manually, 
    the AI finds the unusual ones, and humans verify if they are actually problems.
    """)
    
    st.markdown("---")
    
    # Visual workflow
    st.subheader("The 4-Step Process")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        ### Step 1
        **Upload Your Data**
        
        You upload a CSV file with 
        financial journal entries 
        (transactions, amounts, dates, etc.)
        """)
    
    with col2:
        st.markdown("""
        ### Step 2
        **AI Learns Patterns**
        
        The AI (Isolation Forest) 
        learns what normal entries 
        look like in your data.
        """)
    
    with col3:
        st.markdown("""
        ### Step 3  
        **Find Anomalies**
        
        AI flags entries that look 
        different from normal ones.
        These might be problems.
        """)
    
    with col4:
        st.markdown("""
        ### Step 4
        **Human Verifies**
        
        You review flagged entries 
        and tell the system if they are 
        real problems or false alarms.
        """)
    
    st.markdown("---")
    
    # What is an anomaly?
    st.subheader("What is an Anomaly?")
    st.markdown("""
    An **anomaly** is an entry that looks unusual compared to others. Examples:
    
    | Normal Entry | Anomaly (Suspicious) |
    |-------------|---------------------|
    | Amount: $500 | Amount: $500,000 (way higher than usual) |
    | Posted on Tuesday | Posted on Sunday at 3 AM |
    | Account: Office Supplies | Account: Never seen before |
    | User: Regular employee | User: Someone who rarely posts |
    
    **Important:** Not all anomalies are fraud. Some are just unusual but valid entries.
    That is why **humans need to verify** each one.
    """)
    
    st.markdown("---")
    
    # What is Human-in-the-Loop?
    st.subheader("What is Human-in-the-Loop?")
    st.success("""
    **Human-in-the-Loop (HITL)** means the AI does not make final decisions alone.
    
    1. AI finds suspicious entries
    2. Human auditor reviews each one
    3. Human says "Yes, this is a problem" OR "No, this is fine"
    4. System learns from human feedback to get better over time
    
    This combines **AI speed** (checking 100,000 entries in seconds) 
    with **human judgment** (understanding context and nuance).
    """)
    
    st.markdown("---")
    
    # Current status
    st.subheader("Current Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_count = len(st.session_state.current_data) if st.session_state.current_data is not None else 0
        st.metric("Entries Loaded", f"{data_count:,}")
    
    with col2:
        anomalies = 0
        if st.session_state.predictions is not None:
            anomalies = (st.session_state.predictions == -1).sum()
        st.metric("Anomalies Found", anomalies)
    
    with col3:
        feedback_count = len(st.session_state.feedback_system.get_all_feedback())
        st.metric("Reviews Completed", feedback_count)


def show_data_upload():
    st.header("Step 1: Upload Your Data")
    
    st.info("""
    **Instructions:**
    Upload a CSV file containing your financial journal entries.
    The file should have columns like: amount, date, account, user_id, etc.
    """)
    
    # Expected format
    with st.expander("What should my CSV look like? (Click to expand)"):
        st.markdown("""
        Your CSV file should have financial transaction data. Common columns:
        
        | Column | Description | Example |
        |--------|-------------|--------|
        | amount | Transaction amount | 1500.00 |
        | posting_date | When it was posted | 2024-01-15 |
        | account | Account name/code | "Office Supplies" |
        | user_id | Who created it | "user123" |
        
        Note: The system can handle different column names automatically.
        """)
    
    uploaded_file = st.file_uploader("Drop your CSV file here", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Try different delimiters
            try:
                df = pd.read_csv(uploaded_file, sep=';', on_bad_lines='skip', encoding='utf-8')
            except:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='utf-8')
            st.session_state.current_data = df
        
            st.success(f"**Success!** Loaded {len(df):,} entries from your file")
        
            st.subheader("Data Preview (first 10 rows)")
            st.caption("This is just a preview - your full data has been loaded")
            st.dataframe(df.head(10))
            
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Total Rows:** {df.shape[0]:,}")
                st.write(f"**Total Columns:** {df.shape[1]}")
                st.write("**Column Names:**")
                for col in df.columns:
                    st.write(f"  - {col}")
            
            with col2:
                st.write("**Data Types:**")
                for col, dtype in df.dtypes.items():
                    st.write(f"  - {col}: `{dtype}`")
            
            # Missing values
            missing = df.isnull().sum()
            if missing.sum() > 0:
                st.warning("Some data is missing (will be handled automatically)")
                with st.expander("View missing values"):
                    st.write(missing[missing > 0])
            
            st.success("**Next Step:** Go to 'Step 2: Train Model' in the sidebar")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.info("**Tips:** Make sure your file is a valid CSV with proper formatting")


def show_model_training():
    st.header("Step 2: Train the AI Model")
    
    st.info("""
    **Process:**
    The AI (called "Isolation Forest") learns what normal entries look like.
    Then it can spot entries that are different from normal.
    """)
    
    if st.session_state.current_data is None:
        st.warning("First, upload your data in Step 1")
        return
    
    # Explain the slider
    st.subheader("Settings")
    
    with st.expander("What is Contamination? (Click to learn)"):
        st.markdown("""
        **Contamination** = How many anomalies do you expect in your data?
        
        | Setting | Meaning |
        |---------|--------|
        | 0.01 (1%) | Very few anomalies expected |
        | 0.05 (5%) | Some anomalies expected |
        | 0.10 (10%) | More anomalies expected |
        
        **Tip:** Start with 0.05-0.10. If you get too many false alarms, lower it.
        """)
    
    contamination = st.slider(
        "How many anomalies do you expect? (as % of data)",
        min_value=0.005,
        max_value=0.5,
        value=0.01,  # Default ~1% per actual anomaly rate of 0.94%
        step=0.005,
        format="%.1f%%",
        key="contamination_slider"
    )
    
    st.caption(f"This means: AI will flag approximately {contamination*100:.1f}% of entries as suspicious")
    st.caption("Tip: Actual anomaly rate in sample data is ~0.94%. Start low and adjust.")
    
    # Check if contamination changed from previously trained model
    needs_retrain = False
    if st.session_state.detector is not None:
        if hasattr(st.session_state.detector, 'contamination'):
            if st.session_state.detector.contamination != contamination:
                needs_retrain = True
                st.info("You changed the percentage. Click 'Train' to update the AI model.")
    
    button_label = "Retrain AI with New Percentage" if needs_retrain else "Train the AI Now"
    
    if st.button(button_label, type="primary"):
        with st.spinner("AI is learning patterns from your data..."):
            try:
                # Process data
                processor = st.session_state.data_processor
                processor.identify_features(st.session_state.current_data)
                df = processor.handle_missing_values(st.session_state.current_data.copy())
                df = processor.encode_categorical(df)
                df = processor.create_features(df)
                features = processor.get_features_for_training(df)
                
                if features.shape[1] == 0:
                    st.error("Could not find usable columns in your data")
                    return
                
                # Train model
                detector = AnomalyDetector(contamination=contamination)
                detector.fit(features)
                
                st.session_state.detector = detector
                st.session_state.processed_data = df
                
                st.success(f"**AI Training Complete!**")
                
                # Show what it learned
                st.subheader("What the AI Learned")
                st.write(f"Analyzed **{len(features):,}** entries")
                st.write(f"Used **{len(detector.feature_names)}** features:")
                for feat in detector.feature_names:
                    st.write(f"  - {feat}")
                
                st.success("**Next Step:** Go to 'Step 3: Detect Anomalies' in the sidebar")
            except Exception as e:
                st.error(f"Error during training: {str(e)}")


def show_anomaly_detection():
    st.header("Step 3: Detect Anomalies")
    
    st.info("""
    **Process:**
    The trained AI scans ALL your entries and flags the suspicious ones.
    Entries that are different from normal will be marked as anomalies.
    """)
    
    if st.session_state.detector is None:
        st.warning("First, train the AI in Step 2")
        return
    
    if st.session_state.current_data is None:
        st.warning("First, upload your data in Step 1")
        return
    
    if st.button("Scan for Anomalies Now", type="primary"):
        with st.spinner("Scanning all entries for suspicious patterns..."):
            try:
                # Use processed data if available
                if hasattr(st.session_state, 'processed_data') and st.session_state.processed_data is not None:
                    df = st.session_state.processed_data
                else:
                    processor = st.session_state.data_processor
                    processor.identify_features(st.session_state.current_data)
                    df = processor.handle_missing_values(st.session_state.current_data.copy())
                    df = processor.encode_categorical(df)
                    df = processor.create_features(df)
                
                processor = st.session_state.data_processor
                features = processor.get_features_for_training(df)
                
                # Predict
                predictions, scores = st.session_state.detector.predict_with_scores(features)
                
                st.session_state.predictions = predictions
                
                # Add to dataframe
                df['prediction'] = predictions
                df['anomaly_score'] = scores
                df['is_anomaly'] = (predictions == -1).astype(int)
                
                st.session_state.current_data = df
                
                anomaly_count = (predictions == -1).sum()
                normal_count = len(df) - anomaly_count
                
                st.success(f"**Scan Complete!**")
                
                # Results summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Normal Entries", f"{normal_count:,}")
                with col2:
                    st.metric("Anomalies Found", f"{anomaly_count:,}")
                with col3:
                    pct = (anomaly_count / len(df)) * 100
                    st.metric("Anomaly Rate", f"{pct:.1f}%")
                
                st.success("**Next Step:** Go to 'Step 4: Human Review' to verify the flagged entries")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    # Show results
    if st.session_state.predictions is not None:
        df = st.session_state.current_data
        anomalies = df[df['prediction'] == -1]
        
        st.markdown("---")
        st.subheader(f"Flagged Anomalies ({len(anomalies):,} entries)")
        
        st.caption("""
        Orange = Anomaly (flagged as suspicious)
        Green = Normal (looks fine)
        
        The chart below shows how suspicious each entry is. 
        Entries on the left (more negative scores) are MORE suspicious.
        """)
        
        # Visualization
        fig = px.histogram(df, x='anomaly_score', color='is_anomaly',
                          title='Distribution of Suspicion Scores',
                          labels={'anomaly_score': 'Suspicion Score (lower = more suspicious)',
                                 'is_anomaly': 'Is Anomaly?'},
                          color_discrete_map={0: 'green', 1: 'orange'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show anomalies table
        st.write("**List of Flagged Entries:**")
        st.caption("These are the entries that need human review")
        st.dataframe(anomalies.head(100))  # Limit display for performance
        
        # Show BOTH models per PDF spec
        st.markdown("---")
        st.subheader("Combined Results: Isolation Forest + Preference Model")
        
        st.caption("""
        **Per PDF Specification:**
        The dashboard shows results from BOTH models:
        - Isolation Forest (IF): Finds statistical outliers
        - Preference Model (PM): Finds entries similar to confirmed anomalies
        """)
        
        # Create combined flag per PDF spec
        df['if_flag'] = (df['prediction'] == -1).astype(int)
        
        if st.session_state.preference_model.is_fitted:
            try:
                processor = st.session_state.data_processor
                features = processor.get_features_for_training(
                    st.session_state.processed_data if hasattr(st.session_state, 'processed_data')
                    else st.session_state.current_data
                )
                
                # Get preference model scores
                pref_scores = st.session_state.preference_model.predict_proba(features)
                df['pref_score'] = pref_scores
                df['pref_flag'] = (pref_scores > 0.5).astype(int)
                
                # Combined flag: flagged by either model
                df['combined_flag'] = ((df['if_flag'] == 1) | (df['pref_flag'] == 1)).astype(int)
                
                st.session_state.current_data = df
                
                # Show comparison
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("IF Anomalies", f"{df['if_flag'].sum():,}")
                with col2:
                    st.metric("PM Anomalies", f"{df['pref_flag'].sum():,}")
                with col3:
                    st.metric("Combined", f"{df['combined_flag'].sum():,}")
                
                # Show entries flagged by preference model
                pref_flagged = df[df['pref_flag'] == 1].sort_values('pref_score', ascending=False)
                
                if len(pref_flagged) > 0:
                    st.write(f"**Entries flagged by Preference Model ({len(pref_flagged)}):**")
                    st.caption("These look similar to past confirmed anomalies")
                    display_cols = ['pref_score'] + [c for c in pref_flagged.columns if c not in ['pref_score', 'prediction', 'anomaly_score', 'is_anomaly', 'if_flag', 'pref_flag', 'combined_flag']]
                    st.dataframe(pref_flagged[display_cols].head(20))
            except Exception as e:
                st.info(f"Preference model results not available: {e}")
        else:
            df['combined_flag'] = df['if_flag']
            st.info("Preference Model not trained yet. Run simulation first to train it.")


def show_feedback_review():
    st.header("Step 4: Human Review")
    
    st.info("""
    **Instructions:**
    Look at each flagged entry and decide if it is a REAL problem or just unusual but OK.
    Your feedback helps the system learn and improve.
    """)
    
    if st.session_state.current_data is None:
        st.warning("First, upload data and run detection")
        return
    
    if 'prediction' not in st.session_state.current_data.columns:
        st.warning("First, run anomaly detection in Step 3")
        return
    
    df = st.session_state.current_data
    anomalies = df[df['prediction'] == -1].copy()
    
    if len(anomalies) == 0:
        st.success("No anomalies to review - your data looks clean!")
        return
    
    st.subheader(f"Review Queue: {len(anomalies):,} entries to check")
    
    # Progress tracker
    reviewed = len(st.session_state.feedback_system.get_all_feedback())
    st.progress(min(reviewed / len(anomalies), 1.0) if len(anomalies) > 0 else 0)
    st.caption(f"Reviewed: {reviewed} / {len(anomalies)}")
    
    # Select entry to review
    st.write("---")
    entry_idx = st.selectbox(
        "Select an entry to review:", 
        range(len(anomalies)),
        format_func=lambda x: f"Entry #{x+1}"
    )
    entry = anomalies.iloc[entry_idx]
    actual_idx = anomalies.index[entry_idx]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Entry Details:**")
        st.caption("Here is what this entry looks like")
        
        # Show entry as a nice table
        entry_data = []
        for col, val in entry.items():
            if col not in ['prediction', 'anomaly_score', 'is_anomaly']:
                entry_data.append({"Field": col, "Value": str(val)})
        st.table(entry_data[:10])  # Show first 10 fields
    
    with col2:
        st.write("**Why AI Flagged This Entry:**")
        st.caption("Analysis based on fraud indicators")
        
        if st.session_state.detector is not None:
            try:
                processor = st.session_state.data_processor
                features = processor.get_features_for_training(
                    st.session_state.processed_data if hasattr(st.session_state, 'processed_data') 
                    else st.session_state.current_data
                )
                
                # Get original data for human-readable values
                original_df = st.session_state.current_data
                
                # Get risk level
                risk_level = st.session_state.detector.get_risk_level(features, actual_idx, original_df)
                
                # Show risk badge
                if risk_level == 'HIGH':
                    st.error(f"**RISK LEVEL: HIGH**")
                elif risk_level == 'MEDIUM':
                    st.warning(f"**RISK LEVEL: MEDIUM**")
                else:
                    st.success(f"**RISK LEVEL: LOW**")
                
                # Get reasons with original dataframe for human-readable values
                reasons = st.session_state.detector.get_top_reasons(
                    features, actual_idx, top_n=5, original_df=original_df
                )
                
                if reasons:
                    for reason in reasons:
                        # Strip emoji prefixes for cleaner display
                        clean_reason = reason.lstrip('\U0001f6a8\u26a0\ufe0f\u2705\u2139\ufe0f ')
                        if reason.startswith(("HIGH", "CRITICAL")):
                            st.error(clean_reason)
                        elif reason.startswith(("MEDIUM", "WARNING")):
                            st.warning(clean_reason)
                        elif reason.startswith(("LOW", "NORMAL", "OK")):
                            st.success(clean_reason)
                        else:
                            st.write(f"- {clean_reason}")
                else:
                    st.info("No specific indicators identified")
            except Exception as e:
                st.info(f"Explanation not available: {e}")
        else:
            st.info("Train model first for explanations")
    
    # Feedback form
    st.markdown("---")
    st.subheader("Your Decision")
    
    with st.form("feedback_form"):
        st.write("**Is this entry a REAL problem?**")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **True Positive (Real Problem)**
            - This is actual fraud/error
            - Needs investigation
            - AI was correct to flag it
            """)
        with col2:
            st.markdown("""
            **False Positive (False Alarm)**
            - This is legitimate
            - Just looks unusual
            - AI made a mistake
            """)
        
        label = st.radio(
            "Your verdict:",
            ["True Positive (Real Problem)", "False Positive (False Alarm)"],
            horizontal=True
        )
        
        auditor_id = st.text_input("Your Name/ID (optional)")
        comments = st.text_area("Notes (optional)", placeholder="Add any notes about this entry...")
        
        submitted = st.form_submit_button("Submit My Decision", type="primary")
        
        if submitted:
            entry_id = str(entry.get('id', entry_idx))
            auditor_label = 1 if "True Positive" in label else 0
            
            st.session_state.feedback_system.add_feedback(
                entry_id=entry_id,
                prediction=-1,
                auditor_label=auditor_label,
                confidence=entry.get('anomaly_score'),
                auditor_id=auditor_id,
                comments=comments
            )
            
            st.success("Feedback saved! Select the next entry to continue.")
            
            # Track confirmed anomalies for preference model
            if auditor_label == 1:
                st.session_state.confirmed_indices.append(actual_idx)


def show_simulation():
    """Simulate auditor feedback using ground truth labels (per PDF spec)."""
    st.header("Simulate Auditor Feedback")
    
    st.info("""
    **How simulation works (per PDF spec):**
    Your CSV has a 'label' column (0=normal, 1=anomaly) which is the ground truth.
    We use this to simulate what an auditor would say when reviewing flagged entries.
    """)
    
    if st.session_state.predictions is None:
        st.warning("Run anomaly detection first (Step 3)")
        return
    
    df = st.session_state.current_data
    if 'prediction' not in df.columns:
        st.warning("Run anomaly detection first")
        return
    
    # Check for label column
    if 'label' not in df.columns:
        st.error("No 'label' column found in your data. This is needed for simulation.")
        st.info("The 'label' column should contain: 0 = normal entry, 1 = anomaly")
        return
    
    anomalies = df[df['prediction'] == -1]
    
    st.write(f"**Anomalies to simulate feedback for:** {len(anomalies):,}")
    
    # Show ground truth stats
    st.subheader("Ground Truth in Your Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Entries", f"{len(df):,}")
    with col2:
        actual_anomalies = (df['label'] == 1).sum()
        st.metric("Actual Anomalies (label=1)", f"{actual_anomalies:,}")
    with col3:
        pct = (actual_anomalies / len(df)) * 100 if len(df) > 0 else 0
        st.metric("Anomaly Rate", f"{pct:.2f}%")
    
    st.markdown("---")
    
    # Check if we have enough anomalies
    if len(anomalies) < 2:
        st.warning("Not enough anomalies detected to simulate. Try lowering the contamination rate in Step 2.")
        return

    n_simulate = st.slider(
        "Number of flagged entries to simulate",
        min_value=1,
        max_value=len(anomalies),
        value=min(50, len(anomalies))
    )
    
    if st.button("Run Simulation (using ground truth)", type="primary"):
        with st.spinner("Simulating auditor feedback using ground truth labels..."):
            simulator = FeedbackSimulator(label_column='label')
            
            # Sample anomalies
            sample = anomalies.head(n_simulate)
            simulated = simulator.simulate_feedback(sample)
            
            # Add simulated feedback to system
            confirmed = []
            for idx, row in simulated.iterrows():
                label = int(row['simulated_label'])
                st.session_state.feedback_system.add_feedback(
                    entry_id=str(idx),
                    prediction=-1,
                    auditor_label=label,
                    confidence=row.get('anomaly_score'),
                    auditor_id='ground_truth_simulator',
                    comments=f"Simulated from ground truth (label={label})"
                )
                if label == 1:
                    confirmed.append(idx)
            
            # Update confirmed indices
            st.session_state.confirmed_indices.extend(confirmed)
            
            # Train preference model on SIMULATED FEEDBACK DATA ONLY (not full dataset!)
            # BUG FIX: Previously trained on full dataset labels, now trains on actual feedback
            if len(simulated) > 0:
                processor = st.session_state.data_processor
                # Get features ONLY for the reviewed entries
                reviewed_features = processor.get_features_for_training(simulated)
                # Use the SIMULATED labels (feedback), NOT the original ground truth
                feedback_labels = simulated['simulated_label'].values
                
                # Only train if we have enough samples and both classes
                if len(reviewed_features) >= 10:
                    if feedback_labels.sum() > 0 and (feedback_labels == 0).sum() > 0:
                        st.session_state.preference_model.fit_with_split(reviewed_features, feedback_labels)
                        st.info(f"Preference model trained on {len(reviewed_features)} reviewed entries ({int(feedback_labels.sum())} anomalies, {int((feedback_labels == 0).sum())} normal)")
                    else:
                        st.warning("Cannot train model: need both positive and negative examples in feedback")
            
            # Show results
            summary = simulator.get_feedback_summary(simulated)
            
            st.success(f"Simulated {n_simulate} reviews using ground truth!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("True Positives", summary['true_positives'])
            with col2:
                st.metric("False Positives", summary['false_positives'])
            with col3:
                st.metric("Precision", f"{summary['precision']:.0%}")
    
    # Show preference model status
    st.markdown("---")
    st.subheader("Preference Model Status")
    
    if st.session_state.preference_model.is_fitted:
        metrics = st.session_state.preference_model.train_metrics
        st.success("Preference model trained!")
        
        if metrics:
            st.write("**Model Performance on Test Set:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{metrics.get('precision', 0):.0%}")
            with col2:
                st.metric("Recall", f"{metrics.get('recall', 0):.0%}")
            with col3:
                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.0%}")
            with col4:
                st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2f}")
    else:
        st.info("Preference model will be trained after simulation.")


def show_metrics():
    st.header("Results & Performance")
    
    st.info("""
    **Overview:**
    How well is the AI performing? These metrics are based on YOUR feedback.
    """)
    
    feedback_system = st.session_state.feedback_system
    all_feedback = feedback_system.get_all_feedback()
    
    if len(all_feedback) == 0:
        st.warning("No feedback yet! Go to Step 4 and review some entries first.")
        st.markdown("""
        **What you will see here once you provide feedback:**
        - **Precision**: How accurate is the AI? (% of flagged entries that are real problems)
        - **Recall**: Is the AI missing problems? (% of real problems the AI found)
        - **False Positive Rate**: How often does AI make false alarms?
        """)
        return
    
    # Current metrics
    metrics = feedback_system.calculate_feedback_metrics()
    
    st.subheader("AI Performance Scores")
    
    # Explain metrics
    with st.expander("What do these numbers mean? (Click to learn)"):
        st.markdown("""
        | Metric | What it measures | Good score |
        |--------|-----------------|------------|
        | **Precision** | When AI flags something, how often is it right? | Higher = better (>70% is good) |
        | **Recall** | Of all real problems, how many did AI catch? | Higher = better (>70% is good) |
        | **F1 Score** | Balance between Precision and Recall | Higher = better |
        | **Accuracy** | Overall correct predictions | Higher = better (>70%) |
        | **ROC-AUC** | How well can AI distinguish good from bad? | Higher = better (>0.7) |
        | **False Positive Rate** | How often does AI cry wolf? | Lower = better (<30%) |
        """)
    
    # Row 1: Main metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Precision", f"{metrics['precision']:.0%}")
        st.caption("AI accuracy")
    with col2:
        st.metric("Recall", f"{metrics['recall']:.0%}")
        st.caption("Problems caught")
    with col3:
        st.metric("F1 Score", f"{metrics['f1_score']:.0%}")
        st.caption("Overall balance")
    
    # Row 2: Additional metrics (per professor's requirements)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.0%}")
        st.caption("Correct predictions")
    with col2:
        st.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.2f}")
        st.caption("Discrimination ability")
    with col3:
        st.metric("False Alarms", f"{metrics['fpr']:.0%}")
        st.caption("Wrong flags")
    
    # Feedback summary
    st.markdown("---")
    st.subheader("Your Feedback Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # True vs False Positives
        tp = len(feedback_system.get_true_positives())
        fp = len(feedback_system.get_false_positives())
        
        st.write(f"**Total Reviews:** {tp + fp}")
        st.write(f"- Real Problems Found: **{tp}**")
        st.write(f"- False Alarms: **{fp}**")
        
        if tp + fp > 0:
            fig = go.Figure(data=[go.Pie(
                labels=['Real Problems', 'False Alarms'],
                values=[tp, fp],
                marker_colors=['#ff6b6b', '#4ecdc4']
            )])
            fig.update_layout(title='What Did You Find?')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Feedback over time
        all_feedback['timestamp'] = pd.to_datetime(all_feedback['timestamp'])
        feedback_time = all_feedback.groupby(all_feedback['timestamp'].dt.date).size()
        
        fig = px.line(x=feedback_time.index, y=feedback_time.values,
                     labels={'x': 'Date', 'y': 'Reviews'},
                     title='Your Review Activity')
        st.plotly_chart(fig, use_container_width=True)
        
    # Metrics history
    st.markdown("---")
    st.subheader("Performance Over Time")
    metrics_history = feedback_system.get_metrics_history()
    
    if len(metrics_history) > 0:
        st.caption("As you provide more feedback, the system tracks how performance changes")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=metrics_history['timestamp'], 
                                y=metrics_history['precision'],
                                name='Precision', line=dict(color='#3498db')))
        fig.add_trace(go.Scatter(x=metrics_history['timestamp'], 
                                y=metrics_history['recall'],
                                name='Recall', line=dict(color='#2ecc71')))
        fig.add_trace(go.Scatter(x=metrics_history['timestamp'], 
                                y=metrics_history['false_positive_rate'],
                                name='False Alarm Rate', line=dict(color='#e74c3c')))
        fig.update_layout(title='How Is AI Improving?', xaxis_title='Date', 
                        yaxis_title='Score', yaxis=dict(tickformat='.0%'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Performance history will appear as you provide more feedback over time")
    
    # Threshold adjustment
    st.markdown("---")
    st.subheader("Should You Adjust Settings?")
    
    suggested = feedback_system.suggest_threshold_adjustment()
    if suggested:
        st.warning(f"""
        **Suggestion:** The AI seems to be flagging too many false alarms.
        
        Consider lowering the contamination setting to **{suggested:.0%}** in Step 2.
        This will make the AI more selective (fewer flags, but more accurate).
        """)
    else:
        st.success("Current settings look good! No adjustment needed.")


if __name__ == "__main__":
    main()
