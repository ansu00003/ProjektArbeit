"""
Complete HITL Anomaly Detection System Demo

This script demonstrates:
1. Data loading and preprocessing
2. Isolation Forest training
3. Anomaly detection with SHAP explanations
4. Simulated human feedback
5. Preference model training
6. Model stability analysis (temporal split + cross-validation)
7. Before/After comparison
8. Comprehensive metrics and visualizations
"""

import sys
import os
import pandas as pd
import numpy as np
import plotly.io as pio

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.anomaly_detector import AnomalyDetector
from src.models.preference_model import PreferenceModel, FeedbackSimulator
from src.hitl.feedback_system import FeedbackSystem
from src.utils.data_processor import DataProcessor
from src.utils.metrics import MetricsCalculator
from src.utils.evaluation import ModelEvaluator


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70 + "\n")


def main():
    print_section("🎯 HITL ANOMALY DETECTION SYSTEM - COMPLETE DEMO")
    
    # ==========================================
    # PHASE 1: DATA LOADING & PREPROCESSING
    # ==========================================
    print_section("📊 PHASE 1: Data Loading & Preprocessing")
    
    # Check if we have sample data
    sample_data_path = 'data/raw/sample_journal_entries.csv'
    
    if not os.path.exists(sample_data_path):
        print("⚠️  No sample data found. Generating synthetic data...")
        from main import generate_sample_data
        df = generate_sample_data(n_samples=1000, n_anomalies=50)
        os.makedirs('data/raw', exist_ok=True)
        df.to_csv(sample_data_path, index=False)
        print(f"✅ Generated and saved sample data to {sample_data_path}")
    else:
        df = pd.read_csv(sample_data_path)
        print(f"✅ Loaded existing data: {len(df)} entries")
    
    # Preprocess
    processor = DataProcessor()
    processor.identify_features(df)
    df = processor.handle_missing_values(df.copy())
    df = processor.encode_categorical(df)
    df = processor.create_features(df)
    features = processor.get_features_for_training(df)
    labels = processor.get_label_column(df)
    
    print(f"📐 Feature matrix shape: {features.shape}")
    print(f"📋 Features used: {list(features.columns)}")
    
    if labels is not None:
        anomaly_rate = (labels == 1).sum() / len(labels) * 100
        print(f"🎯 Ground truth anomaly rate: {anomaly_rate:.2f}%")
    
    # ==========================================
    # PHASE 2: ISOLATION FOREST TRAINING
    # ==========================================
    print_section("🌲 PHASE 2: Isolation Forest Training")
    
    contamination = 0.1
    detector = AnomalyDetector(contamination=contamination)
    detector.fit(features)
    
    print(f"✅ Isolation Forest trained (contamination={contamination})")
    print(f"📊 Training samples: {len(features)}")
    
    # ==========================================
    # PHASE 3: ANOMALY DETECTION
    # ==========================================
    print_section("🔍 PHASE 3: Anomaly Detection & SHAP Explanations")
    
    predictions, scores = detector.predict_with_scores(features)
    df['prediction'] = predictions
    df['anomaly_score'] = scores
    
    anomaly_count = (predictions == -1).sum()
    detected_rate = anomaly_count / len(df) * 100
    
    print(f"🚨 Anomalies detected: {anomaly_count} ({detected_rate:.2f}%)")
    
    # Show SHAP explanation for top anomaly
    anomaly_indices = np.where(predictions == -1)[0]
    if len(anomaly_indices) > 0:
        top_anomaly_idx = anomaly_indices[0]
        print(f"\n📝 SHAP Explanation for Entry #{top_anomaly_idx}:")
        reasons = detector.get_top_reasons(features, top_anomaly_idx, top_n=5)
        for i, reason in enumerate(reasons, 1):
            print(f"   {i}. {reason}")
    
    # ==========================================
    # PHASE 4: INITIAL METRICS (BEFORE FEEDBACK)
    # ==========================================
    print_section("📊 PHASE 4: Initial Performance Metrics (Before Feedback)")
    
    if labels is not None:
        calculator = MetricsCalculator()
        metrics_before = calculator.calculate_metrics(labels.values, predictions)
        calculator.print_metrics(metrics_before)
    else:
        print("⚠️  No ground truth labels - skipping initial metrics")
        metrics_before = None
    
    # ==========================================
    # PHASE 5: SIMULATED HUMAN FEEDBACK
    # ==========================================
    print_section("👥 PHASE 5: Simulated Human Feedback")
    
    feedback_system = FeedbackSystem()
    
    if labels is not None:
        # Simulate feedback using ground truth
        simulator = FeedbackSimulator(label_column='is_anomaly')
        anomalies = df[df['prediction'] == -1].head(100)  # Simulate 100 reviews
        
        simulated = simulator.simulate_feedback(anomalies)
        
        # Add to feedback system
        for idx, row in simulated.iterrows():
            feedback_system.add_feedback(
                entry_id=str(idx),
                prediction=-1,
                auditor_label=int(row['simulated_label']),
                confidence=row.get('anomaly_score'),
                auditor_id='ground_truth_simulator'
            )
        
        summary = simulator.get_feedback_summary(simulated)
        print(f"✅ Simulated {summary['total_reviewed']} auditor reviews")
        print(f"   True Positives:  {summary['true_positives']}")
        print(f"   False Positives: {summary['false_positives']}")
        print(f"   Precision:       {summary['precision']:.2%}")
    else:
        print("⚠️  No ground truth - skipping feedback simulation")
    
    # ==========================================
    # PHASE 6: PREFERENCE MODEL TRAINING
    # ==========================================
    print_section("🧠 PHASE 6: Preference Model Training")
    
    if labels is not None:
        preference_model = PreferenceModel()
        train_metrics = preference_model.fit_with_split(features, labels.values, test_size=0.3)
        
        print(f"✅ Preference Model trained")
        print(f"\n📊 Test Set Performance:")
        print(f"   Precision: {train_metrics['precision']:.3f}")
        print(f"   Recall:    {train_metrics['recall']:.3f}")
        print(f"   F1 Score:  {train_metrics['f1_score']:.3f}")
        print(f"   ROC-AUC:   {train_metrics['roc_auc']:.3f}")
        print(f"   Accuracy:  {train_metrics['accuracy']:.3f}")
        
        # Feature importance
        print(f"\n🔑 Top 5 Important Features:")
        importances = preference_model.get_feature_importance()
        for i, (feat, imp) in enumerate(list(importances.items())[:5], 1):
            print(f"   {i}. {feat}: {imp:.4f}")
    else:
        print("⚠️  No ground truth - skipping preference model")
        preference_model = None
    
    # ==========================================
    # PHASE 7: MODEL STABILITY ANALYSIS
    # ==========================================
    print_section("🔬 PHASE 7: Model Stability & Generalization Analysis")
    
    if labels is not None:
        evaluator = ModelEvaluator()
        
        # Temporal split evaluation
        print("📅 Running temporal split evaluation...")
        date_col = df['date_parsed'] if 'date_parsed' in df.columns else None
        temporal_results = evaluator.temporal_split_evaluation(
            detector, features, labels.values, date_col=date_col, train_ratio=0.7
        )
        
        # Cross-validation
        print("🔄 Running 5-fold cross-validation...")
        cv_results = evaluator.cross_validation_evaluation(detector, features, labels.values, cv_folds=5)
        
        # Print stability report
        print("\n" + evaluator.generate_stability_report(temporal_results, cv_results))
        
        # Generate and save ROC curve
        print("\n📈 Generating ROC curve...")
        roc_fig = evaluator.generate_roc_curve(labels.values, scores, title="Isolation Forest ROC Curve")
        os.makedirs('results/figures', exist_ok=True)
        pio.write_html(roc_fig, 'results/figures/roc_curve.html')
        print("   ✅ Saved to results/figures/roc_curve.html")
        
        # Generate Precision-Recall curve
        print("📈 Generating Precision-Recall curve...")
        pr_fig = evaluator.generate_precision_recall_curve(labels.values, scores, 
                                                           title="Isolation Forest PR Curve")
        pio.write_html(pr_fig, 'results/figures/pr_curve.html')
        print("   ✅ Saved to results/figures/pr_curve.html")
    else:
        print("⚠️  No ground truth - skipping stability analysis")
    
    # ==========================================
    # PHASE 8: AFTER FEEDBACK METRICS
    # ==========================================
    print_section("📊 PHASE 8: Performance After Feedback")
    
    feedback_metrics = feedback_system.calculate_feedback_metrics(scores)
    
    print("Current System Performance (based on feedback):")
    print(f"   Precision: {feedback_metrics['precision']:.3f}")
    print(f"   Recall:    {feedback_metrics['recall']:.3f}")
    print(f"   F1 Score:  {feedback_metrics['f1_score']:.3f}")
    print(f"   Accuracy:  {feedback_metrics['accuracy']:.3f}")
    print(f"   ROC-AUC:   {feedback_metrics['roc_auc']:.3f}")
    print(f"   FP Rate:   {feedback_metrics['fpr']:.3f}")
    
    # Save metrics
    feedback_system.save_metrics(feedback_metrics, contamination)
    print("\n✅ Metrics saved to database")
    
    # ==========================================
    # PHASE 9: BEFORE/AFTER COMPARISON
    # ==========================================
    print_section("📊 PHASE 9: Before/After Feedback Comparison")
    
    if metrics_before is not None and labels is not None:
        evaluator = ModelEvaluator()
        
        # Convert feedback metrics to match format
        metrics_after = {
            'precision': feedback_metrics['precision'],
            'recall': feedback_metrics['recall'],
            'f1_score': feedback_metrics['f1_score'],
            'accuracy': feedback_metrics['accuracy'],
            'roc_auc': feedback_metrics['roc_auc'],
            'fpr': feedback_metrics['fpr']
        }
        
        comparison = evaluator.compare_before_after(metrics_before, metrics_after)
        
        print("┌────────────────┬──────────────┬──────────────┬──────────────┐")
        print("│     Metric     │   Before     │   After      │  Improvement │")
        print("├────────────────┼──────────────┼──────────────┼──────────────┤")
        
        for metric_name, data in comparison.items():
            if metric_name == 'fpr':
                print(f"│ FP Rate        │    {data['before']:.3f}     │    {data['after']:.3f}     │    {data['reduction']:+.3f}     │")
            else:
                print(f"│ {metric_name.replace('_', ' ').title():<14} │    {data['before']:.3f}     │    {data['after']:.3f}     │    {data['improvement']:+.3f}     │")
        
        print("└────────────────┴──────────────┴──────────────┴──────────────┘")
        
        # Generate comparison chart
        print("\n📊 Generating comparison chart...")
        comp_fig = evaluator.generate_comparison_chart(comparison)
        pio.write_html(comp_fig, 'results/figures/before_after_comparison.html')
        print("   ✅ Saved to results/figures/before_after_comparison.html")
    else:
        print("⚠️  Insufficient data for comparison")
    
    # ==========================================
    # PHASE 10: SAVE RESULTS
    # ==========================================
    print_section("💾 PHASE 10: Save Results")
    
    # Save detector model
    os.makedirs('results/models', exist_ok=True)
    detector.save('results/models/isolation_forest.pkl')
    print("✅ Isolation Forest model saved")
    
    # Save processed data with predictions
    df.to_csv('results/processed_data_with_predictions.csv', index=False)
    print("✅ Data with predictions saved")
    
    # Save anomalies only
    anomalies = df[df['prediction'] == -1]
    anomalies.to_csv('results/detected_anomalies.csv', index=False)
    print(f"✅ Detected anomalies saved ({len(anomalies)} entries)")
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print_section("✨ DEMO COMPLETE - SUMMARY")
    
    print("📊 System Statistics:")
    print(f"   Total Entries Processed:  {len(df):,}")
    print(f"   Anomalies Detected:       {anomaly_count:,} ({detected_rate:.2f}%)")
    print(f"   Feedback Reviews:         {len(feedback_system.get_all_feedback()):,}")
    
    if metrics_before:
        print(f"\n📈 Performance Metrics:")
        print(f"   Initial Precision:        {metrics_before['precision']:.3f}")
        print(f"   Current Precision:        {feedback_metrics['precision']:.3f}")
        print(f"   Initial FP Rate:          {metrics_before['false_positive_rate']:.3f}")
        print(f"   Current FP Rate:          {feedback_metrics['fpr']:.3f}")
    
    print("\n📁 Generated Files:")
    print("   ├── results/models/isolation_forest.pkl")
    print("   ├── results/processed_data_with_predictions.csv")
    print("   ├── results/detected_anomalies.csv")
    print("   ├── results/figures/roc_curve.html")
    print("   ├── results/figures/pr_curve.html")
    print("   └── results/figures/before_after_comparison.html")
    
    print("\n🎯 Next Steps:")
    print("   1. Run dashboard: python3 -m streamlit run src/dashboard/app.py")
    print("   2. Start API server: python3 api_server.py")
    print("   3. Open visualizations in browser")
    print("   4. Review flagged anomalies in results/detected_anomalies.csv")
    
    print("\n" + "="*70)
    print("Demo completed successfully! 🎉")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
