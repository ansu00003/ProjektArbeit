"""
Example script for running the complete HITL anomaly detection pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.anomaly_detector import AnomalyDetector
from src.hitl.feedback_system import FeedbackSystem
from src.utils.data_processor import DataProcessor
from src.utils.metrics import MetricsCalculator
import pandas as pd
import numpy as np


def generate_sample_data(n_samples=1000, n_anomalies=50):
    """Generate sample journal entry data for testing."""
    np.random.seed(42)
    
    # Normal entries
    normal_amounts = np.random.normal(1000, 200, n_samples - n_anomalies)
    
    # Anomalous entries (unusual amounts)
    anomalous_amounts = np.concatenate([
        np.random.normal(5000, 500, n_anomalies // 2),  # Very high amounts
        np.random.normal(50, 10, n_anomalies // 2)       # Very low amounts
    ])
    
    amounts = np.concatenate([normal_amounts, anomalous_amounts])
    
    # Create dataframe
    data = {
        'id': range(len(amounts)),
        'amount': amounts,
        'account_code': np.random.choice(['ACC001', 'ACC002', 'ACC003'], len(amounts)),
        'date': pd.date_range('2024-01-01', periods=len(amounts), freq='H'),
        'description': np.random.choice(['Payment', 'Transfer', 'Adjustment'], len(amounts)),
        'is_anomaly': np.concatenate([np.zeros(n_samples - n_anomalies), np.ones(n_anomalies)])
    }
    
    df = pd.DataFrame(data)
    return df.sample(frac=1).reset_index(drop=True)  # Shuffle


def main():
    print("="*60)
    print("HUMAN-IN-THE-LOOP ANOMALY DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Generate/Load Data
    print("\n[1] Loading Data...")
    df = generate_sample_data()
    print(f"✓ Loaded {len(df)} journal entries")
    
    # Save sample data
    sample_path = 'data/raw/sample_journal_entries.csv'
    df.to_csv(sample_path, index=False)
    print(f"✓ Saved sample data to {sample_path}")
    
    # Step 2: Preprocess Data
    print("\n[2] Preprocessing Data...")
    processor = DataProcessor()
    processor.identify_features(df)
    df_processed = processor.handle_missing_values(df.copy())
    df_processed = processor.encode_categorical(df_processed)
    df_processed = processor.create_features(df_processed)
    features = processor.get_features_for_training(df_processed)
    print(f"✓ Feature shape: {features.shape}")
    
    # Step 3: Train Anomaly Detector
    print("\n[3] Training Isolation Forest...")
    detector = AnomalyDetector(contamination=0.1)
    detector.fit(features)
    print("✓ Model trained")
    
    # Save model
    detector.save('results/anomaly_detector.pkl')
    print("✓ Model saved")
    
    # Step 4: Detect Anomalies
    print("\n[4] Detecting Anomalies...")
    predictions, scores = detector.predict_with_scores(features)
    df['prediction'] = predictions
    df['anomaly_score'] = scores
    
    anomaly_count = (predictions == -1).sum()
    print(f"✓ Found {anomaly_count} anomalies")
    
    # Step 5: Calculate Initial Metrics
    print("\n[5] Calculating Initial Metrics...")
    calculator = MetricsCalculator()
    initial_metrics = calculator.calculate_metrics(
        df['is_anomaly'].values, 
        predictions
    )
    calculator.print_metrics(initial_metrics)
    
    # Step 6: Simulate Human Feedback
    print("\n[6] Simulating Human Feedback...")
    feedback_system = FeedbackSystem()
    
    # Get flagged anomalies
    anomalies = df[df['prediction'] == -1]
    
    # Simulate auditor review (80% accuracy)
    for idx, row in anomalies.iterrows():
        entry_id = str(row['id'])
        true_label = int(row['is_anomaly'])
        
        # Simulate auditor (sometimes makes mistakes)
        if np.random.random() < 0.8:
            auditor_label = true_label
        else:
            auditor_label = 1 - true_label
        
        feedback_system.add_feedback(
            entry_id=entry_id,
            prediction=-1,
            auditor_label=auditor_label,
            confidence=row['anomaly_score'],
            auditor_id='AUDITOR_001'
        )
    
    print(f"✓ Collected feedback for {len(anomalies)} entries")
    
    # Step 7: Calculate Feedback Metrics
    print("\n[7] Calculating Feedback-based Metrics...")
    feedback_metrics = feedback_system.calculate_feedback_metrics()
    print(f"Precision: {feedback_metrics['precision']:.3f}")
    print(f"Recall: {feedback_metrics['recall']:.3f}")
    print(f"F1 Score: {feedback_metrics['f1_score']:.3f}")
    print(f"False Positive Rate: {feedback_metrics['fpr']:.3f}")
    
    # Save metrics
    feedback_system.save_metrics(feedback_metrics, detector.contamination)
    
    # Step 8: Adjust Model Based on Feedback
    print("\n[8] Checking for Model Adjustment...")
    suggested = feedback_system.suggest_threshold_adjustment()
    if suggested:
        print(f"⚠️  Suggested contamination adjustment: {suggested}")
        print("Retraining with new threshold...")
        detector.adjust_threshold(suggested)
        detector.fit(features)
        
        # Re-evaluate
        predictions_new, scores_new = detector.predict_with_scores(features)
        new_metrics = calculator.calculate_metrics(
            df['is_anomaly'].values,
            predictions_new
        )
        
        print("\n[9] Comparing Metrics...")
        improvements = calculator.compare_metrics(initial_metrics, new_metrics)
        print(f"Precision improvement: {improvements['precision_improvement']:+.3f}")
        print(f"FPR reduction: {improvements['fpr_reduction']:+.3f}")
    else:
        print("✓ Current threshold is appropriate")
    
    # Step 9: Save Results
    print("\n[10] Saving Results...")
    df.to_csv('results/detected_anomalies.csv', index=False)
    print("✓ Results saved to results/detected_anomalies.csv")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run dashboard: streamlit run src/dashboard/app.py")
    print("2. Review flagged anomalies")
    print("3. Provide feedback through UI")
    print("4. Monitor metrics over time")


if __name__ == "__main__":
    main()
