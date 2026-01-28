"""Preference Model - Learns from auditor feedback to find similar anomalies.

This model learns what patterns auditors confirmed as real anomalies,
then finds similar instances in future data.

Per PDF spec: Uses RandomForestClassifier (supervised learning)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from typing import List, Dict, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PreferenceModel:
    """
    Learns from auditor feedback (or ground truth labels) to find anomalies.
    
    Per PDF Specification:
    - Uses RandomForestClassifier (supervised learning)
    - Learns from confirmed anomalies (label=1)
    - Predicts probability of being an anomaly
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize preference model.
        
        Args:
            n_estimators: Number of trees in RandomForest
            random_state: Random seed for reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            class_weight='balanced'  # Important: handles imbalanced data
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        self.train_metrics = {}  # Store metrics from training
        
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'PreferenceModel':
        """
        Train on labeled data (from feedback or ground truth).
        
        Args:
            X: Feature dataframe
            y: Labels (0=normal, 1=anomaly)
        """
        if len(X) == 0 or len(y) == 0:
            logger.warning("No data to train on")
            self.is_fitted = False
            return self
        
        # Check for at least some positive cases
        if y.sum() == 0:
            logger.warning("No positive cases (anomalies) in training data")
            self.is_fitted = False
            return self
            
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        logger.info(f"Preference model trained on {len(X)} samples ({y.sum()} anomalies)")
        return self
    
    def fit_with_split(self, X: pd.DataFrame, y: np.ndarray, test_size: float = 0.3) -> Dict:
        """
        Train with train/test split and return evaluation metrics.
        
        Args:
            X: Feature dataframe
            y: Labels (0=normal, 1=anomaly)
            test_size: Fraction for test set
            
        Returns:
            Dictionary with train and test metrics
        """
        # Split data (simulates Year 1 train, Year 2 test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size
        )
        
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        y_scores = self.model.predict_proba(X_test_scaled)[:, 1]
        
        self.train_metrics = {
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_scores) if y_test.sum() > 0 else 0,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'anomalies_train': int(y_train.sum()),
            'anomalies_test': int(y_test.sum())
        }
        
        logger.info(f"Preference model trained. Test F1: {self.train_metrics['f1_score']:.3f}")
        return self.train_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomaly labels.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of being an anomaly.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of probabilities (0 to 1, higher = more likely anomaly)
        """
        if not self.is_fitted:
            return np.zeros(len(X))
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def predict_similarity(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        For compatibility - returns probability scores.
        
        Returns:
            Tuple of (probability_scores, predicted_labels)
        """
        proba = self.predict_proba(X)
        preds = self.predict(X)
        return proba, preds
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importances from RandomForest.
        
        Returns:
            Dictionary of feature -> importance
        """
        if not self.is_fitted:
            return {}
        
        importances = self.model.feature_importances_
        return dict(sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1], reverse=True
        ))
    
    def get_suspicious_entries(self, X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Get entries flagged by preference model.
        
        Args:
            X: Data to check
            threshold: Minimum probability score (0-1)
            
        Returns:
            DataFrame with suspicious entries and scores
        """
        if not self.is_fitted:
            return pd.DataFrame()
            
        scores = self.predict_proba(X)
        
        # Find entries above threshold
        suspicious_mask = scores >= threshold
        
        result = X[suspicious_mask].copy()
        result['pref_score'] = scores[suspicious_mask]
        
        return result.sort_values('pref_score', ascending=False)


class FeedbackSimulator:
    """
    Simulates auditor feedback using ground truth labels.
    
    Per PDF Spec: "Since real auditors are not available, feedback must be simulated.
    The dataset already has ground truth labels, so use these to simulate
    what an auditor would say."
    """
    
    def __init__(self, label_column: str = 'label'):
        """
        Initialize simulator.
        
        Args:
            label_column: Column name containing ground truth (0=normal, 1=anomaly)
        """
        self.label_column = label_column
        
    def simulate_feedback(self, flagged_entries: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate auditor feedback using ground truth labels.
        
        Per PDF Spec:
        - If label == 1: True Positive (yes, real anomaly)
        - If label == 0: False Positive (no, false alarm)
        
        Args:
            flagged_entries: DataFrame of flagged entries (must have label column)
            
        Returns:
            DataFrame with simulated feedback
        """
        result = flagged_entries.copy()
        
        if self.label_column in result.columns:
            # Use actual ground truth
            result['simulated_label'] = result[self.label_column].astype(int)
            result['is_true_positive'] = result[self.label_column] == 1
            result['feedback'] = result.apply(
                lambda x: 'True Positive' if x[self.label_column] == 1 else 'False Positive',
                axis=1
            )
        else:
            # No ground truth - default to unknown
            result['simulated_label'] = 0
            result['is_true_positive'] = False
            result['feedback'] = 'Unknown (no label column)'
            logger.warning(f"No '{self.label_column}' column found - cannot simulate properly")
        
        return result
    
    def get_feedback_summary(self, simulated: pd.DataFrame) -> Dict[str, int]:
        """
        Summarize simulated feedback.
        
        Returns:
            Dictionary with counts
        """
        tp = (simulated['simulated_label'] == 1).sum()
        fp = (simulated['simulated_label'] == 0).sum()
        
        return {
            'total_reviewed': len(simulated),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'precision': tp / len(simulated) if len(simulated) > 0 else 0
        }
