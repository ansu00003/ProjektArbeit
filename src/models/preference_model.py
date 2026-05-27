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
        # Pinning random_state so the split is the same every run — otherwise
        # the metrics jump around. Stratify on y so we don't accidentally end
        # up with zero anomalies in the test set (they're only ~1% of data).
        stratify = y if (np.sum(y == 1) >= 2 and np.sum(y == 0) >= 2) else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
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
    Pretends to be the auditor by reading the ground-truth column.

    Some older scripts used 'is_anomaly' instead of 'label', so we accept
    both names. 'label' is the one we prefer.
    """

    LABEL_CANDIDATES = ('label', 'is_anomaly')

    def __init__(self, label_column: Optional[str] = None):
        """
        Args:
            label_column: Set this only if you want to force a specific
                column name. Leave it None for auto-detect.
        """
        self.label_column = label_column

    def _resolve_label_column(self, df: pd.DataFrame) -> Optional[str]:
        if self.label_column and self.label_column in df.columns:
            return self.label_column
        for name in self.LABEL_CANDIDATES:
            if name in df.columns:
                return name
        return None

    def simulate_feedback(self, flagged_entries: pd.DataFrame) -> pd.DataFrame:
        """
        Pretend the auditor reviewed the given entries.

        label == 1 -> True Positive, label == 0 -> False Positive.

        We raise instead of returning fake labels if no ground-truth column
        is found — the old version silently labelled everything 0 and that
        quietly broke every downstream metric.
        """
        result = flagged_entries.copy()
        col = self._resolve_label_column(result)

        if col is None:
            raise ValueError(
                "FeedbackSimulator: no ground-truth column found. "
                f"Looked for any of {self.LABEL_CANDIDATES}. "
                "Rename your label column to 'label' or pass "
                "`label_column=...` explicitly."
            )

        result['simulated_label'] = result[col].astype(int)
        result['is_true_positive'] = result[col] == 1
        result['feedback'] = np.where(result[col] == 1,
                                      'True Positive', 'False Positive')
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
