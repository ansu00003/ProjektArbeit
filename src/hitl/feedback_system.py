"""
Human-in-the-Loop feedback system for anomaly detection.
"""

import os
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Always store the DB inside the project folder, no matter where the script
# is started from. Streamlit on Windows was crashing because the relative
# path "data/feedback/..." didn't exist in its working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "feedback" / "feedback.db")


class FeedbackSystem:
    """Stores auditor feedback in SQLite and computes simple metrics."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: Where to store the feedback database. Leave it as None
                and it will land inside the project folder.
        """
        # Make sure the path is absolute, otherwise different OSes behave differently.
        if db_path is None:
            db_path = DEFAULT_DB_PATH
        elif not os.path.isabs(db_path):
            db_path = str(PROJECT_ROOT / db_path)
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
        parent = os.path.dirname(self.db_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Feedback table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entry_id TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                auditor_label INTEGER NOT NULL,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                auditor_id TEXT,
                comments TEXT
            )
        ''')
        
        # Model performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                precision REAL,
                recall REAL,
                f1_score REAL,
                false_positive_rate REAL,
                contamination REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def add_feedback(self, entry_id: str, prediction: int, auditor_label: int, 
                     confidence: float = None, auditor_id: str = None, 
                     comments: str = None):
        """
        Add auditor feedback for a flagged entry.
        
        Args:
            entry_id: ID of the journal entry
            prediction: Model prediction (-1 for anomaly, 1 for normal)
            auditor_label: Auditor's label (1 for True Positive, 0 for False Positive)
            confidence: Confidence score
            auditor_id: ID of the auditor
            comments: Additional comments
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO feedback (entry_id, prediction, auditor_label, confidence, 
                                 auditor_id, comments)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (entry_id, prediction, auditor_label, confidence, auditor_id, comments))
        
        conn.commit()
        conn.close()
        logger.info(f"Feedback added for entry {entry_id}")
    
    def get_all_feedback(self) -> pd.DataFrame:
        """Retrieve all feedback from database."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM feedback", conn)
        conn.close()
        return df
    
    def get_recent_feedback(self, n: int = 100) -> pd.DataFrame:
        """
        Get most recent feedback entries.
        
        Args:
            n: Number of recent entries to retrieve
            
        Returns:
            DataFrame with recent feedback
        """
        conn = sqlite3.connect(self.db_path)
        query = f"SELECT * FROM feedback ORDER BY timestamp DESC LIMIT {n}"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_false_positives(self) -> pd.DataFrame:
        """Get all false positive feedback."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM feedback WHERE prediction = -1 AND auditor_label = 0"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def get_true_positives(self) -> pd.DataFrame:
        """Get all true positive feedback."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM feedback WHERE prediction = -1 AND auditor_label = 1"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    
    def calculate_feedback_metrics(
        self,
        scores: np.ndarray = None,
        ground_truth: Optional[pd.DataFrame] = None,
        label_column: str = "label",
    ) -> Dict[str, float]:
        """
        Two ways to compute the metrics:

        1) Feedback-only — we only see the entries the auditor reviewed,
           so we cannot know about anomalies that were never flagged.
           Recall, F1 and accuracy don't really make sense here, so we
           return NaN for them and only report precision honestly.

        2) Full ground-truth — pass the whole dataset (with the label
           column) and we compute real precision / recall / F1 / accuracy
           / FPR / ROC-AUC against everything. This is what's needed for a
           fair before/after comparison.

        Args:
            scores: anomaly scores, only used for ROC-AUC.
            ground_truth: the full dataset with a ground-truth label column.
                If passed, mode 2 is used.
            label_column: name of the ground-truth column.
        """
        feedback = self.get_all_feedback()

        empty = {
            'precision': 0.0, 'recall': 0.0, 'fpr': 0.0,
            'f1_score': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0,
            'mode': 'empty',
        }
        if len(feedback) == 0:
            return empty

        # Mode 2: we have ground truth, so real metrics are possible.
        if ground_truth is not None and label_column in ground_truth.columns:
            y_true = ground_truth[label_column].astype(int).to_numpy()
            if 'prediction' in ground_truth.columns:
                y_pred = (ground_truth['prediction'] == -1).astype(int).to_numpy()
            else:
                # No prediction column — assume reviewed entries are the flagged ones.
                flagged_ids = set(feedback['entry_id'].astype(str))
                y_pred = np.array(
                    [str(i) in flagged_ids for i in ground_truth.index],
                    dtype=int,
                )

            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if (precision + recall) > 0 else 0.0)
            accuracy = (tp + tn) / len(y_true)

            roc_auc = 0.0
            if scores is not None and len(np.unique(y_true)) > 1:
                try:
                    roc_auc = roc_auc_score(y_true, scores)
                except Exception as e:
                    logger.warning(f"Could not calculate ROC-AUC: {e}")

            return {
                'precision': precision, 'recall': recall, 'fpr': fpr,
                'f1_score': f1, 'accuracy': accuracy, 'roc_auc': roc_auc,
                'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                'mode': 'full_ground_truth',
            }

        # Mode 1: only feedback is available — precision and a "review-set
        # FP rate" are the only honest numbers we can give.
        anomalies = feedback[feedback['prediction'] == -1]
        if len(anomalies) == 0:
            return empty

        y_true = anomalies['auditor_label'].values
        true_positives = int((y_true == 1).sum())
        false_positives = int((y_true == 0).sum())
        n_reviewed = len(anomalies)

        precision = (true_positives / (true_positives + false_positives)
                     if (true_positives + false_positives) > 0 else 0.0)
        # FP rate within the reviewed subset only — not the population FPR.
        review_fp_rate = false_positives / n_reviewed if n_reviewed > 0 else 0.0

        roc_auc = 0.0
        if 'confidence' in anomalies.columns and anomalies['confidence'].notna().any():
            try:
                conf_scores = anomalies['confidence'].fillna(0).values
                if len(np.unique(y_true)) > 1:
                    roc_auc = roc_auc_score(y_true, -conf_scores)
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")

        return {
            'precision': precision,
            'recall': float('nan'),       # we don't see missed anomalies here
            'fpr': review_fp_rate,        # only on the reviewed entries
            'f1_score': float('nan'),     # needs real recall
            'accuracy': float('nan'),     # needs true negatives
            'roc_auc': roc_auc,
            'tp': true_positives,
            'fp': false_positives,
            'n_reviewed': n_reviewed,
            'mode': 'feedback_only',
        }
    
    def save_metrics(self, metrics: Dict[str, float], contamination: float):
        """
        Save model metrics to database.
        
        Args:
            metrics: Dictionary with performance metrics
            contamination: Current contamination parameter
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO model_metrics (precision, recall, f1_score, 
                                      false_positive_rate, contamination)
            VALUES (?, ?, ?, ?, ?)
        ''', (metrics['precision'], metrics['recall'], metrics['f1_score'], 
              metrics['fpr'], contamination))
        
        conn.commit()
        conn.close()
        logger.info("Metrics saved to database")
    
    def get_metrics_history(self) -> pd.DataFrame:
        """Get historical metrics."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query("SELECT * FROM model_metrics ORDER BY timestamp", conn)
        conn.close()
        return df
    
    def suggest_threshold_adjustment(self) -> Optional[float]:
        """
        Suggest new contamination threshold based on feedback.
        
        Returns:
            Suggested contamination value or None
        """
        metrics = self.calculate_feedback_metrics()
        
        # If FPR is too high, decrease contamination
        if metrics['fpr'] > 0.5:
            return 0.05
        # If precision is low, decrease contamination
        elif metrics['precision'] < 0.5:
            return 0.05
        # Otherwise keep current
        else:
            return None
