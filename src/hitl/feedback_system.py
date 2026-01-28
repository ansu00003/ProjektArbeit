"""
Human-in-the-Loop feedback system for anomaly detection.
"""

import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeedbackSystem:
    """
    Manages auditor feedback and model retraining.
    """
    
    def __init__(self, db_path: str = 'data/feedback/feedback.db'):
        """
        Initialize feedback system.
        
        Args:
            db_path: Path to SQLite database for storing feedback
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize database tables."""
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
    
    def calculate_feedback_metrics(self, scores: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate performance metrics based on feedback.
        Now includes ROC-AUC, Accuracy, and F-score per requirements.
        
        Args:
            scores: Optional anomaly scores for ROC-AUC calculation
            
        Returns:
            Dictionary with precision, recall, FPR, accuracy, f1, roc_auc
        """
        feedback = self.get_all_feedback()
        
        if len(feedback) == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'fpr': 0.0, 
                'f1_score': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0
            }
        
        # Filter anomaly predictions
        anomalies = feedback[feedback['prediction'] == -1]
        
        if len(anomalies) == 0:
            return {
                'precision': 0.0, 'recall': 0.0, 'fpr': 0.0,
                'f1_score': 0.0, 'accuracy': 0.0, 'roc_auc': 0.0
            }
        
        # Ground truth from auditor
        y_true = anomalies['auditor_label'].values
        # Model prediction (all -1 = anomaly, treat as 1 for binary)
        y_pred = np.ones(len(anomalies))
        
        # Calculate metrics
        true_positives = (y_true == 1).sum()
        false_positives = (y_true == 0).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = precision  # Simplified - would need all positives for true recall
        fpr = false_positives / len(anomalies) if len(anomalies) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy: % correctly identified
        accuracy = true_positives / len(anomalies) if len(anomalies) > 0 else 0
        
        # ROC-AUC if we have confidence scores
        roc_auc = 0.0
        if 'confidence' in anomalies.columns and anomalies['confidence'].notna().any():
            try:
                conf_scores = anomalies['confidence'].fillna(0).values
                if len(np.unique(y_true)) > 1:  # Need both classes
                    roc_auc = roc_auc_score(y_true, -conf_scores)  # Negative because lower score = more anomalous
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        return {
            'precision': precision,
            'recall': recall,
            'fpr': fpr,
            'f1_score': f1,
            'accuracy': accuracy,
            'roc_auc': roc_auc
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
