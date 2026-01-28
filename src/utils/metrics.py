"""
Evaluation metrics for anomaly detection performance.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """
    Calculate and track performance metrics for anomaly detection.
    """
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: Ground truth labels (1 for anomaly, 0 for normal)
            y_pred: Predicted labels (-1 for anomaly, 1 for normal)
            
        Returns:
            Dictionary with metrics
        """
        # Convert predictions to binary (1 for anomaly, 0 for normal)
        y_pred_binary = np.where(y_pred == -1, 1, 0)
        
        # Calculate metrics
        precision = precision_score(y_true, y_pred_binary, zero_division=0)
        recall = recall_score(y_true, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        # False positive rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False negative rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fpr,
            'false_negative_rate': fnr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
        
        logger.info(f"Metrics calculated: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict[str, float]):
        """Print metrics in formatted way."""
        print("\n" + "="*50)
        print("PERFORMANCE METRICS")
        print("="*50)
        print(f"Precision:           {metrics['precision']:.3f}")
        print(f"Recall:              {metrics['recall']:.3f}")
        print(f"F1 Score:            {metrics['f1_score']:.3f}")
        print(f"False Positive Rate: {metrics['false_positive_rate']:.3f}")
        print(f"False Negative Rate: {metrics['false_negative_rate']:.3f}")
        print("\nConfusion Matrix:")
        print(f"  True Positives:  {metrics['true_positives']}")
        print(f"  False Positives: {metrics['false_positives']}")
        print(f"  True Negatives:  {metrics['true_negatives']}")
        print(f"  False Negatives: {metrics['false_negatives']}")
        print("="*50 + "\n")
    
    @staticmethod
    def compare_metrics(before: Dict[str, float], after: Dict[str, float]) -> Dict[str, float]:
        """
        Compare metrics before and after feedback.
        
        Args:
            before: Metrics before feedback
            after: Metrics after feedback
            
        Returns:
            Dictionary with improvements
        """
        improvements = {
            'precision_improvement': after['precision'] - before['precision'],
            'recall_improvement': after['recall'] - before['recall'],
            'f1_improvement': after['f1_score'] - before['f1_score'],
            'fpr_reduction': before['false_positive_rate'] - after['false_positive_rate']
        }
        
        logger.info("Metrics comparison:")
        for key, value in improvements.items():
            logger.info(f"  {key}: {value:+.3f}")
        
        return improvements
