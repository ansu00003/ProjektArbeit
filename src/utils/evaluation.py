"""
Enhanced evaluation module for model stability and performance analysis.

Includes:
- Temporal split testing (train on older data, test on newer)
- Cross-validation for stability
- ROC curve and PR curve generation
- Before/After feedback comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    confusion_matrix
)
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Tuple, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation including stability analysis.
    """
    
    def __init__(self):
        self.metrics_before = None
        self.metrics_after = None
        
    def temporal_split_evaluation(self, model, X: pd.DataFrame, y: np.ndarray, 
                                  date_col: Optional[pd.Series] = None,
                                  train_ratio: float = 0.7) -> Dict:
        """
        Evaluate model with temporal split (simulates Year 1 train, Year 2 test).
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            date_col: Optional date column for temporal splitting
            train_ratio: Ratio of data for training
            
        Returns:
            Dictionary with train/test metrics
        """
        if date_col is not None and not date_col.isna().all():
            # Sort by date and split
            sorted_indices = date_col.sort_values().index
            split_idx = int(len(sorted_indices) * train_ratio)
            
            train_idx = sorted_indices[:split_idx]
            test_idx = sorted_indices[split_idx:]
            
            X_train = X.loc[train_idx]
            X_test = X.loc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            
            logger.info(f"Temporal split: {len(train_idx)} train, {len(test_idx)} test")
        else:
            # Random split as fallback
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_ratio, random_state=42
            )
            logger.info(f"Random split: {len(X_train)} train, {len(X_test)} test")
        
        # Train on train set
        model.fit(X_train)
        
        # Predict on both sets
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Convert predictions (-1, 1) to binary (1, 0)
        train_pred_binary = np.where(train_pred == -1, 1, 0)
        test_pred_binary = np.where(test_pred == -1, 1, 0)
        
        # Calculate metrics for both sets
        train_metrics = self._calculate_full_metrics(y_train, train_pred_binary)
        test_metrics = self._calculate_full_metrics(y_test, test_pred_binary)
        
        # Calculate stability (difference between train and test)
        stability = {
            'precision_diff': abs(train_metrics['precision'] - test_metrics['precision']),
            'recall_diff': abs(train_metrics['recall'] - test_metrics['recall']),
            'f1_diff': abs(train_metrics['f1_score'] - test_metrics['f1_score']),
            'is_stable': abs(train_metrics['f1_score'] - test_metrics['f1_score']) < 0.1
        }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'stability': stability,
            'train_size': len(y_train),
            'test_size': len(y_test)
        }
    
    def cross_validation_evaluation(self, model, X: pd.DataFrame, y: np.ndarray,
                                    cv_folds: int = 5) -> Dict:
        """
        K-fold cross-validation to assess model stability.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with CV results
        """
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Store metrics for each fold
        fold_metrics = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and predict
            model.fit(X_train)
            val_pred = model.predict(X_val)
            val_pred_binary = np.where(val_pred == -1, 1, 0)
            
            # Calculate metrics
            metrics = self._calculate_full_metrics(y_val, val_pred_binary)
            metrics['fold'] = fold_idx + 1
            fold_metrics.append(metrics)
        
        # Calculate mean and std across folds
        metrics_df = pd.DataFrame(fold_metrics)
        
        cv_results = {
            'mean_precision': metrics_df['precision'].mean(),
            'std_precision': metrics_df['precision'].std(),
            'mean_recall': metrics_df['recall'].mean(),
            'std_recall': metrics_df['recall'].std(),
            'mean_f1': metrics_df['f1_score'].mean(),
            'std_f1': metrics_df['f1_score'].std(),
            'mean_accuracy': metrics_df['accuracy'].mean(),
            'std_accuracy': metrics_df['accuracy'].std(),
            'fold_details': fold_metrics,
            'is_stable': metrics_df['f1_score'].std() < 0.1
        }
        
        logger.info(f"CV Results: F1={cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        
        return cv_results
    
    def compare_before_after(self, metrics_before: Dict, metrics_after: Dict) -> Dict:
        """
        Compare metrics before and after human feedback.
        
        Args:
            metrics_before: Metrics before feedback
            metrics_after: Metrics after feedback
            
        Returns:
            Comparison dictionary with improvements
        """
        self.metrics_before = metrics_before
        self.metrics_after = metrics_after
        
        comparison = {
            'precision': {
                'before': metrics_before.get('precision', 0),
                'after': metrics_after.get('precision', 0),
                'improvement': metrics_after.get('precision', 0) - metrics_before.get('precision', 0),
                'improvement_pct': ((metrics_after.get('precision', 0) - metrics_before.get('precision', 0)) / 
                                   max(metrics_before.get('precision', 0.001), 0.001) * 100)
            },
            'recall': {
                'before': metrics_before.get('recall', 0),
                'after': metrics_after.get('recall', 0),
                'improvement': metrics_after.get('recall', 0) - metrics_before.get('recall', 0),
                'improvement_pct': ((metrics_after.get('recall', 0) - metrics_before.get('recall', 0)) / 
                                   max(metrics_before.get('recall', 0.001), 0.001) * 100)
            },
            'f1_score': {
                'before': metrics_before.get('f1_score', 0),
                'after': metrics_after.get('f1_score', 0),
                'improvement': metrics_after.get('f1_score', 0) - metrics_before.get('f1_score', 0),
                'improvement_pct': ((metrics_after.get('f1_score', 0) - metrics_before.get('f1_score', 0)) / 
                                   max(metrics_before.get('f1_score', 0.001), 0.001) * 100)
            },
            'accuracy': {
                'before': metrics_before.get('accuracy', 0),
                'after': metrics_after.get('accuracy', 0),
                'improvement': metrics_after.get('accuracy', 0) - metrics_before.get('accuracy', 0),
                'improvement_pct': ((metrics_after.get('accuracy', 0) - metrics_before.get('accuracy', 0)) / 
                                   max(metrics_before.get('accuracy', 0.001), 0.001) * 100)
            },
            'roc_auc': {
                'before': metrics_before.get('roc_auc', 0),
                'after': metrics_after.get('roc_auc', 0),
                'improvement': metrics_after.get('roc_auc', 0) - metrics_before.get('roc_auc', 0),
                'improvement_pct': ((metrics_after.get('roc_auc', 0) - metrics_before.get('roc_auc', 0)) / 
                                   max(metrics_before.get('roc_auc', 0.001), 0.001) * 100)
            },
            'fpr': {
                'before': metrics_before.get('fpr', 0),
                'after': metrics_after.get('fpr', 0),
                'reduction': metrics_before.get('fpr', 0) - metrics_after.get('fpr', 0),
                'reduction_pct': ((metrics_before.get('fpr', 0) - metrics_after.get('fpr', 0)) / 
                                 max(metrics_before.get('fpr', 0.001), 0.001) * 100)
            }
        }
        
        return comparison
    
    def generate_roc_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                          title: str = "ROC Curve") -> go.Figure:
        """
        Generate ROC curve visualization.
        
        Args:
            y_true: True labels (0=normal, 1=anomaly)
            y_scores: Anomaly scores (higher = more anomalous)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, -y_scores)  # Negative because lower score = more anomalous
        roc_auc = roc_auc_score(y_true, -y_scores)
        
        # Create figure
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC (AUC = {roc_auc:.3f})',
            line=dict(color='#3498db', width=2)
        ))
        
        # Diagonal reference line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(color='gray', width=1, dash='dash')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def generate_precision_recall_curve(self, y_true: np.ndarray, y_scores: np.ndarray,
                                       title: str = "Precision-Recall Curve") -> go.Figure:
        """
        Generate Precision-Recall curve visualization.
        
        Args:
            y_true: True labels
            y_scores: Anomaly scores
            title: Chart title
            
        Returns:
            Plotly figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, -y_scores)
        avg_precision = average_precision_score(y_true, -y_scores)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR (AP = {avg_precision:.3f})',
            line=dict(color='#2ecc71', width=2),
            fill='tozeroy'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def generate_comparison_chart(self, comparison: Dict) -> go.Figure:
        """
        Generate before/after comparison bar chart.
        
        Args:
            comparison: Comparison dictionary from compare_before_after
            
        Returns:
            Plotly figure
        """
        metrics = ['precision', 'recall', 'f1_score', 'accuracy', 'roc_auc']
        before_values = [comparison[m]['before'] for m in metrics]
        after_values = [comparison[m]['after'] for m in metrics]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Before Feedback',
            x=metrics,
            y=before_values,
            marker_color='#e74c3c'
        ))
        
        fig.add_trace(go.Bar(
            name='After Feedback',
            x=metrics,
            y=after_values,
            marker_color='#2ecc71'
        ))
        
        fig.update_layout(
            title='Model Performance: Before vs After Human Feedback',
            xaxis_title='Metric',
            yaxis_title='Score',
            barmode='group',
            width=800,
            height=500,
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _calculate_full_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Calculate all evaluation metrics."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn)
        }
    
    def generate_stability_report(self, temporal_results: Dict, cv_results: Dict) -> str:
        """
        Generate a text report on model stability.
        
        Args:
            temporal_results: Results from temporal_split_evaluation
            cv_results: Results from cross_validation_evaluation
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*60)
        report.append("MODEL STABILITY ANALYSIS")
        report.append("="*60)
        report.append("")
        
        # Temporal split results
        report.append("📅 TEMPORAL SPLIT (Train on Year 1, Test on Year 2)")
        report.append("-" * 60)
        train = temporal_results['train_metrics']
        test = temporal_results['test_metrics']
        
        report.append(f"Training Set Performance:")
        report.append(f"  Precision: {train['precision']:.3f}")
        report.append(f"  Recall:    {train['recall']:.3f}")
        report.append(f"  F1 Score:  {train['f1_score']:.3f}")
        report.append("")
        
        report.append(f"Test Set Performance (Future Data):")
        report.append(f"  Precision: {test['precision']:.3f}")
        report.append(f"  Recall:    {test['recall']:.3f}")
        report.append(f"  F1 Score:  {test['f1_score']:.3f}")
        report.append("")
        
        stability = temporal_results['stability']
        report.append(f"Stability Assessment:")
        report.append(f"  F1 Difference: {stability['f1_diff']:.3f}")
        report.append(f"  Verdict: {'✅ STABLE' if stability['is_stable'] else '⚠️ UNSTABLE'}")
        report.append("")
        
        # Cross-validation results
        report.append("🔄 CROSS-VALIDATION (5-Fold)")
        report.append("-" * 60)
        report.append(f"Mean Precision: {cv_results['mean_precision']:.3f} ± {cv_results['std_precision']:.3f}")
        report.append(f"Mean Recall:    {cv_results['mean_recall']:.3f} ± {cv_results['std_recall']:.3f}")
        report.append(f"Mean F1 Score:  {cv_results['mean_f1']:.3f} ± {cv_results['std_f1']:.3f}")
        report.append(f"Mean Accuracy:  {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
        report.append("")
        report.append(f"Consistency: {'✅ CONSISTENT' if cv_results['is_stable'] else '⚠️ VARIABLE'}")
        report.append("")
        
        report.append("="*60)
        report.append("CONCLUSION")
        report.append("="*60)
        
        if temporal_results['stability']['is_stable'] and cv_results['is_stable']:
            report.append("✅ Model demonstrates good stability and generalization.")
            report.append("   Performance is consistent across different data splits.")
        else:
            report.append("⚠️ Model shows some instability or overfitting.")
            report.append("   Consider: more data, feature engineering, or regularization.")
        
        return "\n".join(report)
