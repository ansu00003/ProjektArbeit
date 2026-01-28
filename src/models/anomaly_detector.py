"""
Isolation Forest model for anomaly detection in journal entries.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Tuple, Dict, List, Optional
import logging
import shap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# DOMAIN KNOWLEDGE: Human-readable mappings and fraud indicators
# ============================================================================

# User name mappings (reverse of label encoding)
USER_MAPPING = {
    0: 'Alice', 1: 'Bob', 2: 'Charlie', 3: 'David', 4: 'Fred', 5: 'Max'
}

# Weekend mappings
WEEKEND_MAPPING = {
    0: 'Weekday',
    1: 'Saturday', 
    2: 'Sunday'
}

# Marking (fraud pattern) descriptions
MARKING_MAPPING = {
    0: 'Normal',
    1: 'Pattern #1 - Duplicate entry',
    2: 'Pattern #2 - Round number',
    3: 'Pattern #3 - Split transaction',
    4: 'Pattern #4 - Unusual timing',
    5: 'Pattern #5 - Account anomaly',
    6: 'Pattern #6 - Payment reversal'
}

# Features that should NOT be flagged when they have majority/normal values
# Format: {feature_name: normal_value}
NORMAL_VALUES = {
    'weekend': 0,        # Weekday is normal (87.6%)
    'promptly': 1,       # Promptly=1 is normal (99%+)
    'nwh': 0,            # Working hours is normal
    'high_cash': 0,      # Bank transfer is normal
    'marking': 0,        # No fraud pattern is normal
}

# HIGH RISK indicators - these are the PRIMARY fraud signals
HIGH_RISK_INDICATORS = ['high_cash', 'marking']

# MEDIUM RISK indicators
MEDIUM_RISK_INDICATORS = ['nwh', 'weekend', 'top_n']

# Categorical features that should NOT use numeric comparison
CATEGORICAL_FEATURES = ['user_encoded', 'user', 'gl_account', 'account_code']


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for journal entries.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in dataset
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame) -> 'AnomalyDetector':
        """
        Train the anomaly detection model.
        
        Args:
            X: Training data (journal entries)
            
        Returns:
            self
        """
        logger.info(f"Training Isolation Forest with {len(X)} samples")
        self.feature_names = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled)
        
        logger.info("Training complete")
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict anomalies.
        
        Args:
            X: Data to predict on
            
        Returns:
            Array of predictions (-1 for anomaly, 1 for normal)
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_with_scores(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies with anomaly scores.
        
        Args:
            X: Data to predict on
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.score_samples(X_scaled)
        
        return predictions, scores
    
    def adjust_threshold(self, new_contamination: float):
        """
        Adjust the contamination threshold based on feedback.
        
        Args:
            new_contamination: New contamination parameter
        """
        logger.info(f"Adjusting threshold: {self.contamination} -> {new_contamination}")
        self.contamination = new_contamination
        self.model.contamination = new_contamination
    
    def save(self, path: str):
        """Save model to disk."""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'contamination': self.contamination,
            'feature_names': self.feature_names
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.contamination = data['contamination']
        self.feature_names = data['feature_names']
        logger.info(f"Model loaded from {path}")
    
    def explain_anomaly(self, X: pd.DataFrame, idx: int) -> Dict[str, float]:
        """
        Explain why a specific entry is flagged as anomaly.
        
        Args:
            X: Feature dataframe
            idx: Index of the anomaly to explain
            
        Returns:
            Dictionary of feature contributions
        """
        X_scaled = self.scaler.transform(X)
        
        # Use TreeExplainer for Isolation Forest
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_scaled[idx:idx+1])
        
        # Create feature importance dict
        contributions = {}
        for i, feature in enumerate(self.feature_names):
            contributions[feature] = float(shap_values[0][i])
        
        # Sort by absolute contribution
        contributions = dict(sorted(contributions.items(), 
                                   key=lambda x: abs(x[1]), reverse=True))
        return contributions
    
    def get_top_reasons(self, X: pd.DataFrame, idx: int, top_n: int = 5, 
                         original_df: pd.DataFrame = None) -> List[str]:
        """
        Get human-readable reasons for anomaly with proper domain logic.
        
        Args:
            X: Feature dataframe
            idx: Index of anomaly
            top_n: Number of top reasons
            original_df: Original dataframe with all columns (for human-readable values)
            
        Returns:
            List of reason strings
        """
        # Get actual values for this entry
        row = X.iloc[idx] if idx < len(X) else None
        orig_row = original_df.iloc[idx] if original_df is not None and idx < len(original_df) else None
        
        high_risk = []
        medium_risk = []
        info_notes = []
        
        # ===== CHECK HIGH RISK INDICATORS FIRST =====
        # These are the PRIMARY fraud signals per ground truth: anomaly = (high_cash=1 AND marking>0)
        
        # Check high_cash (CRITICAL fraud indicator)
        high_cash_val = self._get_value(row, orig_row, 'high_cash')
        if high_cash_val == 1:
            amount = self._get_value(row, orig_row, 'amount', 'amount_numeric', 'amount_abs')
            amount_str = f"EUR {amount:,.2f}" if amount else "large amount"
            high_risk.append(f"HIGH RISK: CASH PAYMENT of {amount_str} (unusual for large amounts)")
        
        # Check marking (fraud pattern indicator)
        marking_val = self._get_value(row, orig_row, 'marking')
        if marking_val and marking_val > 0:
            pattern_desc = MARKING_MAPPING.get(int(marking_val), f'Pattern #{int(marking_val)}')
            high_risk.append(f"HIGH RISK: Fraud {pattern_desc} detected")
        
        # ===== CHECK MEDIUM RISK INDICATORS =====
        
        # Check weekend (only flag if weekend=1 or 2, NOT 0)
        weekend_val = self._get_value(row, orig_row, 'weekend')
        if weekend_val and weekend_val > 0:
            day_name = WEEKEND_MAPPING.get(int(weekend_val), 'Weekend')
            medium_risk.append(f"MEDIUM RISK: Entered on {day_name.upper()}")
        
        # Check non-working hours (only flag if nwh=1)
        nwh_val = self._get_value(row, orig_row, 'nwh')
        if nwh_val == 1:
            medium_risk.append(f"MEDIUM RISK: Entered during NON-WORKING HOURS")
        
        # Check top_n (unusually high amount)
        top_n_val = self._get_value(row, orig_row, 'top_n')
        if top_n_val == 1:
            medium_risk.append(f"MEDIUM RISK: Amount in TOP percentile of transactions")
        
        # Check promptly (only flag if NOT promptly, i.e., 2 or 3)
        promptly_val = self._get_value(row, orig_row, 'promptly')
        if promptly_val and promptly_val > 1:
            medium_risk.append(f"MEDIUM RISK: Entry was DELAYED (not posted promptly)")
        
        # ===== AMOUNT CHECK (only if significant) =====
        amount_val = self._get_value(row, orig_row, 'amount', 'amount_numeric', 'amount_abs')
        if amount_val and 'amount' in X.columns:
            mean_amount = X['amount'].mean() if 'amount' in X.columns else 0
            std_amount = X['amount'].std() if 'amount' in X.columns else 1
            if std_amount > 0:
                z_score = (amount_val - mean_amount) / std_amount
                if z_score > 3:  # Only flag if VERY high (>3 std)
                    # Check if it's cash - if not cash, it's less suspicious
                    if high_cash_val != 1:
                        info_notes.append(f"NOTE: Amount EUR {amount_val:,.2f} is high but paid via normal bank transfer")
                    else:
                        medium_risk.append(f"MEDIUM RISK: Amount EUR {amount_val:,.2f} is unusually HIGH")
        
        # ===== BUILD FINAL REASON LIST =====
        reasons = []
        
        # Add high risk first
        reasons.extend(high_risk[:top_n])
        
        # Add medium risk if we have room
        remaining = top_n - len(reasons)
        if remaining > 0:
            reasons.extend(medium_risk[:remaining])
        
        # Add info notes if we have room
        remaining = top_n - len(reasons)
        if remaining > 0:
            reasons.extend(info_notes[:remaining])
        
        # If no reasons found, entry is likely LOW RISK
        if not reasons:
            reasons.append("LOW RISK: No major fraud indicators detected")
            # Add normal status info
            normal_items = []
            if high_cash_val == 0:
                normal_items.append("Bank Transfer (normal)")
            if weekend_val == 0:
                normal_items.append("Weekday (normal)")
            if nwh_val == 0:
                normal_items.append("Working hours (normal)")
            if promptly_val == 1:
                normal_items.append("Posted promptly (normal)")
            if normal_items:
                reasons.append(f"Normal indicators: {', '.join(normal_items[:3])}")
        
        return reasons
    
    def _get_value(self, row, orig_row, *col_names):
        """
        Get value from row, trying multiple column names.
        Prefers original dataframe, falls back to feature dataframe.
        """
        for col in col_names:
            # Try original row first (has more columns)
            if orig_row is not None and col in orig_row.index:
                val = orig_row[col]
                if pd.notna(val):
                    return val
            # Fall back to feature row
            if row is not None and col in row.index:
                val = row[col]
                if pd.notna(val):
                    return val
        return None
    
    def get_risk_level(self, X: pd.DataFrame, idx: int, 
                       original_df: pd.DataFrame = None) -> str:
        """
        Get risk level classification for an entry.
        
        Returns:
            'HIGH', 'MEDIUM', or 'LOW'
        """
        row = X.iloc[idx] if idx < len(X) else None
        orig_row = original_df.iloc[idx] if original_df is not None and idx < len(original_df) else None
        
        # Check high risk indicators
        high_cash = self._get_value(row, orig_row, 'high_cash')
        marking = self._get_value(row, orig_row, 'marking')
        
        # Ground truth rule: anomaly = (high_cash=1 AND marking>0)
        if high_cash == 1 and marking and marking > 0:
            return 'HIGH'
        elif high_cash == 1 or (marking and marking > 0):
            return 'MEDIUM'
        
        # Check medium risk indicators
        nwh = self._get_value(row, orig_row, 'nwh')
        weekend = self._get_value(row, orig_row, 'weekend')
        top_n_val = self._get_value(row, orig_row, 'top_n')
        
        medium_count = sum([
            nwh == 1,
            weekend and weekend > 0,
            top_n_val == 1
        ])
        
        if medium_count >= 2:
            return 'MEDIUM'
        
        return 'LOW'
