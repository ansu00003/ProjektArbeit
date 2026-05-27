"""Data preprocessing utilities for journal entries.

Per PDF Specification, the key columns are:
- amount: Transaction amount
- user: User who created entry (categorical -> user_encoded)
- weekend: 0=weekday, 1=Saturday, 2=Sunday
- nwh: 1=non-working hours, 0=normal
- promptly: 1=on time, 2/3=delayed
- top_n: 1=unusually high amount
- high_cash: 1=high cash transaction
- marking: 0=normal, 1-6=various flags
- label: 0=normal, 1=ANOMALY (ground truth)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Features for realistic detection
# Per user feedback: include high_cash, marking for explanation purposes
# But they are accessible from original_df for analysis
PDF_FEATURES = ['amount', 'weekend', 'nwh', 'promptly', 'user_encoded']

# ALL features including fraud indicators (for analysis/explanation)
ALL_ANALYSIS_FEATURES = ['amount', 'weekend', 'nwh', 'promptly', 'user_encoded', 
                         'high_cash', 'marking', 'top_n', 'gl_account']


class DataProcessor:
    """
    Preprocess journal entry data for anomaly detection.
    """
    
    def __init__(self):
        self.numeric_features = []
        self.categorical_features = []
        
    def load_data(self, filepath: str, sep: str = ',') -> pd.DataFrame:
        """Load journal entry data from CSV."""
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath, sep=sep, on_bad_lines='skip', encoding='utf-8')
        except:
            df = pd.read_csv(filepath, sep=';', on_bad_lines='skip', encoding='utf-8')
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df
    
    def identify_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Identify numeric and categorical features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with identified features
        """
        self.numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        
        logger.info(f"Numeric features: {self.numeric_features}")
        logger.info(f"Categorical features: {self.categorical_features}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info(f"Missing values before: {df.isnull().sum().sum()}")
        
        # Fill numeric with median
        for col in self.numeric_features:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val if pd.notna(median_val) else 0)
        
        # Fill categorical with mode or 'Unknown'
        for col in self.categorical_features:
            if col in df.columns and df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if len(mode_val) > 0 else 'Unknown')
        
        logger.info(f"Missing values after: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables.
        Per PDF: encode 'user' column as 'user_encoded'
        """
        # Per PDF spec: encode 'user' column specifically
        if 'user' in df.columns:
            le = LabelEncoder()
            df['user_encoded'] = le.fit_transform(df['user'].astype(str))
            logger.info(f"Encoded 'user' column with {len(le.classes_)} unique values")
        
        # Also encode other categorical features
        for col in self.categorical_features:
            if col in df.columns and col != 'user':  # user already encoded
                df[f'{col}_encoded'] = pd.factorize(df[col])[0]
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for anomaly detection.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with additional features
        """
        # Amount-based features (check various column names)
        amount_col = None
        for col in ['amount', 'Amount', 'AMOUNT', 'betrag', 'Betrag']:
            if col in df.columns:
                amount_col = col
                break
        
        if amount_col:
            df['amount_numeric'] = pd.to_numeric(df[amount_col], errors='coerce').fillna(0)
            df['amount_abs'] = df['amount_numeric'].abs()
            df['amount_log'] = np.log1p(df['amount_abs'])
        
        # Date-based features (check various column names)
        date_col = None
        for col in ['date', 'Date', 'posting_date', 'document_date', 'entered_date']:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            try:
                df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
                df['day_of_week'] = df['date_parsed'].dt.dayofweek.fillna(0).astype(int)
                df['month'] = df['date_parsed'].dt.month.fillna(1).astype(int)
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            except:
                pass
        
        return df
    
    def get_features_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features for model training.
        
        Per PDF Specification:
        features = ['amount', 'weekend', 'nwh', 'promptly', 'top_n', 'high_cash', 'user_encoded']
        """
        # First, try to use PDF-specified features
        pdf_features_available = [col for col in PDF_FEATURES if col in df.columns]
        
        if len(pdf_features_available) >= 3:
            # Use PDF-specified features
            logger.info(f"Using PDF-specified features: {pdf_features_available}")
            return df[pdf_features_available].fillna(0)
        
        # Fallback: build feature list from available columns
        feature_cols = []
        
        # Add numeric features
        for col in self.numeric_features:
            if col in df.columns:
                feature_cols.append(col)
        
        # Add encoded categorical features
        for col in self.categorical_features:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                feature_cols.append(encoded_col)
        
        # Add engineered features
        engineered = ['amount_abs', 'amount_log', 'amount_numeric', 'day_of_week', 'month', 'is_weekend']
        for col in engineered:
            if col in df.columns:
                feature_cols.append(col)
        
        # Remove duplicates and filter
        feature_cols = list(dict.fromkeys([col for col in feature_cols if col in df.columns]))
        
        if len(feature_cols) == 0:
            logger.warning("No features found! Using all numeric columns.")
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Training features: {feature_cols}")
        return df[feature_cols].fillna(0)
    
    # Some old scripts used 'is_anomaly' instead of 'label'. Try both so we
    # don't crash on older CSVs. Keep this list aligned with
    # FeedbackSimulator.LABEL_CANDIDATES.
    LABEL_CANDIDATES = ('label', 'is_anomaly')

    def get_label_column(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Return the ground-truth column (0 = normal, 1 = anomaly), or None."""
        for name in self.LABEL_CANDIDATES:
            if name in df.columns:
                return df[name]
        return None
    
    def process_pipeline(self, filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Complete preprocessing pipeline.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Tuple of (original_df, features_df)
        """
        df = self.load_data(filepath)
        df = self.identify_features(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical(df)
        df = self.create_features(df)
        
        features = self.get_features_for_training(df)
        
        logger.info(f"Pipeline complete. Feature shape: {features.shape}")
        return df, features
