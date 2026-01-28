# Model configuration
MODEL_CONFIG = {
    'contamination': 0.1,
    'random_state': 42,
    'n_estimators': 100
}

# Data paths
DATA_PATHS = {
    'raw': 'data/raw/',
    'processed': 'data/processed/',
    'feedback': 'data/feedback/'
}

# Feedback system
FEEDBACK_DB = 'data/feedback/feedback.db'

# Thresholds
METRICS_THRESHOLDS = {
    'min_precision': 0.7,
    'max_fpr': 0.3
}
