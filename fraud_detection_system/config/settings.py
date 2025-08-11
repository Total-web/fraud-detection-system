import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Database settings
DATABASE_PATH = DATA_DIR / "fraud_detection.db"

# Model settings
MODEL_CONFIG = {
    'contamination': 0.1,  # Expected fraud rate
    'random_state': 42,
    'n_estimators': 100
}

# UI settings
UI_CONFIG = {
    'window_size': '1200x800',
    'theme': 'clam',
    'colors': {
        'normal': '#28a745',
        'suspicious': '#ffc107', 
        'high_risk': '#dc3545',
        'background': '#f8f9fa'
    }
}

# Demo settings
DEMO_CONFIG = {
    'auto_skip_normal': True,
    'feedback_threshold': 10,  # Minimum feedback for retraining
    'confidence_threshold': 0.7
}