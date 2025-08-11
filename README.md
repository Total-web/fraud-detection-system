# Cyber Guard Nepal

**AI-Powered Fraud Detection System with Investigation Management**

A comprehensive fraud detection system built with Python and Machine Learning, featuring real-time transaction analysis, investigation workflow management, and adaptive learning capabilities.

## 🚀 Features

### Core Fraud Detection
- **Advanced ML Models**: Isolation Forest for unsupervised detection + Random Forest for supervised learning
- **Real-time Analysis**: Process transactions with immediate risk assessment
- **Explainable AI**: Detailed explanations for every fraud prediction
- **Adaptive Learning**: System improves accuracy through user feedback

### Investigation Management
- **Priority-based Queue**: Low, Medium, High, Urgent priority levels with color coding
- **Team Collaboration**: Assign investigations to team members
- **Resolution Workflow**: Track investigations from flagging to resolution
- **Comprehensive Reporting**: Summary, detailed, and statistical reports

### Professional Features
- **Interactive GUI**: Built with Tkinter for cross-platform compatibility
- **Database Integration**: SQLite for persistent storage and audit trails
- **Export Capabilities**: CSV/Excel export for results and reports
- **Model Versioning**: Track improvements and performance over time

## 📋 Requirements

- Python 3.8+
- Required packages listed in `requirements.txt`

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/fraud-detection-system.git
   cd fraud-detection-system



Required 
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
fraud_detection_env/
venv/
env/

# Database files
*.db
*.sqlite
*.sqlite3

# Logs
*.log
logs/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Model files (optional - include if you want to share trained models)
models/*.pkl
models/*.joblib

# Data files (optional - exclude sensitive data)
data/*.csv
data/*.xlsx
uploads/

# Temporary files
temp/
tmp/
