# Human-in-the-Loop Anomaly Detection for Journal Entry Testing

## Project Overview
This project implements a Human-in-the-Loop (HITL) system for detecting anomalies in financial journal entries using machine learning (Isolation Forest) combined with auditor feedback to reduce false positives.

## Structure
```
projektarbeit code/
├── data/                   # CSV data files
│   ├── raw/               # Original journal entry data
│   ├── processed/         # Cleaned data
│   └── feedback/          # Auditor feedback logs
├── src/                   # Source code
│   ├── models/            # ML models
│   ├── hitl/              # Human-in-the-Loop logic
│   ├── dashboard/         # Streamlit dashboard
│   └── utils/             # Helper functions
├── notebooks/             # Jupyter notebooks for analysis
├── tests/                 # Unit tests
├── config/                # Configuration files
└── results/               # Model outputs and metrics
```

## Setup
```bash
pip install -r requirements.txt
```

## Run Dashboard
```bash
streamlit run src/dashboard/app.py
```

## Key Features
- **Isolation Forest** for anomaly detection
- **HITL Feedback Loop** for continuous improvement
- **Real-time Dashboard** for auditor review
- **Metrics Tracking**: Precision, Recall, False Positive Rate
