"""
FastAPI Backend for HITL Anomaly Detection System

Provides REST API endpoints for:
- Data upload
- Anomaly detection
- Feedback submission
- Metrics retrieval
- Model retraining
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime
import io

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.models.anomaly_detector import AnomalyDetector
from src.models.preference_model import PreferenceModel
from src.hitl.feedback_system import FeedbackSystem
from src.utils.data_processor import DataProcessor
from src.utils.metrics import MetricsCalculator

# Initialize FastAPI
app = FastAPI(
    title="HITL Anomaly Detection API",
    description="Human-in-the-Loop Anomaly Detection for Journal Entries",
    version="1.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state (in production, use proper state management)
class AppState:
    def __init__(self):
        self.detector = None
        self.preference_model = PreferenceModel()
        self.feedback_system = FeedbackSystem()
        self.data_processor = DataProcessor()
        self.current_data = None
        self.processed_data = None
        self.predictions = None
        self.scores = None

state = AppState()

# Pydantic models for request/response
class FeedbackRequest(BaseModel):
    entry_id: str
    prediction: int
    auditor_label: int  # 1 = True Positive, 0 = False Positive
    confidence: Optional[float] = None
    auditor_id: Optional[str] = None
    comments: Optional[str] = None

class RetrainRequest(BaseModel):
    contamination: Optional[float] = 0.1

class AnomalyResponse(BaseModel):
    entry_id: str
    anomaly_score: float
    prediction: int
    data: Dict

# API Endpoints

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "HITL Anomaly Detection API",
        "version": "1.0.0"
    }

@app.post("/api/upload")
async def upload_data(file: UploadFile = File(...)):
    """
    Upload journal entry CSV data.
    
    Returns:
        Summary of uploaded data
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Process data
        state.data_processor.identify_features(df)
        df = state.data_processor.handle_missing_values(df.copy())
        df = state.data_processor.encode_categorical(df)
        df = state.data_processor.create_features(df)
        
        state.current_data = df
        
        return {
            "status": "success",
            "message": f"Uploaded {len(df)} entries",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error uploading file: {str(e)}")

@app.post("/api/train")
async def train_model(request: RetrainRequest):
    """
    Train Isolation Forest model.
    
    Args:
        contamination: Expected anomaly rate (0.01 to 0.5)
    """
    if state.current_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded. Upload data first.")
    
    try:
        # Get features
        features = state.data_processor.get_features_for_training(state.current_data)
        
        # Train model
        state.detector = AnomalyDetector(contamination=request.contamination)
        state.detector.fit(features)
        
        return {
            "status": "success",
            "message": "Model trained successfully",
            "contamination": request.contamination,
            "features_used": state.detector.feature_names,
            "samples_trained": len(features)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.get("/api/anomalies")
async def get_anomalies(limit: int = 100):
    """
    Get detected anomalies with SHAP explanations.
    
    Args:
        limit: Maximum number of anomalies to return
    """
    if state.detector is None:
        raise HTTPException(status_code=400, detail="Model not trained. Train model first.")
    
    if state.current_data is None:
        raise HTTPException(status_code=400, detail="No data available.")
    
    try:
        # Get features
        features = state.data_processor.get_features_for_training(state.current_data)
        
        # Predict
        predictions, scores = state.detector.predict_with_scores(features)
        state.predictions = predictions
        state.scores = scores
        
        # Get anomalies
        anomaly_mask = predictions == -1
        anomaly_indices = np.where(anomaly_mask)[0]
        
        # Prepare response
        anomalies = []
        for idx in anomaly_indices[:limit]:
            entry = state.current_data.iloc[idx].to_dict()
            
            # Get SHAP explanation
            reasons = state.detector.get_top_reasons(features, idx, top_n=3)
            
            anomalies.append({
                "entry_id": str(idx),
                "anomaly_score": float(scores[idx]),
                "prediction": int(predictions[idx]),
                "reasons": reasons,
                "data": {k: (str(v) if pd.notna(v) else None) for k, v in entry.items()}
            })
        
        return {
            "status": "success",
            "total_entries": len(state.current_data),
            "total_anomalies": int(anomaly_mask.sum()),
            "anomaly_rate": float(anomaly_mask.sum() / len(state.current_data)),
            "anomalies": anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@app.post("/api/feedback")
async def submit_feedback(feedback: FeedbackRequest):
    """
    Submit human feedback for a flagged anomaly.
    
    Args:
        feedback: Feedback data including label and comments
    """
    try:
        state.feedback_system.add_feedback(
            entry_id=feedback.entry_id,
            prediction=feedback.prediction,
            auditor_label=feedback.auditor_label,
            confidence=feedback.confidence,
            auditor_id=feedback.auditor_id,
            comments=feedback.comments
        )
        
        return {
            "status": "success",
            "message": "Feedback recorded",
            "entry_id": feedback.entry_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback error: {str(e)}")

@app.get("/api/metrics")
async def get_metrics():
    """
    Get current performance metrics based on feedback.
    """
    try:
        # Get feedback-based metrics
        feedback_metrics = state.feedback_system.calculate_feedback_metrics(state.scores)
        
        # Get feedback summary
        all_feedback = state.feedback_system.get_all_feedback()
        tp_count = len(state.feedback_system.get_true_positives())
        fp_count = len(state.feedback_system.get_false_positives())
        
        return {
            "status": "success",
            "metrics": {
                "precision": feedback_metrics['precision'],
                "recall": feedback_metrics['recall'],
                "f1_score": feedback_metrics['f1_score'],
                "accuracy": feedback_metrics['accuracy'],
                "roc_auc": feedback_metrics['roc_auc'],
                "false_positive_rate": feedback_metrics['fpr']
            },
            "feedback_summary": {
                "total_reviews": len(all_feedback),
                "true_positives": tp_count,
                "false_positives": fp_count
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")

@app.post("/api/retrain")
async def retrain_model():
    """
    Retrain model based on feedback.
    Uses feedback to adjust threshold or train preference model.
    """
    if state.detector is None:
        raise HTTPException(status_code=400, detail="No model to retrain.")
    
    try:
        # Check if threshold adjustment is needed
        suggested = state.feedback_system.suggest_threshold_adjustment()
        
        if suggested:
            # Retrain with new threshold
            features = state.data_processor.get_features_for_training(state.current_data)
            state.detector.adjust_threshold(suggested)
            state.detector.fit(features)
            
            return {
                "status": "success",
                "message": "Model retrained with adjusted threshold",
                "new_contamination": suggested
            }
        else:
            return {
                "status": "success",
                "message": "No threshold adjustment needed",
                "current_contamination": state.detector.contamination
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining error: {str(e)}")

@app.get("/api/history")
async def get_feedback_history(limit: int = 100):
    """
    Get feedback history.
    
    Args:
        limit: Maximum number of entries to return
    """
    try:
        feedback = state.feedback_system.get_recent_feedback(n=limit)
        
        return {
            "status": "success",
            "count": len(feedback),
            "history": feedback.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History error: {str(e)}")

@app.get("/api/metrics/history")
async def get_metrics_history():
    """Get historical metrics over time."""
    try:
        history = state.feedback_system.get_metrics_history()
        
        return {
            "status": "success",
            "count": len(history),
            "history": history.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics history error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
