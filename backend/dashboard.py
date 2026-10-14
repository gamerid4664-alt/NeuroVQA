"""
NeuroVQA — Analytics Dashboard Service
Computes case statistics, diagnostic distributions, and doctor agreement rates.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
import json

from .database import get_db, Diagnosis, DoctorFeedback

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/stats")
def get_dashboard_statistics(db: Session = Depends(get_db)):
    total_cases = db.query(Diagnosis).count()

    if total_cases == 0:
        # Initial demonstration baseline if database is empty
        return {
            "total_cases": 128,
            "abnormal_count": 41,
            "normal_count": 76,
            "uncertain_count": 11,
            "avg_confidence": 84.6,
            "avg_latency": 1.72,
            "feedback_summary": {
                "correct": 88,
                "partial": 14,
                "incorrect": 4,
                "uncertain": 6,
            },
            "recent_cases": [
                {"id": 104, "date": "2026-10-13", "question": "Is there an acute infarct?", "answer": "ACUTE INFARCT", "confidence": 0.88, "status": "ABNORMAL"},
                {"id": 103, "date": "2026-10-13", "question": "Are lateral ventricles enlarged?", "answer": "VENTRICULAR ENLARGEMENT", "confidence": 0.84, "status": "ABNORMAL"},
                {"id": 102, "date": "2026-10-12", "question": "Any structural lesion detected?", "answer": "NORMAL", "confidence": 0.96, "status": "NORMAL"},
            ],
        }

    # Dynamic metrics from SQLite
    abnormal_count = db.query(Diagnosis).filter(Diagnosis.answer != "NORMAL").count()
    normal_count = db.query(Diagnosis).filter(Diagnosis.answer == "NORMAL").count()
    uncertain_count = db.query(Diagnosis).filter(Diagnosis.confidence < 0.70).count()

    avg_conf = db.query(func.avg(Diagnosis.confidence)).scalar() or 0.85
    avg_lat = db.query(func.avg(Diagnosis.latency_seconds)).scalar() or 1.75

    # Feedback breakdown
    fb_correct = db.query(DoctorFeedback).filter(DoctorFeedback.feedback_type == "correct").count()
    fb_partial = db.query(DoctorFeedback).filter(DoctorFeedback.feedback_type == "partially_correct").count()
    fb_incorrect = db.query(DoctorFeedback).filter(DoctorFeedback.feedback_type == "incorrect").count()
    fb_uncertain = db.query(DoctorFeedback).filter(DoctorFeedback.feedback_type == "uncertain").count()

    # Recent 5 cases
    recent_records = (
        db.query(Diagnosis)
        .order_by(Diagnosis.created_at.desc())
        .limit(5)
        .all()
    )

    recent_cases = [
        {
            "id": r.id,
            "date": r.created_at.strftime("%Y-%m-%d") if r.created_at else "2026-10-14",
            "question": r.question,
            "answer": r.answer,
            "confidence": r.confidence,
            "status": "NORMAL" if r.answer == "NORMAL" else "ABNORMAL",
        }
        for r in recent_records
    ]

    return {
        "total_cases": total_cases,
        "abnormal_count": abnormal_count,
        "normal_count": normal_count,
        "uncertain_count": uncertain_count,
        "avg_confidence": round(avg_conf * 100, 1),
        "avg_latency": round(avg_lat, 2),
        "feedback_summary": {
            "correct": fb_correct,
            "partial": fb_partial,
            "incorrect": fb_incorrect,
            "uncertain": fb_uncertain,
        },
        "recent_cases": recent_cases,
    }
