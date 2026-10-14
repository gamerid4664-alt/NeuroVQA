"""
NeuroVQA — Audit Trail Service
Maintains local clinical audit logs of model predictions, uploads, reports, and feedback.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from .database import get_db, AuditLog

router = APIRouter(prefix="/api/audit", tags=["Audit"])


@router.get("/logs")
def get_audit_logs(limit: int = 50, db: Session = Depends(get_db)):
    logs = (
        db.query(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "total": len(logs),
        "logs": [
            {
                "id": log.id,
                "user_id": log.user_id,
                "diagnosis_id": log.diagnosis_id,
                "action": log.action,
                "model_version": log.model_version,
                "model_checkpoint": log.model_checkpoint,
                "inference_latency": log.inference_latency,
                "created_at": log.created_at.isoformat() if log.created_at else None,
            }
            for log in logs
        ],
    }
