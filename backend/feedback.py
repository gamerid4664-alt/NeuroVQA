"""
NeuroVQA — Doctor Feedback Service
Collects human-in-the-loop clinician reviews (Correct, Partially Correct, Incorrect, Uncertain).
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session

from .database import get_db, DoctorFeedback, Diagnosis, AuditLog

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


class FeedbackCreate(BaseModel):
    diagnosis_id: int
    feedback_type: str  # correct | partially_correct | incorrect | uncertain
    doctor_note: Optional[str] = None


@router.post("")
def submit_feedback(data: FeedbackCreate, db: Session = Depends(get_db)):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == data.diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Diagnosis record not found",
        )

    fb = DoctorFeedback(
        diagnosis_id=data.diagnosis_id,
        feedback_type=data.feedback_type.lower(),
        doctor_note=data.doctor_note,
    )
    db.add(fb)
    db.commit()
    db.refresh(fb)

    # Log audit event
    log = AuditLog(
        diagnosis_id=data.diagnosis_id,
        action="FEEDBACK_SUBMITTED",
    )
    db.add(log)
    db.commit()

    return {"status": "success", "message": "Clinician feedback recorded successfully.", "id": fb.id}
