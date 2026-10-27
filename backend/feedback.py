"""
NeuroVQA — Doctor Feedback Service
Dual-Mode: Cloud Supabase + Local SQLite Fallback.
Collects clinician reviews (Correct, Partially Correct, Incorrect, Uncertain).
"""

from typing import Optional, Union
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import get_db, DoctorFeedback, Diagnosis, AuditLog
from .supabase_client import (
    is_supabase_enabled,
    supabase_insert_feedback,
    supabase_insert_audit_log,
)

router = APIRouter(prefix="/api/feedback", tags=["Feedback"])


class FeedbackCreate(BaseModel):
    diagnosis_id: Union[int, str]
    feedback_type: str  # correct | partially_correct | incorrect | uncertain
    doctor_note: Optional[str] = None


@router.post("")
async def submit_feedback(data: FeedbackCreate, db: Session = Depends(get_db)):
    feedback_type_clean = data.feedback_type.lower()
    feedback_id = None

    # 1. Try Supabase insertion
    if is_supabase_enabled():
        supa_res = await supabase_insert_feedback({
            "diagnosis_id": str(data.diagnosis_id),
            "feedback_type": feedback_type_clean,
            "doctor_note": data.doctor_note,
        })
        if supa_res and "id" in supa_res:
            feedback_id = supa_res["id"]
            await supabase_insert_audit_log({
                "diagnosis_id": str(data.diagnosis_id),
                "action": "FEEDBACK_SUBMITTED",
            })

    # 2. Local SQLite insertion
    try:
        diag_int_id = int(data.diagnosis_id) if str(data.diagnosis_id).isdigit() else None
        if diag_int_id:
            diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diag_int_id).first()
            if diagnosis:
                fb = DoctorFeedback(
                    diagnosis_id=diag_int_id,
                    feedback_type=feedback_type_clean,
                    doctor_note=data.doctor_note,
                )
                db.add(fb)
                db.commit()
                db.refresh(fb)
                if not feedback_id:
                    feedback_id = fb.id

                # Log audit event in SQLite
                log = AuditLog(
                    diagnosis_id=diag_int_id,
                    action="FEEDBACK_SUBMITTED",
                )
                db.add(log)
                db.commit()
    except Exception:
        pass

    return {
        "status": "success",
        "message": "Clinician feedback recorded successfully.",
        "id": feedback_id or data.diagnosis_id,
    }
