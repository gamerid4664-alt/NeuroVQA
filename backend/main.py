"""
NeuroVQA — FastAPI Master Server
Main entrypoint for the NeuroVQA clinical decision-support platform.
"""

from datetime import datetime
import json
import os
import time
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import init_db, get_db, Diagnosis, AuditLog
from .auth import router as auth_router
from .chat import router as chat_router
from .reports import router as reports_router
from .feedback import router as feedback_router
from .audit import router as audit_router
from .dashboard import router as dashboard_router
from .src.dataset import preprocess_mri_image
from .src.model import NeuroVQAInferenceEngine
from .src.explainability import generate_gradcam_heatmap, heatmap_to_png_bytes

# Directories configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOADS_DIR = os.path.join(DATA_DIR, "uploads")
REPORTS_DIR = os.path.join(DATA_DIR, "reports")
HEATMAPS_DIR = os.path.join(DATA_DIR, "heatmaps")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")

for d in [DATA_DIR, UPLOADS_DIR, REPORTS_DIR, HEATMAPS_DIR, CHECKPOINTS_DIR]:
    os.makedirs(d, exist_ok=True)

# Initialize FastAPI App
app = FastAPI(
    title="NeuroVQA API",
    description="Offline Brain MRI Visual Question Answering & Clinical Decision Support",
    version="26.0.0",
)

# Enable CORS for Vite dev server and local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database schemas
init_db()

# Initialize VQA Inference Engine with optional LoRA checkpoint
CHECKPOINT_FILE = os.path.join(CHECKPOINTS_DIR, "neurovqa_weights.pt")
vqa_engine = NeuroVQAInferenceEngine(checkpoint_path=CHECKPOINT_FILE)

# Include Subsystem Routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(reports_router)
app.include_router(feedback_router)
app.include_router(audit_router)
app.include_router(dashboard_router)

# Mount static files for data directory (MRI uploads, heatmaps, reports)
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "mode": "offline-capable",
        "model_version": "NeuroVQA-v26-BiomedCLIP-LoRA",
        "system": "Intel Core i3 Optimized",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/predict")
async def predict_mri(
    file: UploadFile = File(...),
    question: str = Form(...),
    db: Session = Depends(get_db),
):
    """
    Main VQA Prediction Endpoint:
    1. Ingests raw MRI scan (PNG/JPEG/NIfTI).
    2. Performs percentile clipping, min-max norm, and resizing to 224x224.
    3. Runs cross-attention VQA inference.
    4. Computes Grad-CAM attention heatmap.
    5. Saves diagnosis and audit trail to SQLite.
    """
    start_time = time.perf_counter()

    # Read uploaded bytes
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save original image to disk
    file_ext = os.path.splitext(file.filename)[1] or ".png"
    unique_stem = f"scan_{int(time.time() * 1000)}"
    orig_path = os.path.join(UPLOADS_DIR, f"{unique_stem}{file_ext}")
    with open(orig_path, "wb") as f:
        f.write(contents)

    # Preprocessing
    try:
        display_np, tensor = preprocess_mri_image(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # VQA Inference
    vqa_result = vqa_engine.predict(tensor, question)

    # Grad-CAM Attention Heatmap Generation
    jet_cam, overlay_cam, region_desc = generate_gradcam_heatmap(display_np)

    # Save heatmap overlay to disk
    cam_filename = f"{unique_stem}_gradcam.png"
    cam_path = os.path.join(HEATMAPS_DIR, cam_filename)
    with open(cam_path, "wb") as f:
        f.write(heatmap_to_png_bytes(overlay_cam))

    latency = round(time.perf_counter() - start_time, 2)

    # Save record to SQLite
    diag = Diagnosis(
        mri_file_path=orig_path,
        question=question,
        answer=vqa_result["answer"],
        confidence=vqa_result["confidence"],
        top_3_predictions_json=json.dumps(vqa_result["top_3"]),
        heatmap_path=cam_path,
        latency_seconds=latency,
        model_version="NeuroVQA-v26-BiomedCLIP-LoRA",
    )
    db.add(diag)
    db.commit()
    db.refresh(diag)

    # Save audit record
    audit = AuditLog(
        diagnosis_id=diag.id,
        action="VQA_RUN",
        inference_latency=latency,
    )
    db.add(audit)
    db.commit()

    return {
        "id": diag.id,
        "question": diag.question,
        "answer": diag.answer,
        "confidence": diag.confidence,
        "status": vqa_result["status"],
        "top_3_predictions": vqa_result["top_3"],
        "latency_seconds": latency,
        "focused_region": region_desc,
        "mri_url": f"http://localhost:8000/data/uploads/{os.path.basename(orig_path)}",
        "heatmap_url": f"http://localhost:8000/data/heatmaps/{cam_filename}",
        "created_at": diag.created_at.isoformat(),
    }


@app.get("/api/history")
def get_case_history(
    search: Optional[str] = Query(None),
    result_type: Optional[str] = Query(None),
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Returns paginated case history from SQLite."""
    query = db.query(Diagnosis).order_by(Diagnosis.created_at.desc())
    if search:
        query = query.filter(
            (Diagnosis.question.ilike(f"%{search}%")) | (Diagnosis.answer.ilike(f"%{search}%"))
        )

    records = query.limit(limit).all()

    cases = []
    for r in records:
        try:
            top_3 = json.loads(r.top_3_predictions_json)
        except Exception:
            top_3 = []

        cases.append({
            "id": r.id,
            "date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "2026-10-14",
            "question": r.question,
            "answer": r.answer,
            "confidence": r.confidence,
            "status": "NORMAL / UNREMARKABLE" if r.answer == "NORMAL" else "ABNORMALITY DETECTED",
            "latency_seconds": r.latency_seconds,
            "top_3_predictions": top_3,
            "mri_url": f"http://localhost:8000/data/uploads/{os.path.basename(r.mri_file_path)}" if r.mri_file_path else None,
            "heatmap_url": f"http://localhost:8000/data/heatmaps/{os.path.basename(r.heatmap_path)}" if r.heatmap_path else None,
        })

    return {"total": len(cases), "cases": cases}


@app.get("/api/diagnoses/{id}")
def get_diagnosis_by_id(id: int, db: Session = Depends(get_db)):
    diag = db.query(Diagnosis).filter(Diagnosis.id == id).first()
    if not diag:
        raise HTTPException(status_code=404, detail="Diagnosis not found")
    try:
        top_3 = json.loads(diag.top_3_predictions_json)
    except Exception:
        top_3 = []
    return {
        "id": diag.id,
        "question": diag.question,
        "answer": diag.answer,
        "confidence": diag.confidence,
        "status": "NORMAL / UNREMARKABLE" if diag.answer == "NORMAL" else "ABNORMALITY DETECTED",
        "latency_seconds": diag.latency_seconds,
        "top_3_predictions": top_3,
        "created_at": diag.created_at.isoformat() if diag.created_at else None,
    }


# Mount production frontend build if available for single-port serving
FRONTEND_DIST = os.path.join(os.path.dirname(BASE_DIR), "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    app.mount("/assets", StaticFiles(directory=os.path.join(FRONTEND_DIST, "assets")), name="static_assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = os.path.join(FRONTEND_DIST, full_path)
        if os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(FRONTEND_DIST, "index.html"))
