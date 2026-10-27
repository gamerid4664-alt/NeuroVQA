"""
NeuroVQA — FastAPI Master Server
Dual-Mode: Cloud Supabase + Local SQLite Fallback.
Main entrypoint for the NeuroVQA clinical decision-support platform.
"""

from datetime import datetime
import json
import os
import time
from typing import Optional, Union, List
from dotenv import load_dotenv

# Initialize environment configuration
_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_root_dir, ".env"))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
load_dotenv()
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import init_db, get_db, Diagnosis, AuditLog
from .supabase_client import (
    is_supabase_enabled,
    supabase_upload_storage,
    supabase_insert_diagnosis,
    supabase_get_diagnoses,
    supabase_get_diagnosis,
    supabase_insert_audit_log,
    supabase_get_user,
)
from .auth import router as auth_router
from .chat import router as chat_router
from .reports import router as reports_router
from .feedback import router as feedback_router
from .audit import router as audit_router
from .dashboard import router as dashboard_router
from PIL import Image
from .src.dataset import preprocess_mri_image, pack_25d_stack, is_nifti_data
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
    description="Offline-Capable & Supabase Cloud Brain MRI Visual Question Answering & Clinical Decision Support",
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
        "mode": "supabase-hybrid" if is_supabase_enabled() else "offline-sqlite",
        "database": "supabase-postgres" if is_supabase_enabled() else "sqlite-local",
        "model_version": "NeuroVQA-v26-BiomedCLIP-LoRA",
        "system": "Intel Core i3 Optimized",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/predict")
async def predict_mri(
    file: Optional[UploadFile] = File(None),
    files: Optional[List[UploadFile]] = File(None),
    question: str = Form(...),
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Main VQA Prediction Endpoint (FR-1, FR-2, FR-3):
    1. Ingests raw MRI scan (PNG/JPEG/NIfTI) or 3-slice 2.5D stack (k-1, k, k+1).
    2. Performs percentile clipping, min-max norm, and resizing to 224x224.
    3. Runs cross-attention VQA inference.
    4. Computes Grad-CAM attention heatmap.
    5. Saves diagnosis and audit trail to Supabase and SQLite.
    """
    start_time = time.perf_counter()

    # Extract user token and identity if provided
    token = None
    user_id = None
    if authorization and "Bearer " in authorization:
        token = authorization.split("Bearer ")[1].strip()
        supa_user = await supabase_get_user(token)
        if supa_user and "id" in supa_user:
            user_id = supa_user["id"]

    is_25d = False
    # Check if multi-slice 2.5D volumetric stack is provided (FR-1)
    if files and len(files) == 3:
        is_25d = True
        c0 = await files[0].read()
        c1 = await files[1].read()
        c2 = await files[2].read()
        if not c0 or not c1 or not c2:
            raise HTTPException(status_code=400, detail="One or more slices in the 2.5D stack are empty.")

        # Center slice (k) is used as primary preview scan
        contents = c1
        file_ext = os.path.splitext(files[1].filename or "scan.png")[1] or ".png"
        unique_stem = f"scan_25d_{int(time.time() * 1000)}"
        orig_path = os.path.join(UPLOADS_DIR, f"{unique_stem}{file_ext}")
        with open(orig_path, "wb") as f:
            f.write(contents)
        preview_filename = os.path.basename(orig_path)

        try:
            stacked_np = pack_25d_stack(c0, c1, c2)
            display_np, tensor = preprocess_mri_image(stacked_np)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"2.5D stack preprocessing failed: {str(e)}")
    else:
        # Single 2D slice or Volumetric NIfTI (.nii / .nii.gz) (FR-1)
        upload_item = file or (files[0] if files and len(files) > 0 else None)
        if not upload_item:
            raise HTTPException(status_code=400, detail="No MRI image provided. Upload a single slice, 3-slice 2.5D stack, or NIfTI volume.")

        contents = await upload_item.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        file_ext = os.path.splitext(upload_item.filename or "scan.png")[1] or ".png"
        unique_stem = f"scan_{int(time.time() * 1000)}"
        orig_path = os.path.join(UPLOADS_DIR, f"{unique_stem}{file_ext}")
        with open(orig_path, "wb") as f:
            f.write(contents)

        is_nii = file_ext.lower() in (".nii", ".gz") or is_nifti_data(contents)
        try:
            display_np, tensor = preprocess_mri_image(contents)
            preview_filename = os.path.basename(orig_path)
            if is_nii:
                is_25d = True
                preview_filename = f"{unique_stem}_preview.png"
                preview_path = os.path.join(UPLOADS_DIR, preview_filename)
                Image.fromarray(display_np).save(preview_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

    # VQA Inference
    vqa_result = vqa_engine.predict(tensor, question)

    # Grad-CAM Attention Heatmap Generation
    jet_cam, overlay_cam, region_desc = generate_gradcam_heatmap(display_np)

    # Save heatmap overlay to disk
    cam_filename = f"{unique_stem}_gradcam.png"
    cam_path = os.path.join(HEATMAPS_DIR, cam_filename)
    cam_bytes = heatmap_to_png_bytes(overlay_cam)
    with open(cam_path, "wb") as f:
        f.write(cam_bytes)

    latency = round(time.perf_counter() - start_time, 2)

    # Determine default local static URLs
    mri_url = f"http://localhost:8000/data/uploads/{preview_filename}"
    heatmap_url = f"http://localhost:8000/data/heatmaps/{cam_filename}"
    assigned_id = None

    # 1. Cloud Sync: Upload to Supabase Storage & Insert to Postgres
    if is_supabase_enabled():
        cloud_mri = await supabase_upload_storage("mri-scans", f"{unique_stem}{file_ext}", contents, content_type="image/png", access_token=token)
        cloud_cam = await supabase_upload_storage("heatmaps", cam_filename, cam_bytes, content_type="image/png", access_token=token)

        if cloud_mri:
            mri_url = cloud_mri
        if cloud_cam:
            heatmap_url = cloud_cam

        diag_payload = {
            "mri_file_path": mri_url,
            "question": question,
            "answer": vqa_result["answer"],
            "confidence": float(vqa_result["confidence"]),
            "top_3_predictions": vqa_result["top_3"],
            "heatmap_path": heatmap_url,
            "latency_seconds": latency,
            "model_version": "NeuroVQA-v26-BiomedCLIP-LoRA",
        }
        if user_id:
            diag_payload["user_id"] = user_id

        supa_diag = await supabase_insert_diagnosis(diag_payload, access_token=token)

        if supa_diag and "id" in supa_diag:
            assigned_id = supa_diag["id"]
            for action_name, action_lat in [
                ("MRI_UPLOADED", None),
                ("HEATMAP_GENERATED", None),
                ("VQA_RUN", latency),
            ]:
                audit_payload = {
                    "diagnosis_id": assigned_id,
                    "action": action_name,
                    "inference_latency": action_lat,
                }
                if user_id:
                    audit_payload["user_id"] = user_id
                await supabase_insert_audit_log(audit_payload, access_token=token)

    # 2. Local Fallback Sync: Save record to SQLite
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

    # Save comprehensive audit records locally (FR-15, Design 5.5)
    for action_name, action_lat in [
        ("MRI_UPLOADED", None),
        ("HEATMAP_GENERATED", None),
        ("VQA_RUN", latency),
    ]:
        audit = AuditLog(
            diagnosis_id=diag.id,
            action=action_name,
            inference_latency=action_lat,
        )
        db.add(audit)
    db.commit()

    return {
        "id": assigned_id or diag.id,
        "question": diag.question,
        "answer": diag.answer,
        "confidence": diag.confidence,
        "status": vqa_result["status"],
        "top_3_predictions": vqa_result["top_3"],
        "latency_seconds": latency,
        "focused_region": region_desc,
        "mri_url": mri_url,
        "heatmap_url": heatmap_url,
        "slice_mode": "2.5D Volumetric Stack (3 Contiguous Axial Slices)" if is_25d else "2D Single Axial Slice",
        "created_at": diag.created_at.isoformat(),
    }


@app.get("/api/history")
async def get_case_history(
    search: Optional[str] = Query(None),
    result_type: Optional[str] = Query(None),
    limit: int = 50,
    db: Session = Depends(get_db),
):
    """Returns paginated case history from Supabase with SQLite fallback."""
    # 1. Try Supabase
    if is_supabase_enabled():
        supa_cases = await supabase_get_diagnoses(search=search, limit=limit)
        if supa_cases is not None and len(supa_cases) > 0:
            cases = []
            for r in supa_cases:
                top_3 = r.get("top_3_predictions") or []
                created_dt = r.get("created_at", "")[:16].replace("T", " ")
                cases.append({
                    "id": r.get("id"),
                    "date": created_dt or "2026-10-26",
                    "question": r.get("question"),
                    "answer": r.get("answer"),
                    "confidence": r.get("confidence"),
                    "status": "NORMAL / UNREMARKABLE" if r.get("answer") == "NORMAL" else "ABNORMALITY DETECTED",
                    "latency_seconds": r.get("latency_seconds", 1.75),
                    "top_3_predictions": top_3,
                    "mri_url": r.get("mri_file_path"),
                    "heatmap_url": r.get("heatmap_path"),
                })
            return {"total": len(cases), "cases": cases}

    # 2. Local SQLite fallback
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

        fb_type = r.feedbacks[0].feedback_type if r.feedbacks and len(r.feedbacks) > 0 else None
        cases.append({
            "id": r.id,
            "date": r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "2026-10-26",
            "question": r.question,
            "answer": r.answer,
            "confidence": r.confidence,
            "status": "NORMAL / UNREMARKABLE" if r.answer == "NORMAL" else "ABNORMALITY DETECTED",
            "latency_seconds": r.latency_seconds,
            "top_3_predictions": top_3,
            "feedback": fb_type,
            "mri_url": f"http://localhost:8000/data/uploads/{os.path.basename(r.mri_file_path)}" if r.mri_file_path else None,
            "heatmap_url": f"http://localhost:8000/data/heatmaps/{os.path.basename(r.heatmap_path)}" if r.heatmap_path else None,
        })

    return {"total": len(cases), "cases": cases}


@app.get("/api/diagnoses/{id}")
async def get_diagnosis_by_id(id: str, db: Session = Depends(get_db)):
    # 1. Try Supabase
    if is_supabase_enabled():
        supa_diag = await supabase_get_diagnosis(id)
        if supa_diag:
            top_3 = supa_diag.get("top_3_predictions") or []
            return {
                "id": supa_diag.get("id"),
                "question": supa_diag.get("question"),
                "answer": supa_diag.get("answer"),
                "confidence": supa_diag.get("confidence"),
                "status": "NORMAL / UNREMARKABLE" if supa_diag.get("answer") == "NORMAL" else "ABNORMALITY DETECTED",
                "latency_seconds": supa_diag.get("latency_seconds", 1.75),
                "top_3_predictions": top_3,
                "mri_url": supa_diag.get("mri_file_path"),
                "heatmap_url": supa_diag.get("heatmap_path"),
                "created_at": supa_diag.get("created_at"),
            }

    # 2. Local SQLite fallback
    diag_int = int(id) if id.isdigit() else None
    if diag_int:
        diag = db.query(Diagnosis).filter(Diagnosis.id == diag_int).first()
        if diag:
            try:
                top_3 = json.loads(diag.top_3_predictions_json)
            except Exception:
                top_3 = []
            mri_filename = os.path.basename(diag.mri_file_path) if diag.mri_file_path else None
            cam_filename = os.path.basename(diag.heatmap_path) if diag.heatmap_path else None
            return {
                "id": diag.id,
                "question": diag.question,
                "answer": diag.answer,
                "confidence": diag.confidence,
                "status": "NORMAL / UNREMARKABLE" if diag.answer == "NORMAL" else "ABNORMALITY DETECTED",
                "latency_seconds": diag.latency_seconds,
                "top_3_predictions": top_3,
                "mri_url": f"http://localhost:8000/data/uploads/{mri_filename}" if mri_filename else None,
                "heatmap_url": f"http://localhost:8000/data/heatmaps/{cam_filename}" if cam_filename else None,
                "created_at": diag.created_at.isoformat() if diag.created_at else None,
            }

    raise HTTPException(status_code=404, detail="Diagnosis not found")


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
