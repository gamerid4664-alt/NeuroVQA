"""
NeuroVQA — PDF Clinical Report Generation Service
Dual-Mode: Cloud Supabase + Local SQLite Fallback.
Generates structured clinical reports using ReportLab with MRI scan, Grad-CAM, top-3 candidates, and disclaimers.
"""

from datetime import datetime
import json
import os
from typing import Optional, Union, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db, Diagnosis, DoctorFeedback, AuditLog
from .supabase_client import is_supabase_enabled, supabase_get_diagnosis, supabase_insert_audit_log

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/reports", tags=["Reports"])


class DiagnosisReportView:
    """Unified view model across Supabase and SQLite diagnosis records."""
    def __init__(self, data: Any):
        if isinstance(data, dict):
            self.id = data.get("id")
            self.question = data.get("question", "")
            self.answer = data.get("answer", "")
            self.confidence = float(data.get("confidence", 0.0))
            self.mri_file_path = data.get("mri_file_path", "")
            self.heatmap_path = data.get("heatmap_path", "")
            self.model_version = data.get("model_version", "NeuroVQA-v26-BiomedCLIP-LoRA")
            self.latency_seconds = float(data.get("latency_seconds", 1.75))
            created = data.get("created_at")
            if isinstance(created, str):
                try:
                    self.created_at = datetime.fromisoformat(created.replace("Z", "+00:00"))
                except Exception:
                    self.created_at = datetime.utcnow()
            elif isinstance(created, datetime):
                self.created_at = created
            else:
                self.created_at = datetime.utcnow()
            top3 = data.get("top_3_predictions") or data.get("top_3_predictions_json", "[]")
            self.top_3_predictions_json = json.dumps(top3) if not isinstance(top3, str) else top3
        else:
            self.id = getattr(data, "id", None)
            self.question = getattr(data, "question", "")
            self.answer = getattr(data, "answer", "")
            self.confidence = getattr(data, "confidence", 0.0)
            self.mri_file_path = getattr(data, "mri_file_path", "")
            self.heatmap_path = getattr(data, "heatmap_path", "")
            self.model_version = getattr(data, "model_version", "NeuroVQA-v26-BiomedCLIP-LoRA")
            self.latency_seconds = getattr(data, "latency_seconds", 1.75)
            self.created_at = getattr(data, "created_at", datetime.utcnow())
            self.top_3_predictions_json = getattr(data, "top_3_predictions_json", "[]")


def _resolve_image_path(path_or_url: Optional[str]) -> Optional[str]:
    """Resolves local paths, relative filenames, or remote URLs to a usable local image file."""
    if not path_or_url:
        return None
    if os.path.exists(path_or_url):
        return path_or_url

    # Check local data directories
    fname = os.path.basename(path_or_url.split("?")[0])
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "data", "uploads", fname),
        os.path.join(base_dir, "data", "heatmaps", fname),
        os.path.join(base_dir, "data", "demo_samples", fname),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    # Remote URL cache fallback
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        try:
            import httpx
            cache_file = os.path.join(REPORTS_DIR, f"cached_{fname}")
            if os.path.exists(cache_file):
                return cache_file
            with httpx.Client(timeout=10.0) as client:
                r = client.get(path_or_url)
                if r.status_code == 200:
                    with open(cache_file, "wb") as f:
                        f.write(r.content)
                    return cache_file
        except Exception:
            pass

    return None


def build_pdf_report(diagnosis: Any, feedback: Optional[Any] = None) -> str:
    """
    Builds a structured clinical PDF report for the diagnosis record.
    Returns the absolute path to the generated PDF.
    """
    diag = DiagnosisReportView(diagnosis)
    safe_id = str(diag.id).replace("-", "")[:8]
    pdf_path = os.path.join(REPORTS_DIR, f"NeuroVQA_Case_{safe_id}.pdf")

    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Table,
            TableStyle,
            Image as RLImage,
            HRFlowable,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

        doc = SimpleDocTemplate(
            pdf_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TitleStyle",
            parent=styles["Heading1"],
            fontSize=18,
            textColor=colors.HexColor("#1E1B4B"),
            spaceAfter=4,
        )
        subtitle_style = ParagraphStyle(
            "SubTitleStyle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#6B7280"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "SectionStyle",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#312E81"),
            spaceBefore=8,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#1F2937"),
            leading=12,
        )
        bold_style = ParagraphStyle(
            "BoldBody",
            parent=body_style,
            fontName="Helvetica-Bold",
        )
        disclaimer_style = ParagraphStyle(
            "Disclaimer",
            parent=styles["Normal"],
            fontSize=7,
            textColor=colors.HexColor("#9CA3AF"),
            leading=9,
        )

        elements = []

        # 1. Header Banner
        elements.append(Paragraph("NeuroVQA — Clinical Decision Support Report", title_style))
        elements.append(
            Paragraph(
                f"Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} | Architecture: BiomedCLIP + LoRA Cross-Attention | Intel Core i3 Optimized",
                subtitle_style,
            )
        )
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#4F46E5"), spaceAfter=10))

        # 2. Case Metadata Table
        created_str = diag.created_at.strftime("%Y-%m-%d %H:%M:%S UTC") if diag.created_at else "N/A"
        meta_data = [
            [Paragraph("Case ID:", bold_style), Paragraph(str(diag.id), body_style), Paragraph("Date / Time:", bold_style), Paragraph(created_str, body_style)],
            [Paragraph("Modality:", bold_style), Paragraph("Brain MRI (Axial Slice / 2.5D Stack)", body_style), Paragraph("Model Version:", bold_style), Paragraph(str(diag.model_version), body_style)],
            [Paragraph("Clinical Question:", bold_style), Paragraph(diag.question, body_style), Paragraph("Inference Latency:", bold_style), Paragraph(f"{diag.latency_seconds:.2f}s", body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[110, 160, 100, 170])
        meta_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ])
        )
        elements.append(meta_table)
        elements.append(Spacer(1, 10))

        # 3. Diagnostic Finding & Confidence
        elements.append(Paragraph("1. Primary Model Finding", section_style))
        conf_pct = diag.confidence * 100
        is_normal = diag.answer == "NORMAL"
        badge_color = "#10B981" if is_normal else "#E11D48"

        finding_data = [
            [
                Paragraph("Predicted Class", bold_style),
                Paragraph(f"<font color='{badge_color}'><b>{diag.answer}</b></font>", bold_style),
            ],
            [
                Paragraph("Cross-Attention Confidence", bold_style),
                Paragraph(f"<b>{conf_pct:.1f}%</b> ({'High Confidence' if conf_pct >= 80 else 'Moderate / Borderline'})", body_style),
            ],
            [
                Paragraph("Clinical Status", bold_style),
                Paragraph("NORMAL / UNREMARKABLE" if is_normal else "ABNORMALITY DETECTED", bold_style),
            ],
        ]
        finding_table = Table(finding_data, colWidths=[180, 360])
        finding_table.setStyle(
            TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 5),
            ])
        )
        elements.append(finding_table)
        elements.append(Spacer(1, 10))

        # 4. Top-3 Candidate Predictions
        elements.append(Paragraph("2. Top-3 Ranked Predictions", section_style))
        try:
            top_3 = json.loads(diag.top_3_predictions_json)
        except Exception:
            top_3 = []

        top_rows = [[Paragraph("Rank", bold_style), Paragraph("Candidate Finding", bold_style), Paragraph("Confidence", bold_style)]]
        for i, item in enumerate(top_3[:3]):
            top_rows.append([
                Paragraph(f"#{i+1}", body_style),
                Paragraph(item.get("answer", "N/A"), body_style),
                Paragraph(f"{float(item.get('confidence', 0.0)) * 100:.1f}%", body_style),
            ])
        if len(top_rows) == 1:
            top_rows.append([Paragraph("#1", body_style), Paragraph(diag.answer, body_style), Paragraph(f"{conf_pct:.1f}%", body_style)])

        top_table = Table(top_rows, colWidths=[60, 340, 140])
        top_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEF2FF")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("PADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(top_table)
        elements.append(Spacer(1, 10))

        # 5. Visual Evidence (MRI & Grad-CAM)
        elements.append(Paragraph("3. Visual Evidence & Spatial Attention", section_style))
        img_cells = []
        resolved_mri = _resolve_image_path(diag.mri_file_path)
        if resolved_mri and os.path.exists(resolved_mri):
            img_cells.append(RLImage(resolved_mri, width=200, height=200))
        else:
            img_cells.append(Paragraph("Original MRI Scan (Local file archived)", body_style))

        resolved_cam = _resolve_image_path(diag.heatmap_path)
        if resolved_cam and os.path.exists(resolved_cam):
            img_cells.append(RLImage(resolved_cam, width=200, height=200))
        else:
            img_cells.append(Paragraph("Grad-CAM Attention Overlay (Archived)", body_style))

        img_table = Table([[img_cells[0], img_cells[1]]], colWidths=[270, 270])
        img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE")]))
        elements.append(img_table)
        elements.append(Spacer(1, 10))

        # 6. Attending Doctor Notes & Feedback
        elements.append(Paragraph("4. Clinician Evaluation & Feedback", section_style))
        fb_note = getattr(feedback, "doctor_note", "No doctor notes attached.") if feedback else "No feedback recorded yet."
        fb_type = getattr(feedback, "feedback_type", "Pending Review") if feedback else "Pending Review"

        fb_data = [
            [Paragraph("Clinician Review Status:", bold_style), Paragraph(fb_type.upper(), body_style)],
            [Paragraph("Clinical Notes:", bold_style), Paragraph(fb_note or "No additional notes.", body_style)],
        ]
        fb_table = Table(fb_data, colWidths=[150, 390])
        fb_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")), ("PADDING", (0, 0), (-1, -1), 5)]))
        elements.append(fb_table)
        elements.append(Spacer(1, 15))

        # 7. Regulatory & Ethical Disclaimer
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#CBD5E1"), spaceAfter=6))
        elements.append(
            Paragraph(
                "<b>RESEARCH & DECISION-SUPPORT DISCLAIMER:</b> NeuroVQA is an experimental medical AI assistant designed "
                "strictly for secondary decision support and educational research under human supervision. This report does NOT "
                "constitute a definitive medical diagnosis, surgical plan, or prescription. All findings must be independently "
                "verified by a qualified board-certified radiologist.",
                disclaimer_style,
            )
        )

        doc.build(elements)

    except ImportError:
        # Fallback text file if reportlab is not installed
        with open(pdf_path, "w") as f:
            f.write(
                f"NeuroVQA Clinical Report - Case #{diag.id}\nQuestion: {diag.question}\nAnswer: {diag.answer}\nConfidence: {diag.confidence}\n"
            )

    return pdf_path


@router.get("/{diagnosis_id}/download")
async def download_clinical_report(diagnosis_id: str, db: Session = Depends(get_db)):
    diag_record = None

    # 1. Try Supabase lookup
    if is_supabase_enabled():
        supa_diag = await supabase_get_diagnosis(diagnosis_id)
        if supa_diag:
            diag_record = supa_diag

    # 2. Local SQLite fallback
    if not diag_record:
        diag_int = int(diagnosis_id) if diagnosis_id.isdigit() else None
        if diag_int:
            diag_record = db.query(Diagnosis).filter(Diagnosis.id == diag_int).first()

    if not diag_record:
        raise HTTPException(status_code=404, detail="Diagnosis record not found.")

    feedback = None
    if isinstance(diag_record, Diagnosis):
        feedback = db.query(DoctorFeedback).filter(DoctorFeedback.diagnosis_id == diag_record.id).first()

    pdf_path = build_pdf_report(diag_record, feedback)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Report generation failed.")

    # Record REPORT_DOWNLOADED audit event (FR-15, NFR-10)
    try:
        if is_supabase_enabled():
            await supabase_insert_audit_log({
                "diagnosis_id": str(diagnosis_id),
                "action": "REPORT_DOWNLOADED",
            })
        diag_int = int(diagnosis_id) if str(diagnosis_id).isdigit() else None
        if diag_int:
            log = AuditLog(diagnosis_id=diag_int, action="REPORT_DOWNLOADED")
            db.add(log)
            db.commit()
    except Exception:
        pass

    safe_filename = f"NeuroVQA_Report_Case_{str(diagnosis_id)[:8]}.pdf"
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=safe_filename,
    )
