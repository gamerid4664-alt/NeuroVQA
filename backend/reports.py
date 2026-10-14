"""
NeuroVQA — PDF Clinical Report Generation Service
Generates structured clinical reports using ReportLab with MRI scan, Grad-CAM, top-3 candidates, and disclaimers.
"""

from datetime import datetime
import json
import os
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .database import get_db, Diagnosis, DoctorFeedback

REPORTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

router = APIRouter(prefix="/api/reports", tags=["Reports"])


def build_pdf_report(diagnosis: Diagnosis, feedback: Optional[DoctorFeedback] = None) -> str:
    """
    Builds a structured clinical PDF report for the diagnosis record.
    Returns the absolute path to the generated PDF.
    """
    pdf_path = os.path.join(REPORTS_DIR, f"NeuroVQA_Case_{diagnosis.id}.pdf")

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
            "SubStyle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#4B5563"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "SecStyle",
            parent=styles["Heading2"],
            fontSize=12,
            textColor=colors.HexColor("#312E81"),
            spaceBefore=10,
            spaceAfter=6,
        )
        text_style = ParagraphStyle(
            "TextStyle",
            parent=styles["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#1F2937"),
            leading=13,
        )
        disclaimer_style = ParagraphStyle(
            "DiscStyle",
            parent=styles["Normal"],
            fontSize=7.5,
            textColor=colors.HexColor("#9CA3AF"),
            leading=10,
        )

        elements = []

        # Header Title
        elements.append(Paragraph("<b>NEUROVQA — CLINICAL AI DECISION SUPPORT REPORT</b>", title_style))
        elements.append(
            Paragraph(
                f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} | Case ID: <b>#{diagnosis.id}</b> | Model: {diagnosis.model_version}",
                subtitle_style,
            )
        )
        elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#CBD5E1"), spaceAfter=14))

        # Case Overview Table
        overview_data = [
            [Paragraph("<b>Clinical Question:</b>", text_style), Paragraph(diagnosis.question, text_style)],
            [Paragraph("<b>AI Predicted Finding:</b>", text_style), Paragraph(f"<b>{diagnosis.answer}</b>", text_style)],
            [Paragraph("<b>Confidence Score:</b>", text_style), Paragraph(f"<b>{(diagnosis.confidence * 100):.1f}%</b> (Cross-Attention)", text_style)],
            [Paragraph("<b>Inference Latency:</b>", text_style), Paragraph(f"{diagnosis.latency_seconds:.2f} seconds (Local CPU)", text_style)],
            [Paragraph("<b>Clinical Status:</b>", text_style), Paragraph(diagnosis.answer if "NORMAL" in diagnosis.answer else "ABNORMALITY DETECTED", text_style)],
        ]
        if feedback:
            overview_data.append([Paragraph("<b>Doctor Feedback:</b>", text_style), Paragraph(f"{feedback.feedback_type.upper()} {f'— Note: {feedback.doctor_note}' if feedback.doctor_note else ''}", text_style)])

        t_overview = Table(overview_data, colWidths=[140, 400])
        t_overview.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(t_overview)
        elements.append(Spacer(1, 14))

        # Top-3 Ranked Candidates Table
        elements.append(Paragraph("<b>Top-3 Ranked Candidate Predictions</b>", section_style))
        try:
            top_3 = json.loads(diagnosis.top_3_predictions_json)
        except Exception:
            top_3 = []

        candidate_data = [["Rank", "Pathological Finding / Answer", "Cross-Attention Confidence"]]
        for i, item in enumerate(top_3):
            candidate_data.append([f"#{i+1}", item.get("answer", "N/A"), f"{(item.get('confidence', 0) * 100):.1f}%"])

        t_candidates = Table(candidate_data, colWidths=[50, 340, 150])
        t_candidates.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4338CA")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ])
        )
        elements.append(t_candidates)
        elements.append(Spacer(1, 14))

        # Visual Grounding & Images
        elements.append(Paragraph("<b>Visual Evidence Grounding (Grad-CAM Attention Overlay)</b>", section_style))
        elements.append(
            Paragraph(
                "The Grad-CAM attention heatmap highlights anatomical regions that most heavily influenced the VQA cross-attention classification.",
                text_style,
            )
        )
        elements.append(Spacer(1, 10))

        # Attach images if available on disk
        img_row = []
        if diagnosis.mri_file_path and os.path.exists(diagnosis.mri_file_path):
            img_row.append(RLImage(diagnosis.mri_file_path, width=180, height=180))
        if diagnosis.heatmap_path and os.path.exists(diagnosis.heatmap_path):
            img_row.append(RLImage(diagnosis.heatmap_path, width=180, height=180))

        if img_row:
            t_imgs = Table([img_row])
            elements.append(t_imgs)
            elements.append(Spacer(1, 14))

        # Medical Disclaimer
        elements.append(Spacer(1, 16))
        elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0"), spaceAfter=8))
        elements.append(
            Paragraph(
                "<b>MANDATORY MEDICAL DISCLAIMER:</b> NeuroVQA is an investigative clinical decision-support assistant designed to aid clinical workflows. It does not replace professional radiological interpretation or licensed clinical judgment. Definitive diagnosis, surgical planning, and medication prescription must always be performed by a qualified physician.",
                disclaimer_style,
            )
        )

        doc.build(elements)

    except ImportError:
        # Fallback text file if reportlab is not installed
        with open(pdf_path, "w") as f:
            f.write(
                f"NeuroVQA Clinical Report - Case #{diagnosis.id}\nQuestion: {diagnosis.question}\nAnswer: {diagnosis.answer}\nConfidence: {diagnosis.confidence}\n"
            )

    return pdf_path


@router.get("/{diagnosis_id}/download")
def download_clinical_report(diagnosis_id: int, db: Session = Depends(get_db)):
    diagnosis = db.query(Diagnosis).filter(Diagnosis.id == diagnosis_id).first()
    if not diagnosis:
        raise HTTPException(status_code=404, detail="Diagnosis record not found.")

    feedback = db.query(DoctorFeedback).filter(DoctorFeedback.diagnosis_id == diagnosis_id).first()
    pdf_path = build_pdf_report(diagnosis, feedback)

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=500, detail="Report generation failed.")

    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"NeuroVQA_Report_Case_{diagnosis_id}.pdf",
    )
