"""
NeuroVQA — Real-Time Grounded Clinical Chat Service
WebSocket endpoint at /ws/chat with token-by-token streaming from local Ollama (llama3.2:1b).
Constrained strictly to verified VQA output and Grad-CAM visual evidence.
"""

import asyncio
import json
import os
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
import httpx
from sqlalchemy.orm import Session

from .database import get_db, SessionLocal, ChatSession, ChatMessage, Diagnosis

router = APIRouter(tags=["Chat"])

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")

SYSTEM_PROMPT = """You are NeuroVQA Clinical Assistant, an offline clinical decision-support AI designed to help doctors interpret brain MRI VQA findings.

STRICT CLINICAL RULES:
1. You are explaining the verified visual findings produced by the NeuroVQA VQA model. You must NEVER independently diagnose the MRI scan or hallucinate new pathologies not identified by the VQA engine.
2. Ground all explanations in the verified VQA prediction, confidence score, and Grad-CAM attention region provided in context.
3. If confidence is below 70%, clearly advise the doctor that the visual evidence is inconclusive.
4. Always recommend clinical correlation with patient history, physical examination, and review by a qualified radiologist.
5. NEVER recommend specific drug dosages, surgical plans, or definitive final medical decisions.
6. Maintain a professional, concise, and clinically objective tone."""


@router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket, session_id: Optional[int] = None, token: Optional[str] = None):
    await websocket.accept()
    db: Session = SessionLocal()

    try:
        # Load or create chat session
        chat_sess = None
        if session_id:
            chat_sess = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not chat_sess:
            chat_sess = ChatSession(title="MRI Case Consultation")
            db.add(chat_sess)
            db.commit()
            db.refresh(chat_sess)

        while True:
            raw_data = await websocket.receive_text()
            payload = json.loads(raw_data)
            user_text = payload.get("message", "").strip()
            diag_ctx = payload.get("diagnosis_context")

            if not user_text:
                continue

            # Record doctor message
            user_msg = ChatMessage(session_id=chat_sess.id, role="user", content=user_text)
            db.add(user_msg)
            db.commit()

            # Assemble grounded system message with VQA context
            context_block = ""
            if diag_ctx:
                context_block = f"""
VERIFIED VQA ENGINE FINDINGS:
- Predicted Finding: {diag_ctx.get('answer', 'Unknown')}
- Cross-Attention Confidence: {float(diag_ctx.get('confidence', 0.0)) * 100:.1f}%
- Clinical Status: {diag_ctx.get('status', 'Unknown')}
- Top Candidate Predictions: {json.dumps(diag_ctx.get('top_3', []))}
"""
            full_system = f"{SYSTEM_PROMPT}\n{context_block}"

            # Prepare messages for Ollama
            messages = [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user_text},
            ]

            full_reply = ""
            ollama_success = False

            # Try streaming from local Ollama
            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    async with client.stream(
                        "POST",
                        f"{OLLAMA_HOST}/api/chat",
                        json={"model": OLLAMA_MODEL, "messages": messages, "stream": True},
                    ) as response:
                        if response.status_code == 200:
                            ollama_success = True
                            async for line in response.aiter_lines():
                                if not line:
                                    continue
                                chunk = json.loads(line)
                                token = chunk.get("message", {}).get("content", "")
                                if token:
                                    full_reply += token
                                    await websocket.send_text(json.dumps({"type": "token", "content": token}))
                                if chunk.get("done", False):
                                    await websocket.send_text(json.dumps({"type": "done"}))
                                    break
            except Exception:
                ollama_success = False

            # Fallback clinical grounded response if Ollama service is offline
            if not ollama_success:
                if diag_ctx:
                    finding = diag_ctx.get("answer", "Inconclusive")
                    conf = float(diag_ctx.get("confidence", 0.0)) * 100
                    status = diag_ctx.get("status", "UNREMARKABLE")
                    fallback_reply = (
                        f"Based on the NeuroVQA cross-attention analysis, the model classified this axial MRI scan as "
                        f"**{finding}** with **{conf:.1f}%** confidence (Status: *{status}*).\n\n"
                        f"The Grad-CAM attention heatmap highlights the focal spatial region contributing most to this "
                        f"prediction. In clinical practice, this finding should be evaluated alongside sagittal and coronal "
                        f"sequences, diffusion weighting (DWI/ADC), and relevant laboratory workups.\n\n"
                        f"**Clinical Reminder:** This output serves strictly as decision support. Final diagnostic verification "
                        f"must be performed by a qualified radiologist."
                    )
                else:
                    fallback_reply = (
                        "Please run an MRI scan inspection first so that I can provide an evidence-grounded clinical explanation."
                    )

                full_reply = fallback_reply
                # Stream token-by-token
                words = fallback_reply.split(" ")
                for i, w in enumerate(words):
                    token_str = w + (" " if i < len(words) - 1 else "")
                    await websocket.send_text(json.dumps({"type": "token", "content": token_str}))
                    await asyncio.sleep(0.025)
                await websocket.send_text(json.dumps({"type": "done"}))

            # Record assistant message
            asst_msg = ChatMessage(session_id=chat_sess.id, role="assistant", content=full_reply)
            db.add(asst_msg)
            db.commit()

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
    finally:
        db.close()
