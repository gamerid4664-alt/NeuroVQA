# NeuroVQA Updated Documentation Manifest

This manifest confirms the final updates applied to all three NeuroVQA Markdown documents.

## 1. NeuroVQ-prd.md

Updated sections:
- Cloud-training, local-inference deployment model
- Google Colab / Kaggle T4 GPU assigned to dataset preparation, LoRA fine-tuning, evaluation, and checkpoint export
- Local Intel i3 11th Gen / 8 GB RAM / 512 GB SSD / integrated graphics used only for offline inference and doctor-facing workflow
- Local model training explicitly out of scope
- MRI VQA, Grad-CAM explainability, real-time chat, dashboard, PDF reports, case history, feedback, audit trail, MRI comparison, and viewer tools
- Functional requirements FR-1 to FR-18
- Non-functional requirements including local memory limit, VQA latency, chat streaming, privacy, traceability, and report integrity
- Success criteria aligned with Colab training and offline local demonstration

## 2. NeuroVQA-setup.md

Updated sections:
- Hybrid Google Colab training + local inference setup
- Local hardware profile: Intel Core i3 11th Gen, 8 GB RAM, 512 GB SSD, integrated graphics
- Resource limits: VQA inference under 2.5 GB memory; small quantized chat LLM
- Complete backend and frontend dependency setup
- Ollama model recommendation: llama3.2:1b default; phi3:mini optional if memory permits
- Local model asset structure for BiomedCLIP, PubMedBERT, LoRA adapter, demo MRI samples, reports, and optional RAG knowledge
- Google Colab T4 training workflow
- Checkpoint export to Google Drive and download to local backend/checkpoints/
- Local FastAPI, React, and Ollama startup instructions
- Dashboard, history, PDF report, feedback, and audit-trail workflow
- Expanded offline verification checklist

## 3. NeuroVQA-design.md

Updated sections:
- MRI-grounded doctor-assistant product concept
- Cloud-training, local-inference architecture
- VQA engine as the only MRI interpretation component
- LLM chatbot receives only verified VQA context
- Complete MRI preprocessing, dual-encoder, cross-attention, classification, and Grad-CAM workflows
- Real-time WebSocket chat with Ollama streaming
- Lightweight chat model selection for 8 GB RAM
- Doctor dashboard, case history, PDF reports, feedback, audit trail, MRI comparison, and viewer tools
- SQLite database schema
- Final technology stack
- Mandate dark-slate UI specification
- Execution profiles for Google Colab training and local inference

## Final Deployment Flow

```text
Google Colab / Kaggle T4 GPU
Train NeuroVQA + export LoRA adapter
        ↓ neurovqa_weights.pt
Local Intel i3 laptop
8 GB RAM | 512 GB SSD | Integrated graphics
        ↓
React + FastAPI + PyTorch CPU inference
+ Grad-CAM + real-time Ollama chat
```
