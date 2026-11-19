# NeuroVQA Updated Documentation & Codebase Manifest (Version 2.0)

This manifest confirms the updates applied across the Markdown documentation and the full project codebase.

## 1. Documentation Synchronized
- **[NeuroVQA-prd.md](file:///k:/NeuroVQA%20V26/NeuroVQA-prd.md)** & **[NeuroVQ-prd.md](file:///k:/NeuroVQA%20V26/NeuroVQ-prd.md)**:
  - NFR-2 (Sequential Memory $\le 4.6\text{ GB}$) and NFR-3 (VQA Latency $\le 4.0\text{ sec}$ after warm-up on i3 CPU).
  - 4-Dataset Composite: VQA-RAD + SLAKE + VQA-Med 2019 + Kaggle Brain Tumor MRI (~12,157 images, 35,000+ QA pairs).
  - 15 Fixed Canonical Classes & `CLASS_TO_IDX` mapping.
  - Focal Loss ($\gamma=2.0, \alpha=0.25$), Patient-level 80/10/10 split, Horizontal Flip forbidden.
  - Model Card, 3 Fixed Demo Cases, Error Analysis, Airplane Mode test protocol, and Phase 1 Scope Cuts.
- **[NeuroVQA-design.md](file:///k:/NeuroVQA%20V26/NeuroVQA-design.md)**:
  - Gated Multimodal Fusion formula & code (`gate * v_img + (1-gate) * v_txt`).
  - 2.5D Volumetric Preprocessing with BiomedCLIP mean-embedding ($[k-1, k, k+1]$).
  - Attention $\times$ Gradient Grad-CAM (top 20% patches, alpha 0.45 overlay).
  - Sequential Loading Mermaid State and Sequence diagrams.
  - Complete REST & WebSocket API Contract JSON matching all schemas.
  - SQLite database DDL.
- **[NeuroVQA-setup.md](file:///k:/NeuroVQA%20V26/NeuroVQA-setup.md)**:
  - 4-Dataset download instructions and synthetic VQA generation script.
  - Colab T4 training setup with BiomedCLIP unfreezing, PubMedBERT LoRA, and Focal Loss.
  - Ollama setup strictly restricted to `llama3.2:1b` with `num_predict: 140` and `keep_alive: "0s"` (`phi3:mini` strictly prohibited).
  - Step-by-step sequential memory verification and offline Airplane Mode checklist.

## 2. Codebase Updates Applied
- **[generate_synthetic_vqa.py](file:///k:/NeuroVQA%20V26/generate_synthetic_vqa.py)** & **[backend/scripts/generate_synthetic_vqa.py](file:///k:/NeuroVQA%20V26/backend/scripts/generate_synthetic_vqa.py)**:
  - 5 deterministic clinical question templates producing `{image_path, question, answer}` records mapped to the 15 canonical classes.
- **[backend/src/model.py](file:///k:/NeuroVQA%20V26/backend/src/model.py)**:
  - `GatedMultimodalFusion` module with learned sigmoid gating.
  - `FocalLoss(alpha=0.25, gamma=2.0)`.
  - 15 fixed canonical classes + `CLASS_TO_IDX` + `IDX_TO_CLASS`.
  - `predict_25d` with tri-slice mean-embedding, calibrated priors, and uncertainty head ($<0.60 \rightarrow$ `"Uncertain - Needs Radiologist Review"`).
  - Returned dictionary includes both `top3` and `top_3_predictions`.
  - CPU threading restraint: `torch.set_num_threads(2)`.
- **[backend/main.py](file:///k:/NeuroVQA%20V26/backend/main.py)**:
  - `evict_ollama_from_ram()` sends `POST keep_alive: 0` before VQA execution.
  - `POST /api/predict` endpoint returns `{id, question, answer, confidence, status, top3, top_3_predictions, heatmap_url, latency_ms, model_version}`.
- **[backend/chat.py](file:///k:/NeuroVQA%20V26/backend/chat.py)**:
  - Strictly uses `llama3.2:1b` with `num_predict: 140`, `num_ctx: 1024`, `temperature: 0.2`, `top_p: 0.9`, and `keep_alive: "0s"`.
  - Enforces strict VQA evidence grounding and disclaimer: *"For clinician review only."*
  - Dual-compatible streaming frames: `{"token": "..."}` and `{"type": "token", "content": "..."}`.
- **[generate_notebook.py](file:///k:/NeuroVQA%20V26/generate_notebook.py)** & **[train_colab.ipynb](file:///k:/NeuroVQA%20V26/train_colab.ipynb)**:
  - Fully regenerated with 4-dataset composite corpus, 15 classes, Gated Fusion, Focal Loss, 2.5D mean embedding, and 3-way ablation study.
- **[frontend/src/components/ChatPanel.tsx](file:///k:/NeuroVQA%20V26/frontend/src/components/ChatPanel.tsx)**:
  - Configured with the 3 prompt chips: `[Why this answer?]`, `[What region?]`, `[How confident?]`.
  - Transmits grounded `vqa_context` with `case_id`.
- **[frontend/src/constants.ts](file:///k:/NeuroVQA%20V26/frontend/src/constants.ts)**:
  - Standardized to the 3 fixed demo cases (Case 1: Clear Tumor 0.92, Case 2: Normal Scan 0.89, Case 3: Uncertain 0.55).
