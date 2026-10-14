# Product Requirements Document (PRD) — NeuroVQA

## 1. Executive Summary
**NeuroVQA** is an offline, MRI-grounded clinical decision-support assistant designed to help doctors and clinical researchers analyze brain MRI scans and discuss findings through natural-language dialogue. The system accepts an axial brain MRI slice or contiguous 2.5D slice stack along with a clinical question, returns an image-grounded answer with confidence scores and Grad-CAM attention evidence, and supports real-time follow-up questions through an AI chat interface.

NeuroVQA means **Neuro Visual Question Answering**: the system answers questions about neuroimaging by combining visual MRI evidence with clinical language understanding.

---

## 2. Problem Statement
Radiological evaluation of brain MRI data is time-intensive and susceptible to diagnostic backlog and inter-observer variability. Existing automated tools often provide only single-label classification or rigid segmentation masks, without interactive explanation or conversational exploration.

NeuroVQA addresses this by providing an interactive, MRI-grounded assistant in which the VQA model performs image-based analysis and the chat assistant explains the verified result, answers follow-up doubts, and helps the doctor explore the evidence. The system is intended to assist—not replace—qualified radiologists and clinicians.

---

## 3. Target Users & Stakeholders
* **Radiologists and Clinicians:** Medical professionals seeking an interactive secondary reader for MRI exploration, triage support, and explainable diagnostic assistance.
* **Clinical Researchers:** Researchers studying multimodal medical AI, visual question answering, and explainability.
* **Academic Reviewers & Examiners:** Faculty evaluation committees assessing postgraduate computer science research rigor, architecture novelty, and empirical validity.

---

## 4. Deployment Model

NeuroVQA uses a **cloud-training, local-inference architecture**.

| Stage | Environment | Purpose |
|---|---|---|
| Dataset preparation | Google Colab / Kaggle | Download and preprocess VQA-RAD data |
| Model fine-tuning | Google Colab / Kaggle T4 GPU | Train LoRA adapters using GPU acceleration |
| Model export | Google Colab / Kaggle | Export lightweight LoRA adapter checkpoint |
| MRI VQA inference | Local laptop | Run trained adapter on CPU |
| Grad-CAM explainability | Local laptop | Generate attention heatmap locally |
| Doctor authentication | Local laptop | SQLite + JWT |
| Real-time chat | Local laptop | Ollama with a small quantized LLM |
| Dashboard, history, reports | Local laptop | Offline clinical workflow tools |

The local laptop is used only for inference, chat, dashboard, reports, and history. All model training and heavy GPU computation occur in Google Colab or Kaggle.

---

## 5. Scope of the System

### 5.1 In-Scope (Phase 1 Deliverables)
* Processing 2D axial brain MRI slices and contiguous 3-slice 2.5D stacks (PNG, JPEG, and sliced arrays from NIfTI volumes).
* Answering closed diagnostic questions:
  * Abnormality presence: *"Is there an acute lesion present?"*
  * Anatomical or sub-structure identification: *"Which ventricle is enlarged?"*
  * Pathological classification: *"What condition is indicated in the parietal lobe?"*
* Parameter-Efficient Fine-Tuning using LoRA adapters (PEFT) on biomedical vision-language foundation models (BiomedCLIP / PubMedBERT) in Google Colab or Kaggle.
* Visual grounding through Grad-CAM attention heatmaps highlighting the image regions supporting each prediction.
* Real-time MRI-grounded chat assistant for follow-up questions, explanation of results, and discussion of uncertainty.
* Doctor authentication, protected workspace access, and local chat-history storage.
* Offline-first deployment using locally stored model weights, SQLite, and Ollama.
* Analysis dashboard with case statistics, confidence trends, latency metrics, and feedback summaries.
* Downloadable PDF clinical analysis report for each MRI VQA prediction.
* Case history with search, filtering, reopening, and report download.
* Doctor feedback collection for AI answer quality and clinical usefulness.
* Audit trail for MRI uploads, predictions, chat activity, report downloads, and feedback.
* Side-by-side MRI comparison for current and previous scans.
* Basic MRI viewer tools including zoom, pan, brightness/contrast adjustment, region selection, and doctor notes.
* Optional future online mode for higher-quality cloud LLM conversation, disabled by default.

### 5.2 Out-of-Scope (Phase 1)
* Training or fine-tuning models on the local Intel i3 laptop.
* Autonomous generation of final radiology reports without clinician review.
* Definitive diagnosis, treatment recommendation, or medication guidance.
* Direct integration with hospital PACS / DICOM routers.
* Full uncompressed 3D NIfTI volume processing on consumer CPUs.
* Sending MRI images, clinical questions, or chat content to external AI services in offline mode.

---

## 6. Functional Requirements (FR)

| ID | Requirement | Description |
|---|---|---|
| **FR-1** | MRI Ingestion | Accepts PNG, JPEG, and NIfTI-derived axial MRI slices; supports 2D grayscale images and 3-slice 2.5D stacks. |
| **FR-2** | MRI Preprocessing | Applies optional percentile clipping, min-max normalization, resizing to 224 x 224, and PyTorch tensor conversion. |
| **FR-3** | Clinical Question Understanding | Tokenizes natural-language diagnostic questions up to 64 tokens using PubMedBERT-compatible tokenization. |
| **FR-4** | MRI-Grounded VQA | Performs cross-attention between MRI patch tokens and question tokens to produce a top-1 answer and top-3 ranked predictions with confidence scores. |
| **FR-5** | Visual Explainability | Generates a Grad-CAM attention heatmap, identifies the focused spatial region, and overlays it on the MRI scan. |
| **FR-6** | Real-Time Clinical Chat | Provides a ChatGPT-like streaming chat interface where doctors can ask follow-up questions about the MRI result and Grad-CAM evidence. |
| **FR-7** | Grounded Chat Context | Passes only the verified VQA answer, confidence, top predictions, heatmap region, and conversation history to the LLM. |
| **FR-8** | Doctor Authentication | Supports doctor signup, login, JWT-based sessions, and protected access to the diagnostic workspace. |
| **FR-9** | Conversation History | Stores chat sessions and messages locally in SQLite. |
| **FR-10** | Offline Operation | Performs inference, authentication, chat, and history storage fully offline after initial model and dependency setup. |
| **FR-11** | Analysis Dashboard | Displays total cases, abnormal/normal/uncertain counts, confidence distribution, latency metrics, recent cases, and doctor-feedback summaries. |
| **FR-12** | PDF Report Generation | Generates and downloads a structured PDF report containing MRI metadata, clinical question, AI answer, confidence, top-3 predictions, original scan, Grad-CAM heatmap, model version, latency, doctor notes, feedback, and disclaimer. |
| **FR-13** | Case History | Stores and displays previous MRI analyses with search, filtering, reopening, and report-download support. |
| **FR-14** | Doctor Feedback | Allows doctors to mark an AI result as Correct, Partially Correct, Incorrect, or Uncertain and add optional clinical notes. |
| **FR-15** | Audit Trail | Records user actions, timestamps, model version, inference latency, and diagnosis-related events. |
| **FR-16** | MRI Comparison | Compares current and previous MRI scans side-by-side, including VQA answers, confidence scores, and Grad-CAM overlays. |
| **FR-17** | MRI Viewer Tools | Provides zoom, pan, brightness/contrast adjustment, crosshair, region selection, and doctor notes attached to selected regions. |
| **FR-18** | Cloud Training Pipeline | Provides a Google Colab/Kaggle notebook for VQA-RAD dataset loading, LoRA fine-tuning, evaluation, and adapter export. |

---

## 7. Non-Functional Requirements (NFR)

* **NFR-1 (Cloud Training Feasibility):** Complete LoRA/PEFT fine-tuning within 45 minutes on a free-tier NVIDIA T4 GPU in Google Colab or Kaggle.
* **NFR-2 (Local Execution Constraints):** Local inference must run fully offline on an Intel Core i3 11th Gen system with 8 GB RAM, 512 GB SSD, and integrated graphics.
* **NFR-3 (Local Memory Constraint):** The local VQA inference pipeline should consume less than 2.5 GB of system memory; the chat LLM must be small enough to run alongside it.
* **NFR-4 (VQA Latency):** Single-slice MRI VQA inference must complete within 2.5 seconds on the target local CPU after model warm-up.
* **NFR-5 (Chat Responsiveness):** The chat assistant must stream response tokens progressively to provide a real-time conversational experience.
* **NFR-6 (Accuracy Target):** Achieve at least 72% classification accuracy on closed-ended VQA-RAD question subsets.
* **NFR-7 (Safety):** The system must clearly identify itself as a clinical decision-support tool, display uncertainty, and never provide definitive diagnoses or treatment recommendations.
* **NFR-8 (Privacy):** In offline mode, MRI images, clinical questions, chat messages, and inference results must remain on the local device.
* **NFR-9 (Modularity):** Separate the React frontend, FastAPI API layer, VQA model, preprocessing module, explainability module, chat service, reporting service, feedback service, audit service, and database layer.
* **NFR-10 (Traceability):** Every VQA prediction, chat session, report download, and doctor-feedback action must be recorded in local audit records.
* **NFR-11 (Report Integrity):** Every downloadable report must include the model version, inference timestamp, confidence score, Grad-CAM evidence, and medical disclaimer.

---

## 8. System Boundaries

```text
Google Colab / Kaggle T4 GPU
  ↓ Train LoRA adapter
  ↓ Export neurovqa_weights.pt

Local Doctor Laptop
  ↓ React frontend
  ↓ FastAPI backend
  ↓ NeuroVQA VQA inference
  ↓ Grad-CAM heatmap
  ↓ Ollama chat assistant
  ↓ SQLite dashboard/history/reports
```

The VQA engine is the only component permitted to interpret the MRI image. The LLM chatbot may explain and discuss the VQA output but must not independently diagnose the MRI.

---

## 9. Additional Feature Scope

### 9.1 Analysis Dashboard
The dashboard provides doctors with an overview of MRI analysis activity, including total cases, abnormal/normal/uncertain distribution, average confidence, average latency, recent cases, and feedback summaries.

### 9.2 Downloadable PDF Report
After each VQA analysis, doctors can download a structured PDF report containing the MRI image, Grad-CAM heatmap, clinical question, AI answer, confidence, top-3 predictions, model version, latency, doctor notes, feedback, and medical disclaimer.

### 9.3 Case History
Doctors can browse previous analyses, search and filter cases, reopen an MRI case, continue its chat session, and download its historical report.

### 9.4 Doctor Feedback
Doctors can evaluate each AI prediction as Correct, Partially Correct, Incorrect, or Uncertain. Optional notes support human-in-the-loop evaluation and quality monitoring.

### 9.5 Audit Trail
The system records major actions locally, including MRI upload, VQA inference, heatmap generation, chat activity, report download, and feedback submission.

### 9.6 MRI Comparison
Doctors can compare a current MRI with a previous scan side-by-side, including VQA answers, confidence scores, and Grad-CAM overlays.

### 9.7 MRI Viewer Tools
The MRI viewer supports zoom, pan, brightness/contrast adjustment, crosshair navigation, region selection, and doctor notes attached to selected regions.

---

## 10. Project Success Criteria
1. LoRA/PEFT training completes successfully on Google Colab or Kaggle T4 GPU.
2. VQA training loss converges below 0.35 on VQA-RAD.
3. Exported LoRA adapter checkpoint remains below 80 MB, with an expected size of approximately 50 MB.
4. System completes an offline MRI upload, VQA prediction, Grad-CAM generation, and follow-up chat interaction without internet access.
5. Chat responses visibly stream in real time and remain grounded in the latest VQA result.
6. Doctors can download a PDF report for each MRI analysis.
7. Doctors can view historical cases, submit feedback, and view dashboard analytics.
8. Documented ablation study compares Vision-Only, Text-Only, and Multimodal configurations.
9. Every MRI-related chat response includes appropriate uncertainty and clinician-review guidance.
