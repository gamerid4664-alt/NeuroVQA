# System Architecture, Subsystem Workflows & Technical Design — NeuroVQA

## 1. Product Concept

**NeuroVQA** is an offline, MRI-grounded clinical decision-support assistant for doctors. It combines:

1. **MRI Visual Question Answering** — analyzes the uploaded brain MRI and answers clinical questions.
2. **Explainable AI** — shows Grad-CAM evidence for the model’s prediction.
3. **Real-time clinical chat** — allows doctors to ask follow-up questions and discuss the result.
4. **Grounded response generation** — the LLM explains verified VQA output; it does not independently diagnose the MRI.
5. **Clinical workflow tools** — dashboard, case history, PDF reports, feedback, audit trail, comparison, and MRI viewer tools.

The VQA engine is the source of truth for image-based findings. The LLM is a cautious explanation and conversation layer.

---

## 2. Hybrid Deployment Architecture

NeuroVQA uses a **cloud-training, local-inference** architecture.

```text
GOOGLE COLAB / KAGGLE T4 GPU
- VQA-RAD dataset preparation
- BiomedCLIP + PubMedBERT feature extraction
- PEFT / LoRA fine-tuning
- Evaluation and checkpoint export
          |
          | Export neurovqa_weights.pt
          ▼
LOCAL INTEL i3 LAPTOP
8 GB RAM | 512 GB SSD | Integrated Graphics
  |
  ├── React Frontend
  ├── FastAPI Backend
  ├── PyTorch CPU VQA Inference
  ├── Grad-CAM Explainability
  ├── Ollama Small Chat LLM
  └── SQLite Local Storage
```

The local laptop does **not** perform model training. It performs offline inference and provides the doctor-facing interface.

---

## 3. Master System Workflow

```text
Doctor
  |
  ▼
React Frontend
- Landing page
- Login/signup
- MRI upload
- VQA result panel
- Grad-CAM viewer
- Real-time chat
- Dashboard
- Case history
- PDF reports
  |
  | REST /api/predict
  | WebSocket /ws/chat
  ▼
FastAPI Backend
  |
  ├──> NeuroVQA VQA Engine
  |    BiomedCLIP + PubMedBERT + Cross-Attention
  |    Output: answer, confidence, top-3, Grad-CAM
  |
  └──> Ollama Local LLM
       Small quantized model
       Input: verified VQA context + chat history
       Output: streamed grounded explanation
```

---

## 4. MRI-Grounded VQA Pipeline

### Module A: MRI Ingestion and Preprocessing (`src/dataset.py`)

```text
[Raw NIfTI / DICOM / PNG / JPG MRI]
|
├── Volumetric NIfTI/DICOM?
|   ├── Yes -> Select axial slice k
|   |        -> Extract slices k-1, k, k+1
|   |        -> Stack as 3 channels -> (H, W, 3)
|   └── No  -> Load 2D grayscale MRI
|            -> Replicate across 3 channels -> (H, W, 3)
|
▼
[Optional MRI Intensity Clipping] (1st–99th percentile)
|
▼
[Min-Max Normalization -> [0.0, 1.0]]
|
▼
[Bilinear Resize -> 224 x 224 x 3]
|
▼
[Torchvision Normalization]
mean = [0.48145466, 0.4578275, 0.40821073]
std  = [0.26862954, 0.26130258, 0.27577711]
|
▼
[Output Tensor: (B, 3, 224, 224)]
```

Do not use CT HU windowing or CT brain-window values for MRI.

---

### Module B: Dual-Encoder Feature Extraction (`src/model.py`)

```text
VISUAL STREAM                              TEXT STREAM
[MRI Tensor: (B, 3, 224, 224)]             [Clinical Question]
|                                          |
▼                                          ▼
[BiomedCLIP ViT-B/16]                      [PubMedBERT Tokenizer]
196 patch tokens + 1 CLS token             Up to 64 tokens
|                                          |
▼                                          ▼
[Visual Linear Adapter]                    [Text Linear Adapter]
Linear(512, 512) + GELU                    Linear(512, 512) + GELU
|                                          |
▼                                          ▼
[Projected Visual Tokens]                  [Projected Text Tokens]
(B, 197, 512)                              (B, 64, 512)
```

---

### Module C: Cross-Modal Attention and Classification (`src/model.py`)

```text
Text Tokens -> Queries (Q)
Image Tokens -> Keys (K), Values (V)
|
▼
[Scaled Dot-Product Cross-Attention]
|
▼
[Attention over 196 MRI patch tokens]
|
▼
[Weighted visual evidence aggregation]
|
▼
[LayerNorm and sequence pooling]
|
▼
[MLP classifier]
|
▼
[Top-1 answer + confidence + top-3 predictions]
|
▼
[Cross-entropy training loss]
```

---

### Module D: Grad-CAM Explainability (`src/explainability.py`)

```text
[Winning class score]
|
▼
[Gradient with respect to attention activations]
|
▼
[Global average pooling -> channel importance]
|
▼
[Weighted attention map -> ReLU]
|
▼
[Reshape 196 patch tokens -> 14 x 14 grid]
|
▼
[Bilinear upsample -> 224 x 224]
|
▼
[Normalize and apply Jet colormap]
|
▼
[Overlay heatmap on MRI scan]
|
▼
[Return heatmap and focused region to frontend]
```

---

## 5. Doctor Workspace Features

### 5.1 Analysis Dashboard

```text
+--------------------------------------------------------------------+
| NeuroVQA Analysis Dashboard                                        |
+----------------------+----------------------+----------------------+
| Total Cases: 128     | Abnormal: 41         | Normal: 76           |
| Uncertain: 11        | Avg Confidence: 83%  | Avg Latency: 1.72s   |
+----------------------+----------------------+----------------------+
| Confidence Distribution Chart / Recent Cases / Feedback Summary     |
+--------------------------------------------------------------------+
```

Metrics:

- Total MRI cases analyzed
- Abnormal, normal, and uncertain prediction counts
- Average VQA confidence
- Average inference latency
- Recent cases
- Confidence distribution
- Doctor-feedback summary
- Model version and checkpoint information

### 5.2 Downloadable PDF Report

After each VQA analysis, doctors can download a structured PDF report.

Report contents:

- NeuroVQA branding and report title
- Doctor name and analysis timestamp
- MRI file metadata
- Clinical question
- VQA answer
- Confidence score
- Top-3 predictions with percentage bars
- Original MRI image
- Grad-CAM heatmap
- Focused region
- Model version and checkpoint
- Inference latency
- Doctor notes
- Doctor feedback
- Medical disclaimer

Use FastAPI + ReportLab for server-side PDF generation. Store report metadata in SQLite and generated PDFs under `backend/data/reports/`.

### 5.3 Case History

The history page lets doctors reopen previous MRI cases.

Each case includes:

- MRI thumbnail
- Analysis date
- Clinical question
- AI answer
- Confidence score
- Abnormal/normal badge
- Grad-CAM thumbnail
- Linked chat session
- Doctor feedback
- Download report button

Doctors can search and filter by date, result type, confidence range, and feedback status.

### 5.4 Doctor Feedback

After each prediction, doctors can provide feedback:

```text
Was this AI finding helpful?
[ Correct ] [ Partially Correct ] [ Incorrect ] [ Uncertain ]
```

Doctors may add an optional note:

> The heatmap region matches the suspected lesion, but the confidence appears too high.

Feedback supports human-in-the-loop evaluation and measures agreement between AI predictions and expert review.

### 5.5 Audit Trail

Record major actions locally:

```text
audit_logs
- id
- user_id
- diagnosis_id
- action
- model_version
- model_checkpoint
- inference_latency
- created_at
```

Actions include:

- `MRI_UPLOADED`
- `VQA_RUN`
- `HEATMAP_GENERATED`
- `CHAT_STARTED`
- `REPORT_DOWNLOADED`
- `FEEDBACK_SUBMITTED`

### 5.6 MRI Comparison

Allow doctors to compare the current MRI with a previous scan.

```text
+----------------------+----------------------+
| Current MRI          | Previous MRI         |
| Grad-CAM overlay     | Grad-CAM overlay     |
| Answer: Positive 87% | Answer: Negative 74% |
+----------------------+----------------------+
```

### 5.7 MRI Viewer Tools

Provide basic viewer tools:

- Zoom and pan
- Brightness and contrast adjustment
- Crosshair navigation
- Rectangle or freehand region selection
- Distance measurement
- Doctor notes attached to a region
- “Ask AI about this region” action

---

## 6. Real-Time Clinical Chat Workflow

### 6.1 Chat Principle

```text
MRI image + doctor question
        ↓
NeuroVQA VQA Engine
        ↓
Verified answer + confidence + heatmap region
        ↓
Ollama LLM receives constrained context
        ↓
Grounded explanation and follow-up discussion
```

The LLM receives only:

- Latest VQA answer
- Confidence score
- Top-3 predictions
- Grad-CAM focused region
- Relevant retrieved medical context, if RAG is enabled
- Recent conversation history

### 6.2 Lightweight Chat Model

Because the local laptop has 8 GB RAM, integrated graphics, and must also run NeuroVQA inference, use a small quantized model.

| Priority | Model | Recommendation |
|---|---|---|
| Best for 8 GB RAM | `llama3.2:1b` | Use this as the default chat model |
| Better quality if memory allows | `phi3:mini` | Test only after VQA warm-up |
| Avoid | 7B/8B models | Too heavy alongside NeuroVQA |

### 6.3 WebSocket Flow

```text
React Chat Panel
      |
      | WebSocket /ws/chat
      ▼
FastAPI Chat Service
      |
      | Builds grounded prompt
      ▼
Ollama /api/chat with stream=true
      |
      ▼
Token-by-token NDJSON stream
      |
      ▼
FastAPI forwards each token to React
      |
      ▼
React renders response progressively
```

### 6.4 Example Interaction

**Doctor:**  
“Is there an acute lesion present?”

**VQA Engine:**

```json
{
  "answer": "Yes",
  "confidence": 0.87,
  "top_3": [
    {"answer": "Yes", "confidence": 0.87},
    {"answer": "No", "confidence": 0.08},
    {"answer": "Uncertain", "confidence": 0.05}
  ],
  "focused_region": "Right parietal region"
}
```

**Doctor:**  
“Why do you think there is a lesion?”

**NeuroVQA Assistant:**  
“The VQA model classified the scan as positive for an acute lesion with 87% confidence. The Grad-CAM attention map highlights the right parietal region as the main area supporting this prediction. This finding should be correlated with clinical history, additional MRI sequences, and expert radiologist review.”

---

## 7. Safety and Clinical Governance

### Mandatory Chat Rules

* NeuroVQA is a **clinical decision-support assistant**, not an autonomous diagnostic system.
* The VQA model is the only component allowed to interpret the MRI.
* The LLM must not generate a new MRI finding that is absent from the VQA result.
* If confidence is low, the assistant must clearly state that evidence is insufficient.
* The assistant must recommend correlation with clinical history, additional sequences, and radiologist review.
* The assistant must never recommend medication, surgery, or definitive treatment.
* A medical disclaimer must remain visible in the chat interface.

### Recommended Disclaimer

> NeuroVQA is a research and clinical decision-support tool. It does not replace professional radiological interpretation or clinical judgment. All findings should be verified by a qualified radiologist or clinician.

---

## 8. Optional Local RAG Layer

```text
Doctor follow-up question
      ↓
Local embedding model: nomic-embed-text
      ↓
ChromaDB vector search
      ↓
Retrieve relevant radiology notes / guidelines
      ↓
Combine with VQA result and Grad-CAM region
      ↓
Ollama generates grounded explanation
```

Enable RAG only if memory permits. On the 8 GB target laptop, keep the retrieval corpus small and use a lightweight embedding model.

---

## 9. Database Design

Use SQLite for offline local storage.

```text
users
- id
- full_name
- email
- hashed_password
- created_at

diagnoses
- id
- user_id
- mri_file_path
- question
- answer
- confidence
- top_3_predictions_json
- heatmap_path
- report_path
- model_version
- latency_seconds
- created_at

chat_sessions
- id
- user_id
- diagnosis_id
- title
- created_at

chat_messages
- id
- session_id
- role
- content
- created_at

feedback
- id
- diagnosis_id
- doctor_id
- feedback_type
- doctor_note
- created_at

audit_logs
- id
- user_id
- diagnosis_id
- action
- model_version
- model_checkpoint
- inference_latency
- created_at
```

Do not store MRI image binaries inside SQLite. Store images, heatmaps, and generated PDF reports on the local filesystem; save only metadata and results in the database.

---

## 10. Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | React 18 + TypeScript + Vite | Doctor-facing multi-page web application |
| Styling | Tailwind CSS v4 | Mandate dark-slate responsive UI |
| Animation | Motion / Framer Motion | Page transitions, chat animations, micro-interactions |
| Routing | React Router | `/`, `/login`, `/signup`, `/app`, `/dashboard`, `/history` |
| Real-time chat | WebSocket + FastAPI | Live token streaming and interactive chat |
| Chat LLM | Ollama + Llama 3.2 1B or Phi-3 Mini | Lightweight offline conversational explanation |
| API | FastAPI + Uvicorn | MRI upload, VQA inference, auth, chat, reports |
| Database | SQLite + SQLAlchemy + Alembic | Users, diagnoses, chat sessions, feedback, audit logs |
| Authentication | JWT + bcrypt | Secure local doctor authentication |
| VQA Vision Model | BiomedCLIP ViT-B/16 | MRI visual token extraction |
| VQA Text Model | PubMedBERT | Clinical question understanding |
| Fine-Tuning | PEFT / LoRA on Google Colab T4 | Lightweight adapter training |
| Explainability | Grad-CAM + OpenCV | Attention heatmap and focused-region evidence |
| Dashboard Charts | Recharts or Chart.js | Case statistics, confidence trends, and feedback visualization |
| PDF Reports | ReportLab | Downloadable structured MRI analysis reports |
| Optional RAG | nomic-embed-text + ChromaDB | Local medical-knowledge retrieval |

---

## 11. UI/UX Design System

### 11.1 Color Tokens

* Canvas background: `#0B0D11`
* Surface cards: `#161922`
* Elevated inputs: `#1E2230`
* Subtle borders: `#2A2F42`
* Primary accent: `#342CB3`
* Normal / verified: `#10B981`
* Abnormal / risk: `#E11D48`

### 11.2 Application Routes

| Route | Purpose |
|---|---|
| `/` | Landing page introducing NeuroVQA |
| `/login` | Doctor login |
| `/signup` | Doctor registration |
| `/app` | Protected MRI diagnostic workspace |
| `/dashboard` | Analysis dashboard and metrics |
| `/history` | Previous MRI cases and reports |
| Chat panel | Real-time MRI-grounded chat |

### 11.3 Workspace Layout

```text
+--------------------------------------------------------------------+
| NeuroVQA | Clinical AI Diagnostic Assistant | Doctor: Dr. Anfino    |
+------------------+--------------------------+----------------------+
| MRI Upload       | MRI Viewer               | Clinical Chat        |
| Demo Cases       | Raw Scan                 | Streaming responses  |
| Scan Metadata    | Grad-CAM Heatmap         | Follow-up questions  |
| VQA Question     | Prediction Result        | Confidence context   |
| Run Inspection   | Top-3 Predictions        | Medical disclaimer   |
+------------------+--------------------------+----------------------+
```

---

## 12. Execution Profiles

| Dimension | Google Colab Training | Local Doctor Workstation |
|---|---|---|
| Hardware | Google Colab / Kaggle T4 16 GB VRAM | Intel Core i3 11th Gen, 8 GB RAM, integrated graphics |
| Compute mode | FP16 mixed precision with backpropagation | CPU inference with `torch.no_grad()` |
| Batch size | 16 | 1 |
| Network | Required for training/setup | Fully offline after setup |
| VQA latency | Training-oriented | <= 2.5 seconds per single-slice query |
| Chat latency | Not applicable | Progressive token streaming |
| Model artifacts | LoRA adapter checkpoint, approximately 50 MB | Local BiomedCLIP, PubMedBERT, LoRA adapter, small Ollama model |
| Data privacy | Training data handled in cloud environment | MRI and chat data remain local |
