# Environment Setup & Execution Manual — NeuroVQA

This manual describes the hybrid deployment setup for NeuroVQA:

- **Google Colab / Kaggle T4 GPU** for dataset handling and LoRA fine-tuning.
- **Local Intel i3 laptop (8 GB RAM, 512 GB SSD, integrated graphics)** for offline inference, real-time chat, dashboard, reports, history, feedback, and audit trail.

---

## 1. Deployment Architecture

```text
CLOUD TRAINING ENVIRONMENT                     LOCAL DOCTOR WORKSTATION
+------------------------------+               +----------------------------------+
| Google Colab / Kaggle T4 GPU |               | Intel Core i3 / 8 GB RAM laptop  |
| - PyTorch CUDA               |               | - React + Vite frontend          |
| - BiomedCLIP + PubMedBERT    | --Export-->   | - FastAPI backend                |
| - PEFT / LoRA training       |   Adapter     | - PyTorch CPU inference          |
| - VQA-RAD training           |   Weights     | - Ollama small local LLM         |
| - No local GPU required      |               | - SQLite local database          |
+------------------------------+               | - 100% offline after setup       |
                                                +----------------------------------+
```

**Important:** Do not train NeuroVQA on the local laptop. The local system is optimized only for lightweight CPU inference and the doctor-facing interface.

---

## 2. Local Hardware Profile

* Intel Core i3 11th Gen
* 8 GB RAM
* 512 GB SSD
* Integrated graphics
* Windows 64-bit
* No dedicated GPU required

Recommended local resource limits:

| Component | Target |
|---|---|
| NeuroVQA VQA inference memory | < 2.5 GB |
| Chat LLM | Small quantized 1B–3B model |
| SQLite database | Local file |
| Browser + React frontend | Lightweight production build preferred |
| Concurrent heavy tasks | Avoid training, large browser tabs, and multiple LLM sessions |

---

## 3. Project Directory Structure

```text
NeuroVQA/
├── NeuroVQ-prd.md
├── NeuroVQA-design.md
├── NeuroVQA-setup.md
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── auth.py                 # JWT authentication
│   ├── database.py             # SQLite + SQLAlchemy
│   ├── chat.py                 # WebSocket chat and Ollama streaming
│   ├── reports.py              # PDF report generation
│   ├── feedback.py             # Doctor feedback endpoints
│   ├── audit.py                # Audit trail service
│   ├── dashboard.py            # Dashboard analytics endpoints
│   ├── requirements.txt
│   ├── checkpoints/
│   │   ├── biomedclip/         # Local BiomedCLIP weights/config
│   │   ├── pubmedbert/         # Local PubMedBERT model/tokenizer
│   │   └── neurovqa_weights.pt # LoRA adapter checkpoint
│   ├── data/
│   │   ├── demo_samples/       # Curated MRI demo cases
│   │   ├── reports/            # Generated PDF reports
│   │   └── knowledge/          # Optional local RAG documents
│   └── src/
│       ├── __init__.py
│       ├── dataset.py          # MRI preprocessing and 2.5D packing
│       ├── model.py            # BiomedCLIP + cross-attention VQA
│       ├── explainability.py   # Grad-CAM heatmap generation
│       └── rag.py              # Optional local retrieval module
├── frontend/
│   ├── src/
│   │   ├── pages/              # Home, Login, Signup, Workspace, Dashboard, History
│   │   ├── components/         # Navbar, ChatPanel, MRIViewer, ResultPanel
│   │   └── routes/             # React Router configuration
│   ├── package.json
│   └── vite.config.ts
└── train_colab.ipynb
```

---

## 4. Backend Setup

### 4.1 Create Python Environment

```bash
python --version

mkdir NeuroVQA
cd NeuroVQA

python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# Windows Command Prompt
.\venv\Scripts\activate.bat
```

### 4.2 Create `backend/requirements.txt`

```text
# API server
fastapi>=0.110.0
uvicorn>=0.29.0
python-multipart>=0.0.9
websockets>=12.0
httpx>=0.27.0

# Authentication and database
sqlalchemy>=2.0.0
passlib[bcrypt]>=1.7.4
python-jose[cryptography]>=3.3.0
alembic>=1.13.0

# PyTorch CPU build
torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
torchvision>=0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Multimodal VQA models
open_clip_torch>=2.24.0
transformers>=4.38.0
peft>=0.8.0
accelerate>=0.29.0

# Image processing and explainability
Pillow>=10.2.0
opencv-python>=4.9.0
matplotlib>=3.8.0
numpy>=1.24.0
SimpleITK>=2.3.1

# PDF report generation
reportlab>=4.1.0

# Optional local RAG
chromadb>=0.5.0
sentence-transformers>=2.7.0
```

Install:

```bash
pip install -r backend/requirements.txt
```

---

## 5. Frontend Setup

```bash
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
npm install react-router-dom motion axios zustand react-hook-form zod lucide-react recharts
npm install tailwindcss @tailwindcss/vite
```

Configure `frontend/vite.config.ts`:

```ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
});
```

Add Tailwind to `frontend/src/index.css`:

```css
@import "tailwindcss";
```

---

## 6. Ollama Chat Model Setup

Because the laptop has only 8 GB RAM and integrated graphics, use a **small quantized chat model**. Do not use an 8B model while NeuroVQA VQA models are loaded.

Recommended:

```bash
ollama pull llama3.2:1b
```

Alternative:

```bash
ollama pull phi3:mini
```

For the most reliable offline demo on 8 GB RAM, start with `llama3.2:1b`. If memory remains available after loading NeuroVQA, test `phi3:mini`.

Verify:

```bash
ollama run llama3.2:1b
```

Ollama runs locally on port `11434` and supports streaming chat responses.

---

## 7. Prepare Offline Model Assets

Before enabling Airplane Mode, download and store these assets locally:

```text
backend/checkpoints/biomedclip/
├── open_clip_pytorch_model.bin
└── open_clip_config.json

backend/checkpoints/pubmedbert/
├── config.json
├── tokenizer.json
├── tokenizer_config.json
├── vocab.txt
└── model weights

backend/checkpoints/neurovqa_weights.pt
backend/data/demo_samples/case1.png
backend/data/demo_samples/case2.png
backend/data/demo_samples/case3.png
```

Set offline environment variables before starting the backend:

PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Command Prompt:

```bat
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

---

## 8. Google Colab Training Pipeline

### 8.1 Create Colab Notebook

1. Open Google Colab.
2. Create `NeuroVQA_Training.ipynb`.
3. Select **Runtime > Change runtime type > T4 GPU**.

### 8.2 Install Training Dependencies

```python
!pip install -q open_clip_torch transformers datasets accelerate peft albumentations scikit-learn
!nvidia-smi
```

### 8.3 Mount Google Drive

```python
from google.colab import drive
import os

drive.mount('/content/drive')

CHECKPOINT_DIR = "/content/drive/MyDrive/NeuroVQA_Checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
```

### 8.4 Train and Export LoRA Adapter

```python
import torch

torch.save({
    "model_state_dict": model.fusion_head.state_dict(),
    "class_to_idx": train_dataset.class_to_idx,
    "idx_to_class": train_dataset.idx_to_class
}, f"{CHECKPOINT_DIR}/neurovqa_weights.pt")
```

Download the exported checkpoint into:

```text
backend/checkpoints/neurovqa_weights.pt
```

Save checkpoints to Google Drive during training to avoid losing them if the Colab session disconnects.

---

## 9. Run NeuroVQA Locally

### 9.1 Start FastAPI Backend

```bash
cd NeuroVQA
.\venv\Scripts\activate
uvicorn backend.main:app --reload --port 8000
```

Backend API documentation:

<http://localhost:8000/docs>

### 9.2 Start React Frontend

Open a second terminal:

```bash
cd NeuroVQA/frontend
npm run dev
```

Frontend:

<http://localhost:5173>

For the viva, build the frontend first and serve the production build through FastAPI to reduce browser memory usage:

```bash
cd frontend
npm run build
```

### 9.3 Start Ollama

```bash
ollama serve
```

---

## 10. Additional Feature Workflow

```text
1. Doctor uploads MRI and runs VQA analysis
2. System saves diagnosis, heatmap, latency, and model version
3. Doctor asks follow-up questions through real-time chat
4. Doctor downloads a structured PDF report
5. Doctor submits feedback: Correct / Partially Correct / Incorrect / Uncertain
6. Dashboard updates case statistics, confidence trends, and feedback summary
7. Audit trail records every major action locally
8. Doctor can reopen historical cases and compare current and previous MRIs
```

---

## 11. Offline Verification Checklist

- [ ] React frontend dependencies installed
- [ ] FastAPI backend dependencies installed
- [ ] BiomedCLIP weights stored locally
- [ ] PubMedBERT model and tokenizer stored locally
- [ ] LoRA adapter checkpoint available
- [ ] Small Ollama chat model downloaded
- [ ] Demo MRI scans available
- [ ] SQLite database initialized
- [ ] Doctor signup/login tested
- [ ] MRI upload and VQA prediction tested
- [ ] Grad-CAM heatmap displayed
- [ ] Real-time chat tested
- [ ] PDF report generated and downloaded
- [ ] Dashboard metrics displayed
- [ ] Case history searchable
- [ ] Doctor feedback submitted
- [ ] Audit trail recorded
- [ ] Airplane mode enabled and full workflow retested
