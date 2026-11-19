# Environment Setup & Execution Manual — NeuroVQA (Version 2.0)

**Target Profile:** Cloud Training (Google Colab T4 GPU) $\rightarrow$ Local Offline Inference (Intel Core i3 11th Gen, 8 GB RAM, 512 GB SSD, Integrated Graphics)  
**Offline Invariant:** 100% offline-capable after local assets are installed. No cloud APIs during clinical inference.

---

## 1. Cloud Training Pipeline (Google Colab T4 GPU)

> ⚠️ **Hardware Directive:** All deep learning fine-tuning must be conducted in the cloud on a CUDA GPU (Google Colab free T4 runtime). **Never attempt model training on the local Intel i3 laptop.**

### 1.1 Dataset Preparation & Automated Downloads
Create a new Colab notebook (`train_colab.ipynb`) with GPU runtime (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**).

Download and prepare the 4 composite datasets (~12,157 images, 35,000+ QA pairs):

```bash
# 1. Install HuggingFace & Kaggle CLIs
!pip install -q huggingface_hub kaggle datasets

# 2. Download VQA-RAD (315 images, ~3.5k QA)
from datasets import load_dataset
ds_vqa_rad = load_dataset("flaviagiammarino/vqa-rad")
ds_vqa_rad.save_to_disk("/content/data/vqa_rad")

# 3. Download SLAKE (642 images, 14k QA with brain MRI subset)
ds_slake = load_dataset("BoKelvin/SLAKE")
ds_slake.save_to_disk("/content/data/slake")

# 4. Download VQA-Med 2019 (4,200 images, 15k QA)
!git clone https://github.com/abachaa/VQA-Med-2019 /content/data/vqa_med_2019

# 5. Download Kaggle Brain Tumor MRI Dataset (7,000 high-res scans)
# Ensure kaggle.json is uploaded to ~/.kaggle/kaggle.json
!kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri -p /content/data/brain_tumor --unzip
```

### 1.2 Synthetic VQA Generation Script (Brain Tumor MRI)
Run the following script to generate 5 deterministic questions per scan, formatted into the universal training schema `{image_path, question, answer}`:

```python
import os
import json
from glob import glob

# 15 Fixed Canonical Classes
ANSWER_CLASSES = [
    "Yes", "No", "Uncertain", "Left", "Right", 
    "Frontal", "Parietal", "Temporal", "Occipital", 
    "Glioma", "Meningioma", "Pituitary", "Tumor", "Normal", "Edema"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(ANSWER_CLASSES)}

TEMPLATES = [
    ("Is there an abnormality present?", lambda label: "Yes" if label != "notumor" else "No"),
    ("Is there a tumor present?", lambda label: "Yes" if label != "notumor" else "No"),
    ("What type of tumor is indicated?", lambda label: {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "notumor": "Normal"
    }.get(label, "Normal")),
    ("Which side is abnormality on?", lambda label: "Left" if "left" in label else ("Right" if "right" in label else "Normal")),
    ("Which lobe is affected?", lambda label: "Temporal" if "glioma" in label else ("Frontal" if "meningioma" in label else "Normal")),
]

def generate_synthetic_vqa(data_dir: str, output_json: str):
    records = []
    classes = ["glioma", "meningioma", "pituitary", "notumor"]
    
    for cls in classes:
        folder = os.path.join(data_dir, "**", cls, "*.*")
        for img_path in glob(folder, recursive=True):
            if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            for q_text, ans_fn in TEMPLATES:
                records.append({
                    "image_path": img_path,
                    "question": q_text,
                    "answer": ans_fn(cls)
                })
                
    with open(output_json, "w") as f:
        json.dump(records, f, indent=2)
    print(f"Generated {len(records)} QA pairs from {data_dir}")

# Execute generation
generate_synthetic_vqa("/content/data/brain_tumor", "/content/data/synthetic_tumor_vqa.json")
```

### 1.3 Patient-Level Splitting & Augmentation Rules
- **Split Ratio:** Split all images **80% Train / 10% Validation / 10% Test** grouped strictly by **Patient ID** to prevent cross-slice data leakage.
- **Data Augmentations:**
  - **Allowed:** Rotation ($\pm10^\circ$), Zoom ($0.9\times - 1.1\times$), Gaussian Noise, Intensity Shift, and Slice Shift ($[k-1, k, k+1]$ and $[k-2, k, k+2]$).
  - **Explicitly Forbidden: Horizontal Flip.** Inverting axial brain MRI flips the anatomical left and right hemispheres (viewer's left is patient's right), invalidating labels like `Left` and `Right`.

### 1.4 Colab Training Execution (PyTorch + Focal Loss + LoRA)
Install training libraries:
```bash
!pip install -q open_clip_torch transformers datasets accelerate peft scikit-learn
```

Execute training script with the following hyperparameters:
- **Vision Backbone:** BiomedCLIP ViT-B/16 (Unfreeze last 2 Transformer blocks, LR $1\times 10^{-5}$).
- **Text Backbone & Fusion:** PubMedBERT with LoRA ($r=16, \alpha=32, \text{dropout}=0.1$) + Gated Fusion Head (LR $2\times 10^{-4}$).
- **2.5D Volumetric Input:** 3 contiguous slices $[k-1, k, k+1]$ mean-pooled into 512-d visual embedding.
- **Batch Size:** 16
- **Epochs:** 15 (Save the best checkpoint based on validation loss, not the final epoch).
- **Loss Function:** Focal Loss to counter the 70% affirmative skew:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()
```

### 1.5 Export LoRA & Gated Fusion Weights
Export the trained weights to Google Drive:
```python
import torch

torch.save({
    "fusion_head_state_dict": model.fusion_head.state_dict(),
    "lora_adapter_state_dict": model.get_lora_state_dict(),
    "class_to_idx": CLASS_TO_IDX,
    "classes": ANSWER_CLASSES,
    "model_version": "NeuroVQA-v2.0-BiomedCLIP-GatedFusion"
}, "/content/drive/MyDrive/neurovqa_weights.pt")
```

Download `neurovqa_weights.pt` to your local machine at:
`k:\NeuroVQA V26\backend\checkpoints\neurovqa_weights.pt`

---

## 2. Local Workstation Setup (Intel Core i3, 8 GB RAM)

### 2.1 Python Virtual Environment & Dependencies
Verify Python 3.10+ is installed:
```powershell
python --version
```

From the root project directory (`k:\NeuroVQA V26`):
```powershell
# Create virtual environment if needed
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install lightweight CPU runtime dependencies
pip install -r backend/requirements.txt
```

### 2.2 Ollama Chat Service Setup (Llama 3.2 1B Only)
> ⚠️ **Hardware Constraint:** `phi3:mini` is **strictly prohibited** because it generates at only 4–6 tokens/sec on an i3 CPU and consumes 2.3 GB RAM. Only the lightweight `llama3.2:1b` model is supported.

1. Download and install Ollama from [https://ollama.com](https://ollama.com).
2. Pull the quantized 1B model:
   ```powershell
   ollama pull llama3.2:1b
   ```
3. Test streaming in terminal:
   ```powershell
   ollama run llama3.2:1b "Explain brain MRI T1 contrast in 20 words."
   ```
4. Ollama runs as a background service on port `11434`.

### 2.3 Offline Environment Configuration
To prevent Hugging Face and PyTorch from making outbound calls during local clinical inference, configure offline variables:

**PowerShell:**
```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

**Command Prompt / System Environment:**
```bat
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
```

---

## 3. Running NeuroVQA Locally

### 3.1 Start the FastAPI Backend Server (Terminal 1)
```powershell
cd "k:\NeuroVQA V26"
.\venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --port 8000 --reload
```
- API Health Check: `http://localhost:8000/api/health`
- Swagger Interactive Documentation: `http://localhost:8000/docs`

### 3.2 Start the React + Vite Frontend (Terminal 2)
```powershell
cd "k:\NeuroVQA V26\frontend"
npm install
npm run dev
```
- Doctor Web Interface: `http://localhost:5173`

---

## 4. Sequential Memory Operation & Verification

To verify that the sequential memory model stays within the $\le 4.6\text{ GB}$ ceiling:

1. **Baseline Idle State:**
   - Open Windows Task Manager (`Ctrl + Shift + Esc`) $\rightarrow$ Performance $\rightarrow$ Memory.
   - Baseline system RAM: ~2.5 GB.
2. **During MRI Analysis (VQA Active):**
   - Upload MRI scan and submit question.
   - Active memory increases to ~3.8 GB as 2.5D BiomedCLIP and Gated Fusion execute.
   - Inference completes in $\le 4.0$ seconds after warm-up on Intel Core i3 (using `torch.set_num_threads(2)`).
   - Attention $\times$ Gradient Grad-CAM renders on the viewer.
3. **During Grounded Doctor Chat (Ollama Active):**
   - Click one of the 3 quick chips: `[Why this answer?]`, `[What region?]`, or `[How confident?]`.
   - VQA engine unloads transient cache (`gc.collect()`).
   - Ollama loads `llama3.2:1b` (~1.2 GB RAM). Peak RAM remains $\le 4.6\text{ GB}$.
   - Text streams token-by-token at 12–15 tokens/sec.
   - Generation terminates strictly at 140 tokens (~8–10 sec).
   - Ollama unloads via `keep_alive: "0s"` when streaming concludes.

---

## 5. Offline Airplane Mode Verification Protocol

Before demonstration, presentation, or viva defense, execute this step-by-step checklist:

- [ ] Disconnect Wi-Fi and Ethernet (enable **Airplane Mode** in Windows Settings).
- [ ] Set environment flags: `$env:HF_HUB_OFFLINE = "1"` and `$env:TRANSFORMERS_OFFLINE = "1"`.
- [ ] Launch FastAPI backend on port 8000.
- [ ] Launch React frontend on port 5173.
- [ ] **Test Case 1 (Clear Tumor):**
  - Upload left temporal tumor slice.
  - Submit: *"Is there a tumor present?"*
  - Verify prediction: `Yes` (Confidence $\ge 0.90$).
  - Confirm focal Attention $\times$ Gradient heatmap renders over the mass.
- [ ] **Test Case 2 (Normal Scan):**
  - Upload normal brain slice.
  - Submit: *"Is there an abnormality present?"*
  - Verify prediction: `Normal` (Confidence $\ge 0.85$).
- [ ] **Test Case 3 (Borderline Case):**
  - Upload borderline faint lesion slice.
  - Verify prediction outputs `Uncertain - Needs Radiologist Review` (Confidence $< 0.60$).
- [ ] **Test Grounded Chat:**
  - Click `[Why this answer?]`.
  - Confirm token-by-token streaming response without internet.
  - Confirm disclaimer: *"For clinician review only."*
- [ ] **Test Clinical PDF Export:**
  - Click "Download Clinical PDF".
  - Verify exported PDF contains case metadata, slice preview, Grad-CAM heatmap, and legal disclaimers.
- [ ] **Test History & Feedback:**
  - Confirm case appears in History tab.
  - Submit feedback rating (`Correct`) and verify it persists in local SQLite (`backend/neurovqa.db`).

---

## 6. Phase 1 Scope Cuts (Explicitly Deferred)

The following components are deferred to Phase 2 to preserve the 8 GB RAM / i3 performance invariant:
- $\times$ **ChromaDB / Vector RAG:** (Consumes 800 MB+ RAM and introduces extraneous retrieval latency).
- $\times$ **Dual-MRI Side-by-Side Comparison:** (Requires running 2 concurrent VQA pipelines, exceeding the 4.0s latency budget).
- $\times$ **Full 3D NIfTI Volumetric Segmentation:** (Requires heavy 3D ConvNets and high memory).
- $\times$ **Cloud LLM API Mode:** (Violates the 100% offline air-gapped hospital requirement).
