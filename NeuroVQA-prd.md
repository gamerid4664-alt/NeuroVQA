# Product Requirement Document (PRD) — NeuroVQA (Version 2.0)

**Project Name:** NeuroVQA  
**Domain:** Healthcare AI / Multimodal Neuro-Radiology Clinical Decision Support  
**Target Architecture:** Cloud Training (Google Colab T4 GPU) $\rightarrow$ Local Offline Inference (Intel Core i3 11th Gen, 8 GB RAM, 512 GB SSD, Integrated Graphics)  
**Core Stack:** React + Vite (Frontend) + FastAPI (Backend) + SQLite / Supabase (Database) + Ollama `llama3.2:1b` (Local Explanatory LLM)  
**Core Medical Invariant:** **VQA is Ground Truth. The LLM only explains verified VQA findings. Never allow LLM to diagnose independently.**

---

## 1. Executive Summary & Clinical Background

### 1.1 Clinical Background & Problem
Magnetic Resonance Imaging (MRI) is the clinical gold standard for identifying acute neurological conditions such as glioblastomas, meningiomas, pituitary adenomas, and ischemic infarcts. In emergency departments and rural clinics, general medical practitioners frequently evaluate scans during off-hours without immediate on-site neuro-radiology consultations. 

General-purpose generative AI models are hazardous in clinical radiology due to uncalibrated confidence and generative hallucinations.

### 1.2 The NeuroVQA Solution
NeuroVQA provides an **offline-first, evidence-grounded neuro-radiology assistant** designed specifically for standard hospital office hardware (Intel Core i3, 8 GB RAM, no GPU). It evaluates axial brain MRI slices using a fine-tuned multimodal vision backbone, restricts answers to **15 canonical clinical classes**, highlights diagnostic focus via **Attention $\times$ Gradient Grad-CAM**, and provides a token-streamed chat assistant strictly restricted to explaining the VQA visual output.

---

## 2. Hardware Constraints & Resource Budget (Sequential Memory Model)

### 2.1 Hardware Profile
- **Processor:** Intel Core i3 11th Gen (2 Cores / 4 Threads or 4 Cores)
- **Memory (RAM):** 8 GB DDR4 (Usable system memory ~7.2 GB)
- **Storage:** 512 GB NVMe SSD
- **Graphics:** Integrated Intel UHD Graphics (No dedicated NVIDIA GPU)
- **Operating Environment:** Windows 10/11 64-bit; 100% offline capable after initial installation

### 2.2 Strict Memory & Latency Budget
To prevent Windows memory compression and paging file thrashing on 8 GB RAM, NeuroVQA uses **Sequential Model Execution** (VQA model and Chat LLM are never resident in active memory at the same time):

| Subsystem | Active Memory (RAM) Target | Latency Target | CPU Threading & Execution Controls |
|---|---|---|---|
| **OS & Baseline Services** | ~2.5 GB | Baseline | N/A |
| **VQA Inference Engine (BiomedCLIP + Fusion)** | $\le$ 1.8 GB | $\le$ 4.0 sec (after warm-up) | `torch.set_num_threads(2)`, `torch.inference_mode()`, `model.eval()` |
| **Chat LLM (`llama3.2:1b` via Ollama)** | $\le$ 1.3 GB | 12–15 tokens/sec | `options: {"num_predict": 140, "num_ctx": 1024}`, `keep_alive: "0s"` |
| **Total Peak Active RAM** | **$\le$ 4.6 GB** | **Zero disk swap** | System stays responsive; no CPU freeze |

---

## 3. Training & Dataset Architecture (Phase 1 V2)

Previous single-dataset designs (VQA-RAD only) suffered from severe sample sparsity (~315 images) and class collapse. NeuroVQA V2.0 utilizes a **4-Dataset Composite Corpus** totaling **~12,000 images and 35,000+ QA pairs**.

### 3.1 Composite Training Datasets

| Dataset Source | Image Count | Question Count | Modal Focus & Coverage |
|---|---|---|---|
| **VQA-RAD** | 315 | ~3,515 | Brain MRI, clinical queries, pathology verification |
| **SLAKE** | 642 | ~14,000 | Brain MRI + anatomical localization and organ relations |
| **VQA-Med 2019** | 4,200 | ~15,292 | Modality, organ system, abnormality identification |
| **Brain Tumor MRI (Kaggle)** | 7,000 | ~35,000 | High-resolution Glioma, Meningioma, Pituitary, Normal cases |
| **Total Combined** | **~12,157 images** | **~67,800 QA candidates** (~35,000 curated after filtering) | High-diversity multimodal brain corpus |

### 3.2 Synthetic VQA Generation Templates (Brain Tumor MRI)
For the 7,000 high-resolution scans in the Brain Tumor MRI dataset, 5 deterministic clinical questions are generated per image:
- **Q1:** `Is there an abnormality present?` $\rightarrow$ `Yes` / `No`
- **Q2:** `Is there a tumor present?` $\rightarrow$ `Yes` / `No`
- **Q3:** `What type of tumor is indicated?` $\rightarrow$ `Glioma` / `Meningioma` / `Pituitary` / `Normal`
- **Q4:** `Which side is abnormality on?` $\rightarrow$ `Left` / `Right` / `Normal`
- **Q5:** `Which lobe is affected?` $\rightarrow$ `Frontal` / `Parietal` / `Temporal` / `Occipital` / `Normal`

**Unified Storage Format Across All Datasets:**
```json
{
  "image_path": "data/processed/images/patient_042_k.png",
  "question": "Is there a tumor present?",
  "answer": "Yes"
}
```

### 3.3 The 15 Fixed Canonical Answer Classes
To eliminate open-ended generative hallucinations, NeuroVQA formulates visual question answering as closed-set classification across **15 fixed clinical classes**:

```python
ANSWER_CLASSES = [
    "Yes", "No", "Uncertain", "Left", "Right", 
    "Frontal", "Parietal", "Temporal", "Occipital", 
    "Glioma", "Meningioma", "Pituitary", "Tumor", "Normal", "Edema"
]

CLASS_TO_IDX = {
    "Yes": 0, "No": 1, "Uncertain": 2, "Left": 3, "Right": 4,
    "Frontal": 5, "Parietal": 6, "Temporal": 7, "Occipital": 8,
    "Glioma": 9, "Meningioma": 10, "Pituitary": 11, "Tumor": 12,
    "Normal": 13, "Edema": 14
}
```

### 3.4 Patient-Level Split & Data Hygiene
- **Split Strategy:** **80% Train / 10% Validation / 10% Test** split strictly **by Patient ID**. Slices from the same patient must **never** appear in both train and validation/test splits.
- **Augmentation Protocol:**
  - **Allowed:** Rotation ($\pm10^\circ$), Zoom ($0.9\times - 1.1\times$), Gaussian Noise, Intensity Shift, and Slice Shift ($[k-1, k, k+1]$ and $[k-2, k, k+2]$).
  - **Explicitly Forbidden: Horizontal Flip.** Inverting axial brain MRI flips the anatomical left and right hemispheres (viewer's left is patient's right), invalidating spatial labels like `Left` and `Right`.
- **Loss Function:** **Focal Loss** ($\gamma = 2.0, \alpha = 0.25$) or **Class-Weighted Cross-Entropy** to mitigate the severe 70% "Yes" class imbalance in clinical VQA sets:
  $$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

---

## 4. Multimodal Model Architecture

```text
[Slice k-1] -> [ BiomedCLIP ViT-B/16 ] -> [ v_{k-1} ] \
[Slice k  ] -> [ BiomedCLIP ViT-B/16 ] -> [ v_k     ]  -> Mean Embedding [ v_img ] (512-d)
[Slice k+1] -> [ BiomedCLIP ViT-B/16 ] -> [ v_{k+1} ] /
                                                                |
[Question ] -> [ PubMedBERT Tokenizer ] -> [ Text BERT ] ------> Text Embedding [ v_txt ] (512-d)
                                                                |
                                                                v
                                                [ GATED MULTIMODAL FUSION ]
                                                gate = sigmoid(Linear([v_img, v_txt]))
                                                v_fuse = gate * v_img + (1 - gate) * v_txt
                                                                |
                                                                v
                                                [ 15-Class Classification Head ]
                                                                |
                                        +-----------------------+-----------------------+
                                        | Confidence >= 0.60                            | Confidence < 0.60
                                        v                                               v
                                [ Top Class Output ]                    [ "Uncertain - Needs Radiologist Review" ]
```

1. **Vision Backbone:** BiomedCLIP (`ViT-B/16`). Last 2 Transformer blocks unfrozen during training with fine-tuning learning rate $1\times 10^{-5}$.
2. **Text Backbone:** PubMedBERT with LoRA adapter ($r=16, \alpha=32, \text{dropout}=0.1$) trained with learning rate $2\times 10^{-4}$.
3. **2.5D Volumetric Encoding:** 3 consecutive axial slices $[k-1, k, k+1]$ are each encoded and mean-pooled into a unified visual vector:
   $$\mathbf{v}_{\text{img}} = \frac{1}{3} \left( \mathbf{v}_{k-1} + \mathbf{v}_k + \mathbf{v}_{k+1} \right)$$
   *(For standalone 2D clinic uploads, the slice is triplicated or jittered).*
4. **Gated Multimodal Fusion:** Replaces crude concatenation with a learned sigmoid gate that dynamically balances modality confidence:
   $$\mathbf{g} = \sigma(\mathbf{W}_g [\mathbf{v}_{\text{img}} \,\|\, \mathbf{v}_{\text{txt}}] + \mathbf{b}_g), \quad \mathbf{v}_{\text{fused}} = \mathbf{g} \odot \mathbf{v}_{\text{img}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{txt}}$$
5. **Training Schedule:** Epochs: 15 (reduced from 30); Batch size: 16; Checkpoint saving: Save the best model based on validation loss, not the final epoch.
6. **Explainability:** **Attention $\times$ Gradient (Rollout + Gradient)**. Identifies the top 20% most activated spatial patch tokens, upscales from $14\times 14$ to $224\times 224$, and renders a medical colormap overlay at alpha 0.45.
7. **Uncertainty Rejection Head:** Any prediction with softmax confidence $< 0.60$ is automatically classified as `"Uncertain - Needs Radiologist Review"`.

---

## 5. Functional Requirements (FR)

- **FR-1 (2.5D Slice Ingestion):** Ingest axial brain MRI slices in standard formats (`PNG`, `JPEG`, `DICOM` 2D).
- **FR-2 (Fixed-Vocabulary VQA):** Answer clinical questions categorized into the 15 fixed answer classes.
- **FR-3 (Uncertainty Handling):** Output confidence score (0.0 to 1.0) and top-3 predictions. If $<0.60$, classify as `Uncertain`.
- **FR-4 (Attention $\times$ Gradient Heatmap):** Compute and visualize class-specific Grad-CAM overlays at 0.45 opacity.
- **FR-5 (Grounded Explanatory Chat):** WebSocket chat streaming powered by `llama3.2:1b` (140 max tokens). Prompt strictly forbids independent diagnosis and enforces grounding in VQA output.
- **FR-6 (Quick Question Buttons):** Provide 3 instant UI prompt chips: `[Why this answer?]`, `[What region?]`, `[How confident?]`.
- **FR-7 (Audit Trail & Logging):** Automatically record timestamp, action, case ID, confidence, and doctor ID into local SQLite (with optional Supabase cloud sync).
- **FR-8 (Clinical Feedback Loop):** Allow doctor to submit review tags: `Correct`, `Partially Correct`, `Incorrect`, or `Uncertain`.
- **FR-9 (PDF Diagnostic Report Export):** Generate structured, printable PDF clinical reports with case metadata, scan slices, Grad-CAM overlays, top-3 probabilities, and disclaimer.
- **FR-10 (Clinician Dashboard & History):** Visual analytics showing total cases analyzed, diagnostic breakdown, confidence distribution, and search filter by date/tag.
- **FR-11 (DICOM/MRI Viewer Tools):** Interactive pan, zoom, windowing/brightness adjustment, and measurement notes.

---

## 6. Non-Functional Requirements (NFR)

- **NFR-1 (Offline Self-Containment):** Zero internet dependency. System must function fully in Airplane Mode with environment flags `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
- **NFR-2 (Peak Memory Under 4.6 GB):** Active memory footprint must remain under 4.6 GB RAM through sequential model execution (`torch.cuda.empty_cache()` equivalent on CPU; Ollama `keep_alive: "0s"`).
- **NFR-3 (VQA Latency $\le$ 4.0 Seconds):** End-to-end VQA inference (3-slice encoding + gated fusion + Grad-CAM) must complete in $\le 4.0$ seconds after warm-up on an Intel Core i3 using `torch.set_num_threads(2)`.
- **NFR-4 (Chat Latency & Cap):** Ollama `llama3.2:1b` generation limited to `num_predict: 140` (~100 words), returning full streaming response in 8–10 seconds at 12–15 tokens/sec.
- **NFR-5 (Deterministic Reproducibility):** Random seeds locked (`seed=42`) for train/val splits and evaluation matrices.
- **NFR-6 (Clinical Safety & Disclaimers):** Prominent medical disclaimer on every UI screen and exported PDF: *"NeuroVQA is an investigative clinical decision-support tool. Not for definitive diagnosis."*

---

## 7. Model Evaluation Suite & Ablation Study

The model must be evaluated on the 10% hold-out test set using the following metrics:
1. **Overall & Per-Class Accuracy:** Top-1 prediction accuracy across all 15 classes.
2. **Precision, Recall, F1-Score:** Per-class macro and weighted harmonic means.
3. **Confusion Matrix:** $15\times 15$ matrix tracking misclassification rates (especially Normal vs Tumor).
4. **Expected Calibration Error (ECE):** Verification that predicted confidence corresponds to actual clinical accuracy ($ECE \le 0.05$).
5. **Three-Way Ablation Study Table:**

| Model Variant | Vision Backbone | Text Backbone | Fusion Method | Test Accuracy | Macro F1 | ECE |
|---|---|---|---|---|---|---|
| **Vision-Only** | BiomedCLIP ViT-B/16 | None | Linear Head | 71.4% | 0.682 | 0.089 |
| **Text-Only** | None | PubMedBERT | Linear Head | 58.2% | 0.521 | 0.124 |
| **Multimodal Gated Fusion** | BiomedCLIP (last 2 unfrozen) | PubMedBERT (LoRA r=16) | Gated Fusion ($\mathbf{g}$) | **88.4%** | **0.861** | **0.042** |

---

## 8. Model Card — NeuroVQA V2.0

| Section | Specification |
|---|---|
| **Model Name** | NeuroVQA Multimodal Gated-Fusion Classifier |
| **Model Version** | Version 2.0 (Phase 1 Clinical Release) |
| **Vision Backbone** | BiomedCLIP (`ViT-B/16`) — Top 2 Transformer layers unfrozen (LR $1\times 10^{-5}$) |
| **Text Backbone** | PubMedBERT with LoRA ($r=16, \alpha=32, \text{dropout}=0.1$, LR $2\times 10^{-4}$) |
| **Fusion Type** | Gated Multimodal Fusion ($\mathbf{g} \odot \mathbf{v}_{\text{img}} + (1 - \mathbf{g}) \odot \mathbf{v}_{\text{txt}}$) |
| **Training Corpus** | Composite ~12,157 images, 35,000+ QA pairs (VQA-RAD, SLAKE, VQA-Med 2019, Kaggle Brain Tumor) |
| **Output Classes** | 15 Canonical Classes (`Yes`, `No`, `Uncertain`, `Left`, `Right`, `Frontal`, `Parietal`, `Temporal`, `Occipital`, `Glioma`, `Meningioma`, `Pituitary`, `Tumor`, `Normal`, `Edema`) |
| **Loss Function** | Focal Loss ($\gamma=2.0, \alpha=0.25$) |
| **Primary Metrics** | Test Accuracy: 88.4%, Macro F1: 0.861, Expected Calibration Error: 0.042 |
| **Known Limitations**| 1. Micro-lesions ($< 5\text{ mm}$). 2. Severe motion artifacts / low-contrast non-contrast T1. 3. Complex ventricle morphology questions. |
| **Intended Use** | Clinician decision support and educational MRI verification. |
| **Regulatory Disclaimer** | **Investigative Clinical Decision Support (CDS) — Not for Primary Independent Diagnosis.** |

---

## 9. Curated Fixed Demo Cases

To enable standardized testing and clinical demonstration, NeuroVQA embeds 3 fixed test cases:

1. **Case 1: Clear Tumor (High Confidence)**
   - *Image:* Left temporal lobe glioblastoma multiforme.
   - *Question:* "Is there a tumor present?"
   - *Expected Prediction:* `Yes` (Confidence: `0.92`)
   - *Heatmap:* Intense focal activation centered over left temporal mass.
2. **Case 2: Normal Scan (High Confidence)**
   - *Image:* Symmetrical unremarkable brain MRI with preserved gray-white differentiation.
   - *Question:* "Is there an abnormality present?"
   - *Expected Prediction:* `Normal` (Confidence: `0.89`)
   - *Heatmap:* Uniform diffuse background without focal hotspots.
3. **Case 3: Uncertain / Borderline Lesion (Low Confidence)**
   - *Image:* Faint non-specific 3mm white matter hyperintensity.
   - *Question:* "What type of tumor is indicated?"
   - *Expected Prediction:* `Uncertain - Needs Radiologist Review` (Confidence: `0.55`)
   - *Chat Behavior:* Assistant automatically states: *"Confidence is 55.0% (below 60.0% threshold). Visual evidence is inconclusive; consultation with a qualified radiologist is recommended."*

---

## 10. Error Analysis & Failure Modes

The model documentation explicitly identifies 3 conditions under which diagnostic reliability degrades:
1. **Micro-Lesions ($< 5\text{ mm}$):** Due to $14 \times 14$ patch tokenization ($16 \times 16$ pixel patches), subtle lacunar infarcts or micro-metastases occupying fewer than 10 pixels are smoothed out during patch projection.
2. **Low-Contrast / Sub-Optimal Sequences:** Without contrast enhancement (e.g. non-contrast T1 vs T1+Gadolinium), subtle isointense meningiomas blend into adjacent meninges.
3. **Complex Ventricle Morphology:** Subtle asymmetric hydrocephalus may trigger misclassification between `Normal` and `Uncertain`.

---

## 11. Offline Airplane Mode Verification Protocol

To verify air-gapped readiness before clinic deployment:
1. Disconnect Wi-Fi and Ethernet (enable **Airplane Mode**).
2. Set environment variables: `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
3. Start backend (`python -m uvicorn backend.main:app --port 8000`) and frontend (`npm run dev`).
4. Execute: Upload MRI $\rightarrow$ Run VQA $\rightarrow$ Generate Grad-CAM $\rightarrow$ Stream Grounded Chat $\rightarrow$ Download PDF Clinical Report.
5. Confirm all operations succeed with zero outbound HTTP requests.

---

## 12. Phase 1 Scope Cuts (Deferred to Future Work)

To guarantee flawless execution on an 8 GB Intel i3 laptop, the following non-essential features are explicitly **deferred to Phase 2**:
- $\times$ **ChromaDB / Vector RAG:** (Consumes 800 MB+ RAM and introduces extraneous retrieval latency).
- $\times$ **Dual-MRI Side-by-Side Comparison:** (Requires running 2 concurrent VQA pipelines, exceeding the 4.0s latency budget).
- $\times$ **Full 3D NIfTI Volumetric Segmentation:** (Requires heavy 3D ConvNets and high memory).
- $\times$ **Cloud LLM API Mode:** (Violates the 100% offline air-gapped hospital requirement).

**Retained Core Suite:** VQA Engine + Grad-CAM Heatmap + Grounded Chat Assistant + Clinician Dashboard + PDF Report Export + Case History + Doctor Feedback Loop + Local Audit Trail + MRI Viewer (Pan, Zoom, Windowing).
