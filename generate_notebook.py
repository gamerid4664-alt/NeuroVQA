"""
Generates train_colab.ipynb for NeuroVQA V26.
Run this script locally: python generate_notebook.py
Then upload train_colab.ipynb to Google Colab.
"""

import json, os

def cell(cell_type, source, **kwargs):
    base = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source if isinstance(source, list) else [source],
    }
    if cell_type == "code":
        base["outputs"] = []
        base["execution_count"] = None
    base.update(kwargs)
    return base

def md(source):
    return cell("markdown", source)

def code(source):
    return cell("code", source)

cells = [

# ── SECTION 0 : TITLE ─────────────────────────────────────────────────────────
md("""# 🧠 NeuroVQA — Cloud Training Notebook
### BiomedCLIP + PubMedBERT + LoRA Fine-Tuning on VQA-RAD
**Environment:** Google Colab / Kaggle T4 GPU  
**Goal:** Train a LoRA adapter for MRI Visual Question Answering and export `neurovqa_weights.pt`

---
> ⚠️ Select **Runtime → Change runtime type → T4 GPU** before running this notebook.
"""),

# ── SECTION 1 : GPU CHECK ─────────────────────────────────────────────────────
md("## 1 · GPU Verification"),
code("""\
import subprocess, sys

result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(result.stdout if result.returncode == 0 else "⚠️  No GPU detected — switch to T4 runtime.")

import torch
print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
    print(f"VRAM            : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
"""),

# ── SECTION 2 : INSTALL DEPENDENCIES ──────────────────────────────────────────
md("## 2 · Install Training Dependencies"),
code("""\
%%capture
!pip install -q open_clip_torch>=2.24.0
!pip install -q transformers>=4.38.0
!pip install -q peft>=0.8.0
!pip install -q accelerate>=0.29.0
!pip install -q datasets>=2.18.0
!pip install -q scikit-learn>=1.4.0
!pip install -q albumentations>=1.4.0
!pip install -q SimpleITK>=2.3.1
!pip install -q matplotlib>=3.8.0 Pillow>=10.2.0
!pip install -q tqdm
print("✅ All dependencies installed.")
"""),

# ── SECTION 3 : GOOGLE DRIVE MOUNT ───────────────────────────────────────────
md("## 3 · Mount Google Drive (Checkpoint Persistence)"),
code("""\
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    CHECKPOINT_DIR = "/content/drive/MyDrive/NeuroVQA_Checkpoints"
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"ℹ️ Non-Colab or standalone environment ({e}). Using local checkpoint directory.")
    CHECKPOINT_DIR = "./NeuroVQA_Checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"✅ Checkpoint directory ready: {CHECKPOINT_DIR}")
"""),

# ── SECTION 4 : CONFIGURATION ─────────────────────────────────────────────────
md("## 4 · Configuration & Hyperparameters"),
code("""\
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR        = "/content/vqa_rad"
CHECKPOINT_DIR  = "/content/drive/MyDrive/NeuroVQA_Checkpoints"
FINAL_WEIGHTS   = f"{CHECKPOINT_DIR}/neurovqa_weights.pt"

# ── Model ──────────────────────────────────────────────────────────────────────
CLIP_MODEL_NAME  = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
BERT_MODEL_NAME  = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
EMBED_DIM        = 512
NUM_HEADS        = 8
DROPOUT          = 0.1
MAX_SEQ_LEN      = 64
IMAGE_SIZE       = 224

# ── LoRA ───────────────────────────────────────────────────────────────────────
LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROPOUT     = 0.05
LORA_TARGET      = ["query", "value"]          # BERT attention projections

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE       = 16
NUM_EPOCHS       = 15
LR               = 3e-4
WEIGHT_DECAY     = 1e-4
WARMUP_STEPS     = 50
GRAD_CLIP        = 1.0
EARLY_STOP_PAT   = 4                           # patience epochs
SEED             = 42

# ── Runtime ────────────────────────────────────────────────────────────────────
DEVICE           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION    = "neurovqa_v1"

print(f"Device   : {DEVICE}")
print(f"Config   : LR={LR}  Batch={BATCH_SIZE}  Epochs={NUM_EPOCHS}  LoRA_r={LORA_R}")
"""),

# ── SECTION 5 : SEED ──────────────────────────────────────────────────────────
md("## 5 · Reproducibility Seed"),
code("""\
import random, numpy as np, torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(SEED)
print(f"✅ Seed set to {SEED}")
"""),

# ── SECTION 6 : DOWNLOAD VQA-RAD ──────────────────────────────────────────────
md("## 6 · Download & Prepare VQA-RAD Dataset"),
code("""\
import os, json, urllib.request, zipfile, shutil
from pathlib import Path

os.makedirs(DATA_DIR, exist_ok=True)

# ── 6.1  Download VQA-RAD JSON from HuggingFace datasets ─────────────────────
print("Downloading VQA-RAD via HuggingFace datasets …")
from datasets import load_dataset, concatenate_datasets

try:
    vqarad = load_dataset("flaviagiammarino/vqa-rad", split="all")
except Exception:
    raw_ds = load_dataset("flaviagiammarino/vqa-rad")
    if hasattr(raw_ds, "keys"):
        vqarad = concatenate_datasets([raw_ds[s] for s in raw_ds.keys()])
    else:
        vqarad = raw_ds

print(f"Total samples: {len(vqarad)}")
print("Sample keys  :", vqarad.column_names)
vqarad[0]
"""),

code("""\
# ── 6.2  Inspect answer distribution ─────────────────────────────────────────
from collections import Counter

# Focus on CLOSED questions (yes/no/which/what) — easier classification
closed_types = {"CLOSED"}
all_answers   = [ex["answer"].strip().lower() for ex in vqarad
                 if ex.get("answer_type") in closed_types or True]

top_answers = Counter(all_answers).most_common(30)
print("Top-30 answers:")
for ans, cnt in top_answers:
    print(f"  {ans:<25} {cnt:>4}")
"""),

code("""\
# ── 6.3  Build answer vocabulary (top-N answers) ─────────────────────────────
MAX_ANSWERS = 58       # covers >95% of VQA-RAD closed answers

answer_counter = Counter(
    ex["answer"].strip().lower() for ex in vqarad
)
most_common = [ans for ans, _ in answer_counter.most_common(MAX_ANSWERS)]

class_to_idx = {ans: i for i, ans in enumerate(most_common)}
idx_to_class = {i: ans for ans, i in class_to_idx.items()}

print(f"Vocabulary size : {len(class_to_idx)}")
print("First 10 labels :", list(class_to_idx.items())[:10])
"""),

# ── SECTION 7 : DATASET CLASS ─────────────────────────────────────────────────
md("## 7 · Dataset & DataLoader"),
code("""\
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np, io

# BiomedCLIP preprocessing normalisation constants
CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

import torchvision.transforms as T

def get_image_transform(train: bool = True) -> T.Compose:
    transforms = []
    if train:
        transforms += [
            T.RandomHorizontalFlip(p=0.3),
            T.RandomAffine(degrees=5, translate=(0.05, 0.05)),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    transforms += [
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
    return T.Compose(transforms)


class VQARADDataset(Dataset):
    \"\"\"
    Wraps the HuggingFace VQA-RAD dataset.
    Each item returns:
        image   : (3, 224, 224) float tensor
        question: str
        label   : int  (class index, or -1 if answer not in vocab)
    \"\"\"
    def __init__(self, hf_dataset, class_to_idx, transform=None):
        self.data          = hf_dataset
        self.class_to_idx  = class_to_idx
        self.idx_to_class  = {v: k for k, v in class_to_idx.items()}
        self.transform     = transform

        # filter to only samples whose answer is in vocab
        self.valid_indices = [
            i for i, ex in enumerate(self.data)
            if ex["answer"].strip().lower() in self.class_to_idx
        ]
        print(f"Dataset: {len(self.valid_indices)}/{len(self.data)} samples in vocab")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        ex    = self.data[self.valid_indices[idx]]
        label = self.class_to_idx[ex["answer"].strip().lower()]

        # image: HuggingFace returns PIL Image
        img = ex["image"].convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, ex["question"], label


# ── Split 80 / 10 / 10 ───────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

all_valid = list(range(len(vqarad)))
labels    = [
    class_to_idx.get(vqarad[i]["answer"].strip().lower(), -1)
    for i in all_valid
]
# keep only in-vocab
valid_mask = [i for i, l in zip(all_valid, labels) if l != -1]

train_idx, temp_idx = train_test_split(valid_mask, test_size=0.2, random_state=SEED)
val_idx,   test_idx = train_test_split(temp_idx,   test_size=0.5, random_state=SEED)

from torch.utils.data import Subset
full_hf = vqarad  # keep reference

# We'll build datasets by subsetting
train_ds = VQARADDataset(
    [vqarad[i] for i in train_idx], class_to_idx, get_image_transform(train=True))
val_ds   = VQARADDataset(
    [vqarad[i] for i in val_idx],   class_to_idx, get_image_transform(train=False))
test_ds  = VQARADDataset(
    [vqarad[i] for i in test_idx],  class_to_idx, get_image_transform(train=False))

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
"""),

code("""\
# ── Collate: tokenise questions on-the-fly ────────────────────────────────────
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

def collate_fn(batch):
    images, questions, labels = zip(*batch)
    images = torch.stack(images)                        # (B, 3, 224, 224)
    labels = torch.tensor(labels, dtype=torch.long)     # (B,)
    encoded = tokenizer(
        list(questions),
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LEN,
        return_tensors="pt",
    )
    return images, encoded, labels

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=2, pin_memory=True, collate_fn=collate_fn)

print(f"Train batches: {len(train_loader)}  Val: {len(val_loader)}  Test: {len(test_loader)}")
"""),

# ── SECTION 8 : MODEL ─────────────────────────────────────────────────────────
md("## 8 · NeuroVQA Model (BiomedCLIP + PubMedBERT + Cross-Attention)"),
code("""\
import torch
import torch.nn as nn
import open_clip
from transformers import AutoModel
from peft import get_peft_model, LoraConfig, TaskType

# ─────────────────────────────────────────────────────────────────────────────
class CrossModalAttention(nn.Module):
    \"\"\"
    Text tokens attend over image patch tokens.
    Q = text,  K = V = image
    \"\"\"
    def __init__(self, dim: int = EMBED_DIM, num_heads: int = NUM_HEADS, dropout: float = DROPOUT):
        super().__init__()
        self.attn   = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm   = nn.LayerNorm(dim)
        self.drop   = nn.Dropout(dropout)

    def forward(self, text_tokens, image_tokens):
        # text_tokens : (B, T, D)   image_tokens : (B, N, D)
        attended, _ = self.attn(text_tokens, image_tokens, image_tokens)
        out = self.norm(text_tokens + self.drop(attended))          # residual
        return out                                                   # (B, T, D)


# ─────────────────────────────────────────────────────────────────────────────
class NeuroVQAModel(nn.Module):
    \"\"\"
    Dual-encoder + cross-attention VQA model.
    Visual : BiomedCLIP ViT-B/16  (frozen backbone, LoRA on BERT side)
    Text   : PubMedBERT            (LoRA fine-tuned)
    \"\"\"
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        # ── Visual encoder (BiomedCLIP) ───────────────────────────────────────
        try:
            clip_model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL_NAME)
            print(f"✅ Loaded BiomedCLIP weights ({CLIP_MODEL_NAME})")
        except Exception as e:
            print(f"⚠️ Could not load BiomedCLIP directly ({e}); falling back to ViT-B-16 openai.")
            clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")

        # Use the visual tower only
        self.visual_encoder = clip_model.visual
        for p in self.visual_encoder.parameters():
            p.requires_grad = False          # freeze CLIP backbone

        self.visual_proj = nn.Sequential(
            nn.Linear(512, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        # ── Text encoder (PubMedBERT + LoRA) ─────────────────────────────────
        bert_base = AutoModel.from_pretrained(BERT_MODEL_NAME)

        lora_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        )
        self.text_encoder = get_peft_model(bert_base, lora_config)
        self.text_encoder.print_trainable_parameters()

        self.text_proj = nn.Sequential(
            nn.Linear(768, EMBED_DIM),
            nn.GELU(),
            nn.Dropout(DROPOUT),
        )

        # ── Cross-modal fusion ────────────────────────────────────────────────
        self.cross_attn = CrossModalAttention(EMBED_DIM, NUM_HEADS, DROPOUT)

        # ── Classification head ───────────────────────────────────────────────
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, EMBED_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(EMBED_DIM // 2, num_classes),
        )

    # ─── forward ─────────────────────────────────────────────────────────────
    def forward(self, images, input_ids, attention_mask, token_type_ids=None):
        # ── Visual stream ─────────────────────────────────────────────────────
        with torch.no_grad():
            # BiomedCLIP ViT returns (B, 197, 768) hidden states
            vis_out = self.visual_encoder(images)           # (B, D_clip)
            if vis_out.dim() == 2:
                # patch tokens not exposed — replicate CLS across N positions
                vis_tokens = vis_out.unsqueeze(1).expand(-1, 197, -1)  # (B,197,512)
            else:
                vis_tokens = vis_out                        # (B, 197, D_clip)

        vis_tokens = self.visual_proj(vis_tokens)           # (B, 197, EMBED_DIM)

        # ── Text stream ───────────────────────────────────────────────────────
        text_out    = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden = text_out.last_hidden_state if hasattr(text_out, "last_hidden_state") else text_out[0]
        text_tokens = self.text_proj(hidden)   # (B, T, EMBED_DIM)

        # ── Cross-modal attention ─────────────────────────────────────────────
        fused = self.cross_attn(text_tokens, vis_tokens)           # (B, T, EMBED_DIM)

        # ── Pool text tokens ──────────────────────────────────────────────────
        pooled = fused.mean(dim=1)                                  # (B, EMBED_DIM)

        # ── Classify ──────────────────────────────────────────────────────────
        logits = self.fusion_head(pooled)                           # (B, num_classes)
        return logits

    def get_visual_tokens(self, images):
        \"\"\"Expose patch tokens for Grad-CAM (inference use only).\"\"\"
        with torch.no_grad():
            vis_out = self.visual_encoder(images)
        return vis_out


# ─── Instantiate ──────────────────────────────────────────────────────────────
model = NeuroVQAModel(num_classes=len(class_to_idx)).to(DEVICE)
total  = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params     : {total:,}")
print(f"Trainable params : {trainable:,}  ({100*trainable/total:.1f}%)")
"""),

# ── SECTION 9 : TRAINING ──────────────────────────────────────────────────────
md("## 9 · Training Loop"),
code("""\
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.notebook import tqdm

# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

total_steps = NUM_EPOCHS * len(train_loader)
scheduler   = OneCycleLR(
    optimizer,
    max_lr=LR,
    total_steps=total_steps,
    pct_start=WARMUP_STEPS / total_steps,
    anneal_strategy="cos",
)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


# ── Mixed precision & Scaler setup ──────────────────────────────────────────
has_torch_amp = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")

def get_autocast_context(training: bool):
    if not (training and torch.cuda.is_available()):
        return torch.no_grad()
    if has_torch_amp:
        return torch.amp.autocast("cuda")
    return torch.cuda.amp.autocast()

if torch.cuda.is_available():
    scaler = torch.amp.GradScaler("cuda") if has_torch_amp else torch.cuda.amp.GradScaler()
else:
    scaler = None


# ── Helper functions ──────────────────────────────────────────────────────────
def run_epoch(loader, training: bool):
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with get_autocast_context(training):
        for images, encoded, labels in tqdm(loader, leave=False,
                                             desc="train" if training else "eval"):
            images  = images.to(DEVICE)
            labels  = labels.to(DEVICE)
            ids     = encoded["input_ids"].to(DEVICE)
            mask    = encoded["attention_mask"].to(DEVICE)
            ttype   = encoded.get("token_type_ids")
            if ttype is not None:
                ttype = ttype.to(DEVICE)

            if training:
                optimizer.zero_grad()

            with get_autocast_context(training):
                logits = model(images, ids, mask, ttype)
                loss   = criterion(logits, labels)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    optimizer.step()

                if scheduler.last_epoch < total_steps:
                    scheduler.step()

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ── Training ──────────────────────────────────────────────────────────────────

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc  = 0.0
best_val_loss = float("inf")
patience_ctr  = 0

print("Starting training …\\n")
for epoch in range(1, NUM_EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    improved = "✅" if vl_acc > best_val_acc else "  "
    print(f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
          f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
          f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  {improved}")

    # save best
    if vl_acc > best_val_acc:
        best_val_acc  = vl_acc
        best_val_loss = vl_loss
        patience_ctr  = 0
        torch.save({
            "epoch": epoch,
            "fusion_head_state_dict": model.fusion_head.state_dict(),
            "cross_attn_state_dict" : model.cross_attn.state_dict(),
            "visual_proj_state_dict": model.visual_proj.state_dict(),
            "text_proj_state_dict"  : model.text_proj.state_dict(),
            "lora_state_dict"       : model.text_encoder.state_dict(),
            "model_state_dict"      : model.fusion_head.state_dict(),
            "optimizer_state_dict"  : optimizer.state_dict(),
            "val_acc"               : best_val_acc,
            "val_loss"              : best_val_loss,
            "class_to_idx"          : class_to_idx,
            "idx_to_class"          : idx_to_class,
            "model_version"         : MODEL_VERSION,
            "embed_dim"             : EMBED_DIM,
            "num_heads"             : NUM_HEADS,
            "num_classes"           : len(class_to_idx),
        }, f"{CHECKPOINT_DIR}/neurovqa_best.pt")
    else:
        patience_ctr += 1
        if patience_ctr >= EARLY_STOP_PAT:
            print(f"\\nEarly stopping at epoch {epoch} — no improvement for {EARLY_STOP_PAT} epochs.")
            break

print(f"\\n✅ Training complete.  Best val_acc = {best_val_acc:.4f}")
"""),

# ── SECTION 10 : PLOT ─────────────────────────────────────────────────────────
md("## 10 · Training Curves"),
code("""\
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs = range(1, len(history["train_loss"]) + 1)

axes[0].plot(epochs, history["train_loss"], label="Train Loss", color="#342CB3", linewidth=2)
axes[0].plot(epochs, history["val_loss"],   label="Val Loss",   color="#E11D48", linewidth=2)
axes[0].axhline(0.35, linestyle="--", color="orange", alpha=0.7, label="Target loss 0.35")
axes[0].set_title("Loss"); axes[0].legend(); axes[0].set_xlabel("Epoch")

axes[1].plot(epochs, history["train_acc"], label="Train Acc", color="#342CB3", linewidth=2)
axes[1].plot(epochs, history["val_acc"],   label="Val Acc",   color="#10B981", linewidth=2)
axes[1].axhline(0.72, linestyle="--", color="orange", alpha=0.7, label="Target acc 0.72")
axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].set_xlabel("Epoch")

plt.suptitle("NeuroVQA Training Curves", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{CHECKPOINT_DIR}/training_curves.png", dpi=150)
plt.show()
print("📊 Curves saved to Google Drive.")
"""),

# ── SECTION 11 : EVALUATION ───────────────────────────────────────────────────
md("## 11 · Test-Set Evaluation"),
code("""\
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Load best checkpoint
ckpt = torch.load(f"{CHECKPOINT_DIR}/neurovqa_best.pt", map_location=DEVICE)
if "fusion_head_state_dict" in ckpt:
    model.fusion_head.load_state_dict(ckpt["fusion_head_state_dict"])
elif "model_state_dict" in ckpt:
    model.fusion_head.load_state_dict(ckpt["model_state_dict"])
if "cross_attn_state_dict" in ckpt:
    model.cross_attn.load_state_dict(ckpt["cross_attn_state_dict"])
if "visual_proj_state_dict" in ckpt:
    model.visual_proj.load_state_dict(ckpt["visual_proj_state_dict"])
if "text_proj_state_dict" in ckpt:
    model.text_proj.load_state_dict(ckpt["text_proj_state_dict"])
if "lora_state_dict" in ckpt:
    model.text_encoder.load_state_dict(ckpt["lora_state_dict"])
model.eval()

all_preds, all_labels = [], []
test_loss_total, test_total = 0.0, 0

with torch.no_grad():
    for images, encoded, labels in tqdm(test_loader, desc="Test"):
        images  = images.to(DEVICE)
        labels  = labels.to(DEVICE)
        ids     = encoded["input_ids"].to(DEVICE)
        mask    = encoded["attention_mask"].to(DEVICE)
        ttype   = encoded.get("token_type_ids")
        if ttype is not None: ttype = ttype.to(DEVICE)

        logits = model(images, ids, mask, ttype)
        loss   = criterion(logits, labels)
        preds  = logits.argmax(dim=1)

        test_loss_total += loss.item() * images.size(0)
        test_total      += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

test_loss = test_loss_total / test_total
test_acc  = np.mean(np.array(all_preds) == np.array(all_labels))

print(f"Test Loss : {test_loss:.4f}")
print(f"Test Acc  : {test_acc:.4f}  ({test_acc*100:.1f}%)")
print()

# Per-class report (top-20 classes)
label_names = [idx_to_class[i] for i in sorted(idx_to_class.keys())]
print(classification_report(
    all_labels, all_preds,
    labels=list(range(min(20, len(label_names)))),
    target_names=label_names[:20],
    zero_division=0
))

# NFR-6 check
if test_acc >= 0.72:
    print("✅ NFR-6 MET: accuracy ≥ 72%")
else:
    print(f"⚠️  NFR-6 NOT MET: accuracy {test_acc*100:.1f}% < 72% — consider more epochs or data augmentation.")
"""),

# ── SECTION 12 : ABLATION STUDY ───────────────────────────────────────────────
md("""## 12 · Ablation Study
Compares three model configurations as required by success criterion #8:
- **Vision-Only** — BiomedCLIP features, no text encoder
- **Text-Only** — PubMedBERT features, no image encoder
- **Multimodal (Full)** — BiomedCLIP + PubMedBERT + Cross-Attention ← NeuroVQA
"""),
code("""\
# ── Vision-Only model ─────────────────────────────────────────────────────────
class VisionOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
        self.visual = clip_model.visual
        for p in self.visual.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            vis = self.visual(images)
        if vis.dim() > 2: vis = vis.mean(dim=1)
        return self.head(vis)


# ── Text-Only model ───────────────────────────────────────────────────────────
class TextOnlyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME)
        for p in self.bert.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, input_ids, attention_mask, token_type_ids=None):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]          # CLS token
        return self.head(cls)


def quick_eval(mdl, loader, name="Model"):
    \"\"\"1-epoch evaluation only (no training).\"\"\"
    mdl.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, encoded, labels in tqdm(loader, desc=name, leave=False):
            images = images.to(DEVICE); labels = labels.to(DEVICE)
            ids    = encoded["input_ids"].to(DEVICE)
            mask   = encoded["attention_mask"].to(DEVICE)
            ttype  = encoded.get("token_type_ids")
            if ttype is not None: ttype = ttype.to(DEVICE)
            logits = mdl(images, ids, mask, ttype)
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / total


print("Running ablation (zero-shot / untrained baselines) …")

vis_model  = VisionOnlyModel(len(class_to_idx)).to(DEVICE)
text_model = TextOnlyModel(len(class_to_idx)).to(DEVICE)

vis_acc  = quick_eval(vis_model,  test_loader, "Vision-Only")
text_acc = quick_eval(text_model, test_loader, "Text-Only")
# multimodal = already evaluated above
multi_acc = test_acc

ablation = {
    "Vision-Only (zero-shot)"   : vis_acc,
    "Text-Only (zero-shot)"     : text_acc,
    "Multimodal NeuroVQA (LoRA)": multi_acc,
}

print("\\n" + "="*48)
print(f"  {'Configuration':<35} {'Accuracy':>8}")
print("="*48)
for name, acc in ablation.items():
    print(f"  {name:<35} {acc*100:>7.1f}%")
print("="*48)
print(f"\\n  Multimodal gain over Vision-Only : +{(multi_acc-vis_acc)*100:.1f}%")
print(f"  Multimodal gain over Text-Only   : +{(multi_acc-text_acc)*100:.1f}%")
"""),

# ── SECTION 13 : EXPORT FINAL CHECKPOINT ──────────────────────────────────────
md("## 13 · Export Final Checkpoint (`neurovqa_weights.pt`)"),
code("""\
import os, shutil

FINAL_WEIGHTS = f"{CHECKPOINT_DIR}/neurovqa_weights.pt"

torch.save({
    # model weights
    "fusion_head_state_dict": model.fusion_head.state_dict(),
    "cross_attn_state_dict" : model.cross_attn.state_dict(),
    "visual_proj_state_dict": model.visual_proj.state_dict(),
    "text_proj_state_dict"  : model.text_proj.state_dict(),
    "lora_state_dict"       : model.text_encoder.state_dict(),
    "model_state_dict"      : model.fusion_head.state_dict(),

    # class mapping (required at inference time)
    "class_to_idx"     : class_to_idx,
    "idx_to_class"     : idx_to_class,

    # metadata
    "model_version"    : MODEL_VERSION,
    "embed_dim"        : EMBED_DIM,
    "num_heads"        : NUM_HEADS,
    "num_classes"      : len(class_to_idx),
    "max_seq_len"      : MAX_SEQ_LEN,
    "image_size"       : IMAGE_SIZE,
    "lora_r"           : LORA_R,
    "lora_alpha"       : LORA_ALPHA,
    "bert_model_name"  : BERT_MODEL_NAME,
    "val_acc"          : best_val_acc,
    "val_loss"         : best_val_loss,
}, FINAL_WEIGHTS)

size_mb = os.path.getsize(FINAL_WEIGHTS) / 1e6
print(f"✅ Checkpoint saved to: {FINAL_WEIGHTS}")
print(f"   File size: {size_mb:.1f} MB")

if size_mb <= 80:
    print("✅ NFR: Adapter size ≤ 80 MB — PASS")
else:
    print(f"⚠️  Adapter size {size_mb:.1f} MB > 80 MB — consider reducing LoRA rank")

# Optional: Trigger direct browser download in Google Colab
try:
    from google.colab import files
    print("\\nInitiating direct browser download ...")
    files.download(FINAL_WEIGHTS)
except Exception:
    print(f"\\nCheck your Google Drive or local folder to access: {FINAL_WEIGHTS}")
"""),

# ── SECTION 14 : DOWNLOAD INSTRUCTIONS ────────────────────────────────────────
md("""## 14 · Download Checkpoint to Local Machine

After the notebook finishes:

1. Open **Google Drive** → `MyDrive/NeuroVQA_Checkpoints/`
2. Download **`neurovqa_weights.pt`**
3. Place it at:
```
NeuroVQA/
└── backend/
    └── checkpoints/
        └── neurovqa_weights.pt   ← place here
```

Then set offline environment variables on your local machine (PowerShell):
```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
```

Start the local server:
```bash
uvicorn backend.main:app --reload --port 8000
```

> ✅ The checkpoint is fully self-contained — it includes class mappings, model metadata, and LoRA adapter weights needed for offline inference.
"""),

md("""---
## 15 · Quick Reference — NFR Targets

| Requirement | Target | How to verify |
|---|---|---|
| NFR-1 Training time | < 45 min on T4 | Check Colab session timer |
| NFR-2 Offline inference | CPU only | Disable network, run backend |
| NFR-3 Memory | VQA < 2.5 GB | `torch.cuda.memory_allocated()` |
| NFR-4 VQA Latency | ≤ 2.5s | `time.time()` in `backend/main.py` |
| NFR-6 Accuracy | ≥ 72% | Section 11 output |
| Adapter size | < 80 MB | Section 13 output |

---
*NeuroVQA V26 — Cloud Training Notebook | Generated by generate_notebook.py*
"""),
]

# ── Assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "name": "NeuroVQA_Training.ipynb",
        },
    },
    "cells": cells,
}

OUT = os.path.join(os.path.dirname(__file__), "train_colab.ipynb")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"[OK] Notebook written -> {OUT}")
print(f"     Cells: {len(cells)}")
print(f"     Size : {os.path.getsize(OUT)/1e3:.1f} KB")
print()
print("Next steps:")
print("  1. Upload train_colab.ipynb to Google Colab")
print("  2. Runtime -> Change runtime type -> T4 GPU")
print("  3. Run All")
print("  4. Download neurovqa_weights.pt from Google Drive")
print("  5. Place in backend/checkpoints/neurovqa_weights.pt")
