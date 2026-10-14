"""
NeuroVQA — Module A: MRI Ingestion and Preprocessing
Implements the exact pipeline specified in NeuroVQA-design.md:
Percentile clipping (1st-99th) -> Min-Max norm -> 224x224 resize -> Torchvision normalization.
"""

from io import BytesIO
from typing import Tuple, Union
import numpy as np
from PIL import Image

# Torch & Torchvision imports with safe fallback for lightweight CPU environments
try:
    import torch
    from torchvision import transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Torchvision standard normalization parameters for BiomedCLIP ViT-B/16
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def preprocess_mri_image(image_input: Union[bytes, Image.Image, np.ndarray]) -> Tuple[np.ndarray, any]:
    """
    Preprocess MRI image array or raw bytes into:
    1. Display-ready normalized RGB numpy array (224, 224, 3) in [0, 255] uint8.
    2. Model-ready PyTorch tensor (1, 3, 224, 224) float32 normalized.
    """
    # 1. Load image
    if isinstance(image_input, bytes):
        pil_img = Image.open(BytesIO(image_input)).convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input.astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[0] == 3:
            img_np = np.transpose(img_np, (1, 2, 0))
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")

    # 2. MRI Intensity Percentile Clipping (1st to 99th percentile)
    # WARNING: Do not apply CT HU windowing to MRI
    p1, p99 = np.percentile(img_np, (1.0, 99.0))
    if p99 > p1:
        img_np = np.clip(img_np, p1, p99)

    # 3. Min-Max Normalization to [0.0, 1.0]
    min_val, max_val = img_np.min(), img_np.max()
    if max_val > min_val:
        img_np = (img_np - min_val) / (max_val - min_val)
    else:
        img_np = np.zeros_like(img_np)

    # 4. Bilinear Resize to 224 x 224
    pil_resized = Image.fromarray((img_np * 255).astype(np.uint8)).resize((224, 224), Image.Resampling.BILINEAR)
    display_np = np.array(pil_resized, dtype=np.uint8)

    # 5. Tensor conversion & Torchvision normalization
    if HAS_TORCH:
        transform_pipeline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
        ])
        tensor = transform_pipeline(pil_resized).unsqueeze(0)  # Shape: (1, 3, 224, 224)
    else:
        norm_arr = (np.array(pil_resized, dtype=np.float32) / 255.0 - NORM_MEAN) / NORM_STD
        tensor = np.transpose(norm_arr, (2, 0, 1))[np.newaxis, ...]

    return display_np, tensor


def pack_25d_stack(slice_k_minus_1: np.ndarray, slice_k: np.ndarray, slice_k_plus_1: np.ndarray) -> np.ndarray:
    """
    Stacks 3 contiguous axial slices as 3 channels (H, W, 3) for 2.5D volumetric context.
    """
    stack = np.stack([slice_k_minus_1, slice_k, slice_k_plus_1], axis=-1)
    return stack
