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
    import torch  # type: ignore[import-not-found, import-untyped]
    from torchvision import transforms  # type: ignore[import-not-found, import-untyped]
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    torch = None  # type: ignore
    transforms = None  # type: ignore
    HAS_TORCH = False

# Torchvision standard normalization parameters for BiomedCLIP ViT-B/16
NORM_MEAN = [0.48145466, 0.4578275, 0.40821073]
NORM_STD = [0.26862954, 0.26130258, 0.27577711]


def is_nifti_data(data: bytes) -> bool:
    """Checks if raw bytes represent a NIfTI-1/2 file (uncompressed or gzipped)."""
    if not isinstance(data, (bytes, bytearray)) or len(data) < 352:
        return False
    # Check gzip header: magic 0x1f, 0x8b
    if data[:2] == b"\x1f\x8b":
        return True
    # Check uncompressed NIfTI header sizeof_hdr (348)
    import struct
    try:
        sizeof_hdr_le = struct.unpack("<I", data[:4])[0]
        sizeof_hdr_be = struct.unpack(">I", data[:4])[0]
        if sizeof_hdr_le == 348 or sizeof_hdr_be == 348:
            return True
    except Exception:
        pass
    return False


def load_nifti_volume(nifti_input: Union[bytes, str]) -> np.ndarray:
    """
    Loads a 3D volumetric array from NIfTI bytes or file path.
    Returns float32 3D array oriented as (H, W, Z).
    """
    try:
        import nibabel as nib
        if isinstance(nifti_input, (bytes, bytearray)):
            import io
            import gzip
            if nifti_input[:2] == b"\x1f\x8b":
                decompressed = gzip.decompress(nifti_input)
                fh = nib.FileHolder(fileobj=io.BytesIO(decompressed))
            else:
                fh = nib.FileHolder(fileobj=io.BytesIO(nifti_input))
            nii = nib.Nifti1Image.from_file_map({"header": fh, "image": fh})
        else:
            nii = nib.load(nifti_input)

        vol = np.asarray(nii.get_fdata(dtype=np.float32))
        if vol.ndim > 3:
            vol = vol[..., 0]
        return vol
    except Exception as e:
        raise ValueError(f"Failed to parse NIfTI volume: {e}")


def extract_nifti_axial_slice(
    nifti_input: Union[bytes, str], slice_index: int = None
) -> Tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Extracts the center axial slice k (and optional 2.5D stack k-1, k, k+1) from a 3D NIfTI volume.
    Returns:
    - 2D middle axial slice array (H, W) float32
    - 2.5D stack array (H, W, 3) float32 if volume has >=3 slices, else None
    """
    vol = load_nifti_volume(nifti_input)
    if vol.ndim < 3:
        if vol.ndim == 2:
            return vol, None
        raise ValueError(f"Invalid NIfTI volume dimensions: {vol.ndim}")

    H, W, Z = vol.shape
    k = slice_index if (slice_index is not None and 0 <= slice_index < Z) else Z // 2

    center_slice = vol[:, :, k]

    stack_25d = None
    if Z >= 3:
        k_clamped = max(1, min(Z - 2, k))
        stack_25d = np.stack(
            [vol[:, :, k_clamped - 1], vol[:, :, k_clamped], vol[:, :, k_clamped + 1]], axis=-1
        )

    return center_slice, stack_25d


def preprocess_mri_image(image_input: Union[bytes, Image.Image, np.ndarray, str]) -> Tuple[np.ndarray, any]:
    """
    Preprocess MRI image array, NIfTI volume, or raw bytes into:
    1. Display-ready normalized RGB numpy array (224, 224, 3) in [0, 255] uint8.
    2. Model-ready PyTorch tensor (1, 3, 224, 224) float32 normalized.
    """
    # 1. Load image (supports PNG, JPEG, TIFF, and NIfTI .nii / .nii.gz)
    if isinstance(image_input, (bytes, bytearray)):
        if is_nifti_data(image_input):
            center_slice, stack_25d = extract_nifti_axial_slice(image_input)
            img_np = stack_25d if stack_25d is not None else np.stack([center_slice] * 3, axis=-1)
        else:
            pil_img = Image.open(BytesIO(image_input)).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32)
    elif isinstance(image_input, str):
        if image_input.endswith((".nii", ".nii.gz")):
            center_slice, stack_25d = extract_nifti_axial_slice(image_input)
            img_np = stack_25d if stack_25d is not None else np.stack([center_slice] * 3, axis=-1)
        else:
            pil_img = Image.open(image_input).convert("RGB")
            img_np = np.array(pil_img, dtype=np.float32)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
        img_np = np.array(pil_img, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        img_np = image_input.astype(np.float32)
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.ndim == 3 and img_np.shape[0] == 3 and img_np.shape[2] != 3:
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


def pack_25d_stack(slice_k_minus_1: Union[np.ndarray, bytes], slice_k: Union[np.ndarray, bytes], slice_k_plus_1: Union[np.ndarray, bytes]) -> np.ndarray:
    """
    Stacks 3 contiguous axial slices as 3 channels (H, W, 3) for 2.5D volumetric context (FR-1).
    Accepts numpy arrays or raw image bytes.
    """
    slices = []
    for s in [slice_k_minus_1, slice_k, slice_k_plus_1]:
        if isinstance(s, bytes):
            img = Image.open(BytesIO(s)).convert("L")
            slices.append(np.array(img, dtype=np.float32))
        elif isinstance(s, np.ndarray):
            arr = s.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            slices.append(arr)
        else:
            raise ValueError(f"Unsupported slice type in 2.5D stack: {type(s)}")

    # Align height and width to middle slice (slice_k)
    target_h, target_w = slices[1].shape[:2]
    aligned = []
    for s in slices:
        if s.shape[:2] != (target_h, target_w):
            pil_s = Image.fromarray(s.astype(np.uint8)).resize((target_w, target_h), Image.Resampling.BILINEAR)
            aligned.append(np.array(pil_s, dtype=np.float32))
        else:
            aligned.append(s)

    stack = np.stack(aligned, axis=-1)  # Shape: (H, W, 3)
    return stack
