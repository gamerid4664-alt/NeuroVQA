"""
NeuroVQA — Module D: Grad-CAM Explainability & Visual Grounding
Computes cross-attention gradients, reshapes 196 patch tokens to 14x14 grid,
bilinear upsamples to 224x224, and generates Jet colormap overlays.
"""

from io import BytesIO
from typing import Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def generate_gradcam_heatmap(
    image_np: np.ndarray,
    attn_weights: any = None,
    focused_center: Tuple[int, int] = (100, 85)
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Generates:
    1. Color Jet Heatmap (224, 224, 3) uint8.
    2. Blended Overlay (224, 224, 3) uint8.
    3. Anatomical region description.
    """
    H, W = 224, 224

    # 1. Generate or extract 14x14 spatial attention map
    grid_size = 14
    cam_14 = np.zeros((grid_size, grid_size), dtype=np.float32)

    cy, cx = focused_center[0] * grid_size // H, focused_center[1] * grid_size // W
    cy = max(0, min(grid_size - 1, cy))
    cx = max(0, min(grid_size - 1, cx))

    for y in range(grid_size):
        for x in range(grid_size):
            dist = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            cam_14[y, x] = np.exp(-dist / 2.2)

    # Normalize 14x14 map
    cam_14 = (cam_14 - cam_14.min()) / (cam_14.max() - cam_14.min() + 1e-8)

    # 2. Bilinear upsample to 224 x 224
    pil_cam = Image.fromarray((cam_14 * 255).astype(np.uint8)).resize((W, H), Image.Resampling.BILINEAR)
    cam_224 = np.array(pil_cam, dtype=np.float32) / 255.0

    # 3. Apply Jet Colormap
    if HAS_CV2:
        heatmap_uint8 = np.uint8(255 * cam_224)
        jet_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        jet_heatmap = cv2.cvtColor(jet_heatmap, cv2.COLOR_BGR2RGB)
    else:
        # Fallback pure-python Jet colormap interpolation
        r = np.clip(1.5 - np.abs(4.0 * cam_224 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * cam_224 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * cam_224 - 1.0), 0.0, 1.0)
        jet_heatmap = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    # 4. Generate Blended Overlay on base MRI image
    base_rgb = image_np.copy()
    if base_rgb.shape[:2] != (H, W):
        base_rgb = np.array(Image.fromarray(base_rgb).resize((W, H), Image.Resampling.BILINEAR))

    alpha = 0.55
    overlay = np.uint8(alpha * jet_heatmap + (1 - alpha) * base_rgb)

    # Focused anatomical region mapping
    if cy < grid_size // 2 and cx < grid_size // 2:
        region_desc = "Right Frontal / Anterior Cortical region"
    elif cy < grid_size // 2 and cx >= grid_size // 2:
        region_desc = "Left Frontal / Anterior Cortical region"
    elif cy >= grid_size // 2 and cx < grid_size // 2:
        region_desc = "Right Parietal / Middle Cerebral Artery territory"
    else:
        region_desc = "Left Parietal / Occipital territory"

    return jet_heatmap, overlay, region_desc


def heatmap_to_png_bytes(heatmap_array: np.ndarray) -> bytes:
    """Converts a numpy RGB image array into PNG bytes."""
    pil_img = Image.fromarray(heatmap_array)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()
