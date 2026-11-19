"""
NeuroVQA — Demo MRI Case Generator (Version 2.0)
Generates the 3 canonical demo cases:
- Case 1: Clear Tumor (Glioblastoma, high confidence ~0.92)
- Case 2: Normal Scan (Unremarkable brain MRI, confidence ~0.89)
- Case 3: Uncertain / Small lesion (Faint 3mm non-specific hyperintensity, confidence ~0.55)
"""

import os
import numpy as np
from PIL import Image, ImageFilter

demo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "demo_samples")
os.makedirs(demo_dir, exist_ok=True)

y, x = np.ogrid[:256, :256]

def create_base_mri():
    img = np.zeros((256, 256), dtype=np.float32)
    # Skull contour
    mask = ((x - 128)**2) / 85**2 + ((y - 128)**2) / 105**2 <= 1.0
    img[mask] = 45.0
    # Brain parenchyma
    brain_mask = ((x - 128)**2) / 80**2 + ((y - 128)**2) / 100**2 <= 1.0
    img[brain_mask] = 135.0
    # Sulcal noise
    noise = np.random.normal(0, 10, (256, 256))
    img[brain_mask] += noise[brain_mask]
    # Bilateral ventricles
    v_left = ((x - 116)**2) / 10**2 + ((y - 120)**2) / 28**2 <= 1.0
    v_right = ((x - 140)**2) / 10**2 + ((y - 120)**2) / 28**2 <= 1.0
    img[v_left] = 20.0
    img[v_right] = 20.0
    return img

# Case 1: Clear Tumor (Glioblastoma in left temporal/parietal region)
mri1 = create_base_mri()
gb_outer = ((x - 95)**2) / 26**2 + ((y - 148)**2) / 24**2 <= 1.0
gb_inner = ((x - 95)**2) / 14**2 + ((y - 148)**2) / 12**2 <= 1.0
mri1[gb_outer] = 230.0
mri1[gb_inner] = 65.0
c1_img = Image.fromarray(np.clip(mri1, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
c1_img.save(os.path.join(demo_dir, "case1.png"))
c1_img.save(os.path.join(demo_dir, "case1_clear_tumor.png"))

# Case 2: Normal Scan (Symmetrical unremarkable parenchymal contours)
mri2 = create_base_mri()
c2_img = Image.fromarray(np.clip(mri2, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
c2_img.save(os.path.join(demo_dir, "case2.png"))
c2_img.save(os.path.join(demo_dir, "case2_normal_scan.png"))

# Case 3: Uncertain / Small Lesion (Faint 3mm focal white-matter signal)
mri3 = create_base_mri()
faint_lesion = ((x - 150)**2) / 4**2 + ((y - 105)**2) / 4**2 <= 1.0
mri3[faint_lesion] = np.clip(mri3[faint_lesion] + 35.0, 0, 255)
c3_img = Image.fromarray(np.clip(mri3, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
c3_img.save(os.path.join(demo_dir, "case3.png"))
c3_img.save(os.path.join(demo_dir, "case3_uncertain_small_lesion.png"))

print(f"Generated 3 canonical demo MRI cases in: {demo_dir}")
