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

# 1. Ischemic Stroke
mri1 = create_base_mri()
infarct = ((x - 158)**2) / 18**2 + ((y - 115)**2) / 24**2 <= 1.0
mri1[infarct] = np.clip(mri1[infarct] + 65.0, 0, 255)
Image.fromarray(np.clip(mri1, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case1_ischemic_stroke.png"))
Image.fromarray(np.clip(mri1, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case1.png"))

# 2. Hydrocephalus
mri2 = create_base_mri()
v_dil_l = ((x - 110)**2) / 24**2 + ((y - 120)**2) / 45**2 <= 1.0
v_dil_r = ((x - 146)**2) / 24**2 + ((y - 120)**2) / 45**2 <= 1.0
mri2[v_dil_l] = 20.0
mri2[v_dil_r] = 20.0
Image.fromarray(np.clip(mri2, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case2_hydrocephalus.png"))
Image.fromarray(np.clip(mri2, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case2.png"))

# 3. Normal Brain
mri3 = create_base_mri()
Image.fromarray(np.clip(mri3, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case3_normal_brain.png"))
Image.fromarray(np.clip(mri3, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case3.png"))

# 4. Subdural Hematoma
mri4 = create_base_mri()
crescent = (((x - 128)**2) / 84**2 + ((y - 128)**2) / 104**2 <= 1.0) & (((x - 128)**2) / 72**2 + ((y - 128)**2) / 92**2 > 1.0) & (x < 120)
mri4[crescent] = 220.0
Image.fromarray(np.clip(mri4, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case4_subdural_hematoma.png"))

# 5. Glioblastoma
mri5 = create_base_mri()
gb_outer = ((x - 95)**2) / 25**2 + ((y - 150)**2) / 22**2 <= 1.0
gb_inner = ((x - 95)**2) / 14**2 + ((y - 150)**2) / 12**2 <= 1.0
mri5[gb_outer] = 230.0
mri5[gb_inner] = 60.0
Image.fromarray(np.clip(mri5, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8)).save(os.path.join(demo_dir, "case5_glioblastoma.png"))

print(f"Generated 5 demo MRI scans in: {demo_dir}")
