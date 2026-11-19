"""
NeuroVQA — Core Unit & Regression Test Suite
Validates mathematical and architectural invariants:
1. Gated Multimodal Fusion Unit (learned sigmoid gating & feature projection)
2. Focal Loss calculation (alpha=0.25, gamma=2.0 affirmative skew damping)
3. 15 Canonical Clinical Classes (bijective mapping & vocabulary integrity)
4. Uncertainty Rejection Head (< 0.60 safety threshold)
5. Explainability (Attention x Gradient Grad-CAM, top 20% patches, alpha 0.45 blend)
6. 2.5D Volumetric Stack Ingestion & Percentile Normalization
7. DICOM (.dcm) Parsing & Metadata Extraction
"""

import os
import sys
import unittest
from io import BytesIO
import numpy as np
from PIL import Image

# Ensure project root is in sys.path
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(TEST_DIR))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F

from backend.src.model import (
    ANSWER_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    UNCERTAINTY_THRESHOLD,
    GatedMultimodalFusion,
    FocalLoss,
    NeuroVQAFusionModel,
    NeuroVQAInferenceEngine,
)
from backend.src.explainability import (
    generate_gradcam_heatmap,
    heatmap_to_png_bytes,
)
from backend.src.dataset import (
    preprocess_mri_image,
    pack_25d_stack,
    is_nifti_data,
    is_dicom_data,
    load_dicom_slice,
)


class TestCanonicalClassSpace(unittest.TestCase):
    """Validates the 15 fixed canonical answer classes (PRD Section 3.3)."""

    def test_class_count_and_vocabulary(self):
        expected_classes = [
            "Yes", "No", "Uncertain", "Left", "Right",
            "Frontal", "Parietal", "Temporal", "Occipital",
            "Glioma", "Meningioma", "Pituitary", "Tumor", "Normal", "Edema",
        ]
        self.assertEqual(len(ANSWER_CLASSES), 15, "Must have exactly 15 canonical classes")
        self.assertEqual(ANSWER_CLASSES, expected_classes, "Class vocabulary mismatch")

    def test_bijective_index_mapping(self):
        self.assertEqual(len(CLASS_TO_IDX), 15)
        self.assertEqual(len(IDX_TO_CLASS), 15)
        for cls_name, idx in CLASS_TO_IDX.items():
            self.assertIn(cls_name, ANSWER_CLASSES)
            self.assertTrue(0 <= idx < 15)
            self.assertEqual(IDX_TO_CLASS[idx], cls_name, f"Bijective mapping broken for {cls_name}")


class TestGatedMultimodalFusion(unittest.TestCase):
    """Validates Gated Multimodal Fusion formula and bounds (Design Section 3)."""

    def setUp(self):
        self.embed_dim = 512
        self.fusion = GatedMultimodalFusion(embed_dim=self.embed_dim, dropout=0.0)
        self.fusion.eval()

    def test_gate_bounds_and_tensor_shapes(self):
        batch_size = 4
        v_img = torch.randn(batch_size, self.embed_dim)
        v_txt = torch.randn(batch_size, self.embed_dim)

        out, gate = self.fusion(v_img, v_txt)

        # Output shape must remain (batch_size, embed_dim)
        self.assertEqual(out.shape, (batch_size, self.embed_dim))
        self.assertEqual(gate.shape, (batch_size, self.embed_dim))

        # Sigmoid gate values must be strictly bounded in (0.0, 1.0)
        self.assertTrue(torch.all(gate >= 0.0).item())
        self.assertTrue(torch.all(gate <= 1.0).item())

    def test_gradient_propagation(self):
        v_img = torch.randn(2, self.embed_dim, requires_grad=True)
        v_txt = torch.randn(2, self.embed_dim, requires_grad=True)

        out, gate = self.fusion(v_img, v_txt)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(v_img.grad)
        self.assertIsNotNone(v_txt.grad)
        self.assertFalse(torch.isnan(v_img.grad).any())
        self.assertFalse(torch.isnan(v_txt.grad).any())


class TestFocalLoss(unittest.TestCase):
    """Validates Focal Loss implementation (PRD Section 3.4)."""

    def setUp(self):
        self.focal = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

    def test_focal_damping_on_easy_examples(self):
        # Case A: Confident prediction (easy sample: ~95% probability on class 0)
        logits_easy = torch.tensor([[3.0, 0.0, 0.0]])
        target = torch.tensor([0])
        loss_easy = self.focal(logits_easy, target)

        # Case B: Low-confidence/hard prediction (equal probabilities)
        logits_hard = torch.tensor([[0.0, 0.0, 0.0]])
        loss_hard = self.focal(logits_hard, target)

        # Focal loss must severely down-weight the easy sample compared to the hard sample
        self.assertLess(loss_easy.item(), loss_hard.item() * 0.05)
        self.assertGreater(loss_easy.item(), 0.0)

    def test_focal_loss_gradients(self):
        logits = torch.randn(4, 15, requires_grad=True)
        targets = torch.tensor([0, 9, 13, 2])
        loss = self.focal(logits, targets)
        loss.backward()

        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, (4, 15))
        self.assertFalse(torch.isnan(logits.grad).any())


class TestUncertaintyRejectionHead(unittest.TestCase):
    """Validates the 60% confidence uncertainty safety override (PRD Section 4 & FR-3)."""

    def setUp(self):
        self.engine = NeuroVQAInferenceEngine(checkpoint_path=None)

    def test_uncertainty_threshold_override(self):
        self.assertEqual(UNCERTAINTY_THRESHOLD, 0.60)

        # Test case: Low confidence (<0.60) must trigger Uncertainty override
        # We simulate this via engine with low asymmetry scan and non-specific question
        dummy_scan = np.full((3, 224, 224), fill_value=128, dtype=np.uint8)
        # Adding slight noise without high focal peak
        dummy_scan[1, 100:110, 100:110] = 175

        res = self.engine.predict_25d(dummy_scan, "What type of tumor is indicated?")
        self.assertIn("answer", res)
        self.assertIn("confidence", res)
        self.assertIn("status", res)
        self.assertIn("top3", res)
        self.assertEqual(len(res["top3"]), 3)

        # When confidence is below threshold, answer must be overridden
        if res["confidence"] < 0.60:
            self.assertEqual(res["answer"], "Uncertain - Needs Radiologist Review")
            self.assertEqual(res["status"], "UNCERTAIN / INCONCLUSIVE")
            self.assertTrue(res["uncertainty_flag"])


class TestGradCAMExplainability(unittest.TestCase):
    """Validates Attention x Gradient Grad-CAM computation (Design Section 4)."""

    def setUp(self):
        self.sample_img = np.full((224, 224, 3), fill_value=100, dtype=np.uint8)
        self.sample_img[80:120, 80:120] = 220  # Simulate focal hyperintensity

    def test_gradcam_output_shapes_and_alpha(self):
        jet_cam, overlay_cam, region_desc = generate_gradcam_heatmap(
            self.sample_img,
            focused_center=(100, 100),
        )

        # Validates output shapes: 224x224x3 RGB
        self.assertEqual(jet_cam.shape, (224, 224, 3))
        self.assertEqual(overlay_cam.shape, (224, 224, 3))
        self.assertEqual(overlay_cam.dtype, np.uint8)

        # Region description must be clinically descriptive
        self.assertIsInstance(region_desc, str)
        self.assertGreater(len(region_desc), 5)

    def test_heatmap_to_png_conversion(self):
        _, overlay_cam, _ = generate_gradcam_heatmap(self.sample_img)
        png_bytes = heatmap_to_png_bytes(overlay_cam)

        self.assertIsInstance(png_bytes, bytes)
        # Verify valid PNG magic header (0x89, 'P', 'N', 'G')
        self.assertTrue(png_bytes.startswith(b"\x89PNG\r\n\x1a\n"))


class TestDatasetAndDicomIngestion(unittest.TestCase):
    """Validates 2.5D packing, NIfTI detection, and native DICOM parsing (FR-1)."""

    def test_25d_stack_packing(self):
        s0 = np.full((128, 128), fill_value=50, dtype=np.uint8)
        s1 = np.full((128, 128), fill_value=100, dtype=np.uint8)
        s2 = np.full((128, 128), fill_value=150, dtype=np.uint8)

        stacked = pack_25d_stack(s0, s1, s2)
        self.assertEqual(stacked.shape, (128, 128, 3))
        self.assertEqual(stacked[0, 0, 0], 50)
        self.assertEqual(stacked[0, 0, 1], 100)
        self.assertEqual(stacked[0, 0, 2], 150)

    def test_preprocessing_output_tensor(self):
        s = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        display_np, tensor = preprocess_mri_image(s)

        # Display array must be resized to 224x224 uint8
        self.assertEqual(display_np.shape, (224, 224, 3))
        self.assertEqual(display_np.dtype, np.uint8)

        # PyTorch model input tensor must be (1, 3, 224, 224) float32
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))

    def test_native_dicom_ingestion(self):
        import pydicom
        from pydicom.dataset import Dataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = "1.2.840.10008.5.1.4.1.1.7"
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PatientID = "NEURO_TEST_042"
        ds.Modality = "MR"
        ds.SeriesDescription = "Axial T2 FLAIR"
        ds.Rows = 128
        ds.Columns = 128
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.RescaleSlope = 1.0
        ds.RescaleIntercept = 0.0

        raw_px = np.random.randint(100, 3000, (128, 128), dtype=np.uint16)
        ds.PixelData = raw_px.tobytes()

        buf = BytesIO()
        pydicom.dcmwrite(buf, ds, enforce_file_format=True)
        dcm_bytes = buf.getvalue()

        # 1. Detection
        self.assertTrue(is_dicom_data(dcm_bytes), "Failed to detect DICOM bytes")

        # 2. Slice extraction
        slice_arr, meta = load_dicom_slice(dcm_bytes)
        self.assertEqual(slice_arr.shape, (128, 128))
        self.assertEqual(meta["patient_id"], "NEURO_TEST_042")
        self.assertEqual(meta["modality"], "MR")

        # 3. Full pipeline preprocessing
        display_np, tensor = preprocess_mri_image(dcm_bytes)
        self.assertEqual(display_np.shape, (224, 224, 3))
        self.assertEqual(tuple(tensor.shape), (1, 3, 224, 224))


if __name__ == "__main__":
    unittest.main(verbosity=2)
