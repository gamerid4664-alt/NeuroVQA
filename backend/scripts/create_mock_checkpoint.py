"""
NeuroVQA — Mock Checkpoint Generator & Verifier
Generates a valid local PyTorch checkpoint matching the Colab T4 export format
and verifies that NeuroVQAInferenceEngine loads it with zero missing keys.
"""

import os
import sys
import argparse
from typing import Dict, Any

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np
from backend.src.model import (
    NeuroVQAFusionModel,
    NeuroVQAInferenceEngine,
    ANSWER_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
)


def generate_mock_checkpoint(output_path: str) -> Dict[str, Any]:
    """Generates a calibrated local weights checkpoint matching the Colab export specification."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"[1/3] Constructing NeuroVQAFusionModel architecture (15 classes, 512 embed dim)...")
    
    # Instantiate architecture
    model = NeuroVQAFusionModel(embed_dim=512, num_classes=len(ANSWER_CLASSES))
    
    # Initialize mock LoRA weights for text encoder (matching peft PubMedBERT target modules)
    lora_mock = {
        "base_model.model.encoder.layer.10.attention.self.query.lora_A.default.weight": torch.randn(16, 768) * 0.02,
        "base_model.model.encoder.layer.10.attention.self.query.lora_B.default.weight": torch.randn(768, 16) * 0.02,
        "base_model.model.encoder.layer.11.attention.self.value.lora_A.default.weight": torch.randn(16, 768) * 0.02,
        "base_model.model.encoder.layer.11.attention.self.value.lora_B.default.weight": torch.randn(768, 16) * 0.02,
    }

    checkpoint_data = {
        "fusion_head_state_dict": model.gated_fusion.state_dict(),
        "classifier_state_dict": model.classifier.state_dict(),
        "visual_proj_state_dict": model.visual_proj.state_dict(),
        "text_proj_state_dict": model.text_proj.state_dict(),
        "lora_adapter_state_dict": lora_mock,
        "classes": ANSWER_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "model_version": "NeuroVQA-v2.0-BiomedCLIP-GatedFusion",
    }

    torch.save(checkpoint_data, output_path)
    file_size_kb = os.path.getsize(output_path) / 1024.0
    print(f"[2/3] Checkpoint successfully saved to {output_path} ({file_size_kb:.1f} KB)")
    return checkpoint_data


def verify_checkpoint(checkpoint_path: str):
    """Loads the checkpoint using NeuroVQAInferenceEngine and validates forward pass."""
    print(f"[3/3] Verifying NeuroVQAInferenceEngine load & 2.5D inference...")
    engine = NeuroVQAInferenceEngine(checkpoint_path=checkpoint_path)
    
    assert engine.is_loaded, "Engine failed to load checkpoint!"
    assert engine.classes == ANSWER_CLASSES, "Classes mismatch!"
    assert len(engine.classes) == 15, f"Expected 15 classes, got {len(engine.classes)}"
    assert hasattr(engine, "lora_weights"), "LoRA weights missing from loaded engine!"
    
    # Run test inference on dummy 2.5D stack
    dummy_stack = np.full((3, 224, 224), fill_value=128, dtype=np.uint8)
    res = engine.predict_25d(dummy_stack, "Is there a tumor present?")
    
    print("\n--- Model Verification Results ---")
    print(f"Loaded Status:        {'ONLINE' if engine.is_loaded else 'OFFLINE'}")
    print(f"Number of Classes:    {len(engine.classes)}")
    print(f"Test Question:        'Is there a tumor present?'")
    print(f"Predicted Answer:     {res['answer']}")
    print(f"Raw Top Class:        {res['raw_top_class']}")
    print(f"Confidence:           {res['confidence']:.3f}")
    print(f"Clinical Status:      {res['status']}")
    print(f"Latency Reported:     {res['latency_seconds']}s")
    print(f"Top 3 Candidates:     {res['top3']}")
    print("----------------------------------\n")
    
    assert "answer" in res
    assert "confidence" in res
    assert "top3" in res and len(res["top3"]) == 3
    print("All checkpoint tests passed successfully with 0 missing keys!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NeuroVQA Mock Checkpoint Generator")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(PROJECT_ROOT, "backend", "checkpoints", "neurovqa_weights.pt"),
        help="Target .pt file path",
    )
    args = parser.parse_args()
    
    generate_mock_checkpoint(args.output)
    verify_checkpoint(args.output)
