"""
NeuroVQA — Synthetic Visual Question Answering (VQA) Generator
Generates clinical question-answer pairs from the Brain Tumor MRI Kaggle dataset (7,000 images).
Produces 5 deterministic questions per image mapped to the 15 fixed canonical answer classes.
"""

import os
import json
import argparse
from glob import glob
from typing import List, Dict, Any

# 15 Canonical Answer Classes
ANSWER_CLASSES = [
    "Yes", "No", "Uncertain", "Left", "Right", 
    "Frontal", "Parietal", "Temporal", "Occipital", 
    "Glioma", "Meningioma", "Pituitary", "Tumor", "Normal", "Edema"
]
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(ANSWER_CLASSES)}

# 5 Deterministic Question Templates
TEMPLATES = [
    {
        "question": "Is there an abnormality present?",
        "answer_fn": lambda tumor_type, filename: "No" if tumor_type == "notumor" else "Yes"
    },
    {
        "question": "Is there a tumor present?",
        "answer_fn": lambda tumor_type, filename: "No" if tumor_type == "notumor" else "Yes"
    },
    {
        "question": "What type of tumor is indicated?",
        "answer_fn": lambda tumor_type, filename: {
            "glioma": "Glioma",
            "meningioma": "Meningioma",
            "pituitary": "Pituitary",
            "notumor": "Normal"
        }.get(tumor_type, "Normal")
    },
    {
        "question": "Which side is abnormality on?",
        "answer_fn": lambda tumor_type, filename: (
            "Normal" if tumor_type == "notumor"
            else ("Left" if "left" in filename.lower() else ("Right" if "right" in filename.lower() else "Left"))
        )
    },
    {
        "question": "Which lobe is affected?",
        "answer_fn": lambda tumor_type, filename: (
            "Normal" if tumor_type == "notumor"
            else ("Temporal" if tumor_type == "glioma" else ("Frontal" if tumor_type == "meningioma" else "Parietal"))
        )
    }
]


def generate_synthetic_dataset(data_dir: str, output_file: str) -> None:
    """
    Scans image directories organized by tumor class and writes JSON records with {image_path, question, answer}.
    """
    print(f"Scanning directory: {data_dir}")
    records: List[Dict[str, Any]] = []
    tumor_types = ["glioma", "meningioma", "pituitary", "notumor"]
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for tumor in tumor_types:
        pattern = os.path.join(data_dir, "**", tumor, "*.*")
        matches = glob(pattern, recursive=True)
        img_files = [m for m in matches if os.path.splitext(m)[1].lower() in valid_exts]
        print(f"  - Found {len(img_files)} images for class '{tumor}'")

        for img_path in img_files:
            fname = os.path.basename(img_path)
            for t in TEMPLATES:
                ans = t["answer_fn"](tumor, fname)
                assert ans in ANSWER_CLASSES, f"Generated answer '{ans}' not in 15 canonical classes!"
                records.append({
                    "image_path": os.path.abspath(img_path),
                    "question": t["question"],
                    "answer": ans
                })

    output_dir = os.path.dirname(os.path.abspath(output_file))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"\nSuccessfully generated {len(records)} synthetic VQA pairs!")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic VQA pairs for Brain Tumor MRI")
    parser.add_argument("--data_dir", type=str, default="./data/brain_tumor_mri", help="Path to raw dataset")
    parser.add_argument("--output_file", type=str, default="./data/synthetic_vqa.json", help="Path to output JSON")
    args = parser.parse_args()

    generate_synthetic_dataset(args.data_dir, args.output_file)
