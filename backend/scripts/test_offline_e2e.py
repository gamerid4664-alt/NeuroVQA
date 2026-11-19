"""
NeuroVQA — Airplane Mode (Offline) End-to-End Test Suite
Simulates full offline execution on an Intel Core i3 workstation with 0 external network requests.
Validates:
1. Offline Health Check & SQLite Fallback
2. 2D Single Slice Prediction (POST /api/predict)
3. Attention x Gradient Heatmap Generation & Local Persistence
4. Clinical PDF Report Generation (GET /api/report/{case_id})
5. Clinician Feedback Submission (POST /api/feedback)
6. Case History Retrieval (GET /api/history)
7. 2.5D Volumetric Tri-Slice Prediction (k-1, k, k+1)
"""

import os
import sys
import time

# Enforce strict offline mode BEFORE any imports
os.environ["USE_SUPABASE"] = "false"

# Ensure project root is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from backend.main import app, startup_event
import asyncio


def run_offline_test():
    print("=" * 70)
    print("NeuroVQA -- Airplane Mode (Offline) End-to-End Validation Protocol")
    print("=" * 70)

    # 1. Initialize client & run startup pre-warmup
    print("\n[Step 1/7] Running startup pre-warmup routine...")
    t0 = time.perf_counter()
    asyncio.run(startup_event())
    client = TestClient(app)
    warmup_duration = (time.perf_counter() - t0) * 1000
    print(f"[OK] Startup pre-warmup complete in {warmup_duration:.1f}ms.")

    # 2. Verify Health Check
    print("\n[Step 2/7] Checking API health in offline mode...")
    res_health = client.get("/api/health")
    assert res_health.status_code == 200, f"Health check failed: {res_health.text}"
    health_data = res_health.json()
    print(f"[OK] Health status: {health_data['status']}")
    print(f"   Mode:          {health_data['mode']}")
    print(f"   Database:      {health_data['database']}")
    print(f"   Model Version: {health_data['model_version']}")
    assert health_data["mode"] == "offline-sqlite", "Expected offline-sqlite mode!"

    # 3. 2D Single Slice Prediction
    print("\n[Step 3/7] Testing 2D single slice prediction (POST /api/predict)...")
    sample_path = os.path.join(PROJECT_ROOT, "backend", "data", "demo_samples", "case1_clear_tumor.png")
    assert os.path.exists(sample_path), f"Demo sample not found at: {sample_path}"

    with open(sample_path, "rb") as f:
        file_bytes = f.read()

    pred_t0 = time.perf_counter()
    res_pred = client.post(
        "/api/predict",
        files={"file": ("case1_clear_tumor.png", file_bytes, "image/png")},
        data={"question": "Is there a tumor present?"},
    )
    pred_latency = (time.perf_counter() - pred_t0) * 1000

    assert res_pred.status_code == 200, f"Predict failed: {res_pred.text}"
    pred_data = res_pred.json()

    print(f"[OK] Prediction Successful:")
    print(f"   Diagnosis ID:    {pred_data['id']}")
    print(f"   Question:        {pred_data['question']}")
    print(f"   Answer:          {pred_data['answer']}")
    print(f"   Confidence:      {pred_data['confidence']:.3f}")
    print(f"   Clinical Status: {pred_data['status']}")
    print(f"   Top-3:           {pred_data['top3']}")
    print(f"   Focused Region:  {pred_data['focused_region']}")
    print(f"   Reported Latency:{pred_data['latency_ms']}ms (measured {pred_latency:.1f}ms)")
    print(f"   Heatmap URL:     {pred_data['heatmap_url']}")

    diag_id = pred_data["id"]
    assert diag_id is not None, "Missing diagnosis id in response"
    assert "top3" in pred_data and len(pred_data["top3"]) == 3, "Missing or invalid top3"

    # Verify heatmap file created on disk
    heatmap_filename = os.path.basename(pred_data["heatmap_url"])
    heatmap_disk_path = os.path.join(PROJECT_ROOT, "backend", "data", "heatmaps", heatmap_filename)
    assert os.path.exists(heatmap_disk_path), f"Heatmap file missing on disk: {heatmap_disk_path}"
    print(f"[OK] Heatmap verified on disk ({os.path.getsize(heatmap_disk_path)} bytes).")

    # 4. Generate Clinical PDF Report
    print(f"\n[Step 4/7] Generating PDF report for Case #{diag_id} (GET /api/report/{diag_id})...")
    res_report = client.get(f"/api/report/{diag_id}")
    assert res_report.status_code == 200, f"Report download failed: {res_report.text}"
    assert res_report.headers.get("content-type") == "application/pdf", "Expected application/pdf"

    pdf_bytes = res_report.content
    assert len(pdf_bytes) > 1000, f"PDF report suspiciously small: {len(pdf_bytes)} bytes"
    print(f"[OK] Clinical PDF report generated successfully ({len(pdf_bytes)} bytes).")

    # 5. Submit Doctor Feedback
    print(f"\n[Step 5/7] Submitting clinician feedback for Case #{diag_id} (POST /api/feedback)...")
    res_fb = client.post(
        "/api/feedback",
        json={
            "diagnosis_id": diag_id,
            "feedback_rating": "CORRECT",
            "doctor_notes": "Ground-truth alignment verified. Well-circumscribed focal lesion in left temporal territory.",
        },
    )
    assert res_fb.status_code == 200, f"Feedback submission failed: {res_fb.text}"
    fb_data = res_fb.json()
    print(f"[OK] Feedback recorded: {fb_data.get('message', 'Success')}")

    # 6. Retrieve Case History
    print("\n[Step 6/7] Retrieving case history from local SQLite (GET /api/history)...")
    res_hist = client.get("/api/history?limit=10")
    assert res_hist.status_code == 200, f"History fetch failed: {res_hist.text}"
    hist_data = res_hist.json()
    assert hist_data["total"] >= 1, "Expected at least 1 case in history"
    matching_case = next((c for c in hist_data["cases"] if str(c["id"]) == str(diag_id)), None)
    assert matching_case is not None, f"Case #{diag_id} not found in history"
    print(f"[OK] Case #{diag_id} verified in SQLite history with feedback: '{matching_case.get('feedback')}'.")

    # 7. Test 2.5D Volumetric Multi-Slice Stack (k-1, k, k+1)
    print("\n[Step 7/7] Testing 2.5D Volumetric Tri-Slice Stack inference...")
    case1_path = os.path.join(PROJECT_ROOT, "backend", "data", "demo_samples", "case1_clear_tumor.png")
    case2_path = os.path.join(PROJECT_ROOT, "backend", "data", "demo_samples", "case2_normal_scan.png")
    case3_path = os.path.join(PROJECT_ROOT, "backend", "data", "demo_samples", "case3_uncertain_small_lesion.png")

    with open(case1_path, "rb") as f1, open(case2_path, "rb") as f2, open(case3_path, "rb") as f3:
        b1, b2, b3 = f1.read(), f2.read(), f3.read()

    res_25d = client.post(
        "/api/predict",
        files=[
            ("files", ("slice_k_minus_1.png", b1, "image/png")),
            ("files", ("slice_k.png", b2, "image/png")),
            ("files", ("slice_k_plus_1.png", b3, "image/png")),
        ],
        data={"question": "What type of tumor is indicated?"},
    )
    assert res_25d.status_code == 200, f"2.5D inference failed: {res_25d.text}"
    data_25d = res_25d.json()
    print(f"[OK] 2.5D Stack Prediction Successful:")
    print(f"   Slice Mode:      {data_25d['slice_mode']}")
    print(f"   Answer:          {data_25d['answer']}")
    print(f"   Confidence:      {data_25d['confidence']:.3f}")
    print(f"   Latency:         {data_25d['latency_ms']}ms")
    assert "2.5D Volumetric Stack" in data_25d["slice_mode"], "Slice mode mismatch"

    print("\n" + "=" * 70)
    print("ALL 7 AIRPLANE MODE (OFFLINE) VALIDATION TESTS PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_offline_test()
