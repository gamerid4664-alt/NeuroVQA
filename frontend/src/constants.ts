// API base URL
export const API_URL = "http://localhost:8000";
export const WS_URL = "ws://localhost:8000";

// 3 Fixed Demo Cases defined in NeuroVQA Version 2.0 PRD & Design
export const DEMO_CASES = [
  {
    id: "case1",
    name: "Case #1: Clear Tumor (High Confidence)",
    modality: "Axial T1 with contrast",
    indication: "Left temporal lobe glioblastoma, headache, seizure onset",
    default_question: "Is there a tumor present?",
    expected_answer: "Yes",
    confidence: 0.92,
    status: "ABNORMALITY DETECTED",
    filename: "case1_clear_tumor.png",
  },
  {
    id: "case2",
    name: "Case #2: Normal Scan (High Confidence)",
    modality: "Axial T1 / T2 Unremarkable",
    indication: "Routine medical screening, preserved gray-white differentiation",
    default_question: "Is there an abnormality present?",
    expected_answer: "Normal",
    confidence: 0.89,
    status: "NORMAL / UNREMARKABLE",
    filename: "case2_normal_scan.png",
  },
  {
    id: "case3",
    name: "Case #3: Uncertain / Small Lesion (Borderline Case)",
    modality: "Axial T2 / FLAIR Faint Lesion",
    indication: "Subtle non-specific 3mm white matter hyperintensity",
    default_question: "What type of tumor is indicated?",
    expected_answer: "Uncertain - Needs Radiologist Review",
    confidence: 0.55,
    status: "UNCERTAIN / INCONCLUSIVE",
    filename: "case3_uncertain_small_lesion.png",
  },
];
