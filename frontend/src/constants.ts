// API base URL
export const API_URL = "http://localhost:8000";
export const WS_URL = "ws://localhost:8000";

// Mock/demo data for offline showcase when backend isn't running
export const DEMO_CASES = [
  {
    id: "case1",
    name: "Case #1: Acute Ischemic Infarct",
    modality: "Axial DWI / FLAIR",
    indication: "Sudden right hemiplegia, aphasia, 2h onset",
    default_question: "Is there an acute infarct present in the cerebral cortex?",
    expected_answer: "ACUTE INFARCT",
    status: "ABNORMALITY DETECTED",
    filename: "case1_ischemic_stroke.png",
  },
  {
    id: "case2",
    name: "Case #2: Ventricular Enlargement",
    modality: "Axial T2-Weighted",
    indication: "Progressive cognitive decline, gait disturbance",
    default_question: "Are the lateral ventricles enlarged?",
    expected_answer: "VENTRICULAR ENLARGEMENT",
    status: "ABNORMALITY DETECTED",
    filename: "case2_hydrocephalus.png",
  },
  {
    id: "case3",
    name: "Case #3: Normal Brain",
    modality: "Axial T1-FLAIR",
    indication: "Routine screening, no focal deficits",
    default_question: "Is there any acute structural lesion detected?",
    expected_answer: "NORMAL",
    status: "NORMAL / UNREMARKABLE",
    filename: "case3_normal_brain.png",
  },
  {
    id: "case4",
    name: "Case #4: Subdural Hematoma",
    modality: "Axial T2 / Gradient-Echo (GRE) MRI",
    indication: "Fall, loss of consciousness, headache",
    default_question: "Is an extra-axial mass or hematoma visible?",
    expected_answer: "SUBDURAL HEMATOMA",
    status: "ABNORMALITY DETECTED",
    filename: "case4_subdural_hematoma.png",
  },
  {
    id: "case5",
    name: "Case #5: Glioblastoma",
    modality: "Axial T1 with contrast",
    indication: "New onset seizures, progressive headache",
    default_question: "What pathological condition is indicated in the temporal region?",
    expected_answer: "GLIOBLASTOMA",
    status: "ABNORMALITY DETECTED",
    filename: "case5_glioblastoma.png",
  },
  {
    id: "case6",
    name: "Case #6: 2.5D Volumetric Stack (FR-1)",
    modality: "Axial Contiguous Stack (k-1, k, k+1)",
    indication: "Multislice volumetric temporal lobe assessment",
    default_question: "Does the 3-slice contiguous stack confirm an infiltrative mass?",
    expected_answer: "CONTIGUOUS INFILTRATIVE GLIOMA",
    status: "ABNORMALITY DETECTED",
    filename: "case5_glioblastoma.png",
  },
];
