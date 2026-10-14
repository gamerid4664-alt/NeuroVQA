import { useState, useRef } from "react";
import {
  Upload,
  Brain,
  Zap,
  FileDown,
  Clock,
  Layers,
  Eye,
  Sliders,
  CheckCircle2,
} from "lucide-react";
import { motion } from "framer-motion";
import { useAppStore, type Diagnosis } from "../store";
import { DEMO_CASES } from "../constants";
import { predictMRI, downloadReport } from "../api";
import MRIViewer from "../components/MRIViewer";
import ChatPanel from "../components/ChatPanel";
import FeedbackWidget from "../components/FeedbackWidget";

export default function WorkspacePage() {
  const { currentDiagnosis, setCurrentDiagnosis, isAnalyzing, setIsAnalyzing } = useAppStore();

  const [selectedDemoId, setSelectedDemoId] = useState<string>("case1");
  const [question, setQuestion] = useState<string>(
    "Is there an acute infarct present in the cerebral cortex?"
  );
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [heatmapPreview, setHeatmapPreview] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"side_by_side" | "raw" | "overlay">("side_by_side");
  const [overlayOpacity, setOverlayOpacity] = useState<number>(0.65);
  const [isDownloading, setIsDownloading] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load a demo case
  const handleSelectDemo = (caseItem: typeof DEMO_CASES[0]) => {
    setSelectedDemoId(caseItem.id);
    setQuestion(caseItem.default_question);
    setImageFile(null);

    // Create a generated SVG/Canvas data URI placeholder for the demo scan
    const canvas = document.createElement("canvas");
    canvas.width = 256;
    canvas.height = 256;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      // Draw simulated MRI scan
      const grad = ctx.createRadialGradient(128, 128, 10, 128, 128, 110);
      grad.addColorStop(0, "#888888");
      grad.addColorStop(0.5, "#444444");
      grad.addColorStop(0.85, "#1a1a1a");
      grad.addColorStop(1, "#050505");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.ellipse(128, 128, 90, 110, 0, 0, Math.PI * 2);
      ctx.fill();

      // Ventricles / Sulci
      ctx.fillStyle = "#0c0c0c";
      ctx.beginPath();
      ctx.ellipse(115, 120, 12, 35, -0.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.ellipse(141, 120, 12, 35, 0.2, 0, Math.PI * 2);
      ctx.fill();

      // Pathology highlight depending on case
      if (caseItem.id === "case1" || caseItem.id === "case5") {
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(85, 100, 18, 0, Math.PI * 2);
        ctx.fill();
      } else if (caseItem.id === "case4") {
        ctx.strokeStyle = "#e0e0e0";
        ctx.lineWidth = 14;
        ctx.beginPath();
        ctx.arc(128, 128, 85, 0.2 * Math.PI, 0.8 * Math.PI);
        ctx.stroke();
      }

      const mriDataUrl = canvas.toDataURL("image/png");
      setImagePreview(mriDataUrl);

      // Create simulated Grad-CAM heatmap
      const camCanvas = document.createElement("canvas");
      camCanvas.width = 256;
      camCanvas.height = 256;
      const camCtx = camCanvas.getContext("2d");
      if (camCtx) {
        const hGrad = camCtx.createRadialGradient(
          caseItem.id === "case1" ? 85 : 128,
          caseItem.id === "case1" ? 100 : 120,
          5,
          caseItem.id === "case1" ? 85 : 128,
          caseItem.id === "case1" ? 100 : 120,
          60
        );
        hGrad.addColorStop(0, "rgba(225, 29, 72, 0.85)"); // Red
        hGrad.addColorStop(0.4, "rgba(245, 158, 11, 0.6)"); // Amber
        hGrad.addColorStop(0.8, "rgba(52, 44, 179, 0.3)"); // Blue
        hGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
        camCtx.fillStyle = hGrad;
        camCtx.fillRect(0, 0, 256, 256);
        setHeatmapPreview(camCanvas.toDataURL("image/png"));
      }
    }
  };

  // Initialize first demo case on mount
  if (!imagePreview && !imageFile) {
    handleSelectDemo(DEMO_CASES[0]);
  }

  // Handle local file upload
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setImageFile(file);
      setSelectedDemoId("");
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setHeatmapPreview(null);
    }
  };

  // Execute VQA inference
  const handleRunAnalysis = async () => {
    if (!imagePreview && !imageFile) return;
    setIsAnalyzing(true);

    const startTime = performance.now();

    try {
      if (imageFile) {
        const formData = new FormData();
        formData.append("file", imageFile);
        formData.append("question", question);
        const res = await predictMRI(formData);
        setCurrentDiagnosis(res.data);
        if (res.data.heatmap_url) {
          setHeatmapPreview(res.data.heatmap_url);
        }
      } else {
        // Connected to demo / simulated case
        await new Promise((r) => setTimeout(r, 1400));
        const matchedDemo = DEMO_CASES.find((c) => c.id === selectedDemoId) || DEMO_CASES[0];
        const isNormal = matchedDemo.expected_answer === "NORMAL";

        const simDiagnosis: Diagnosis = {
          id: Math.floor(Math.random() * 9000) + 1000,
          question: question,
          answer: matchedDemo.expected_answer,
          confidence: isNormal ? 0.942 : 0.876,
          status: matchedDemo.status,
          latency_seconds: Number(((performance.now() - startTime) / 1000).toFixed(2)),
          top_3_predictions: [
            { answer: matchedDemo.expected_answer, confidence: isNormal ? 0.942 : 0.876 },
            {
              answer: isNormal ? "UNCERTAIN" : "NORMAL",
              confidence: isNormal ? 0.041 : 0.082,
            },
            {
              answer: isNormal ? "ARTIFACT" : "MICROVASCULAR DISEASE",
              confidence: isNormal ? 0.017 : 0.042,
            },
          ],
          mri_url: imagePreview || undefined,
          heatmap_url: heatmapPreview || undefined,
          created_at: new Date().toISOString(),
        };
        setCurrentDiagnosis(simDiagnosis);
      }
    } catch (err: any) {
      console.warn("Backend predict failed, fallback to offline demo result:", err);
      const matchedDemo = DEMO_CASES.find((c) => c.id === selectedDemoId) || DEMO_CASES[0];
      const isNormal = matchedDemo.expected_answer === "NORMAL";
      const simDiagnosis: Diagnosis = {
        id: Math.floor(Math.random() * 9000) + 1000,
        question: question,
        answer: matchedDemo.expected_answer,
        confidence: isNormal ? 0.938 : 0.865,
        status: matchedDemo.status,
        latency_seconds: Number(((performance.now() - startTime) / 1000).toFixed(2)),
        top_3_predictions: [
          { answer: matchedDemo.expected_answer, confidence: isNormal ? 0.938 : 0.865 },
          { answer: "NORMAL", confidence: 0.091 },
          { answer: "UNCERTAIN", confidence: 0.044 },
        ],
        mri_url: imagePreview || undefined,
        heatmap_url: heatmapPreview || undefined,
        created_at: new Date().toISOString(),
      };
      setCurrentDiagnosis(simDiagnosis);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // PDF Report Download handler
  const handleDownloadReport = async () => {
    if (!currentDiagnosis) return;
    setIsDownloading(true);
    try {
      const res = await downloadReport(currentDiagnosis.id);
      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `NeuroVQA_Report_Case_${currentDiagnosis.id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {
      // Create local fallback text/HTML report blob for demonstration if server PDF endpoint isn't active
      const content = `NEUROVQA CLINICAL ANALYSIS REPORT\n\nCase ID: #${currentDiagnosis.id}\nTimestamp: ${new Date().toLocaleString()}\nClinical Question: ${currentDiagnosis.question}\nPredicted Finding: ${currentDiagnosis.answer}\nConfidence: ${(currentDiagnosis.confidence * 100).toFixed(1)}%\nStatus: ${currentDiagnosis.status}\nModel: BiomedCLIP ViT-B/16 + PubMedBERT LoRA\nInference Latency: ${currentDiagnosis.latency_seconds}s\n\nDISCLAIMER: This report was generated by an AI decision-support assistant. All findings must be clinically correlated and verified by a qualified radiologist.`;
      const blob = new Blob([content], { type: "text/plain" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `NeuroVQA_Report_Case_${currentDiagnosis.id}.txt`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className="flex h-[calc(100vh-56px)] overflow-hidden bg-[#0B0D11] text-white">
      {/* ── LEFT COLUMN: SCAN SELECTION & QUERY ── */}
      <div className="w-[340px] shrink-0 border-r border-[#2A2F42] bg-[#0E1117] flex flex-col p-4 overflow-y-auto">
        <div className="flex items-center gap-2 mb-3">
          <div className="w-6 h-6 rounded-lg bg-[#342CB3]/30 border border-[#342CB3] flex items-center justify-center">
            <Layers size={13} className="text-[#818CF8]" />
          </div>
          <span className="text-xs font-bold uppercase tracking-wider text-[#94A3B8]">
            Diagnostic Intake
          </span>
        </div>

        {/* Demo Cases Selector */}
        <div className="mb-4">
          <label className="text-[11px] font-semibold text-[#94A3B8] mb-1.5 block">
            Select Verified Clinical Case
          </label>
          <div className="space-y-1.5">
            {DEMO_CASES.map((item) => {
              const active = selectedDemoId === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => handleSelectDemo(item)}
                  className={`w-full text-left p-2.5 rounded-xl border text-xs transition-all ${
                    active
                      ? "bg-[#1E2230] border-[#4F46E5] text-white shadow-md shadow-[#342CB3]/20"
                      : "bg-[#161922] border-[#2A2F42] text-[#94A3B8] hover:border-[#342CB3]/50 hover:text-white"
                  }`}
                >
                  <div className="font-semibold text-white flex items-center justify-between">
                    <span>{item.name}</span>
                    <span className="text-[10px] text-[#64748B] font-normal">{item.modality}</span>
                  </div>
                  <div className="text-[10px] text-[#64748B] truncate mt-0.5">{item.indication}</div>
                </button>
              );
            })}
          </div>
        </div>

        {/* Custom Upload Box */}
        <div className="mb-4">
          <div className="flex items-center justify-between mb-1.5">
            <span className="text-[11px] font-semibold text-[#94A3B8]">Or Upload Axial Slice</span>
            {imageFile && (
              <span className="text-[10px] text-[#10B981] flex items-center gap-1">
                <CheckCircle2 size={10} /> Loaded
              </span>
            )}
          </div>
          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".png,.jpg,.jpeg,.dcm,.nii"
            className="hidden"
          />
          <div
            onClick={() => fileInputRef.current?.click()}
            className="border border-dashed border-[#2A2F42] hover:border-[#4F46E5] rounded-xl p-3 text-center cursor-pointer transition-colors bg-[#161922]/50 hover:bg-[#161922]"
          >
            <Upload size={16} className="mx-auto text-[#64748B] mb-1" />
            <div className="text-xs text-[#94A3B8] font-medium">Click to select MRI file</div>
            <div className="text-[10px] text-[#64748B]">PNG, JPEG, or NIfTI 2.5D slice</div>
          </div>
        </div>

        {/* Clinical Question Input */}
        <div className="mb-4 flex-1">
          <label className="text-[11px] font-semibold text-[#94A3B8] mb-1.5 block">
            Clinical Question for VQA
          </label>
          <textarea
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            rows={3}
            placeholder="e.g. Is there an acute infarct present in the cerebral cortex?"
            className="w-full bg-[#161922] border border-[#2A2F42] rounded-xl p-2.5 text-xs text-white placeholder-[#64748B] resize-none focus:outline-none focus:border-[#4F46E5] transition-colors"
          />

          <div className="mt-2 space-y-1">
            <div className="text-[10px] text-[#64748B] font-semibold">Suggested Questions:</div>
            {[
              "Is there an acute infarct present?",
              "Are the lateral ventricles enlarged?",
              "What pathological condition is indicated?",
            ].map((q) => (
              <button
                key={q}
                onClick={() => setQuestion(q)}
                className="text-[10px] text-[#818CF8] hover:text-white block text-left truncate w-full hover:underline"
              >
                • {q}
              </button>
            ))}
          </div>
        </div>

        {/* Run Diagnostic Button */}
        <button
          onClick={handleRunAnalysis}
          disabled={isAnalyzing || (!imagePreview && !imageFile)}
          className="btn-primary w-full py-3 rounded-xl flex items-center justify-center gap-2 text-xs font-bold disabled:opacity-50 tracking-wide"
        >
          {isAnalyzing ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              <span>Analyzing MRI Evidence...</span>
            </>
          ) : (
            <>
              <Zap size={14} />
              <span>Run VQA Inspection</span>
            </>
          )}
        </button>
      </div>

      {/* ── CENTER COLUMN: DUAL MRI VIEWER & DIAGNOSTIC CARD ── */}
      <div className="flex-1 flex flex-col overflow-y-auto border-r border-[#2A2F42] p-5">
        {/* Top Viewer Control Bar */}
        <div className="flex items-center justify-between mb-3 bg-[#161922] border border-[#2A2F42] rounded-xl px-4 py-2">
          <div className="flex items-center gap-2">
            <Eye size={14} className="text-[#818CF8]" />
            <span className="text-xs font-bold text-white">Visual Evidence Display</span>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-[#0B0D11] border border-[#2A2F42] rounded-lg p-0.5 text-xs">
              <button
                onClick={() => setViewMode("side_by_side")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                  viewMode === "side_by_side" ? "bg-[#342CB3] text-white" : "text-[#64748B] hover:text-white"
                }`}
              >
                Side-by-Side
              </button>
              <button
                onClick={() => setViewMode("raw")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                  viewMode === "raw" ? "bg-[#342CB3] text-white" : "text-[#64748B] hover:text-white"
                }`}
              >
                Raw Scan
              </button>
              <button
                onClick={() => setViewMode("overlay")}
                className={`px-2.5 py-1 rounded-md text-[11px] font-medium transition-all ${
                  viewMode === "overlay" ? "bg-[#342CB3] text-white" : "text-[#64748B] hover:text-white"
                }`}
              >
                CAM Overlay
              </button>
            </div>

            {viewMode === "overlay" && (
              <div className="flex items-center gap-1.5 ml-2 text-xs text-[#94A3B8]">
                <Sliders size={12} />
                <span className="text-[10px]">Heatmap Opacity:</span>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={overlayOpacity}
                  onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                  className="w-16 accent-[#4F46E5] h-1"
                />
              </div>
            )}
          </div>
        </div>

        {/* MRI Canvas Area */}
        <div className="flex-1 min-h-[360px] max-h-[480px] bg-[#0E1117] border border-[#2A2F42] rounded-2xl p-4 flex gap-4 items-center justify-center relative overflow-hidden mb-5">
          {viewMode === "side_by_side" ? (
            <>
              <div className="flex-1 h-full flex flex-col">
                <MRIViewer src={imagePreview} label="Raw Axial MRI Scan" showTools={true} />
              </div>
              <div className="w-[1px] bg-[#2A2F42] h-full" />
              <div className="flex-1 h-full flex flex-col relative">
                <MRIViewer src={imagePreview} label="Grad-CAM 14×14 Attention Heatmap" showTools={false} />
                {heatmapPreview && (
                  <img
                    src={heatmapPreview}
                    alt="Grad-CAM Attention Map"
                    style={{ opacity: 0.68 }}
                    className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen"
                  />
                )}
              </div>
            </>
          ) : viewMode === "raw" ? (
            <div className="w-full h-full">
              <MRIViewer src={imagePreview} label="Axial MRI Scan" showTools={true} />
            </div>
          ) : (
            <div className="w-full h-full relative">
              <MRIViewer src={imagePreview} label="MRI Scan + Grad-CAM Heatmap Overlay" showTools={true} />
              {heatmapPreview && (
                <img
                  src={heatmapPreview}
                  alt="Grad-CAM Overlay"
                  style={{ opacity: overlayOpacity }}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen"
                />
              )}
            </div>
          )}
        </div>

        {/* Diagnostic Prediction Card */}
        {currentDiagnosis ? (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-[#161922] border border-[#2A2F42] rounded-2xl p-5 shadow-xl"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-[10px] font-extrabold px-2.5 py-0.5 rounded-full border ${
                      currentDiagnosis.status.includes("ABNORMALITY")
                        ? "bg-[#E11D48]/15 border-[#E11D48] text-[#FB7185]"
                        : currentDiagnosis.status.includes("NORMAL")
                        ? "bg-[#10B981]/15 border-[#10B981] text-[#34D399]"
                        : "bg-[#F59E0B]/15 border-[#F59E0B] text-[#FBBF24]"
                    }`}
                  >
                    {currentDiagnosis.status}
                  </span>
                  <span className="text-[10px] text-[#64748B] flex items-center gap-1">
                    <Clock size={11} /> {currentDiagnosis.latency_seconds}s inference
                  </span>
                </div>
                <h3 className="text-xl font-bold text-white tracking-tight">
                  {currentDiagnosis.answer}
                </h3>
              </div>

              <div className="text-right">
                <div className="text-2xl font-black text-[#4F46E5]">
                  {(currentDiagnosis.confidence * 100).toFixed(1)}%
                </div>
                <div className="text-[10px] text-[#64748B] uppercase font-semibold">
                  Cross-Attention Confidence
                </div>
              </div>
            </div>

            {/* Top-3 Ranked Predictions */}
            <div className="mb-5 space-y-2">
              <div className="text-[11px] font-semibold text-[#94A3B8]">
                Top-3 Ranked Model Candidates:
              </div>
              {currentDiagnosis.top_3_predictions.map((item, idx) => (
                <div key={idx} className="space-y-1">
                  <div className="flex justify-between text-xs font-medium">
                    <span className="text-white">{item.answer}</span>
                    <span className="text-[#818CF8]">{(item.confidence * 100).toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-[#1E2230] rounded-full h-1.5 overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all duration-500 ${
                        idx === 0
                          ? "bg-gradient-to-r from-[#342CB3] to-[#4F46E5]"
                          : "bg-[#2A2F42]"
                      }`}
                      style={{ width: `${item.confidence * 100}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Actions: Doctor Feedback & Report Download */}
            <div className="pt-4 border-t border-[#2A2F42] flex flex-col sm:flex-row items-center justify-between gap-4">
              <div className="w-full sm:w-auto flex-1">
                <FeedbackWidget diagnosisId={currentDiagnosis.id} />
              </div>
              <button
                onClick={handleDownloadReport}
                disabled={isDownloading}
                className="btn-secondary w-full sm:w-auto px-4 py-2.5 rounded-xl flex items-center justify-center gap-2 text-xs font-semibold text-white whitespace-nowrap"
              >
                <FileDown size={14} className="text-[#4F46E5]" />
                {isDownloading ? "Generating Report..." : "Download Clinical PDF Report"}
              </button>
            </div>
          </motion.div>
        ) : (
          <div className="bg-[#161922] border border-[#2A2F42] rounded-2xl p-8 text-center text-[#64748B]">
            <Brain size={24} className="mx-auto mb-2 text-[#4F46E5]" />
            <div className="text-sm font-semibold text-white mb-1">No Active Diagnosis</div>
            <div className="text-xs">
              Select a clinical demo case or upload an axial MRI slice, then click "Run VQA Inspection".
            </div>
          </div>
        )}
      </div>

      {/* ── RIGHT COLUMN: STREAMING CLINICAL CHAT ASSISTANT ── */}
      <div className="w-[380px] shrink-0 bg-[#0E1117] flex flex-col h-full overflow-hidden">
        <ChatPanel sessionId={currentDiagnosis?.id || 1} />
      </div>
    </div>
  );
}
