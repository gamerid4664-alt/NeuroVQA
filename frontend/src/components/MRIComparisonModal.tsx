import { useState, useEffect } from "react";
import { X, GitCompare, Brain } from "lucide-react";
import { type Diagnosis } from "../store";
import { getHistory, getDiagnosis } from "../api";
import { motion, AnimatePresence } from "framer-motion";

interface MRIComparisonModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentDiagnosis: Diagnosis | null;
}

export default function MRIComparisonModal({ isOpen, onClose, currentDiagnosis }: MRIComparisonModalProps) {
  const [historyCases, setHistoryCases] = useState<any[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<string | number>("");
  const [comparisonDiagnosis, setComparisonDiagnosis] = useState<Diagnosis | null>(null);
  const [loadingCompare, setLoadingCompare] = useState<boolean>(false);

  useEffect(() => {
    if (!isOpen) return;
    getHistory()
      .then((res) => {
        if (res.data?.cases?.length > 0) {
          // Filter out the active case from comparison list
          const others = res.data.cases.filter((c: any) => String(c.id) !== String(currentDiagnosis?.id));
          setHistoryCases(others);
          if (others.length > 0) {
            setSelectedCaseId(others[0].id);
          }
        }
      })
      .catch(() => {});
  }, [isOpen, currentDiagnosis?.id]);

  useEffect(() => {
    if (!selectedCaseId) return;
    setLoadingCompare(true);
    getDiagnosis(selectedCaseId)
      .then((res) => {
        if (res.data) setComparisonDiagnosis(res.data);
      })
      .catch(() => {
        // Fallback from historyCases list
        const found = historyCases.find((c) => String(c.id) === String(selectedCaseId));
        if (found) {
          setComparisonDiagnosis({
            id: found.id,
            question: found.question,
            answer: found.answer,
            confidence: found.confidence,
            status: found.status,
            latency_seconds: found.latency_seconds,
            top_3_predictions: [
              { answer: found.answer, confidence: found.confidence },
              { answer: "NORMAL", confidence: (1 - found.confidence) * 0.7 },
            ],
            created_at: found.date,
          });
        }
      })
      .finally(() => setLoadingCompare(false));
  }, [selectedCaseId, historyCases]);

  if (!isOpen) return null;

  const currentConf = (currentDiagnosis?.confidence || 0) * 100;
  const compareConf = (comparisonDiagnosis?.confidence || 0) * 100;
  const confDiff = currentConf - compareConf;

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.95 }}
          className="bg-[#0E1117] border border-[#2A2F42] rounded-2xl w-full max-w-5xl max-h-[90vh] flex flex-col overflow-hidden shadow-2xl text-white"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-[#2A2F42] bg-[#161922]">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-[#342CB3]/30 border border-[#342CB3] flex items-center justify-center">
                <GitCompare size={16} className="text-[#818CF8]" />
              </div>
              <div>
                <h2 className="text-sm font-bold text-white">Longitudinal MRI Case Comparison (FR-16)</h2>
                <p className="text-[11px] text-[#94A3B8]">
                  Side-by-side visual grounding, VQA cross-attention findings, and confidence shift analysis
                </p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg hover:bg-[#1E2230] text-[#94A3B8] hover:text-white transition-colors"
            >
              <X size={18} />
            </button>
          </div>

          {/* Comparative Metrics Banner */}
          <div className="px-6 py-3 bg-[#11141C] border-b border-[#2A2F42] flex flex-wrap items-center justify-between gap-4 text-xs">
            <div className="flex items-center gap-3">
              <span className="text-[#94A3B8]">Select Prior Examination:</span>
              <select
                value={selectedCaseId}
                onChange={(e) => setSelectedCaseId(e.target.value)}
                className="bg-[#1E2230] border border-[#2A2F42] rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-[#4F46E5]"
              >
                {historyCases.length > 0 ? (
                  historyCases.map((c) => (
                    <option key={c.id} value={c.id}>
                      Case #{c.id} — {c.answer} ({Math.round(c.confidence * 100)}%) · {c.date}
                    </option>
                  ))
                ) : (
                  <option value="">No other historical cases</option>
                )}
              </select>
            </div>

            {comparisonDiagnosis && (
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1.5">
                  <span className="text-[#64748B]">Confidence Shift:</span>
                  <span
                    className={`font-bold font-mono ${
                      confDiff > 0 ? "text-[#34D399]" : confDiff < 0 ? "text-[#FB7185]" : "text-white"
                    }`}
                  >
                    {confDiff > 0 ? `+${confDiff.toFixed(1)}%` : `${confDiff.toFixed(1)}%`}
                  </span>
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="text-[#64748B]">Progression:</span>
                  <span className="font-semibold text-white">
                    {comparisonDiagnosis.answer === currentDiagnosis?.answer
                      ? "Stable / Concordant finding"
                      : "Evolution in diagnostic classification"}
                  </span>
                </div>
              </div>
            )}
          </div>

          {/* Comparison Body */}
          <div className="flex-1 p-6 overflow-y-auto grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* LEFT: Current MRI Scan */}
            <div className="bg-[#161922] border border-[#2A2F42] rounded-xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-[#818CF8]">
                  Current Active Scan (Case #{currentDiagnosis?.id || "Current"})
                </span>
                <span
                  className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                    currentDiagnosis?.status?.includes("ABNORMAL")
                      ? "bg-[#E11D48]/15 border-[#E11D48] text-[#FB7185]"
                      : "bg-[#10B981]/15 border-[#10B981] text-[#34D399]"
                  }`}
                >
                  {currentDiagnosis?.answer || "NORMAL"}
                </span>
              </div>

              {/* Scan / Overlay Image */}
              <div className="relative aspect-square w-full bg-[#0B0D11] rounded-lg overflow-hidden border border-[#2A2F42] mb-3 flex items-center justify-center">
                {currentDiagnosis?.heatmap_url || currentDiagnosis?.mri_url ? (
                  <img
                    src={currentDiagnosis.heatmap_url || currentDiagnosis.mri_url}
                    alt="Current MRI"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <Brain size={48} className="text-[#2A2F42]" />
                )}
              </div>

              {/* Clinical Details */}
              <div className="space-y-1.5 text-xs text-[#94A3B8]">
                <div>
                  <span className="text-[#64748B]">Clinical Question: </span>
                  <span className="text-white font-medium">{currentDiagnosis?.question || "Is there an acute infarct?"}</span>
                </div>
                <div className="flex justify-between items-center pt-1">
                  <span>VQA Confidence:</span>
                  <span className="font-bold text-white font-mono">{currentConf.toFixed(1)}%</span>
                </div>
                <div className="w-full bg-[#1E2230] rounded-full h-1.5 overflow-hidden">
                  <div className="bg-[#4F46E5] h-full" style={{ width: `${currentConf}%` }} />
                </div>
              </div>
            </div>

            {/* RIGHT: Prior Comparison Scan */}
            <div className="bg-[#161922] border border-[#2A2F42] rounded-xl p-4 flex flex-col">
              <div className="flex items-center justify-between mb-3">
                <span className="text-xs font-bold uppercase tracking-wider text-[#94A3B8]">
                  Prior Examination (Case #{comparisonDiagnosis?.id || "—"})
                </span>
                {comparisonDiagnosis && (
                  <span
                    className={`px-2 py-0.5 rounded text-[10px] font-bold border ${
                      comparisonDiagnosis.status?.includes("ABNORMAL")
                        ? "bg-[#E11D48]/15 border-[#E11D48] text-[#FB7185]"
                        : "bg-[#10B981]/15 border-[#10B981] text-[#34D399]"
                    }`}
                  >
                    {comparisonDiagnosis.answer}
                  </span>
                )}
              </div>

              {/* Scan / Overlay Image */}
              <div className="relative aspect-square w-full bg-[#0B0D11] rounded-lg overflow-hidden border border-[#2A2F42] mb-3 flex items-center justify-center">
                {loadingCompare ? (
                  <div className="flex flex-col items-center gap-2">
                    <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span className="text-xs text-[#64748B]">Loading historical scan...</span>
                  </div>
                ) : comparisonDiagnosis?.heatmap_url || comparisonDiagnosis?.mri_url ? (
                  <img
                    src={comparisonDiagnosis.heatmap_url || comparisonDiagnosis.mri_url}
                    alt="Prior MRI"
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <div className="text-center p-4 text-xs text-[#64748B]">
                    Select a prior case from the selector above to compare scans
                  </div>
                )}
              </div>

              {/* Clinical Details */}
              {comparisonDiagnosis ? (
                <div className="space-y-1.5 text-xs text-[#94A3B8]">
                  <div>
                    <span className="text-[#64748B]">Clinical Question: </span>
                    <span className="text-white font-medium">{comparisonDiagnosis.question}</span>
                  </div>
                  <div className="flex justify-between items-center pt-1">
                    <span>VQA Confidence:</span>
                    <span className="font-bold text-white font-mono">{compareConf.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-[#1E2230] rounded-full h-1.5 overflow-hidden">
                    <div className="bg-[#10B981] h-full" style={{ width: `${compareConf}%` }} />
                  </div>
                </div>
              ) : (
                <div className="text-xs text-[#64748B] text-center pt-4">No comparison case loaded.</div>
              )}
            </div>
          </div>

          {/* Footer */}
          <div className="px-6 py-3 border-t border-[#2A2F42] bg-[#161922] flex items-center justify-between text-xs text-[#64748B]">
            <span>⚠️ Decision Support Only: Longitudinal shifts must be corroborated with radiologist review.</span>
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-[#1E2230] hover:bg-[#2A2F42] text-white font-medium transition-colors"
            >
              Close Comparison
            </button>
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
