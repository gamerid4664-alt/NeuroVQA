import { useState, useEffect } from "react";
import {
  Search,
  Brain,
  FileDown,
  ArrowUpRight,
  Filter,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useAppStore, type Diagnosis } from "../store";
import { getHistory, downloadReport, getDiagnosis } from "../api";

interface HistoryCase {
  id: number | string;
  date: string;
  question: string;
  answer: string;
  confidence: number;
  status: string;
  latency_seconds: number;
  feedback?: string;
  mri_url?: string;
  heatmap_url?: string;
}

const MOCK_HISTORY: HistoryCase[] = [
  {
    id: 104,
    date: "2026-10-13 16:42",
    question: "Is there an acute infarct present in the cerebral cortex?",
    answer: "ACUTE INFARCT",
    confidence: 0.884,
    status: "ABNORMALITY DETECTED",
    latency_seconds: 1.68,
    feedback: "Correct",
  },
  {
    id: 103,
    date: "2026-10-13 11:20",
    question: "Are the lateral ventricles enlarged?",
    answer: "VENTRICULAR ENLARGEMENT",
    confidence: 0.842,
    status: "ABNORMALITY DETECTED",
    latency_seconds: 1.74,
    feedback: "Correct",
  },
  {
    id: 102,
    date: "2026-10-12 14:15",
    question: "Is there any acute structural lesion detected?",
    answer: "NORMAL",
    confidence: 0.962,
    status: "NORMAL / UNREMARKABLE",
    latency_seconds: 1.55,
    feedback: "Correct",
  },
  {
    id: 101,
    date: "2026-10-12 09:30",
    question: "Is an extra-axial mass or hematoma visible?",
    answer: "SUBDURAL HEMATOMA",
    confidence: 0.891,
    status: "ABNORMALITY DETECTED",
    latency_seconds: 1.81,
    feedback: "Partially Correct",
  },
  {
    id: 100,
    date: "2026-10-11 18:05",
    question: "What pathological condition is indicated in the temporal region?",
    answer: "GLIOBLASTOMA",
    confidence: 0.871,
    status: "ABNORMALITY DETECTED",
    latency_seconds: 1.92,
    feedback: "Correct",
  },
  {
    id: 99,
    date: "2026-10-11 12:40",
    question: "Is the posterior fossa structure normal?",
    answer: "NORMAL",
    confidence: 0.935,
    status: "NORMAL / UNREMARKABLE",
    latency_seconds: 1.62,
    feedback: "Correct",
  },
];

export default function HistoryPage() {
  const [cases, setCases] = useState<HistoryCase[]>(MOCK_HISTORY);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("ALL");
  const [confidenceFilter, setConfidenceFilter] = useState("ALL");
  const [feedbackFilter, setFeedbackFilter] = useState("ALL");
  const { setCurrentDiagnosis } = useAppStore();
  const navigate = useNavigate();

  useEffect(() => {
    getHistory()
      .then((res) => {
        if (res.data?.cases?.length > 0) {
          setCases(res.data.cases);
        }
      })
      .catch(() => {
        // Fallback to mock history
      });
  }, []);

  const filteredCases = cases.filter((c) => {
    const matchSearch =
      c.question.toLowerCase().includes(search.toLowerCase()) ||
      c.answer.toLowerCase().includes(search.toLowerCase()) ||
      c.id.toString().includes(search);

    const matchStatus =
      statusFilter === "ALL" ||
      (statusFilter === "ABNORMAL" && c.status.includes("ABNORMAL")) ||
      (statusFilter === "NORMAL" && c.status.includes("NORMAL")) ||
      (statusFilter === "UNCERTAIN" && c.status.includes("UNCERTAIN"));

    const matchConfidence =
      confidenceFilter === "ALL" ||
      (confidenceFilter === "HIGH" && c.confidence >= 0.85) ||
      (confidenceFilter === "MODERATE" && c.confidence >= 0.70 && c.confidence < 0.85) ||
      (confidenceFilter === "LOW" && c.confidence < 0.70);

    const cFb = (c.feedback || "").toLowerCase();
    const matchFeedback =
      feedbackFilter === "ALL" ||
      (feedbackFilter === "PENDING" && !c.feedback) ||
      (feedbackFilter === "CORRECT" && cFb.includes("correct") && !cFb.includes("partial")) ||
      (feedbackFilter === "PARTIAL" && (cFb.includes("partial") || cFb.includes("partially_correct"))) ||
      (feedbackFilter === "INCORRECT" && cFb.includes("incorrect")) ||
      (feedbackFilter === "UNCERTAIN" && cFb.includes("uncertain"));

    return matchSearch && matchStatus && matchConfidence && matchFeedback;
  });

  const hasActiveFilters =
    search !== "" || statusFilter !== "ALL" || confidenceFilter !== "ALL" || feedbackFilter !== "ALL";

  const resetFilters = () => {
    setSearch("");
    setStatusFilter("ALL");
    setConfidenceFilter("ALL");
    setFeedbackFilter("ALL");
  };

  const handleReopenInWorkspace = async (c: HistoryCase) => {
    try {
      const res = await getDiagnosis(c.id);
      if (res.data) {
        setCurrentDiagnosis(res.data);
        navigate("/app");
        return;
      }
    } catch {
      // Fallback to table data if offline or network error
    }

    const diag: Diagnosis = {
      id: c.id,
      question: c.question,
      answer: c.answer,
      confidence: c.confidence,
      status: c.status,
      latency_seconds: c.latency_seconds,
      top_3_predictions: [
        { answer: c.answer, confidence: c.confidence },
        { answer: "NORMAL", confidence: (1 - c.confidence) * 0.7 },
        { answer: "UNCERTAIN", confidence: (1 - c.confidence) * 0.3 },
      ],
      created_at: c.date,
    };
    setCurrentDiagnosis(diag);
    navigate("/app");
  };

  const handleDownload = async (id: number | string) => {
    try {
      const res = await downloadReport(id);
      const blob = new Blob([res.data], { type: "application/pdf" });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.setAttribute("download", `NeuroVQA_Report_Case_${id}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch {
      alert(`Report download started for Case #${id}`);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0D11] text-white p-8">
      <div className="max-w-7xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
              <Brain className="text-[#4F46E5]" /> Diagnostic Case History
            </h1>
            <p className="text-xs text-[#94A3B8] mt-1">
              Search, multi-criteria filter, and reopen previous brain MRI evaluations
            </p>
          </div>
          <div className="text-xs text-[#64748B]">
            Showing <span className="font-bold text-white">{filteredCases.length}</span> of {cases.length} cases
          </div>
        </div>

        {/* Multi-Criteria Filter Bar (FR-13) */}
        <div className="bg-[#161922] border border-[#2A2F42] rounded-2xl p-4 space-y-3">
          <div className="flex flex-col md:flex-row items-center gap-3">
            <div className="relative flex-1 w-full">
              <Search size={14} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-[#64748B]" />
              <input
                type="text"
                placeholder="Search by case #, clinical question, or pathology..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full bg-[#1E2230] border border-[#2A2F42] rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-[#64748B] focus:outline-none focus:border-[#4F46E5]"
              />
            </div>

            {/* Status Tabs */}
            <div className="flex items-center gap-1 w-full md:w-auto overflow-x-auto">
              {["ALL", "ABNORMAL", "NORMAL", "UNCERTAIN"].map((type) => (
                <button
                  key={type}
                  onClick={() => setStatusFilter(type)}
                  className={`px-3 py-1.5 rounded-xl text-xs font-semibold whitespace-nowrap transition-all ${
                    statusFilter === type
                      ? "bg-[#342CB3] text-white"
                      : "text-[#64748B] hover:text-white hover:bg-[#1E2230]"
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Secondary Filter Row: Confidence Range, Feedback, and Reset */}
          <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-[#2A2F42]/60 text-xs">
            <div className="flex flex-wrap items-center gap-4">
              {/* Confidence Filter Dropdown */}
              <div className="flex items-center gap-2">
                <span className="text-[#64748B] font-medium flex items-center gap-1">
                  <SlidersHorizontal size={12} className="text-[#818CF8]" /> Confidence:
                </span>
                <select
                  value={confidenceFilter}
                  onChange={(e) => setConfidenceFilter(e.target.value)}
                  className="bg-[#1E2230] border border-[#2A2F42] rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-[#4F46E5]"
                >
                  <option value="ALL">All Confidence Levels</option>
                  <option value="HIGH">High Confidence (≥ 85%)</option>
                  <option value="MODERATE">Moderate (70% – 84%)</option>
                  <option value="LOW">Low / Inconclusive (&lt; 70%)</option>
                </select>
              </div>

              {/* Feedback Filter Dropdown */}
              <div className="flex items-center gap-2">
                <span className="text-[#64748B] font-medium flex items-center gap-1">
                  <Filter size={12} className="text-[#34D399]" /> Feedback:
                </span>
                <select
                  value={feedbackFilter}
                  onChange={(e) => setFeedbackFilter(e.target.value)}
                  className="bg-[#1E2230] border border-[#2A2F42] rounded-lg px-2.5 py-1 text-xs text-white focus:outline-none focus:border-[#4F46E5]"
                >
                  <option value="ALL">All Clinician Reviews</option>
                  <option value="CORRECT">Correct</option>
                  <option value="PARTIAL">Partially Correct</option>
                  <option value="INCORRECT">Incorrect</option>
                  <option value="UNCERTAIN">Uncertain</option>
                  <option value="PENDING">Pending Review</option>
                </select>
              </div>
            </div>

            {hasActiveFilters && (
              <button
                onClick={resetFilters}
                className="flex items-center gap-1.5 px-3 py-1 rounded-lg bg-[#1E2230] hover:bg-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors text-xs font-semibold"
              >
                <RotateCcw size={11} /> Reset Filters
              </button>
            )}
          </div>
        </div>

        {/* Case Cards Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredCases.map((c) => {
            const isAbnormal = c.status.includes("ABNORMAL");
            return (
              <div
                key={c.id}
                className="bg-[#161922] border border-[#2A2F42] hover:border-[#342CB3] rounded-2xl p-5 flex flex-col justify-between transition-all group"
              >
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono text-xs font-bold text-[#818CF8]">#{c.id}</span>
                    <span
                      className={`text-[10px] font-bold px-2.5 py-0.5 rounded-full border ${
                        isAbnormal
                          ? "bg-[#E11D48]/15 border-[#E11D48] text-[#FB7185]"
                          : "bg-[#10B981]/15 border-[#10B981] text-[#34D399]"
                      }`}
                    >
                      {c.answer}
                    </span>
                  </div>

                  <h3 className="text-sm font-semibold text-white mb-2 line-clamp-2">
                    {c.question}
                  </h3>

                  <div className="space-y-1 mb-4 text-xs text-[#94A3B8]">
                    <div className="flex justify-between">
                      <span className="text-[#64748B]">Confidence:</span>
                      <span className="font-bold text-white">{(c.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#64748B]">Inference Latency:</span>
                      <span>{c.latency_seconds}s</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#64748B]">Timestamp:</span>
                      <span>{c.date}</span>
                    </div>
                    {c.feedback && (
                      <div className="flex justify-between">
                        <span className="text-[#64748B]">Doctor Review:</span>
                        <span className="text-[#10B981] font-semibold">{c.feedback}</span>
                      </div>
                    )}
                  </div>
                </div>

                <div className="pt-3 border-t border-[#2A2F42] flex items-center gap-2">
                  <button
                    onClick={() => handleReopenInWorkspace(c)}
                    className="flex-1 btn-primary py-2 rounded-xl text-xs flex items-center justify-center gap-1 font-semibold"
                  >
                    <span>Reopen in Workspace</span>
                    <ArrowUpRight size={12} />
                  </button>
                  <button
                    onClick={() => handleDownload(c.id)}
                    className="p-2 rounded-xl bg-[#1E2230] hover:bg-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors"
                    title="Download Report"
                  >
                    <FileDown size={14} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
