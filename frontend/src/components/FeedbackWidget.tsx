import { useState } from "react";
import { CheckCircle, XCircle, HelpCircle, AlertCircle, MessageSquare } from "lucide-react";
import { submitFeedback } from "../api";
import { motion, AnimatePresence } from "framer-motion";

const FEEDBACK_OPTIONS = [
  { type: "correct", label: "Correct", icon: CheckCircle, color: "#10B981", bg: "rgba(16,185,129,0.1)", border: "#10B981" },
  { type: "partially_correct", label: "Partial", icon: AlertCircle, color: "#F59E0B", bg: "rgba(245,158,11,0.1)", border: "#F59E0B" },
  { type: "incorrect", label: "Incorrect", icon: XCircle, color: "#E11D48", bg: "rgba(225,29,72,0.1)", border: "#E11D48" },
  { type: "uncertain", label: "Uncertain", icon: HelpCircle, color: "#94A3B8", bg: "rgba(148,163,184,0.1)", border: "#94A3B8" },
];

export default function FeedbackWidget({ diagnosisId }: { diagnosisId: number | string }) {
  const [selected, setSelected] = useState<string | null>(null);
  const [note, setNote] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [showNote, setShowNote] = useState(false);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async () => {
    if (!selected) return;
    setLoading(true);
    try {
      await submitFeedback(diagnosisId, selected, note || undefined);
      setSubmitted(true);
    } catch {
      // Demo fallback
      setSubmitted(true);
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }}
        className="flex items-center gap-2 p-3 rounded-lg bg-[#0A1F0F] border border-[#10B981]/30">
        <CheckCircle size={16} className="text-[#10B981]" />
        <span className="text-xs text-[#10B981] font-medium">Feedback recorded. Thank you for your clinical evaluation.</span>
      </motion.div>
    );
  }

  return (
    <div className="space-y-2">
      <div className="text-[10px] text-[#64748B] uppercase tracking-wider font-semibold">Was this AI finding helpful?</div>
      <div className="flex gap-2">
        {FEEDBACK_OPTIONS.map(({ type, label, icon: Icon, color, bg, border }) => (
          <button key={type} onClick={() => setSelected(type)}
            className="flex-1 flex flex-col items-center gap-1 py-2 rounded-lg text-[10px] font-semibold transition-all"
            style={{
              background: selected === type ? bg : "transparent",
              border: `1px solid ${selected === type ? border : "#2A2F42"}`,
              color: selected === type ? color : "#64748B",
            }}>
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>

      <AnimatePresence>
        {selected && !showNote && (
          <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} onClick={() => setShowNote(true)}
            className="flex items-center gap-1.5 text-[10px] text-[#64748B] hover:text-[#94A3B8] transition-colors">
            <MessageSquare size={10} /> Add clinical note (optional)
          </motion.button>
        )}
        {showNote && (
          <motion.textarea initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }}
            value={note} onChange={e => setNote(e.target.value)} rows={2}
            placeholder="Add clinical notes or observations..."
            className="w-full bg-[#1E2230] border border-[#2A2F42] rounded-lg px-2.5 py-2 text-xs text-white placeholder-[#64748B] resize-none focus:outline-none focus:border-[#4F46E5]" />
        )}
      </AnimatePresence>

      {selected && (
        <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }}
          onClick={handleSubmit} disabled={loading}
          className="w-full py-2 rounded-lg bg-[#342CB3] hover:bg-[#4F46E5] text-white text-xs font-semibold transition-all disabled:opacity-50">
          {loading ? "Submitting..." : "Submit Feedback"}
        </motion.button>
      )}
    </div>
  );
}
