import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  Brain,
  Zap,
  Shield,
  MessageSquare,
  BarChart3,
  FileText,
  ArrowRight,
  Lock,
  Cpu,
  GitBranch,
} from "lucide-react";

const features = [
  { icon: Brain, title: "MRI Visual Q&A", desc: "BiomedCLIP ViT-B/16 + PubMedBERT cross-attention fusion answers clinical questions grounded in scan evidence.", color: "#4F46E5" },
  { icon: Zap, title: "Grad-CAM Explainability", desc: "14×14 spatial attention rollout bilinear-upsampled to 224×224 overlay reveals the exact anatomical focus.", color: "#10B981" },
  { icon: MessageSquare, title: "Real-Time Chat", desc: "Ollama-powered streaming assistant explains VQA results and answers follow-up questions without internet.", color: "#F59E0B" },
  { icon: BarChart3, title: "Analytics Dashboard", desc: "Case statistics, confidence trends, latency metrics, and doctor-feedback summaries at a glance.", color: "#818CF8" },
  { icon: FileText, title: "PDF Clinical Reports", desc: "Downloadable structured reports with MRI scan, heatmap, predictions, confidence, and medical disclaimer.", color: "#38BDF8" },
  { icon: Shield, title: "100% Offline & Private", desc: "All inference, chat, and storage runs locally on your laptop. No patient data ever leaves the device.", color: "#34D399" },
];

const specs = [
  { label: "VQA Latency", value: "≤ 2.5s", sub: "CPU inference" },
  { label: "Checkpoint", value: "~50 MB", sub: "LoRA adapter" },
  { label: "RAM Usage", value: "< 2.5 GB", sub: "Intel i3 OK" },
  { label: "Accuracy", value: "≥ 72%", sub: "VQA-RAD closed Q" },
];

const pipeline = [
  { step: "01", title: "MRI Ingestion", desc: "PNG / JPEG / NIfTI 2.5D axial slice" },
  { step: "02", title: "Dual Encoder", desc: "BiomedCLIP vision + PubMedBERT text" },
  { step: "03", title: "Cross-Attention", desc: "Text queries interrogate 196 patch tokens" },
  { step: "04", title: "VQA Answer", desc: "Top-1 + top-3 predictions with confidence" },
  { step: "05", title: "Grad-CAM", desc: "14×14 heatmap upsampled to 224×224" },
  { step: "06", title: "Chat + Report", desc: "Ollama explanation + PDF download" },
];

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0B0D11] text-white overflow-x-hidden">
      {/* Nav */}
      <header className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center justify-between px-8 border-b border-[#2A2F42] bg-[#0B0D11]/90 backdrop-blur-md">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center shadow-lg">
            <Brain size={16} className="text-white" />
          </div>
          <span className="font-bold text-lg tracking-tight">NeuroVQA</span>
          <span className="text-[#64748B] text-xs ml-1">Clinical AI</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/login" className="text-sm text-[#94A3B8] hover:text-white transition-colors">Sign In</Link>
          <Link to="/signup" className="btn-primary text-sm px-5 py-2 rounded-lg">Get Started</Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative pt-32 pb-24 px-8 text-center overflow-hidden">
        {/* Background glow */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[400px] bg-[#342CB3]/10 rounded-full blur-[120px]" />
          <div className="absolute top-1/2 left-1/4 w-[300px] h-[300px] bg-[#4F46E5]/6 rounded-full blur-[80px]" />
        </div>
        {/* Grid pattern */}
        <div className="absolute inset-0 opacity-[0.03]" style={{ backgroundImage: "linear-gradient(#4F46E5 1px, transparent 1px), linear-gradient(to right, #4F46E5 1px, transparent 1px)", backgroundSize: "60px 60px" }} />

        <motion.div initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7 }} className="relative z-10 max-w-4xl mx-auto">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-[#161922] border border-[#2A2F42] text-xs text-[#94A3B8] mb-8">
            <span className="status-dot" />
            <span>100% Offline · Intel i3 Compatible · No Cloud Dependency</span>
          </div>

          <h1 className="text-5xl sm:text-6xl font-extrabold leading-tight mb-6" style={{ background: "linear-gradient(135deg, #FFFFFF 0%, #818CF8 50%, #4F46E5 100%)", WebkitBackgroundClip: "text", WebkitTextFillColor: "transparent" }}>
            MRI-Grounded<br />Clinical Decision AI
          </h1>

          <p className="text-lg text-[#94A3B8] max-w-2xl mx-auto mb-10 leading-relaxed">
            NeuroVQA answers diagnostic questions about brain MRI scans using BiomedCLIP + PubMedBERT cross-attention, generates Grad-CAM visual evidence, and provides grounded real-time chat — all completely offline.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <Link to="/signup" className="btn-primary flex items-center gap-2 px-8 py-3 text-base rounded-xl">
              Launch Workstation <ArrowRight size={16} />
            </Link>
            <Link to="/login" className="btn-secondary flex items-center gap-2 px-8 py-3 text-base rounded-xl">
              Sign In <Lock size={14} />
            </Link>
          </div>
        </motion.div>

        {/* Specs row */}
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4, duration: 0.6 }}
          className="relative z-10 mt-16 grid grid-cols-2 sm:grid-cols-4 gap-4 max-w-2xl mx-auto">
          {specs.map(({ label, value, sub }) => (
            <div key={label} className="bg-[#161922] border border-[#2A2F42] rounded-xl p-4 text-center">
              <div className="text-2xl font-black text-[#4F46E5]">{value}</div>
              <div className="text-xs font-semibold text-white mt-0.5">{label}</div>
              <div className="text-[10px] text-[#64748B]">{sub}</div>
            </div>
          ))}
        </motion.div>
      </section>

      {/* Features */}
      <section className="py-20 px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold text-white mb-3">Complete Clinical AI Toolkit</h2>
            <p className="text-[#94A3B8] text-base">From MRI upload to downloadable PDF report — entirely on your laptop.</p>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map(({ icon: Icon, title, desc, color }, i) => (
              <motion.div key={title} initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * 0.08 }}
                className="bg-[#161922] border border-[#2A2F42] rounded-2xl p-6 hover:border-[#342CB3] transition-all group">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4" style={{ background: `${color}18`, border: `1px solid ${color}40` }}>
                  <Icon size={18} style={{ color }} />
                </div>
                <h3 className="text-white font-bold mb-2 text-sm">{title}</h3>
                <p className="text-[#64748B] text-xs leading-relaxed">{desc}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Pipeline */}
      <section className="py-20 px-8 bg-[#0E1117]">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-14">
            <h2 className="text-3xl font-bold text-white mb-3">VQA Pipeline Architecture</h2>
            <p className="text-[#94A3B8] text-sm">Cloud-trained LoRA adapter · Local CPU inference · Zero network calls</p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            {pipeline.map(({ step, title, desc }, i) => (
              <motion.div key={step} initial={{ opacity: 0, scale: 0.9 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.08 }}
                className="relative bg-[#161922] border border-[#2A2F42] rounded-xl p-4 text-center">
                <div className="text-[10px] font-black text-[#4F46E5] mb-2">{step}</div>
                <div className="text-xs font-bold text-white mb-1">{title}</div>
                <div className="text-[10px] text-[#64748B] leading-relaxed">{desc}</div>
                {i < pipeline.length - 1 && (
                  <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10">
                    <ArrowRight size={12} className="text-[#2A2F42]" />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Tech stack */}
      <section className="py-16 px-8">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-10">
            <h2 className="text-2xl font-bold text-white mb-2">Technology Stack</h2>
            <p className="text-[#64748B] text-sm">Production-grade components for offline clinical AI</p>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[
              ["React 18 + TypeScript", "Frontend"],
              ["Tailwind CSS v4", "Styling"],
              ["FastAPI + Uvicorn", "Backend API"],
              ["SQLite + SQLAlchemy", "Database"],
              ["PyTorch CPU + LoRA", "VQA Engine"],
              ["Ollama llama3.2:1b", "Chat LLM"],
              ["BiomedCLIP ViT-B/16", "Vision Backbone"],
              ["PubMedBERT", "Text Backbone"],
              ["ReportLab", "PDF Reports"],
            ].map(([name, category]) => (
              <div key={name} className="flex items-center gap-3 bg-[#161922] border border-[#2A2F42] rounded-xl px-4 py-3">
                <Cpu size={14} className="text-[#4F46E5] shrink-0" />
                <div>
                  <div className="text-xs font-semibold text-white">{name}</div>
                  <div className="text-[10px] text-[#64748B]">{category}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-8 text-center">
        <div className="max-w-2xl mx-auto bg-gradient-to-br from-[#161922] to-[#1A1B2E] border border-[#342CB3]/40 rounded-3xl p-12">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center mx-auto mb-6 shadow-lg shadow-[#342CB3]/30">
            <Brain size={28} className="text-white" />
          </div>
          <h2 className="text-3xl font-bold text-white mb-4">Ready for Offline Deployment?</h2>
          <p className="text-[#94A3B8] mb-8 text-sm leading-relaxed">Sign up in seconds. Upload your first MRI scan, run VQA analysis, and get Grad-CAM evidence — no internet required after initial setup.</p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link to="/signup" className="btn-primary flex items-center justify-center gap-2 px-8 py-3 text-base rounded-xl">
              Create Doctor Account <ArrowRight size={16} />
            </Link>
            <Link to="/login" className="btn-secondary flex items-center justify-center gap-2 px-8 py-3 text-base rounded-xl">
              <GitBranch size={14} /> View Demo
            </Link>
          </div>
          <p className="text-[10px] text-[#64748B] mt-6">NeuroVQA is a research and clinical decision-support tool. All findings must be verified by a qualified radiologist.</p>
        </div>
      </section>

      <footer className="border-t border-[#2A2F42] py-8 px-8 text-center text-xs text-[#64748B]">
        <div className="flex items-center justify-center gap-2 mb-2">
          <Brain size={12} className="text-[#4F46E5]" />
          <span className="font-semibold text-[#94A3B8]">NeuroVQA</span>
        </div>
        <p>Clinical AI Decision Support · Cloud Training + Local Inference · 100% Offline After Setup</p>
      </footer>
    </div>
  );
}
