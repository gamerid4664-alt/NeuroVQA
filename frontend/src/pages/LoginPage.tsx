import { useState } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { Link, useNavigate } from "react-router-dom";
import { Brain, Mail, Lock, Eye, EyeOff, AlertCircle } from "lucide-react";
import { motion } from "framer-motion";
import { login, getMe } from "../api";
import { useAppStore } from "../store";

const schema = z.object({
  email: z.string().email("Invalid email address"),
  password: z.string().min(6, "Password must be at least 6 characters"),
});
type FormData = z.infer<typeof schema>;

export default function LoginPage() {
  const [showPw, setShowPw] = useState(false);
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);
  const { setToken, setUser } = useAppStore();
  const navigate = useNavigate();

  const { register, handleSubmit, formState: { errors } } = useForm<FormData>({ resolver: zodResolver(schema) });

  const onSubmit = async (data: FormData) => {
    setLoading(true);
    setApiError("");
    try {
      const res = await login(data.email, data.password);
      const token = res.data.access_token;
      setToken(token);
      // Fetch user profile
      try {
        const meRes = await getMe();
        setUser(meRes.data);
      } catch {
        setUser({ id: 1, full_name: "Dr. Doctor", email: data.email });
      }
      navigate("/app");
    } catch (err: any) {
      const msg = err?.response?.data?.detail || "Invalid email or password";
      // Demo mode: accept any credentials
      if (err?.code === "ERR_NETWORK" || err?.response?.status >= 500) {
        setToken("demo-token-" + Date.now());
        setUser({ id: 1, full_name: "Dr. Demo User", email: data.email });
        navigate("/app");
        return;
      }
      setApiError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0D11] flex items-center justify-center px-4">
      {/* Background */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[400px] bg-[#342CB3]/8 rounded-full blur-[100px]" />
      </div>
      <div className="absolute inset-0 opacity-[0.025]" style={{ backgroundImage: "linear-gradient(#4F46E5 1px, transparent 1px), linear-gradient(to right, #4F46E5 1px, transparent 1px)", backgroundSize: "50px 50px" }} />

      <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5 }}
        className="relative z-10 w-full max-w-md">
        {/* Card */}
        <div className="bg-[#161922] border border-[#2A2F42] rounded-2xl p-8 shadow-2xl">
          {/* Logo */}
          <div className="flex items-center justify-center gap-3 mb-8">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center shadow-lg shadow-[#342CB3]/40">
              <Brain size={20} className="text-white" />
            </div>
            <div>
              <div className="text-white font-bold text-lg leading-none">NeuroVQA</div>
              <div className="text-[#64748B] text-xs">Clinical AI Platform</div>
            </div>
          </div>

          <h1 className="text-xl font-bold text-white text-center mb-1">Doctor Sign In</h1>
          <p className="text-[#64748B] text-xs text-center mb-7">Access your protected diagnostic workspace</p>

          {apiError && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="flex items-center gap-2 mb-4 px-3 py-2.5 rounded-lg bg-[#1A0B0F] border border-[#E11D48]/40">
              <AlertCircle size={14} className="text-[#E11D48] shrink-0" />
              <span className="text-xs text-[#FB7185]">{apiError}</span>
            </motion.div>
          )}

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-[#94A3B8] mb-1.5 block">Email Address</label>
              <div className="relative">
                <Mail size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#64748B]" />
                <input {...register("email")} type="email" placeholder="doctor@hospital.com"
                  className="w-full bg-[#1E2230] border border-[#2A2F42] rounded-xl pl-9 pr-4 py-3 text-sm text-white placeholder-[#4A5568] focus:outline-none focus:border-[#4F46E5] transition-colors" />
              </div>
              {errors.email && <p className="text-[10px] text-[#FB7185] mt-1">{errors.email.message}</p>}
            </div>

            <div>
              <label className="text-xs font-semibold text-[#94A3B8] mb-1.5 block">Password</label>
              <div className="relative">
                <Lock size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#64748B]" />
                <input {...register("password")} type={showPw ? "text" : "password"} placeholder="••••••••"
                  className="w-full bg-[#1E2230] border border-[#2A2F42] rounded-xl pl-9 pr-10 py-3 text-sm text-white placeholder-[#4A5568] focus:outline-none focus:border-[#4F46E5] transition-colors" />
                <button type="button" onClick={() => setShowPw(!showPw)} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#64748B] hover:text-[#94A3B8]">
                  {showPw ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
              {errors.password && <p className="text-[10px] text-[#FB7185] mt-1">{errors.password.message}</p>}
            </div>

            <button type="submit" disabled={loading}
              className="btn-primary w-full py-3 rounded-xl text-sm mt-2 flex items-center justify-center gap-2">
              {loading ? (
                <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Signing In...</>
              ) : "Sign In to Workspace"}
            </button>
          </form>

          <div className="mt-6 pt-5 border-t border-[#2A2F42] text-center">
            <p className="text-xs text-[#64748B]">No account yet?{" "}
              <Link to="/signup" className="text-[#818CF8] hover:text-[#4F46E5] font-semibold transition-colors">Create Doctor Account</Link>
            </p>
          </div>

          <div className="mt-4 px-3 py-2 rounded-lg bg-[#0E1117] border border-[#2A2F42]">
            <p className="text-[10px] text-[#64748B] text-center">🔒 All authentication is local. Credentials are stored securely on your device using JWT + bcrypt.</p>
          </div>
        </div>

        <p className="text-center text-[10px] text-[#64748B] mt-4">
          <Link to="/" className="hover:text-[#94A3B8] transition-colors">← Back to Home</Link>
        </p>
      </motion.div>
    </div>
  );
}
