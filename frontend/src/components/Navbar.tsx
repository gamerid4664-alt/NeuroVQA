import { Link, useNavigate, useLocation } from "react-router-dom";
import {
  Brain,
  LayoutDashboard,
  History,
  LogOut,
  Menu,
  X,
  Activity,
  ChevronRight,
} from "lucide-react";
import { useAppStore } from "../store";

const navLinks = [
  { to: "/app", label: "Workspace", icon: Brain },
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/history", label: "Case History", icon: History },
];

export default function Navbar() {
  const { user, logout, sidebarOpen, setSidebarOpen } = useAppStore();
  const navigate = useNavigate();
  const loc = useLocation();

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 h-14 flex items-center px-4 border-b border-[#2A2F42] bg-[#0B0D11]/95 backdrop-blur-md">
      {/* Logo */}
      <div className="flex items-center gap-2 min-w-[220px]">
        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="p-1.5 rounded-lg hover:bg-[#1E2230] transition-colors mr-1"
        >
          {sidebarOpen ? <X size={16} className="text-[#94A3B8]" /> : <Menu size={16} className="text-[#94A3B8]" />}
        </button>
        <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center">
          <Brain size={14} className="text-white" />
        </div>
        <span className="font-bold text-white text-sm tracking-tight">NeuroVQA</span>
        <span className="text-[#64748B] text-xs">• Clinical AI</span>
      </div>

      {/* Nav links */}
      <div className="flex items-center gap-1 flex-1">
        {navLinks.map(({ to, label, icon: Icon }) => {
          const active = loc.pathname === to;
          return (
            <Link
              key={to}
              to={to}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                active
                  ? "bg-[#1E2230] text-white border border-[#2A2F42]"
                  : "text-[#94A3B8] hover:text-white hover:bg-[#161922]"
              }`}
            >
              <Icon size={13} />
              {label}
              {active && <ChevronRight size={10} className="text-[#4F46E5]" />}
            </Link>
          );
        })}
      </div>

      {/* Status + User */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1.5 text-xs text-[#94A3B8] border border-[#2A2F42] rounded-full px-3 py-1 bg-[#0E1117]">
          <span className="status-dot scale-75" />
          <Activity size={11} className="text-[#10B981]" />
          <span className="text-[#10B981] font-medium">Offline Engine</span>
        </div>
        {user && (
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-full bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center text-xs font-bold text-white">
              {user.full_name?.[0]?.toUpperCase()}
            </div>
            <span className="text-xs text-[#94A3B8] hidden sm:block max-w-[120px] truncate">
              {user.full_name}
            </span>
            <button
              onClick={handleLogout}
              className="p-1.5 rounded-lg hover:bg-[#1E2230] transition-colors text-[#64748B] hover:text-[#E11D48]"
              title="Sign out"
            >
              <LogOut size={13} />
            </button>
          </div>
        )}
      </div>
    </nav>
  );
}
