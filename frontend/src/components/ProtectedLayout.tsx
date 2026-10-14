import { Navigate, Outlet } from "react-router-dom";
import { useAppStore } from "../store";
import Navbar from "./Navbar";

export default function ProtectedLayout() {
  const { token } = useAppStore();
  if (!token) return <Navigate to="/login" replace />;
  return (
    <div className="min-h-screen bg-[#0B0D11]">
      <Navbar />
      <div className="pt-14">
        <Outlet />
      </div>
    </div>
  );
}
