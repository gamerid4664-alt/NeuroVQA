import axios from "axios";
import { useAppStore } from "./store";

const api = axios.create({ baseURL: "http://localhost:8000" });

api.interceptors.request.use((config) => {
  const token = useAppStore.getState().token;
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response?.status === 401) {
      useAppStore.getState().logout();
    }
    return Promise.reject(err);
  }
);

export default api;

// ----- Auth -----
export const login = (email: string, password: string) =>
  api.post("/api/auth/login", { email, password });

export const signup = (full_name: string, email: string, password: string) =>
  api.post("/api/auth/signup", { full_name, email, password });

export const getMe = () => api.get("/api/auth/me");

// ----- VQA Predict -----
export const predictMRI = (formData: FormData) =>
  api.post("/api/predict", formData, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 30000,
  });

// ----- Dashboard -----
export const getDashboardStats = () => api.get("/api/dashboard/stats");

// ----- History -----
export const getHistory = (params?: { search?: string; result_type?: string; page?: number }) =>
  api.get("/api/history", { params });

export const getDiagnosis = (id: number | string) => api.get(`/api/diagnoses/${id}`);

// ----- Feedback -----
export const submitFeedback = (diagnosisId: number | string, feedback_type: string, doctor_note?: string) =>
  api.post("/api/feedback", { diagnosis_id: diagnosisId, feedback_type, doctor_note });

// ----- Reports -----
export const downloadReport = (diagnosisId: number | string) =>
  api.get(`/api/reports/${diagnosisId}/download`, { responseType: "blob" });

// ----- Audit -----
export const getAuditLogs = () => api.get("/api/audit/logs");
