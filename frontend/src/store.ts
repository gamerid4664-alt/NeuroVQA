import { create } from "zustand";
import { persist } from "zustand/middleware";

export interface User {
  id: string | number;
  full_name: string;
  email: string;
}

export interface Diagnosis {
  id: string | number;
  question: string;
  answer: string;
  confidence: number;
  top_3_predictions: Array<{ answer: string; confidence: number }>;
  status: string;
  latency_seconds: number;
  mri_url?: string;
  heatmap_url?: string;
  slice_mode?: string;
  created_at: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
}

interface AppState {
  user: User | null;
  token: string | null;
  currentDiagnosis: Diagnosis | null;
  chatMessages: ChatMessage[];
  isAnalyzing: boolean;
  isChatConnected: boolean;
  sidebarOpen: boolean;

  setUser: (user: User | null) => void;
  setToken: (token: string | null) => void;
  setCurrentDiagnosis: (d: Diagnosis | null) => void;
  addChatMessage: (msg: ChatMessage) => void;
  updateLastMessage: (content: string, done?: boolean) => void;
  clearChat: () => void;
  setIsAnalyzing: (v: boolean) => void;
  setIsChatConnected: (v: boolean) => void;
  setSidebarOpen: (v: boolean) => void;
  logout: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      currentDiagnosis: null,
      chatMessages: [],
      isAnalyzing: false,
      isChatConnected: false,
      sidebarOpen: true,

      setUser: (user) => set({ user }),
      setToken: (token) => set({ token }),
      setCurrentDiagnosis: (d) => set({ currentDiagnosis: d }),
      addChatMessage: (msg) =>
        set((s) => ({ chatMessages: [...s.chatMessages, msg] })),
      updateLastMessage: (content, done = false) =>
        set((s) => {
          const msgs = [...s.chatMessages];
          if (msgs.length > 0) {
            const last = msgs[msgs.length - 1];
            msgs[msgs.length - 1] = {
              ...last,
              content: last.content + content,
              isStreaming: !done,
            };
          }
          return { chatMessages: msgs };
        }),
      clearChat: () => set({ chatMessages: [] }),
      setIsAnalyzing: (v) => set({ isAnalyzing: v }),
      setIsChatConnected: (v) => set({ isChatConnected: v }),
      setSidebarOpen: (v) => set({ sidebarOpen: v }),
      logout: () => set({ user: null, token: null, currentDiagnosis: null, chatMessages: [] }),
    }),
    {
      name: "neurovqa-store",
      partialize: (state) => ({ user: state.user, token: state.token }),
    }
  )
);
