import { useState, useEffect, useRef, useCallback } from "react";
import { Send, Bot, User, Wifi, WifiOff, AlertTriangle, RefreshCw } from "lucide-react";
import { useAppStore, type ChatMessage } from "../store";
import { WS_URL } from "../constants";
import { motion, AnimatePresence } from "framer-motion";

// Simple ID generator
const genId = () => Math.random().toString(36).slice(2);

const DISCLAIMER = "⚠️ NeuroVQA is a research and clinical decision-support tool. All findings must be verified by a qualified radiologist. This system does not provide definitive diagnoses.";

export default function ChatPanel({ sessionId }: { sessionId?: number }) {
  const { token, currentDiagnosis, chatMessages, addChatMessage, updateLastMessage, isChatConnected, setIsChatConnected } = useAppStore();
  const [input, setInput] = useState("");
  const [isConnecting, setIsConnecting] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const scrollToBottom = useCallback(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, []);

  useEffect(() => { scrollToBottom(); }, [chatMessages, scrollToBottom]);

  const connect = useCallback(() => {
    if (!token || !sessionId) return;
    setIsConnecting(true);
    const ws = new WebSocket(`${WS_URL}/ws/chat?token=${token}&session_id=${sessionId}`);
    wsRef.current = ws;

    ws.onopen = () => { setIsChatConnected(true); setIsConnecting(false); };
    ws.onclose = () => { setIsChatConnected(false); setIsConnecting(false); };
    ws.onerror = () => { setIsChatConnected(false); setIsConnecting(false); };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === "token") {
          updateLastMessage(data.content, false);
        } else if (data.type === "done") {
          updateLastMessage("", true);
        } else if (data.type === "error") {
          updateLastMessage(`\n\n⚠️ ${data.message}`, true);
        }
      } catch {
        updateLastMessage(e.data, false);
      }
    };
  }, [token, sessionId, setIsChatConnected, updateLastMessage]);

  useEffect(() => {
    if (sessionId && token) connect();
    return () => wsRef.current?.close();
  }, [sessionId, token]);

  const sendMessage = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    setInput("");

    const userMsg: ChatMessage = { id: genId(), role: "user", content: text, timestamp: new Date() };
    addChatMessage(userMsg);

    const assistantMsg: ChatMessage = { id: genId(), role: "assistant", content: "", timestamp: new Date(), isStreaming: true };
    addChatMessage(assistantMsg);

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        message: text,
        diagnosis_context: currentDiagnosis ? {
          answer: currentDiagnosis.answer,
          confidence: currentDiagnosis.confidence,
          status: currentDiagnosis.status,
          top_3: currentDiagnosis.top_3_predictions,
        } : null,
      }));
    } else {
      // Fallback demo response
      const demoReply = currentDiagnosis
        ? `Based on the NeuroVQA analysis, the model classified this scan as **${currentDiagnosis.answer}** with ${(currentDiagnosis.confidence * 100).toFixed(1)}% confidence. The Grad-CAM attention map highlights the focal region supporting this prediction.\n\nAs a clinical decision-support tool, this finding should be correlated with the patient's clinical history, additional MRI sequences, and must be verified by a qualified radiologist before any clinical action is taken.`
        : "Please run the MRI diagnostic analysis first, then I can provide grounded explanations based on the VQA results.";
      
      let i = 0;
      const interval = setInterval(() => {
        if (i < demoReply.length) {
          updateLastMessage(demoReply[i], false);
          i++;
        } else {
          updateLastMessage("", true);
          clearInterval(interval);
        }
      }, 12);
    }
  }, [input, addChatMessage, updateLastMessage, currentDiagnosis, wsRef]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const QUICK_PROMPTS = [
    "Why do you think there is a lesion?",
    "What does the heatmap region indicate?",
    "How confident is the model, and why?",
    "Should this be urgent?",
  ];

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="flex items-center justify-between mb-3 px-1">
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-[#342CB3] to-[#4F46E5] flex items-center justify-center">
            <Bot size={12} className="text-white" />
          </div>
          <span className="text-xs font-bold text-white">Clinical AI Chat</span>
        </div>
        <div className="flex items-center gap-2">
          {isChatConnected ? (
            <span className="flex items-center gap-1 text-[10px] text-[#10B981]"><Wifi size={10} /> Live</span>
          ) : (
            <span className="flex items-center gap-1 text-[10px] text-[#64748B]"><WifiOff size={10} /> Demo Mode</span>
          )}
          {!isChatConnected && !isConnecting && sessionId && (
            <button onClick={connect} className="p-1 rounded hover:bg-[#1E2230] text-[#64748B]" title="Reconnect">
              <RefreshCw size={10} />
            </button>
          )}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="mb-2 px-2 py-1.5 rounded-lg bg-[#1A1019] border border-[#E11D48]/30 flex items-start gap-1.5">
        <AlertTriangle size={11} className="text-[#E11D48] mt-0.5 shrink-0" />
        <p className="text-[10px] text-[#94A3B8] leading-relaxed">{DISCLAIMER}</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 min-h-0">
        {chatMessages.length === 0 && (
          <div className="text-center py-8">
            <div className="w-12 h-12 rounded-full bg-[#1E2230] border border-[#2A2F42] flex items-center justify-center mx-auto mb-3">
              <Bot size={20} className="text-[#4F46E5]" />
            </div>
            <p className="text-[#64748B] text-xs">Run an MRI analysis, then ask follow-up questions here.</p>
            {currentDiagnosis && (
              <div className="mt-3 flex flex-wrap gap-1.5 justify-center">
                {QUICK_PROMPTS.map((q) => (
                  <button key={q} onClick={() => { setInput(q); inputRef.current?.focus(); }}
                    className="text-[10px] px-2 py-1 rounded-full bg-[#1E2230] border border-[#2A2F42] text-[#94A3B8] hover:border-[#4F46E5] hover:text-white transition-all">
                    {q}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        <AnimatePresence>
          {chatMessages.map((msg) => (
            <motion.div key={msg.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={`flex gap-2 ${msg.role === "user" ? "flex-row-reverse" : "flex-row"}`}>
              <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 ${msg.role === "user" ? "bg-[#342CB3]" : "bg-[#1E2230] border border-[#2A2F42]"}`}>
                {msg.role === "user" ? <User size={12} className="text-white" /> : <Bot size={12} className="text-[#4F46E5]" />}
              </div>
              <div className={`max-w-[85%] rounded-xl px-3 py-2 text-xs leading-relaxed ${msg.role === "user" ? "bg-[#342CB3] text-white rounded-tr-sm" : "bg-[#161922] border border-[#2A2F42] text-[#E2E8F0] rounded-tl-sm"}`}>
                {msg.content || (msg.isStreaming && (
                  <div className="typing-indicator flex gap-1">
                    <span /><span /><span />
                  </div>
                ))}
                {msg.isStreaming && msg.content && <span className="chat-cursor" />}
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="mt-3 relative">
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask a follow-up question..."
          rows={2}
          className="w-full bg-[#1E2230] border border-[#2A2F42] rounded-xl px-3 py-2 text-xs text-white placeholder-[#64748B] resize-none focus:outline-none focus:border-[#4F46E5] transition-colors pr-10"
        />
        <button
          onClick={sendMessage}
          disabled={!input.trim()}
          className="absolute right-2 bottom-2.5 p-1.5 rounded-lg bg-[#342CB3] hover:bg-[#4F46E5] disabled:opacity-40 disabled:cursor-not-allowed transition-all"
        >
          <Send size={12} className="text-white" />
        </button>
      </div>
    </div>
  );
}
