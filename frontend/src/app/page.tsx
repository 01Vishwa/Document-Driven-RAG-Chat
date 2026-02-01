"use client";

import { useState, useRef, useEffect } from "react";
import { Sidebar } from "@/components/Sidebar";
import { ChatMessage } from "@/components/ChatMessage";
import { ChatInput } from "@/components/ChatInput";
import { LoadingDots } from "@/components/LoadingDots";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  chunks?: RetrievedChunk[];
  isGrounded?: boolean;
  confidence?: number;
  timestamp: Date;
}

interface Citation {
  document_name: string;
  page_number: number | null;
  chunk_id: string;
  relevance_score: number;
  snippet: string;
}

interface RetrievedChunk {
  chunk_id: string;
  content: string;
  document_name: string;
  page_number: number | null;
  vector_score: number | null;
  bm25_score: number | null;
  rerank_score: number | null;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isIndexLoaded, setIsIndexLoaded] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    checkHealth();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      const data = await res.json();
      setIsIndexLoaded(data.index_loaded);
    } catch (error) {
      console.error("Health check failed:", error);
    }
  };

  const handleSend = async (content: string) => {
    if (!content.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const res = await fetch(`${API_BASE_URL}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: content,
          debug: false,
          top_k: 5,
        }),
      });

      if (!res.ok) throw new Error(`API error: ${res.status}`);

      const data = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        citations: data.citations,
        chunks: data.retrieved_chunks,
        isGrounded: data.is_grounded,
        confidence: data.confidence_score,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I couldn't process your request. Please make sure the backend is running and documents are indexed.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <div className="flex h-screen bg-[var(--background)]">
      <Sidebar onClear={clearChat} />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <header className="header-bar bg-[var(--background)] border-b border-[var(--border)] px-6 py-4 flex items-center justify-between">
          <div>
            <h1 className="header-title text-lg font-semibold text-[var(--foreground)]">Aviation Chat</h1>
            <p className="header-subtitle text-sm text-[var(--muted)]">Ask questions about aviation documents</p>
          </div>
          <div className="flex items-center gap-3">
            {/* Search bar removed */}
            <span className={`badge px-2 py-1 text-xs rounded-full border ${isIndexLoaded ? 'bg-green-100 text-green-700 border-green-200' : 'bg-yellow-100 text-yellow-700 border-yellow-200'}`}>
              {isIndexLoaded ? "System Ready" : "Index Not Loaded"}
            </span>
          </div>
        </header>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-lg mx-auto animate-fade-in">
              <div className="w-16 h-16 mb-4 rounded-xl bg-[var(--foreground)] flex items-center justify-center shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold mb-2 text-[var(--foreground)]">Welcome to AviationAI</h2>
              <p className="text-[var(--muted)] text-base mb-8">
                Ask questions about aviation documents. Answers are strictly grounded in indexed content.
              </p>

              {/* Quick questions */}
              <div className="grid grid-cols-1 gap-3 w-full">
                {["What is a VOR?", "Define true airspeed", "Explain density altitude"].map((q) => (
                  <button
                    key={q}
                    onClick={() => handleSend(q)}
                    className="card text-left p-3 rounded-lg border border-[var(--border)] hover:border-[var(--accent)] hover:bg-[var(--sidebar-hover)] transition-all duration-200 text-sm font-medium text-[var(--foreground)] flex items-center gap-3"
                  >
                    <svg className="w-4 h-4 text-[var(--muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  showDebug={false}
                />
              ))}
              {isLoading && (
                <div className="flex items-start gap-4 animate-fade-in">
                  <div className="w-8 h-8 rounded-full bg-[var(--foreground)] flex items-center justify-center shrink-0">
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                  </div>
                  <div className="bg-[var(--sidebar-bg)] border border-[var(--border)] rounded-2xl rounded-tl-none px-4 py-3 shadow-sm">
                    <LoadingDots />
                  </div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Input */}
        <div className="border-t border-[var(--border)] bg-[var(--card-bg)] px-6 py-4">
          <div className="max-w-3xl mx-auto">
            <ChatInput onSend={handleSend} disabled={isLoading} />
          </div>
        </div>
      </main>
    </div>
  );
}
