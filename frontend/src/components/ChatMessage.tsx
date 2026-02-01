"use client";

import { useState, CSSProperties } from "react";

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

interface ChatMessageProps {
    message: Message;
    showDebug?: boolean;
    style?: CSSProperties;
}

export function ChatMessage({ message, showDebug, style }: ChatMessageProps) {
    const [showCitations, setShowCitations] = useState(false);
    const [showChunks, setShowChunks] = useState(false);

    const isUser = message.role === "user";

    // Calculate confidence score display
    const confidencePercent = message.confidence ? Math.round(message.confidence * 100) : 0;
    const confidenceLevel = confidencePercent >= 70 ? 'excellent' : confidencePercent >= 40 ? 'good' : 'warning';

    return (
        <div
            className={`flex items-start gap-3 animate-fade-in ${isUser ? "flex-row-reverse" : ""}`}
            style={style}
        >
            {/* Avatar */}
            <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${isUser ? "bg-[var(--accent)]" : "bg-[var(--foreground)]"
                }`}>
                {isUser ? (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
                    </svg>
                ) : (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                    </svg>
                )}
            </div>

            {/* Message Content */}
            <div className={`max-w-[80%] ${isUser ? "text-right" : ""}`}>
                <div className={`px-4 py-3 text-sm ${isUser ? "message-user" : "message-bot"}`}>
                    <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                </div>

                {/* Metadata for assistant messages */}
                {!isUser && (
                    <div className="mt-2 space-y-2">
                        {/* Confidence Score */}
                        {message.confidence !== undefined && message.confidence > 0 && (
                            <div className="card p-3">
                                <div className="score-label mb-1">
                                    <span className="flex items-center gap-1">
                                        <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                        </svg>
                                        Confidence Score
                                    </span>
                                    <span className={`score-value ${confidenceLevel}`}>{confidencePercent}%</span>
                                </div>
                                <div className="progress-bar">
                                    <div
                                        className={`progress-fill ${confidenceLevel}`}
                                        style={{ width: `${confidencePercent}%` }}
                                    />
                                </div>
                            </div>
                        )}

                        {/* Grounding indicator */}
                        {message.isGrounded !== undefined && (
                            <span className={`badge ${message.isGrounded ? 'badge-ready' : 'badge-error'}`}>
                                {message.isGrounded ? "Grounded" : "Not Grounded"}
                            </span>
                        )}

                        {/* Action buttons */}
                        <div className="flex flex-wrap gap-2">
                            {message.citations && message.citations.length > 0 && (
                                <button
                                    onClick={() => setShowCitations(!showCitations)}
                                    className="citation-badge"
                                >
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                                    </svg>
                                    {message.citations.length} Citation{message.citations.length > 1 ? "s" : ""}
                                </button>
                            )}

                            {showDebug && message.chunks && message.chunks.length > 0 && (
                                <button
                                    onClick={() => setShowChunks(!showChunks)}
                                    className="citation-badge"
                                >
                                    <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4" />
                                    </svg>
                                    {message.chunks.length} Chunks
                                </button>
                            )}
                        </div>
                    </div>
                )}

                {/* Citations Panel */}
                {showCitations && message.citations && (
                    <div className="mt-3 card animate-fade-in">
                        <div className="text-xs font-semibold text-[var(--muted)] uppercase tracking-wide mb-3">Sources</div>
                        <div className="space-y-2">
                            {message.citations.map((citation, idx) => (
                                <div key={idx} className="p-3 rounded border border-[var(--border)] bg-[var(--background)]">
                                    <div className="flex items-start justify-between gap-2 mb-1">
                                        <span className="text-sm font-medium">{citation.document_name}</span>
                                        {citation.page_number && (
                                            <span className="badge badge-processing text-xs">Page {citation.page_number}</span>
                                        )}
                                    </div>
                                    <p className="text-xs text-[var(--muted)] line-clamp-2">{citation.snippet}</p>
                                    <div className="mt-2 text-xs text-[var(--muted)]">
                                        Score: {(citation.relevance_score * 100).toFixed(0)}%
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Debug Chunks Panel */}
                {showChunks && message.chunks && (
                    <div className="mt-3 card bg-slate-900 text-slate-100 animate-fade-in">
                        <div className="text-xs font-semibold uppercase tracking-wide mb-3 text-slate-400">Retrieved Chunks</div>
                        <div className="space-y-2">
                            {message.chunks.map((chunk, idx) => (
                                <div key={idx} className="p-3 rounded bg-slate-800 border border-slate-700 text-xs font-mono">
                                    <div className="flex flex-wrap gap-1 mb-2">
                                        <span className="px-2 py-0.5 rounded bg-blue-900 text-blue-200">
                                            Vec: {chunk.vector_score?.toFixed(3) ?? "N/A"}
                                        </span>
                                        <span className="px-2 py-0.5 rounded bg-green-900 text-green-200">
                                            BM25: {chunk.bm25_score?.toFixed(3) ?? "N/A"}
                                        </span>
                                        <span className="px-2 py-0.5 rounded bg-purple-900 text-purple-200">
                                            Rerank: {chunk.rerank_score?.toFixed(3) ?? "N/A"}
                                        </span>
                                    </div>
                                    <div className="text-slate-400 text-xs mb-1">
                                        {chunk.document_name} {chunk.page_number && `• Page ${chunk.page_number}`}
                                    </div>
                                    <div className="text-slate-300 whitespace-pre-wrap max-h-24 overflow-y-auto">
                                        {chunk.content.slice(0, 200)}...
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Timestamp */}
                <div className={`mt-1 text-xs text-[var(--muted)] ${isUser ? "text-right" : ""}`}>
                    {message.timestamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
            </div>
        </div>
    );
}
