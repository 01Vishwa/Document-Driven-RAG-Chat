"use client";

import { useState, KeyboardEvent } from "react";

interface ChatInputProps {
    onSend: (message: string) => void;
    disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
    const [input, setInput] = useState("");

    const handleSubmit = () => {
        if (input.trim() && !disabled) {
            onSend(input.trim());
            setInput("");
        }
    };

    const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            handleSubmit();
        }
    };

    return (
        <div className="flex items-end gap-3">
            <div className="flex-1">
                <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask a question about aviation documents..."
                    disabled={disabled}
                    rows={1}
                    className="chat-input resize-none"
                    style={{ minHeight: "44px", maxHeight: "120px" }}
                />
            </div>
            <button
                onClick={handleSubmit}
                disabled={disabled || !input.trim()}
                className="btn btn-primary h-[44px] px-5"
            >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
                Send
            </button>
        </div>
    );
}
