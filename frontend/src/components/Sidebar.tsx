"use client";

interface SidebarProps {
    onClear: () => void;
    onIngest: () => void;
    isIndexLoaded: boolean;
    showDebug: boolean;
    onToggleDebug: () => void;
}

export function Sidebar({
    onClear,
}: { onClear: () => void }) {
    return (
        <aside className="sidebar w-60 flex flex-col h-full bg-[var(--sidebar-bg)] border-r border-[var(--border)]">
            {/* Logo */}
            <div className="p-4 border-b border-[var(--border)]">
                <div className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded bg-[var(--foreground)] flex items-center justify-center">
                        <svg
                            className="w-5 h-5 text-white"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                            />
                        </svg>
                    </div>
                    <span className="font-semibold text-[var(--foreground)]">AviationAI</span>
                </div>
            </div>

            {/* Main Navigation */}
            <nav className="flex-1 py-4">
                <div className="sidebar-section px-4 mb-2 text-xs font-semibold text-[var(--muted)] uppercase tracking-wider">Main</div>

                <div className="px-3 space-y-1">
                    <button
                        onClick={onClear}
                        className="sidebar-item active w-full text-left flex items-center gap-3 px-3 py-2 rounded-lg transition-colors hover:bg-[var(--sidebar-hover)] text-[var(--foreground)]"
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                        </svg>
                        Chat
                    </button>
                </div>
            </nav>
        </aside>
    );
}
