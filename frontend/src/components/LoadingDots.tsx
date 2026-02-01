export function LoadingDots() {
    return (
        <div className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-[var(--muted)] animate-pulse" style={{ animationDelay: "0ms" }} />
            <span className="w-2 h-2 rounded-full bg-[var(--muted)] animate-pulse" style={{ animationDelay: "150ms" }} />
            <span className="w-2 h-2 rounded-full bg-[var(--muted)] animate-pulse" style={{ animationDelay: "300ms" }} />
        </div>
    );
}
