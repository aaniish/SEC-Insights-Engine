"use client";

import { ArrowUpRight } from "lucide-react";
import { createContext, type ReactNode, useContext, useMemo, useState } from "react";
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card";
import type { Citation } from "@/lib/ai/tools";
import { cn } from "@/lib/utils";

interface CitationState {
  turnId: string;
  citations: Map<number, Citation>;
  active: number | null;
  setActive: (n: number | null) => void;
}

const CitationContext = createContext<CitationState | null>(null);

export function CitationProvider({
  turnId,
  citations,
  children,
}: {
  turnId: string;
  citations: Map<number, Citation>;
  children: ReactNode;
}) {
  const [active, setActive] = useState<number | null>(null);
  const value = useMemo(() => ({ turnId, citations, active, setActive }), [turnId, citations, active]);
  return <CitationContext.Provider value={value}>{children}</CitationContext.Provider>;
}

function useCitations() {
  const context = useContext(CitationContext);
  if (!context) throw new Error("Citation components must be inside CitationProvider");
  return context;
}

export const noteId = (turnId: string, n: number) => `note-${turnId}-${n}`;

export function sourceLabel(c: Citation) {
  return `${c.ticker} · ${c.form} ${c.period}`;
}

/** Inline "[n]" marker. Hover previews the passage; click jumps to it in the margin. */
export function CitationMarker({ n }: { n: number }) {
  const { turnId, citations, active, setActive } = useCitations();
  const citation = citations.get(n);
  if (!citation) return <sup>[{n}]</sup>;

  const jumpToNote = () => {
    const note = document.getElementById(noteId(turnId, n));
    if (!note || note.offsetParent === null) return; // margin hidden on small screens
    note.scrollIntoView({ block: "nearest", behavior: "smooth" });
    note.classList.remove("cite-flash");
    void note.offsetWidth;
    note.classList.add("cite-flash");
  };

  return (
    <HoverCard openDelay={120} closeDelay={80}>
      <HoverCardTrigger asChild>
        <a
          href={citation.url}
          target="_blank"
          rel="noreferrer"
          onMouseEnter={() => setActive(n)}
          onMouseLeave={() => setActive(null)}
          onFocus={() => setActive(n)}
          onBlur={() => setActive(null)}
          onClick={(event) => {
            if (window.matchMedia("(min-width: 1024px)").matches) {
              event.preventDefault();
              jumpToNote();
            }
          }}
          aria-label={`Source ${n}: ${sourceLabel(citation)}, ${citation.section}`}
          className={cn(
            "mx-px inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-[0.1em] items-center justify-center rounded-[4px] border border-amber/40 px-1 align-middle font-mono text-[0.66rem] font-medium text-amber-ink no-underline transition-colors hover:bg-amber/15",
            active === n && "bg-amber/20",
          )}
        >
          {n}
        </a>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="w-80 p-0">
        <SourceBody citation={citation} clamp="line-clamp-6" />
      </HoverCardContent>
    </HoverCard>
  );
}

function SourceBody({ citation, clamp }: { citation: Citation; clamp: string }) {
  return (
    <div className="space-y-1.5 p-3">
      <div className="flex items-baseline justify-between gap-2 font-mono text-[0.68rem] text-graphite">
        <span className="text-foreground">{sourceLabel(citation)}</span>
        <span>filed {citation.filedAt}</span>
      </div>
      <div className="text-xs font-medium">{citation.section}</div>
      <p className={cn("font-serif text-[0.84rem] leading-snug text-foreground/85", clamp)}>
        {citation.text}
      </p>
      <a
        href={citation.url}
        target="_blank"
        rel="noreferrer"
        className="inline-flex items-center gap-1 text-xs font-medium text-amber-ink hover:underline"
      >
        Read in the filing on sec.gov <ArrowUpRight className="size-3" />
      </a>
    </div>
  );
}

/** Margin notes: the cited passages beside the answer, like an annotated filing. */
export function SourceNotes({
  order,
  className,
  anchored = false,
}: {
  order: number[];
  className?: string;
  /** Only the desktop margin gets element ids, so markers can scroll to it. */
  anchored?: boolean;
}) {
  const { turnId, citations, active, setActive } = useCitations();
  const cited = order.filter((n) => citations.has(n));
  const [showAll, setShowAll] = useState(false);
  const uncited = [...citations.keys()].filter((n) => !cited.includes(n));
  const visible = showAll ? [...cited, ...uncited] : cited;
  if (citations.size === 0) return null;

  return (
    <aside className={className} aria-label="Sources">
      <div className="mb-2 flex items-baseline justify-between font-mono text-[0.68rem] text-graphite uppercase tracking-wider">
        <span>Sources</span>
        <span>
          {cited.length} cited · {citations.size} read
        </span>
      </div>
      <ol className="space-y-1">
        {visible.map((n) => {
          const citation = citations.get(n) as Citation;
          return (
            <li
              key={n}
              id={anchored ? noteId(turnId, n) : undefined}
              onMouseEnter={() => setActive(n)}
              onMouseLeave={() => setActive(null)}
              className={cn(
                "rounded-md border-l-2 border-transparent transition-colors",
                active === n && "border-amber bg-card",
                !cited.includes(n) && "opacity-70",
              )}
            >
              <a href={citation.url} target="_blank" rel="noreferrer" className="block px-2.5 py-2">
                <div className="flex items-baseline gap-2 font-mono text-[0.68rem]">
                  <span className="text-amber-ink">[{n}]</span>
                  <span className="text-foreground">{sourceLabel(citation)}</span>
                </div>
                <div className="mt-0.5 text-[0.72rem] text-graphite">{citation.section}</div>
                <p className="mt-1 line-clamp-3 font-serif text-[0.8rem] leading-snug text-foreground/80">
                  {citation.text}
                </p>
              </a>
            </li>
          );
        })}
      </ol>
      {uncited.length > 0 && (
        <button
          type="button"
          onClick={() => setShowAll((v) => !v)}
          className="mt-2 px-2.5 font-mono text-[0.68rem] text-graphite hover:text-foreground"
        >
          {showAll ? "Hide passages that weren’t cited" : `+ ${uncited.length} more passages read`}
        </button>
      )}
    </aside>
  );
}
