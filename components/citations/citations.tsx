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

const cardId = (turnId: string, n: number) => `source-${turnId}-${n}`;

export function sourceLabel(c: Citation) {
  return `${c.ticker} · ${c.form} ${c.period}`;
}

/** Inline numbered pill. Hover previews the passage; click jumps to its source card. */
export function CitationMarker({ n }: { n: number }) {
  const { turnId, citations, active, setActive } = useCitations();
  const citation = citations.get(n);
  if (!citation) return <sup>[{n}]</sup>;

  const jumpToCard = () => {
    const card = document.getElementById(cardId(turnId, n));
    if (!card) return;
    card.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
    card.classList.remove("cite-flash");
    void card.offsetWidth;
    card.classList.add("cite-flash");
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
            if (window.matchMedia("(pointer: fine)").matches) {
              event.preventDefault();
              jumpToCard();
            }
          }}
          aria-label={`Source ${n}: ${sourceLabel(citation)}, ${citation.section}`}
          className={cn(
            "mx-[0.15em] inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-[0.12em] items-center justify-center rounded-full bg-muted px-1.5 align-middle text-[0.66rem] font-semibold text-graphite no-underline tabular transition-colors hover:bg-navy hover:text-white",
            active === n && "bg-navy text-white",
          )}
        >
          {n}
        </a>
      </HoverCardTrigger>
      <HoverCardContent side="top" className="glass glass-dense w-80 rounded-2xl border-0 p-0 ring-0">
        <div className="space-y-1.5 p-4">
          <div className="flex items-baseline justify-between gap-2 text-[0.72rem] text-graphite">
            <span className="font-semibold text-ink">{sourceLabel(citation)}</span>
            <span>Filed {citation.filedAt}</span>
          </div>
          <div className="text-xs font-medium text-navy-ink">{citation.section}</div>
          <p className="line-clamp-6 text-[0.84rem] leading-snug text-ink/85">{citation.text}</p>
          <a
            href={citation.url}
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1 pt-1 text-xs font-medium text-navy-ink hover:underline"
          >
            Read it in the filing on sec.gov <ArrowUpRight className="size-3" />
          </a>
        </div>
      </HoverCardContent>
    </HoverCard>
  );
}

/** Source cards above the answer: cited passages first, the rest on request. */
export function SourceStrip({ order }: { order: number[] }) {
  const { turnId, citations, active, setActive } = useCitations();
  const [showAll, setShowAll] = useState(false);
  const cited = order.filter((n) => citations.has(n));
  const uncited = [...citations.keys()].filter((n) => !cited.includes(n));
  const visible = showAll ? [...cited, ...uncited] : cited;
  if (cited.length === 0) return null;

  return (
    <section aria-label="Sources">
      <div className="mb-2 flex items-baseline gap-2 text-sm font-medium">
        Sources
        <span className="text-xs font-normal text-graphite">
          {cited.length} cited · {citations.size} read
        </span>
      </div>
      <ol className="scroll-fade-x -mx-4 flex snap-x scroll-px-4 gap-2 overflow-x-auto px-4 pt-1 pb-3">
        {visible.map((n) => {
          const citation = citations.get(n) as Citation;
          return (
            <li
              key={n}
              id={cardId(turnId, n)}
              onMouseEnter={() => setActive(n)}
              onMouseLeave={() => setActive(null)}
              className={cn(
                "glass w-60 shrink-0 snap-start rounded-2xl transition-transform",
                active === n &&
                  "-translate-y-0.5 shadow-[inset_0_1px_0_var(--glass-highlight),0_0_0_1.5px_var(--ring),var(--glass-shadow)]",
                !cited.includes(n) && "opacity-75",
              )}
            >
              <a
                href={citation.url}
                target="_blank"
                rel="noreferrer"
                className="flex h-full flex-col gap-1 p-3"
              >
                <div className="flex items-center gap-2 text-[0.7rem] text-graphite">
                  <span className="grid size-[1.1rem] shrink-0 place-items-center rounded-full bg-muted text-[0.62rem] font-semibold tabular">
                    {n}
                  </span>
                  <span className="truncate font-medium text-ink">{sourceLabel(citation)}</span>
                </div>
                <div className="line-clamp-1 text-xs font-medium text-navy-ink">{citation.section}</div>
                <p className="line-clamp-2 text-[0.78rem] leading-snug text-graphite">{citation.text}</p>
              </a>
            </li>
          );
        })}
        {uncited.length > 0 && (
          <li className="shrink-0">
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="glass flex h-full w-28 flex-col items-start justify-center gap-0.5 rounded-2xl p-3 text-left text-xs text-graphite hover:text-ink"
            >
              <span className="text-base font-semibold text-ink">{showAll ? "−" : `+${uncited.length}`}</span>
              {showAll ? "Show cited only" : "more passages read"}
            </button>
          </li>
        )}
      </ol>
    </section>
  );
}
