"use client";

import { diffWords } from "diff";
import { ArrowUpRight } from "lucide-react";
import { useMemo, useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { FilingDiff } from "@/lib/diff/compare";
import { cn } from "@/lib/utils";

const PREVIEW = 4;

const KIND_LABEL = { added: "New", removed: "Removed", modified: "Reworded" } as const;

function Redline({ before, after }: { before: string; after: string }) {
  const parts = useMemo(() => diffWords(before, after), [before, after]);
  return (
    <p className="font-serif text-[0.9rem] leading-relaxed">
      {parts.map((part, i) =>
        part.added ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: diff parts are positional
          <ins key={i} className="bg-insert-wash text-insert decoration-insert/60 underline-offset-2">
            {part.value}
          </ins>
        ) : part.removed ? (
          // biome-ignore lint/suspicious/noArrayIndexKey: diff parts are positional
          <del key={i} className="bg-redline-wash text-redline decoration-redline/70">
            {part.value}
          </del>
        ) : (
          // biome-ignore lint/suspicious/noArrayIndexKey: diff parts are positional
          <span key={i}>{part.value}</span>
        ),
      )}
    </p>
  );
}

function Expandable<T>({
  items,
  itemKey,
  render,
}: {
  items: T[];
  itemKey: (item: T) => string;
  render: (item: T) => React.ReactNode;
}) {
  const [all, setAll] = useState(false);
  if (items.length === 0) return <p className="py-3 text-sm text-graphite">None.</p>;
  const shown = all ? items : items.slice(0, PREVIEW);
  return (
    <div>
      <ol className="divide-y divide-rule/70">
        {shown.map((item) => (
          <li key={itemKey(item)} className="py-3">
            {render(item)}
          </li>
        ))}
      </ol>
      {items.length > PREVIEW && (
        <button
          type="button"
          onClick={() => setAll((v) => !v)}
          className="font-mono text-[0.7rem] text-graphite hover:text-foreground"
        >
          {all ? "Show fewer" : `Show all ${items.length}`}
        </button>
      )}
    </div>
  );
}

/** "What changed" between two annual reports, rendered as a legal redline. */
export function FilingDiffCard({ diff }: { diff: FilingDiff }) {
  const { stats } = diff;
  const total = stats.unchanged + stats.modified + stats.added + stats.removed;
  const changedShare =
    total > 0 ? Math.round(((stats.modified + stats.added + stats.removed) / total) * 100) : 0;

  return (
    <section
      className="overflow-hidden rounded-lg border bg-card"
      aria-label={`${diff.sectionTitle} changes`}
    >
      <header className="border-b px-4 py-3">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="text-sm font-semibold">
            {diff.sectionTitle} · {diff.ticker}
          </h3>
          <div className="flex gap-3 font-mono text-[0.68rem] text-graphite">
            <a href={diff.from.url} target="_blank" rel="noreferrer" className="hover:text-foreground">
              FY{diff.from.fiscalYear} 10-K <ArrowUpRight className="inline size-3" />
            </a>
            <span aria-hidden="true">→</span>
            <a href={diff.to.url} target="_blank" rel="noreferrer" className="hover:text-foreground">
              FY{diff.to.fiscalYear} 10-K <ArrowUpRight className="inline size-3" />
            </a>
          </div>
        </div>
        <p className="mt-1.5 font-serif text-[0.95rem] leading-snug">{diff.headline}</p>
        <dl className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[0.7rem] text-graphite">
          <div>
            <dt className="sr-only">New paragraphs</dt>
            <dd>
              <span className="text-insert">+{stats.added}</span> new
            </dd>
          </div>
          <div>
            <dt className="sr-only">Removed paragraphs</dt>
            <dd>
              <span className="text-redline">−{stats.removed}</span> removed
            </dd>
          </div>
          <div>
            <dt className="sr-only">Reworded paragraphs</dt>
            <dd>~{stats.modified} reworded</dd>
          </div>
          <div>
            <dt className="sr-only">Unchanged paragraphs</dt>
            <dd>
              {stats.unchanged} unchanged · {changedShare}% of paragraphs changed
            </dd>
          </div>
        </dl>
      </header>

      {diff.highlights.length > 0 && (
        <ul className="divide-y divide-rule/70 border-b">
          {diff.highlights.map((h) => (
            <li key={h.title} className="flex gap-3 px-4 py-2.5">
              <span
                className={cn(
                  "mt-0.5 h-fit shrink-0 rounded-sm border px-1.5 py-px font-mono text-[0.62rem] uppercase",
                  h.kind === "added" && "border-insert/40 text-insert",
                  h.kind === "removed" && "border-redline/40 text-redline",
                  h.kind === "modified" && "text-graphite",
                )}
              >
                {KIND_LABEL[h.kind]}
              </span>
              <div className="min-w-0">
                <div className="text-sm font-medium">
                  {h.title}
                  {h.severity === "high" && (
                    <span className="ml-2 font-mono text-[0.62rem] font-normal text-amber-ink uppercase">
                      material
                    </span>
                  )}
                </div>
                <p className="text-[0.82rem] leading-snug text-graphite">{h.explanation}</p>
              </div>
            </li>
          ))}
        </ul>
      )}

      <Tabs defaultValue="added" className="px-4 pt-3 pb-2">
        <TabsList className="font-mono text-[0.7rem]">
          <TabsTrigger value="added">New ({stats.added})</TabsTrigger>
          <TabsTrigger value="removed">Removed ({stats.removed})</TabsTrigger>
          <TabsTrigger value="modified">Reworded ({stats.modified})</TabsTrigger>
        </TabsList>
        <TabsContent value="added">
          <Expandable
            items={diff.added}
            itemKey={(text) => text}
            render={(text) => (
              <p className="border-l-2 border-insert/60 pl-3 font-serif text-[0.9rem] leading-relaxed">
                {text}
              </p>
            )}
          />
        </TabsContent>
        <TabsContent value="removed">
          <Expandable
            items={diff.removed}
            itemKey={(text) => text}
            render={(text) => (
              <p className="border-l-2 border-redline/60 pl-3 font-serif text-[0.9rem] leading-relaxed text-graphite line-through decoration-redline/50">
                {text}
              </p>
            )}
          />
        </TabsContent>
        <TabsContent value="modified">
          <Expandable
            items={diff.modified}
            itemKey={(m) => `${m.before}→${m.after}`}
            render={(m) => <Redline before={m.before} after={m.after} />}
          />
        </TabsContent>
      </Tabs>
    </section>
  );
}
