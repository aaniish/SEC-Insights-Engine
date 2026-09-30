"use client";

import { AlertTriangle, CornerDownRight, RotateCcw } from "lucide-react";
import { useMemo } from "react";
import { FinancialCharts } from "@/components/charts/financial-charts";
import { CitationProvider, SourceNotes } from "@/components/citations/citations";
import { FilingDiffCard } from "@/components/diff/filing-diff-card";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import type { ChatMessage } from "@/lib/ai/types";
import {
  answerText,
  citationsOf,
  citedNumbers,
  isToolPart,
  linkCitations,
  suggestionsOf,
  type ToolPart,
} from "@/lib/chat/parts";
import { Answer } from "./answer";
import { ResearchLog } from "./research-log";

interface TurnProps {
  question: ChatMessage;
  answer?: ChatMessage;
  working: boolean;
  error?: string;
  onAsk: (question: string) => void;
  onRetry?: () => void;
}

function questionText(message: ChatMessage): string {
  return message.parts.map((p) => (p.type === "text" ? p.text : "")).join(" ");
}

/** Charts and diff cards produced by tools, rendered after the prose answer. */
function Exhibits({ parts }: { parts: ToolPart[] }) {
  return (
    <>
      {parts.map((part) => {
        if (part.state !== "output-available") return null;
        if (part.type === "tool-getFinancials" && part.output.series.length > 0) {
          return (
            <FinancialCharts
              key={part.toolCallId}
              series={part.output.series}
              missing={part.output.missing}
            />
          );
        }
        if (part.type === "tool-compareFilings" && !("error" in part.output)) {
          return <FilingDiffCard key={part.toolCallId} diff={part.output} />;
        }
        return null;
      })}
    </>
  );
}

export function Turn({ question, answer, working, error, onAsk, onRetry }: TurnProps) {
  const citations = useMemo(() => (answer ? citationsOf(answer) : new Map()), [answer]);
  const text = answer ? answerText(answer) : "";
  const markdown = useMemo(() => linkCitations(text, citations), [text, citations]);
  const order = useMemo(() => citedNumbers(text), [text]);
  const toolParts = answer?.parts.filter(isToolPart) ?? [];
  const suggestions = answer ? suggestionsOf(answer) : [];
  // Passages that were read but never cited stay in the research log, not the margin.
  const hasSources = order.some((n) => citations.has(n));

  return (
    <CitationProvider turnId={question.id} citations={citations}>
      <article
        className="grid scroll-mt-20 gap-x-10 lg:grid-cols-[minmax(0,1fr)_17.5rem]"
        data-turn={question.id}
      >
        <div className="min-w-0 space-y-5">
          <h2 className="font-display text-[1.65rem] leading-[1.1] text-balance sm:text-3xl">
            {questionText(question)}
          </h2>

          <ResearchLog parts={toolParts} working={working} />

          {text && <Answer markdown={markdown} streaming={working} />}

          <Exhibits parts={toolParts} />

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-redline/30 bg-redline-wash px-3 py-2 text-sm text-redline">
              <AlertTriangle className="mt-0.5 size-4 shrink-0" />
              <span className="flex-1">{error}</span>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="inline-flex items-center gap-1 font-medium"
                >
                  <RotateCcw className="size-3.5" /> Try again
                </button>
              )}
            </div>
          )}

          {hasSources && (
            <Collapsible className="lg:hidden">
              <CollapsibleTrigger className="font-mono text-[0.72rem] text-graphite hover:text-foreground">
                Show {citations.size} sources
              </CollapsibleTrigger>
              <CollapsibleContent className="pt-2">
                <SourceNotes order={order} />
              </CollapsibleContent>
            </Collapsible>
          )}

          {suggestions.length > 0 && !working && (
            <nav aria-label="Follow-up questions" className="border-t pt-4">
              <div className="mb-1.5 font-mono text-[0.68rem] text-graphite uppercase tracking-wider">
                Keep digging
              </div>
              <ul>
                {suggestions.map((s) => (
                  <li key={s}>
                    <button
                      type="button"
                      onClick={() => onAsk(s)}
                      className="group flex w-full items-start gap-2 rounded-md py-1.5 text-left text-[0.95rem] hover:text-amber-ink"
                    >
                      <CornerDownRight className="mt-1 size-3.5 shrink-0 text-graphite group-hover:text-amber-ink" />
                      {s}
                    </button>
                  </li>
                ))}
              </ul>
            </nav>
          )}
        </div>

        {hasSources && (
          <div className="hidden lg:block">
            <SourceNotes
              anchored
              order={order}
              className="sticky top-20 max-h-[calc(100dvh-7rem)] overflow-y-auto pb-4"
            />
          </div>
        )}
      </article>
    </CitationProvider>
  );
}
