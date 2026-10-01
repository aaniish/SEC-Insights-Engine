"use client";

import { AlertTriangle, Plus, RotateCcw } from "lucide-react";
import { useMemo } from "react";
import { FinancialCharts } from "@/components/charts/financial-charts";
import { CitationProvider, SourceStrip } from "@/components/citations/citations";
import { FilingDiffCard } from "@/components/diff/filing-diff-card";
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

  return (
    <CitationProvider turnId={question.id} citations={citations}>
      <article className="scroll-mt-24 space-y-6" data-turn={question.id}>
        <h2 className="font-display text-[clamp(1.9rem,4.2vw,2.75rem)] leading-[1.05] text-balance">
          {questionText(question)}
        </h2>

        <ResearchLog parts={toolParts} working={working} />

        <SourceStrip order={order} />

        {text && <Answer markdown={markdown} streaming={working} />}

        <Exhibits parts={toolParts} />

        {error && (
          <div className="flex items-start gap-2 rounded-2xl bg-redline-wash px-4 py-3 text-sm text-redline">
            <AlertTriangle className="mt-0.5 size-4 shrink-0" />
            <span className="flex-1">{error}</span>
            {onRetry && (
              <button type="button" onClick={onRetry} className="inline-flex items-center gap-1 font-medium">
                <RotateCcw className="size-3.5" /> Try again
              </button>
            )}
          </div>
        )}

        {suggestions.length > 0 && !working && (
          <nav aria-label="Related questions" className="pt-2">
            <div className="mb-1 font-display text-xl">Related</div>
            <ul className="divide-y divide-rule border-y border-rule">
              {suggestions.map((s) => (
                <li key={s}>
                  <button
                    type="button"
                    onClick={() => onAsk(s)}
                    className="group flex w-full items-center gap-3 py-3 text-left text-[0.95rem] transition-colors hover:text-navy-ink"
                  >
                    <span className="flex-1">{s}</span>
                    <Plus className="size-4 shrink-0 text-graphite transition-transform group-hover:rotate-90 group-hover:text-navy-ink" />
                  </button>
                </li>
              ))}
            </ul>
          </nav>
        )}
      </article>
    </CitationProvider>
  );
}
