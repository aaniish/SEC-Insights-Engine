"use client";

import { AlertTriangle, Check, ChevronRight, Loader2 } from "lucide-react";
import { useState } from "react";
import { type ToolPart, toolName } from "@/lib/chat/parts";
import { metricLabel } from "@/lib/sec/xbrl";
import { cn } from "@/lib/utils";

type Status = "running" | "done" | "error";

interface LogLine {
  id: string;
  status: Status;
  text: string;
  progress?: number;
}

const TOPIC_LABELS: Record<string, string> = {
  business: "Business",
  risk_factors: "Risk Factors",
  mdna: "MD&A",
  market_risk: "Market Risk",
  financial_statements: "Financial Statements",
  legal: "Legal Proceedings",
};

const list = (items: readonly (string | undefined)[] | undefined) => {
  const present = items?.filter(Boolean) ?? [];
  return present.length ? present.join(", ") : "…";
};

function describe(part: ToolPart): LogLine {
  const id = part.toolCallId;
  if (part.state === "output-error") return { id, status: "error", text: part.errorText };
  const done = part.state === "output-available" && !part.preliminary;

  switch (toolName(part)) {
    case "searchFilings": {
      const p = part as Extract<ToolPart, { type: "tool-searchFilings" }>;
      const input = p.input;
      const where = [
        list(input?.tickers),
        input?.topic && input.topic !== "any" ? TOPIC_LABELS[input.topic] : null,
      ]
        .filter(Boolean)
        .join(" · ");
      if (p.state !== "output-available")
        return { id, status: "running", text: `Searching ${where} for “${input?.query ?? "…"}”` };
      const { results, notIndexed } = p.output;
      const missing = notIndexed.length ? ` · ${notIndexed.join(", ")} not indexed yet` : "";
      return { id, status: "done", text: `Searched ${where} · ${results.length} passages${missing}` };
    }
    case "getFinancials": {
      const p = part as Extract<ToolPart, { type: "tool-getFinancials" }>;
      const metrics = (p.input?.metrics ?? [])
        .filter((m): m is string => Boolean(m))
        .map(metricLabel)
        .join(", ")
        .toLowerCase();
      const what = `${metrics || "financials"} for ${list(p.input?.tickers)}`;
      if (p.state !== "output-available") return { id, status: "running", text: `Pulling ${what} from XBRL` };
      // Answers cached before `unavailable` existed don't have it.
      const gaps = [...p.output.missing.map((t) => `${t} (no XBRL)`), ...(p.output.unavailable ?? [])];
      const note = gaps.length ? ` · not reported: ${gaps.join(", ")}` : "";
      if (p.output.series.length === 0) return { id, status: "done", text: `No XBRL data for ${what}` };
      return { id, status: "done", text: `Pulled ${p.output.period} ${what} from XBRL${note}` };
    }
    case "compareFilings": {
      const p = part as Extract<ToolPart, { type: "tool-compareFilings" }>;
      const section = TOPIC_LABELS[p.input?.topic ?? "risk_factors"];
      if (p.state !== "output-available") {
        return {
          id,
          status: "running",
          text: `Comparing ${p.input?.ticker ?? "…"} ${section} across its last two 10-Ks`,
        };
      }
      if ("error" in p.output) return { id, status: "error", text: p.output.error };
      const { ticker, from, to, stats } = p.output;
      return {
        id,
        status: "done",
        text: `Compared ${ticker} ${section} FY${from.fiscalYear} → FY${to.fiscalYear} · ${stats.added} new, ${stats.removed} removed, ${stats.modified} reworded`,
      };
    }
    case "indexFiling": {
      const p = part as Extract<ToolPart, { type: "tool-indexFiling" }>;
      const target = `${p.input?.ticker ?? "…"} ${p.input?.form ?? "10-K"}`;
      if (p.state !== "output-available")
        return { id, status: "running", text: `Finding ${target} on EDGAR`, progress: 0 };
      const out = p.output;
      const label = `${out.ticker} ${out.form}${"period" in out && out.period ? ` ${out.period}` : ""}`;
      if (out.status === "error") return { id, status: "error", text: out.message ?? "Indexing failed" };
      if (out.status === "ready") {
        const passages = "chunks" in out && out.chunks ? ` · ${out.chunks} passages` : "";
        return { id, status: "done", text: `Indexed ${label}${passages}` };
      }
      return {
        id,
        status: done ? "done" : "running",
        text: `Downloading and indexing ${label}`,
        progress: out.progress,
      };
    }
    case "listFilings": {
      const p = part as Extract<ToolPart, { type: "tool-listFilings" }>;
      return {
        id,
        status: p.state === "output-available" ? "done" : "running",
        text: `Listed recent filings for ${p.input?.ticker ?? "…"}`,
      };
    }
    default:
      return { id, status: done ? "done" : "running", text: "Working" };
  }
}

function Line({ line }: { line: LogLine }) {
  return (
    <li className="log-line flex gap-2">
      <span className="mt-[0.2rem] shrink-0" aria-hidden="true">
        {line.status === "running" && <Loader2 className="size-3.5 animate-spin text-navy-ink" />}
        {line.status === "done" && <Check className="size-3.5 text-graphite" />}
        {line.status === "error" && <AlertTriangle className="size-3.5 text-redline" />}
      </span>
      <span className="min-w-0 flex-1">
        <span className={cn(line.status === "error" && "text-redline")}>{line.text}</span>
        {line.status === "running" && line.progress !== undefined && (
          <span className="mt-1.5 block h-1 w-full max-w-64 overflow-hidden rounded-full bg-muted">
            <span
              className="block h-full rounded-full bg-navy transition-[width] duration-500"
              style={{ width: `${Math.max(4, line.progress)}%` }}
            />
          </span>
        )}
      </span>
    </li>
  );
}

/**
 * What the agent did, as a compact log. Expanded while it works; collapses to a
 * one-line summary once the answer is complete.
 */
export function ResearchLog({ parts, working }: { parts: ToolPart[]; working: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const lines = parts.map(describe);

  if (lines.length === 0) {
    if (!working) return null;
    return (
      <ul className="text-[0.84rem] text-graphite" aria-live="polite">
        <Line line={{ id: "thinking", status: "running", text: "Reading the question" }} />
      </ul>
    );
  }

  if (!working && !expanded) {
    const searches = lines.filter((l) => l.text.startsWith("Searched")).length;
    return (
      <button
        type="button"
        onClick={() => setExpanded(true)}
        className="glass group inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs text-graphite hover:text-ink"
      >
        <ChevronRight className="size-3.5 transition-transform group-hover:translate-x-0.5" />
        {lines.length} {lines.length === 1 ? "step" : "steps"}
        {searches > 0 && ` · ${searches} ${searches === 1 ? "search" : "searches"}`}
        {lines.some((l) => l.status === "error") && " · 1 issue"}
      </button>
    );
  }

  return (
    <div className="text-[0.84rem] leading-relaxed text-graphite">
      {!working && (
        <button type="button" onClick={() => setExpanded(false)} className="mb-1.5 text-xs hover:text-ink">
          Hide steps
        </button>
      )}
      <ul className="space-y-1.5" aria-live="polite">
        {lines.map((line) => (
          <Line key={line.id} line={line} />
        ))}
      </ul>
    </div>
  );
}
