"use client";

import { ArrowUpRight } from "lucide-react";
import type { ReactNode } from "react";
import { CURATED_COMPANIES, FEATURED_QUESTIONS, type FeaturedQuestion } from "@/lib/featured";
import type { PickedCompany } from "./company-picker";
import { DriftStatement } from "./drift-statement";

export function Home({
  composer,
  onFeatured,
  onPickCompany,
}: {
  composer: ReactNode;
  onFeatured: (question: FeaturedQuestion) => void;
  onPickCompany: (company: PickedCompany) => void;
}) {
  return (
    <main className="mx-auto w-full max-w-5xl flex-1 px-4 pt-[11vh] pb-24 sm:px-6">
      <section className="text-center">
        <h1 className="font-display text-[clamp(3.4rem,10.5vw,7.5rem)] leading-[0.92]">
          Ask the <span className="whitespace-nowrap">10-K.</span>
        </h1>
        <p className="mx-auto mt-5 max-w-2xl text-[clamp(1.1rem,2.3vw,1.45rem)] leading-snug text-graphite">
          Answers about any public company, cited to the exact passage in its SEC filings.
        </p>
        <div className="mx-auto mt-10 max-w-2xl text-left">{composer}</div>
      </section>

      <section aria-label="Example questions" className="mx-auto mt-8 grid max-w-2xl gap-2.5 sm:grid-cols-2">
        {FEATURED_QUESTIONS.map((q) => (
          <button
            key={q.id}
            type="button"
            onClick={() => onFeatured(q)}
            className="glass group flex flex-col gap-3 rounded-2xl p-4 text-left transition-transform hover:-translate-y-0.5 active:translate-y-0"
          >
            <span className="text-[0.95rem] leading-snug font-medium">{q.question}</span>
            <span className="mt-auto flex items-center gap-1.5 text-xs text-graphite">
              {q.tickers.map((t) => (
                <span
                  key={t}
                  className="rounded-full bg-navy-soft px-2 py-0.5 font-semibold text-navy-ink tabular"
                >
                  {t}
                </span>
              ))}
              <span className="ml-1">{q.shows}</span>
              <ArrowUpRight className="ml-auto size-4 opacity-40 transition-opacity group-hover:opacity-100" />
            </span>
          </button>
        ))}
      </section>

      <section aria-labelledby="coverage-heading" className="mx-auto mt-10 max-w-2xl text-center">
        <h2 id="coverage-heading" className="text-xs font-medium text-graphite">
          Indexed and refreshed daily · any other company on request
        </h2>
        <div className="mt-3 flex flex-wrap justify-center gap-1.5">
          {CURATED_COMPANIES.map((c) => (
            <button
              key={c.ticker}
              type="button"
              onClick={() => onPickCompany(c)}
              title={`Ask about ${c.name}`}
              className="rounded-full bg-muted px-3 py-1.5 text-xs font-semibold text-graphite tabular transition-colors hover:bg-navy-soft hover:text-navy-ink"
            >
              {c.ticker}
            </button>
          ))}
        </div>
      </section>

      <section aria-label="How it works" className="mt-32">
        <DriftStatement />
        <p className="mx-auto mt-10 max-w-xl text-center text-sm leading-relaxed text-graphite">
          Built on SEC EDGAR. Filing text is searched with hybrid semantic and keyword retrieval; financials
          come from the XBRL data companies file with every report. Not investment advice.
        </p>
      </section>
    </main>
  );
}
