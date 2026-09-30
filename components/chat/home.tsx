"use client";

import { ArrowRight } from "lucide-react";
import { CURATED_COMPANIES, FEATURED_QUESTIONS, type FeaturedQuestion } from "@/lib/featured";
import type { PickedCompany } from "./company-picker";

export function Home({
  composer,
  onFeatured,
  onPickCompany,
}: {
  composer: React.ReactNode;
  onFeatured: (question: FeaturedQuestion) => void;
  onPickCompany: (company: PickedCompany) => void;
}) {
  return (
    <main className="mx-auto w-full max-w-3xl flex-1 px-4 pt-[10vh] pb-16 sm:px-6">
      <h1 className="font-display text-[clamp(3rem,11vw,6.5rem)] leading-[0.88] uppercase">Ask the 10-K.</h1>
      <p className="mt-5 max-w-xl font-serif text-lg leading-snug text-graphite sm:text-xl">
        Answers about any US public company, cited to the exact passage in its SEC filings, with reported
        numbers charted straight from XBRL.
      </p>

      <div className="mt-8">{composer}</div>

      <section className="mt-12" aria-labelledby="try-heading">
        <h2 id="try-heading" className="mb-2 font-mono text-[0.68rem] text-graphite uppercase tracking-wider">
          Try one
        </h2>
        <ul className="divide-y border-y">
          {FEATURED_QUESTIONS.map((q) => (
            <li key={q.id}>
              <button
                type="button"
                onClick={() => onFeatured(q)}
                className="group flex w-full items-center gap-4 py-3.5 text-left"
              >
                <span className="flex-1 text-[1.02rem] leading-snug group-hover:text-amber-ink">
                  {q.question}
                </span>
                <span className="hidden shrink-0 font-mono text-[0.68rem] text-graphite sm:block">
                  {q.shows}
                </span>
                <span className="w-24 shrink-0 text-right font-mono text-[0.72rem]">
                  {q.tickers.join(" · ")}
                </span>
                <ArrowRight className="size-4 shrink-0 text-graphite transition-transform group-hover:translate-x-0.5 group-hover:text-amber-ink" />
              </button>
            </li>
          ))}
        </ul>
      </section>

      <section className="mt-10" aria-labelledby="coverage-heading">
        <h2
          id="coverage-heading"
          className="mb-2 font-mono text-[0.68rem] text-graphite uppercase tracking-wider"
        >
          Indexed and refreshed daily
        </h2>
        <div className="flex flex-wrap gap-1.5">
          {CURATED_COMPANIES.map((c) => (
            <button
              key={c.ticker}
              type="button"
              onClick={() => onPickCompany(c)}
              title={`Ask about ${c.name}`}
              className="rounded-md border bg-card px-2 py-1 font-mono text-xs hover:border-amber/60 hover:text-amber-ink"
            >
              {c.ticker}
            </button>
          ))}
        </div>
        <p className="mt-3 text-sm text-graphite">
          Any other US-listed company’s latest 10-K is downloaded and indexed the first time you ask about it.
        </p>
      </section>
    </main>
  );
}
