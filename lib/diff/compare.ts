import { generateText, Output } from "ai";
import { and, eq } from "drizzle-orm";
import { z } from "zod";
import { chatModel, embedTexts, reasoningOptions } from "@/lib/ai/models";
import { findCompany } from "@/lib/companies";
import { db } from "@/lib/db/client";
import { filingDiffs, sections } from "@/lib/db/schema";
import { type Filing, ingestFiling, latestFilings } from "@/lib/ingest/ingest-filing";
import { type Alignment, alignParagraphs, exactMatches, splitParagraphs } from "./align";

export type DiffTopic = "risk_factors" | "mdna";

const TOPIC_TITLES: Record<DiffTopic, string> = {
  risk_factors: "Risk Factors",
  mdna: "Management's Discussion and Analysis",
};

const MAX_LISTED = 30;

const highlightsSchema = z.object({
  headline: z.string().describe("One sentence on the overall nature of the changes."),
  highlights: z
    .array(
      z.object({
        title: z.string().describe("Short name of the change, e.g. 'New tariff exposure risk'."),
        kind: z.enum(["added", "removed", "modified"]),
        severity: z.enum(["high", "medium", "low"]),
        explanation: z.string().describe("One or two sentences on what changed and why it matters."),
      }),
    )
    .max(6),
});

export interface FilingRef {
  accession: string;
  fiscalYear: number | null;
  filedAt: string;
  url: string;
}

export interface FilingDiff {
  ticker: string;
  company: string;
  topic: DiffTopic;
  sectionTitle: string;
  from: FilingRef;
  to: FilingRef;
  stats: {
    unchanged: number;
    modified: number;
    added: number;
    removed: number;
  };
  headline: string;
  highlights: z.infer<typeof highlightsSchema>["highlights"];
  added: string[];
  removed: string[];
  modified: { before: string; after: string; changeRatio: number }[];
}

function ref(filing: Filing): FilingRef {
  return {
    accession: filing.accession,
    fiscalYear: filing.fiscalYear,
    filedAt: filing.filedAt,
    url: filing.docUrl,
  };
}

async function sectionText(accession: string, topic: DiffTopic): Promise<string> {
  const rows = await db()
    .select({ content: sections.content })
    .from(sections)
    .where(and(eq(sections.accession, accession), eq(sections.topic, topic)));
  return rows.map((r) => r.content).join("\n");
}

/** Embeds only paragraphs without an exact counterpart, then aligns the two versions. */
async function align(previousText: string, currentText: string): Promise<Alignment> {
  const previous = splitParagraphs(previousText);
  const current = splitParagraphs(currentText);
  const exact = exactMatches(previous, current);
  const toEmbed = [
    ...previous.filter((_, i) => !exact.prev.has(i)),
    ...current.filter((_, j) => !exact.curr.has(j)),
  ];
  const vectors = toEmbed.length > 0 ? await embedTexts(toEmbed) : [];
  const vectorFor = new Map(toEmbed.map((text, k) => [text, vectors[k]]));
  return alignParagraphs(
    previous.map((text) => ({ text, vector: vectorFor.get(text) })),
    current.map((text) => ({ text, vector: vectorFor.get(text) })),
  );
}

async function summarize(company: string, topic: DiffTopic, alignment: Alignment, years: string) {
  const clip = (s: string) => (s.length > 600 ? `${s.slice(0, 600)}…` : s);
  const { output } = await generateText({
    model: chatModel("fast"),
    output: Output.object({ schema: highlightsSchema }),
    providerOptions: reasoningOptions("low"),
    prompt: `You compare two consecutive annual reports (10-K) of ${company}, section "${TOPIC_TITLES[topic]}" (${years}).
Below are paragraphs that were ADDED, REMOVED, or MODIFIED. Identify the most material changes for an investor (new risks, dropped risks, meaningfully reworded disclosures). Ignore cosmetic edits. Return at most 6 highlights ordered by importance.

ADDED (${alignment.added.length}):
${alignment.added.slice(0, 25).map(clip).join("\n---\n")}

REMOVED (${alignment.removed.length}):
${alignment.removed.slice(0, 25).map(clip).join("\n---\n")}

MODIFIED (${alignment.modified.length}):
${alignment.modified
  .slice(0, 15)
  .map((m) => `BEFORE: ${clip(m.before)}\nAFTER: ${clip(m.after)}`)
  .join("\n---\n")}`,
  });
  return output;
}

/**
 * Compares a section between a company's two most recent 10-Ks. Only section
 * text is needed, so filings are ingested in the cheap "sections" mode.
 */
export async function compareFilings(
  ticker: string,
  topic: DiffTopic,
): Promise<FilingDiff | { error: string }> {
  const company = await findCompany(ticker);
  if (!company) return { error: `Unknown ticker ${ticker}.` };

  const [latest, previous] = await latestFilings(company, "10-K", 2);
  if (!latest || !previous)
    return {
      error: `${company.ticker} doesn't have two annual reports (10-K) on EDGAR.`,
    };

  const [cached] = await db()
    .select()
    .from(filingDiffs)
    .where(
      and(
        eq(filingDiffs.accessionFrom, previous.accession),
        eq(filingDiffs.accessionTo, latest.accession),
        eq(filingDiffs.topic, topic),
      ),
    )
    .limit(1);
  if (cached) return cached.result as FilingDiff;

  for (const filing of [latest, previous]) {
    if (filing.status !== "ready" && filing.status !== "sections")
      await ingestFiling(filing.accession, "sections");
  }
  const [previousText, currentText] = await Promise.all([
    sectionText(previous.accession, topic),
    sectionText(latest.accession, topic),
  ]);
  if (!previousText || !currentText) {
    return {
      error: `Couldn't find the ${TOPIC_TITLES[topic]} section in both of ${company.ticker}'s annual reports.`,
    };
  }

  const alignment = await align(previousText, currentText);
  const years = `FY${previous.fiscalYear} → FY${latest.fiscalYear}`;
  const summary = await summarize(company.name, topic, alignment, years);

  const diff: FilingDiff = {
    ticker: company.ticker,
    company: company.name,
    topic,
    sectionTitle: TOPIC_TITLES[topic],
    from: ref(previous),
    to: ref(latest),
    stats: {
      unchanged: alignment.unchanged,
      modified: alignment.modified.length,
      added: alignment.added.length,
      removed: alignment.removed.length,
    },
    headline: summary.headline,
    highlights: summary.highlights,
    added: alignment.added.slice(0, MAX_LISTED),
    removed: alignment.removed.slice(0, MAX_LISTED),
    modified: alignment.modified.slice(0, MAX_LISTED).map(({ before, after, changeRatio }) => ({
      before,
      after,
      changeRatio,
    })),
  };

  await db()
    .insert(filingDiffs)
    .values({
      cik: company.cik,
      topic,
      accessionFrom: previous.accession,
      accessionTo: latest.accession,
      result: diff,
    })
    .onConflictDoNothing();
  return diff;
}
