import { and, desc, eq, inArray, lt, ne, or } from "drizzle-orm";
import { embedTexts } from "@/lib/ai/models";
import { type Company, findCompany } from "@/lib/companies";
import { db } from "@/lib/db/client";
import { chunks, filings, sections } from "@/lib/db/schema";
import { ensureFinancials } from "@/lib/financials";
import { chunkText } from "@/lib/sec/chunk";
import { secText } from "@/lib/sec/client";
import { type EdgarFiling, fetchCompanyFilings, type SupportedForm } from "@/lib/sec/edgar";
import { htmlToText } from "@/lib/sec/html-to-text";
import { type FilingForm, splitSections } from "@/lib/sec/sections";
import { ensureStorageHeadroom } from "./evict";

export type Filing = typeof filings.$inferSelect;

/**
 * "full" embeds chunks for search; "sections" only stores section text, which
 * is enough for filing diffs and costs no vector storage.
 */
export type IngestMode = "full" | "sections";

const EMBED_BATCH = 128;
const INSERT_BATCH = 150;
/** A "processing" claim older than this is treated as abandoned (e.g. a timed-out function). */
const STALE_CLAIM_MS = 10 * 60_000;

/** Records filings from EDGAR (idempotent) so their status can be tracked. */
export async function registerFilings(cik: number, edgarFilings: EdgarFiling[]): Promise<void> {
  if (edgarFilings.length === 0) return;
  await db()
    .insert(filings)
    .values(
      edgarFilings.map((f) => ({
        accession: f.accession,
        cik,
        form: f.form,
        fiscalYear: f.fiscalYear,
        fiscalPeriod: f.fiscalPeriod,
        periodOfReport: f.reportDate,
        filedAt: f.filedAt,
        docUrl: f.docUrl,
      })),
    )
    .onConflictDoNothing();
}

/** Finds the latest N filings of a form on EDGAR and registers them. */
export async function latestFilings(company: Company, form: SupportedForm, count: number): Promise<Filing[]> {
  const { filings: all } = await fetchCompanyFilings(company.cik, form === "10-K" ? count : 1);
  const wanted = all.filter((f) => f.form === form).slice(0, count);
  await registerFilings(company.cik, wanted);
  if (wanted.length === 0) return [];
  return db()
    .select()
    .from(filings)
    .where(
      inArray(
        filings.accession,
        wanted.map((f) => f.accession),
      ),
    )
    .orderBy(desc(filings.filedAt));
}

async function setProgress(accession: string, values: Partial<Filing>) {
  await db().update(filings).set(values).where(eq(filings.accession, accession));
}

function embeddingHeader(company: Company, filing: Filing, sectionTitle: string): string {
  const period =
    filing.fiscalPeriod === "FY" ? `FY${filing.fiscalYear}` : `${filing.fiscalPeriod} FY${filing.fiscalYear}`;
  return `${company.name} (${company.ticker}) · ${filing.form} ${period} · ${sectionTitle}`;
}

/**
 * Downloads, parses, chunks, embeds, and stores one filing. Safe to re-run:
 * existing sections/chunks for the accession are replaced.
 */
export async function ingestFiling(accession: string, mode: IngestMode = "full"): Promise<Filing> {
  const [filing] = await db().select().from(filings).where(eq(filings.accession, accession)).limit(1);
  if (!filing) throw new Error(`Unknown filing ${accession}`);
  if (filing.status === "ready" || (mode === "sections" && filing.status === "sections")) return filing;
  const company = await findCompany(filing.cik);
  if (!company) throw new Error(`Unknown company for filing ${accession}`);

  // Atomically claim the filing so concurrent requests don't ingest it twice.
  const claimed = await db()
    .update(filings)
    .set({ status: "processing", progress: 5, error: null })
    .where(
      and(
        eq(filings.accession, accession),
        or(ne(filings.status, "processing"), lt(filings.updatedAt, new Date(Date.now() - STALE_CLAIM_MS))),
      ),
    )
    .returning();
  if (claimed.length === 0) return filing;

  try {
    if (mode === "full") await ensureStorageHeadroom();
    // Refreshes XBRL facts, which also corrects this filing's fiscal year/period label.
    await ensureFinancials(company.cik, { force: true }).catch(() => false);
    Object.assign(filing, await getFiling(accession));

    const text = htmlToText(await secText(filing.docUrl));
    const parsed = splitSections(text, filing.form as FilingForm);
    await setProgress(accession, { progress: 15 });

    await db().delete(chunks).where(eq(chunks.accession, accession));
    await db().delete(sections).where(eq(sections.accession, accession));
    await db()
      .insert(sections)
      .values(
        parsed.map((s) => ({
          accession,
          itemCode: s.itemCode,
          title: s.title,
          topic: s.topic,
          content: s.content,
        })),
      );

    if (mode === "sections") {
      await setProgress(accession, { status: "sections", progress: 100 });
      return { ...filing, status: "sections", progress: 100 };
    }

    const pieces = parsed.flatMap((section) =>
      chunkText(section.content).map((chunk) => ({ section, chunk })),
    );

    let stored = 0;
    for (let i = 0; i < pieces.length; i += EMBED_BATCH) {
      const batch = pieces.slice(i, i + EMBED_BATCH);
      const vectors = await embedTexts(
        batch.map(
          ({ section, chunk }) => `${embeddingHeader(company, filing, section.title)}\n\n${chunk.content}`,
        ),
      );
      const rows = batch.map(({ section, chunk }, j) => ({
        accession,
        cik: company.cik,
        ticker: company.ticker,
        companyName: company.name,
        form: filing.form,
        fiscalYear: filing.fiscalYear,
        fiscalPeriod: filing.fiscalPeriod,
        filedAt: filing.filedAt,
        docUrl: filing.docUrl,
        itemCode: section.itemCode,
        topic: section.topic,
        sectionTitle: section.title,
        chunkIndex: chunk.index,
        content: chunk.content,
        anchorText: chunk.anchorText,
        embedding: vectors[j],
      }));
      for (let k = 0; k < rows.length; k += INSERT_BATCH) {
        await db()
          .insert(chunks)
          .values(rows.slice(k, k + INSERT_BATCH));
      }
      stored += rows.length;
      await setProgress(accession, {
        progress: 15 + Math.round((80 * stored) / pieces.length),
      });
    }

    const done = {
      status: "ready" as const,
      progress: 100,
      chunkCount: stored,
    };
    await setProgress(accession, done);
    return { ...filing, ...done };
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    await setProgress(accession, {
      status: "failed",
      error: message.slice(0, 500),
    });
    throw error;
  }
}

export async function filingsForCompany(cik: number): Promise<Filing[]> {
  return db().select().from(filings).where(eq(filings.cik, cik)).orderBy(desc(filings.filedAt));
}

export async function readyFilings(cik: number, form?: SupportedForm): Promise<Filing[]> {
  const conditions = [eq(filings.cik, cik), eq(filings.status, "ready")];
  if (form) conditions.push(eq(filings.form, form));
  return db()
    .select()
    .from(filings)
    .where(and(...conditions))
    .orderBy(desc(filings.filedAt));
}

export async function getFiling(accession: string): Promise<Filing | undefined> {
  const [row] = await db().select().from(filings).where(eq(filings.accession, accession)).limit(1);
  return row;
}
