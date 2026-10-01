import { eq } from "drizzle-orm";
import { syncCompanies } from "@/lib/companies";
import { db } from "@/lib/db/client";
import { companies } from "@/lib/db/schema";
import { env } from "@/lib/env";
import { ensureFinancials } from "@/lib/financials";
import { pruneOldFilings } from "@/lib/ingest/evict";
import { ingestFiling, latestFilings } from "@/lib/ingest/ingest-filing";

export const maxDuration = 300;

/** Filings ingested per run; each takes ~5–20s, well within the function limit. */
const MAX_NEW_FILINGS = 4;

/**
 * Daily (Vercel Cron): refresh the ticker list, then index any new 10-K/10-Q
 * for curated companies and refresh their financials.
 */
export async function GET(req: Request) {
  const secret = env().CRON_SECRET;
  if (!secret || req.headers.get("authorization") !== `Bearer ${secret}`) {
    return Response.json({ error: "Unauthorized" }, { status: 401 });
  }

  const synced = await syncCompanies();
  const curated = await db().select().from(companies).where(eq(companies.isCurated, true));
  const indexed: string[] = [];

  for (const company of curated) {
    const candidates = [
      ...(await latestFilings(company, "10-K", 1)),
      ...(await latestFilings(company, "10-Q", 1)),
    ];
    for (const filing of candidates) {
      if (filing.status === "ready" || indexed.length >= MAX_NEW_FILINGS) continue;
      await ingestFiling(filing.accession, "full");
      indexed.push(`${company.ticker} ${filing.form} ${filing.fiscalPeriod} FY${filing.fiscalYear}`);
    }
    await pruneOldFilings(company.cik);
    await ensureFinancials(company.cik);
  }

  return Response.json({ synced, indexed });
}
