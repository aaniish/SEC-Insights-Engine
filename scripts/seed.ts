/**
 * Seeds the database: the full SEC ticker list, then filings + financials for
 * the curated companies (or the tickers passed as arguments).
 *
 *   pnpm seed            # curated set
 *   pnpm seed TSLA CMG   # specific companies
 */
import { sql } from "drizzle-orm";
import { findCompany, syncCompanies } from "@/lib/companies";
import { db } from "@/lib/db/client";
import { companies } from "@/lib/db/schema";
import { CURATED_TICKERS } from "@/lib/featured";
import { ensureFinancials } from "@/lib/financials";
import { ingestFiling, latestFilings } from "@/lib/ingest/ingest-filing";

const CURATED: readonly string[] = CURATED_TICKERS;

async function seedCompany(ticker: string) {
  const company = await findCompany(ticker);
  if (!company) throw new Error(`Unknown ticker ${ticker}`);
  const started = Date.now();
  const [latestTenK, previousTenK] = await latestFilings(company, "10-K", 2);
  const [latestTenQ] = await latestFilings(company, "10-Q", 1);

  for (const filing of [latestTenK, latestTenQ, previousTenK]) {
    if (!filing) continue;
    const result = await ingestFiling(filing.accession, "full");
    console.log(
      `  ${filing.form} FY${filing.fiscalYear} ${filing.fiscalPeriod} → ${result.status} (${result.chunkCount} chunks)`,
    );
  }
  const hasFacts = await ensureFinancials(company.cik);
  console.log(`  financials: ${hasFacts ? "ok" : "none"} · ${((Date.now() - started) / 1000).toFixed(1)}s`);
}

async function main() {
  const args = process.argv.slice(2).map((t) => t.toUpperCase());
  const tickers = args.length > 0 ? args : CURATED;

  console.log(`Synced ${await syncCompanies()} companies from SEC.`);
  if (args.length === 0) {
    await db().update(companies).set({ isCurated: true }).where(sql`${companies.ticker} in ${CURATED}`);
  }

  for (const ticker of tickers) {
    console.log(`\n${ticker}`);
    await seedCompany(ticker);
  }

  const stats = await db().execute<{ chunks: number; size: string }>(sql`
    select (select count(*)::int from chunks) as chunks,
           pg_size_pretty(pg_database_size(current_database())) as size
  `);
  console.log(`\nDone. ${stats.rows[0].chunks} chunks, database size ${stats.rows[0].size}.`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
