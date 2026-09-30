import { eq, sql } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { companies } from "@/lib/db/schema";
import { fetchTickerList } from "@/lib/sec/edgar";

export type Company = typeof companies.$inferSelect;

const KEEP_UPPER = new Set([
  "LLC",
  "LP",
  "LLP",
  "PLC",
  "AG",
  "NV",
  "SA",
  "SE",
  "II",
  "III",
  "IV",
  "USA",
  "US",
  "REIT",
  "ETF",
  "N.V.",
  "S.A.",
]);

/** SEC titles are often ALL CAPS ("NVIDIA CORP"); make them readable. */
export function displayName(raw: string): string {
  // SEC appends state-of-incorporation markers like "/DE" or "/NEW/".
  const name = raw.replace(/\s*\/[A-Za-z]{2,5}\/?$/, "").trim();
  if (name !== name.toUpperCase()) return name;
  return name
    .split(/\s+/)
    .map((word) => (KEEP_UPPER.has(word) ? word : word.charAt(0) + word.slice(1).toLowerCase()))
    .join(" ");
}

/** Loads/refreshes the full SEC ticker list (~10k filers). */
export async function syncCompanies(): Promise<number> {
  const entries = await fetchTickerList();
  const BATCH = 1000;
  for (let i = 0; i < entries.length; i += BATCH) {
    const rows = entries.slice(i, i + BATCH).map((e) => ({
      cik: e.cik,
      ticker: e.ticker,
      tickers: e.tickers,
      name: displayName(e.name),
      popularityRank: e.rank,
    }));
    await db()
      .insert(companies)
      .values(rows)
      .onConflictDoUpdate({
        target: companies.cik,
        set: {
          ticker: sql`excluded.ticker`,
          tickers: sql`excluded.tickers`,
          name: sql`excluded.name`,
          popularityRank: sql`excluded.popularity_rank`,
        },
      });
  }
  return entries.length;
}

export async function findCompany(tickerOrCik: string | number): Promise<Company | undefined> {
  if (typeof tickerOrCik === "number") {
    const [row] = await db().select().from(companies).where(eq(companies.cik, tickerOrCik)).limit(1);
    return row;
  }
  const ticker = tickerOrCik.trim().toUpperCase();
  const [row] = await db()
    .select()
    .from(companies)
    .where(sql`${companies.ticker} = ${ticker} or ${ticker} = any(${companies.tickers})`)
    .limit(1);
  return row;
}

export type CompanySearchResult = {
  cik: number;
  ticker: string;
  name: string;
  indexed: boolean;
};

/** Ticker-prefix matches first, then name matches; flags companies with searchable filings. */
export async function searchCompanies(query: string, limit = 12): Promise<CompanySearchResult[]> {
  const q = query.trim();
  if (!q) return featuredCompanies(limit);
  const upper = q.toUpperCase();
  const rows = await db().execute<CompanySearchResult>(sql`
    select c.cik, c.ticker, c.name,
      exists (select 1 from filings f where f.cik = c.cik and f.status = 'ready') as indexed
    from companies c
    where c.ticker like ${`${upper}%`} or ${upper} = any(c.tickers) or c.name ilike ${`%${q}%`}
    order by
      (c.ticker = ${upper} or ${upper} = any(c.tickers)) desc,
      (c.ticker like ${`${upper}%`}) desc,
      (c.name ilike ${`${q}%`}) desc,
      c.popularity_rank nulls last
    limit ${limit}
  `);
  return rows.rows;
}

async function featuredCompanies(limit: number): Promise<CompanySearchResult[]> {
  const rows = await db().execute<CompanySearchResult>(sql`
    select c.cik, c.ticker, c.name, true as indexed
    from companies c
    where c.is_curated
    order by c.popularity_rank nulls last
    limit ${limit}
  `);
  return rows.rows;
}

export async function touchCompanies(ciks: number[]): Promise<void> {
  if (ciks.length === 0) return;
  await db().update(companies).set({ lastUsedAt: new Date() }).where(sql`${companies.cik} in ${ciks}`);
}
