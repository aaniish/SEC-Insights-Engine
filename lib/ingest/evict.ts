import { sql } from "drizzle-orm";
import { db } from "@/lib/db/client";

/**
 * Neon's free tier has 0.5 GB. Chunk count is a stable proxy for storage
 * (deleted rows are reused by later inserts, while pg_database_size never shrinks).
 * ~5 KB per chunk including indexes, so 55k chunks ≈ 330 MB, leaving headroom.
 */
const MAX_CHUNKS = Number(process.env.MAX_CHUNKS ?? 55_000);
const TARGET_CHUNKS = Math.floor(MAX_CHUNKS * 0.85);

async function chunkCount(): Promise<number> {
  const result = await db().execute<{ count: number }>(sql`select count(*)::int as count from chunks`);
  return result.rows[0]?.count ?? 0;
}

/**
 * Before indexing a new filing, evicts the least-recently-used non-curated
 * companies' filings until we're under the target size.
 */
export async function ensureStorageHeadroom(): Promise<void> {
  let count = await chunkCount();
  if (count < MAX_CHUNKS) return;

  const candidates = await db().execute<{
    accession: string;
    chunk_count: number;
  }>(sql`
    select f.accession, f.chunk_count
    from filings f join companies c on c.cik = f.cik
    where not c.is_curated and f.status = 'ready'
    order by c.last_used_at asc nulls first, f.filed_at asc
    limit 200
  `);

  for (const { accession, chunk_count } of candidates.rows) {
    if (count < TARGET_CHUNKS) break;
    await db().execute(sql`delete from chunks where accession = ${accession}`);
    await db().execute(sql`
      update filings set status = 'sections', chunk_count = 0, progress = 100 where accession = ${accession}
    `);
    count -= chunk_count;
  }
}

/**
 * Keeps search vectors only for a company's latest two 10-Ks and latest 10-Q;
 * older filings keep their section text (for diffs) but drop their chunks.
 */
export async function pruneOldFilings(cik: number): Promise<void> {
  await db().execute(sql`
    with ranked as (
      select accession, row_number() over (partition by form order by filed_at desc) as rank, form
      from filings where cik = ${cik} and status = 'ready'
    ), stale as (
      select accession from ranked
      where (form = '10-K' and rank > 2) or (form = '10-Q' and rank > 1)
    ), deleted as (
      delete from chunks where accession in (select accession from stale)
    )
    update filings set status = 'sections', chunk_count = 0 where accession in (select accession from stale)
  `);
}
