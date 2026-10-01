import { type SQL, sql } from "drizzle-orm";
import { embedQuery } from "@/lib/ai/models";
import { db } from "@/lib/db/client";
import type { SectionTopic } from "@/lib/sec/sections";
import { balanceByGroup, reciprocalRankFusion } from "./fusion";

export interface SearchFilters {
  ciks?: number[];
  topics?: SectionTopic[];
  forms?: string[];
  fiscalYears?: number[];
}

export interface SearchHit {
  id: number;
  cik: number;
  ticker: string;
  companyName: string;
  form: string;
  fiscalYear: number | null;
  fiscalPeriod: string | null;
  filedAt: string;
  docUrl: string;
  itemCode: string;
  topic: string;
  sectionTitle: string;
  content: string;
  anchorText: string;
  score: number;
}

const CANDIDATES = 40;

const columns = sql`
  id, cik, ticker, company_name as "companyName", form, fiscal_year as "fiscalYear",
  fiscal_period as "fiscalPeriod", filed_at::text as "filedAt", doc_url as "docUrl",
  item_code as "itemCode", topic, section_title as "sectionTitle", content, anchor_text as "anchorText"
`;

function whereClause(filters: SearchFilters): SQL {
  const conditions: SQL[] = [sql`true`];
  if (filters.ciks?.length) conditions.push(sql`cik in ${filters.ciks}`);
  if (filters.topics?.length) conditions.push(sql`topic in ${filters.topics}`);
  if (filters.forms?.length) conditions.push(sql`form in ${filters.forms}`);
  if (filters.fiscalYears?.length) conditions.push(sql`fiscal_year in ${filters.fiscalYears}`);
  return sql.join(conditions, sql` and `);
}

type Row = Omit<SearchHit, "score">;

async function vectorCandidates(query: string, filters: SearchFilters): Promise<Row[]> {
  const vector = JSON.stringify(await embedQuery(query));
  const filtered = Boolean(
    filters.ciks?.length || filters.topics?.length || filters.forms?.length || filters.fiscalYears?.length,
  );
  // With filters, "+ 0" forces an exact scan over the (small) filtered set: perfect
  // recall in a few ms. Unfiltered searches use the HNSW index.
  const distance = filtered
    ? sql`(embedding <=> ${vector}::halfvec) + 0`
    : sql`embedding <=> ${vector}::halfvec`;
  const result = await db().execute<Row>(sql`
    select ${columns} from chunks
    where ${whereClause(filters)}
    order by ${distance}
    limit ${CANDIDATES}
  `);
  return result.rows;
}

async function keywordCandidates(query: string, filters: SearchFilters): Promise<Row[]> {
  // plainto_tsquery ANDs every word; OR them so partial matches still rank.
  const result = await db().execute<Row>(sql`
    select ${columns} from chunks,
      to_tsquery('english', replace(plainto_tsquery('english', ${query})::text, ' & ', ' | ')) as q
    where tsv @@ q and ${whereClause(filters)}
    order by ts_rank_cd(tsv, q) desc
    limit ${CANDIDATES}
  `);
  return result.rows;
}

/** Hybrid retrieval: semantic + keyword candidates fused with Reciprocal Rank Fusion. */
export async function hybridSearch(query: string, filters: SearchFilters, limit = 8): Promise<SearchHit[]> {
  const [semantic, keyword] = await Promise.all([
    vectorCandidates(query, filters),
    keywordCandidates(query, filters),
  ]);
  const fused = reciprocalRankFusion([semantic, keyword], (row) => row.id);
  const comparingCompanies = (filters.ciks?.length ?? 0) > 1;
  return comparingCompanies ? balanceByGroup(fused, (hit) => hit.cik, limit) : fused.slice(0, limit);
}
