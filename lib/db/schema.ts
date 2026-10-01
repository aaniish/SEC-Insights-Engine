import { sql } from "drizzle-orm";
import {
  bigint,
  boolean,
  customType,
  date,
  doublePrecision,
  halfvec,
  index,
  integer,
  jsonb,
  pgTable,
  primaryKey,
  serial,
  text,
  timestamp,
  uniqueIndex,
} from "drizzle-orm/pg-core";

export const EMBEDDING_DIMENSIONS = 512;

const tsvector = customType<{ data: string }>({
  dataType: () => "tsvector",
});

const timestamps = {
  createdAt: timestamp({ withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp({ withTimezone: true })
    .notNull()
    .defaultNow()
    .$onUpdate(() => new Date()),
};

/** Every SEC filer (~10k). `tickers` holds all share classes (GOOGL, GOOG). */
export const companies = pgTable(
  "companies",
  {
    cik: integer().primaryKey(),
    ticker: text().notNull(),
    tickers: text().array().notNull().default(sql`'{}'::text[]`),
    name: text().notNull(),
    /** Position in SEC's ticker file, which is ordered roughly by market cap. */
    popularityRank: integer(),
    fiscalYearEnd: text(),
    isCurated: boolean().notNull().default(false),
    factsFetchedAt: timestamp({ withTimezone: true }),
    lastUsedAt: timestamp({ withTimezone: true }),
  },
  (t) => [uniqueIndex("companies_ticker_idx").on(t.ticker)],
);

export type FilingStatus = "pending" | "processing" | "sections" | "ready" | "failed";

export const filings = pgTable(
  "filings",
  {
    accession: text().primaryKey(),
    cik: integer()
      .notNull()
      .references(() => companies.cik, { onDelete: "cascade" }),
    form: text().notNull(),
    fiscalYear: integer(),
    fiscalPeriod: text(),
    periodOfReport: date(),
    filedAt: date().notNull(),
    docUrl: text().notNull(),
    status: text().$type<FilingStatus>().notNull().default("pending"),
    progress: integer().notNull().default(0),
    error: text(),
    chunkCount: integer().notNull().default(0),
    ...timestamps,
  },
  (t) => [index("filings_cik_form_idx").on(t.cik, t.form, t.filedAt.desc())],
);

export const sections = pgTable(
  "sections",
  {
    id: serial().primaryKey(),
    accession: text()
      .notNull()
      .references(() => filings.accession, { onDelete: "cascade" }),
    itemCode: text().notNull(),
    title: text().notNull(),
    topic: text().notNull(),
    content: text().notNull(),
  },
  (t) => [uniqueIndex("sections_accession_item_idx").on(t.accession, t.itemCode)],
);

/** Retrieval unit. Company/filing fields are denormalized so search needs no joins. */
export const chunks = pgTable(
  "chunks",
  {
    id: bigint({ mode: "number" }).primaryKey().generatedAlwaysAsIdentity(),
    accession: text()
      .notNull()
      .references(() => filings.accession, { onDelete: "cascade" }),
    cik: integer().notNull(),
    ticker: text().notNull(),
    companyName: text().notNull(),
    form: text().notNull(),
    fiscalYear: integer(),
    fiscalPeriod: text(),
    filedAt: date().notNull(),
    docUrl: text().notNull(),
    itemCode: text().notNull(),
    topic: text().notNull(),
    sectionTitle: text().notNull(),
    chunkIndex: integer().notNull(),
    content: text().notNull(),
    anchorText: text().notNull(),
    embedding: halfvec({ dimensions: EMBEDDING_DIMENSIONS }).notNull(),
    tsv: tsvector().generatedAlwaysAs(sql`to_tsvector('english', content)`),
  },
  (t) => [
    index("chunks_embedding_idx").using("hnsw", t.embedding.op("halfvec_cosine_ops")),
    index("chunks_tsv_idx").using("gin", t.tsv),
    index("chunks_cik_topic_idx").on(t.cik, t.topic),
    index("chunks_accession_idx").on(t.accession),
  ],
);

export type PeriodType = "FY" | "Q";

/** Normalized XBRL values, one row per metric per reporting period. */
export const financialFacts = pgTable(
  "financial_facts",
  {
    cik: integer()
      .notNull()
      .references(() => companies.cik, { onDelete: "cascade" }),
    metric: text().notNull(),
    periodType: text().$type<PeriodType>().notNull(),
    fiscalYear: integer().notNull(),
    fiscalPeriod: text().notNull(),
    periodStart: date(),
    periodEnd: date().notNull(),
    value: doublePrecision().notNull(),
    unit: text().notNull(),
    form: text(),
    accession: text(),
  },
  (t) => [primaryKey({ columns: [t.cik, t.metric, t.periodType, t.periodEnd] })],
);

export const filingDiffs = pgTable(
  "filing_diffs",
  {
    id: serial().primaryKey(),
    cik: integer().notNull(),
    topic: text().notNull(),
    accessionFrom: text().notNull(),
    accessionTo: text().notNull(),
    result: jsonb().notNull(),
    createdAt: timestamp({ withTimezone: true }).notNull().defaultNow(),
  },
  (t) => [uniqueIndex("filing_diffs_pair_idx").on(t.accessionFrom, t.accessionTo, t.topic)],
);

/** Daily counters for rate limiting, keyed by hashed IP or "global". */
export const usageCounters = pgTable(
  "usage_counters",
  {
    key: text().notNull(),
    day: date().notNull(),
    kind: text().notNull(),
    count: integer().notNull().default(0),
  },
  (t) => [primaryKey({ columns: [t.key, t.day, t.kind] })],
);

/** Pre-computed answers for the starter questions, replayed at zero cost. */
export const featuredAnswers = pgTable("featured_answers", {
  id: text().primaryKey(),
  question: text().notNull(),
  tickers: text().array().notNull(),
  messages: jsonb().notNull(),
  createdAt: timestamp({ withTimezone: true }).notNull().defaultNow(),
});
