CREATE TABLE "chunks" (
	"id" bigint PRIMARY KEY GENERATED ALWAYS AS IDENTITY (sequence name "chunks_id_seq" INCREMENT BY 1 MINVALUE 1 MAXVALUE 9223372036854775807 START WITH 1 CACHE 1),
	"accession" text NOT NULL,
	"cik" integer NOT NULL,
	"ticker" text NOT NULL,
	"company_name" text NOT NULL,
	"form" text NOT NULL,
	"fiscal_year" integer,
	"fiscal_period" text,
	"filed_at" date NOT NULL,
	"doc_url" text NOT NULL,
	"item_code" text NOT NULL,
	"topic" text NOT NULL,
	"section_title" text NOT NULL,
	"chunk_index" integer NOT NULL,
	"content" text NOT NULL,
	"anchor_text" text NOT NULL,
	"embedding" halfvec(512) NOT NULL,
	"tsv" "tsvector" GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);
--> statement-breakpoint
CREATE TABLE "companies" (
	"cik" integer PRIMARY KEY NOT NULL,
	"ticker" text NOT NULL,
	"tickers" text[] DEFAULT '{}'::text[] NOT NULL,
	"name" text NOT NULL,
	"fiscal_year_end" text,
	"is_curated" boolean DEFAULT false NOT NULL,
	"facts_fetched_at" timestamp with time zone,
	"last_used_at" timestamp with time zone
);
--> statement-breakpoint
CREATE TABLE "featured_answers" (
	"id" text PRIMARY KEY NOT NULL,
	"question" text NOT NULL,
	"tickers" text[] NOT NULL,
	"messages" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "filing_diffs" (
	"id" serial PRIMARY KEY NOT NULL,
	"cik" integer NOT NULL,
	"topic" text NOT NULL,
	"accession_from" text NOT NULL,
	"accession_to" text NOT NULL,
	"result" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "filings" (
	"accession" text PRIMARY KEY NOT NULL,
	"cik" integer NOT NULL,
	"form" text NOT NULL,
	"fiscal_year" integer,
	"fiscal_period" text,
	"period_of_report" date,
	"filed_at" date NOT NULL,
	"doc_url" text NOT NULL,
	"status" text DEFAULT 'pending' NOT NULL,
	"progress" integer DEFAULT 0 NOT NULL,
	"error" text,
	"chunk_count" integer DEFAULT 0 NOT NULL,
	"created_at" timestamp with time zone DEFAULT now() NOT NULL,
	"updated_at" timestamp with time zone DEFAULT now() NOT NULL
);
--> statement-breakpoint
CREATE TABLE "financial_facts" (
	"cik" integer NOT NULL,
	"metric" text NOT NULL,
	"period_type" text NOT NULL,
	"fiscal_year" integer NOT NULL,
	"fiscal_period" text NOT NULL,
	"period_start" date,
	"period_end" date NOT NULL,
	"value" double precision NOT NULL,
	"unit" text NOT NULL,
	"form" text,
	"accession" text,
	CONSTRAINT "financial_facts_cik_metric_period_type_period_end_pk" PRIMARY KEY("cik","metric","period_type","period_end")
);
--> statement-breakpoint
CREATE TABLE "sections" (
	"id" serial PRIMARY KEY NOT NULL,
	"accession" text NOT NULL,
	"item_code" text NOT NULL,
	"title" text NOT NULL,
	"topic" text NOT NULL,
	"content" text NOT NULL
);
--> statement-breakpoint
CREATE TABLE "usage_counters" (
	"key" text NOT NULL,
	"day" date NOT NULL,
	"kind" text NOT NULL,
	"count" integer DEFAULT 0 NOT NULL,
	CONSTRAINT "usage_counters_key_day_kind_pk" PRIMARY KEY("key","day","kind")
);
--> statement-breakpoint
ALTER TABLE "chunks" ADD CONSTRAINT "chunks_accession_filings_accession_fk" FOREIGN KEY ("accession") REFERENCES "public"."filings"("accession") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "filings" ADD CONSTRAINT "filings_cik_companies_cik_fk" FOREIGN KEY ("cik") REFERENCES "public"."companies"("cik") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "financial_facts" ADD CONSTRAINT "financial_facts_cik_companies_cik_fk" FOREIGN KEY ("cik") REFERENCES "public"."companies"("cik") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
ALTER TABLE "sections" ADD CONSTRAINT "sections_accession_filings_accession_fk" FOREIGN KEY ("accession") REFERENCES "public"."filings"("accession") ON DELETE cascade ON UPDATE no action;--> statement-breakpoint
CREATE INDEX "chunks_embedding_idx" ON "chunks" USING hnsw ("embedding" halfvec_cosine_ops);--> statement-breakpoint
CREATE INDEX "chunks_tsv_idx" ON "chunks" USING gin ("tsv");--> statement-breakpoint
CREATE INDEX "chunks_cik_topic_idx" ON "chunks" USING btree ("cik","topic");--> statement-breakpoint
CREATE INDEX "chunks_accession_idx" ON "chunks" USING btree ("accession");--> statement-breakpoint
CREATE UNIQUE INDEX "companies_ticker_idx" ON "companies" USING btree ("ticker");--> statement-breakpoint
CREATE UNIQUE INDEX "filing_diffs_pair_idx" ON "filing_diffs" USING btree ("accession_from","accession_to","topic");--> statement-breakpoint
CREATE INDEX "filings_cik_form_idx" ON "filings" USING btree ("cik","form","filed_at" DESC NULLS LAST);--> statement-breakpoint
CREATE UNIQUE INDEX "sections_accession_item_idx" ON "sections" USING btree ("accession","item_code");