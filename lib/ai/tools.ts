import { tool } from "ai";
import { z } from "zod";
import { findCompany, touchCompanies } from "@/lib/companies";
import { compareFilings } from "@/lib/diff/compare";
import { DERIVED_METRICS, getFinancialSeries } from "@/lib/financials";
import {
  filingsForCompany,
  getFiling,
  ingestFiling,
  latestFilings,
  readyFilings,
} from "@/lib/ingest/ingest-filing";
import { consumeLimit, limitMessage } from "@/lib/rate-limit";
import { hybridSearch, type SearchHit } from "@/lib/retrieval/hybrid-search";
import { fetchCompanyFilings } from "@/lib/sec/edgar";
import { METRIC_KEYS } from "@/lib/sec/xbrl";

export interface ToolContext {
  /** Hashed client identifier for per-visitor limits. */
  clientKey: string;
  deep: boolean;
}

const TOPICS = ["business", "risk_factors", "mdna", "market_risk", "financial_statements", "legal"] as const;
const ALL_METRICS = [
  ...METRIC_KEYS,
  ...(Object.keys(DERIVED_METRICS) as (keyof typeof DERIVED_METRICS)[]),
] as [string, ...string[]];

const tickerList = (max: number) =>
  z
    .array(z.string().min(1).max(10))
    .min(1)
    .max(max)
    .transform((tickers) => [...new Set(tickers.map((t) => t.trim().toUpperCase()))]);

export interface Citation {
  n: number;
  ticker: string;
  company: string;
  form: string;
  period: string;
  section: string;
  filedAt: string;
  url: string;
  text: string;
}

/** Deep link that scrolls the filing to the passage (text fragments work in Chromium & Safari). */
function deepLink(hit: SearchHit): string {
  return hit.anchorText ? `${hit.docUrl}#:~:text=${encodeURIComponent(hit.anchorText)}` : hit.docUrl;
}

function periodLabel(fiscalYear: number | null, fiscalPeriod: string | null): string {
  if (!fiscalYear) return "";
  return fiscalPeriod && fiscalPeriod !== "FY" ? `${fiscalPeriod} FY${fiscalYear}` : `FY${fiscalYear}`;
}

/**
 * Tools are created per request so citation numbers stay unique across every
 * search the agent runs while answering one question.
 */
export function createTools(context: ToolContext) {
  let citationCounter = 0;

  return {
    searchFilings: tool({
      description:
        "Semantic + keyword search over indexed 10-K/10-Q text. Returns numbered passages to cite as [n]. Reports companies that aren't indexed yet.",
      inputSchema: z.object({
        query: z
          .string()
          .min(3)
          .describe("A focused search query, e.g. 'supply chain concentration risks China'."),
        tickers: tickerList(4).describe("Company tickers to search, e.g. ['AAPL']."),
        topic: z
          .enum([...TOPICS, "any"])
          .default("any")
          .describe("Filing section to search within."),
        form: z.enum(["10-K", "10-Q", "any"]).default("any"),
        fiscalYear: z.number().int().optional().describe("Only search filings for this fiscal year."),
      }),
      execute: async ({ query, tickers, topic, form, fiscalYear }) => {
        const unknown: string[] = [];
        const notIndexed: string[] = [];
        const ciks: number[] = [];
        for (const ticker of tickers) {
          const company = await findCompany(ticker);
          if (!company) unknown.push(ticker);
          else if ((await readyFilings(company.cik)).length === 0) notIndexed.push(company.ticker);
          else ciks.push(company.cik);
        }
        if (ciks.length === 0) return { query, results: [] as Citation[], notIndexed, unknown };

        const hits = await hybridSearch(
          query,
          {
            ciks,
            topics: topic === "any" ? undefined : [topic],
            forms: form === "any" ? undefined : [form],
            fiscalYears: fiscalYear ? [fiscalYear] : undefined,
          },
          context.deep ? 10 : 8,
        );
        await touchCompanies(ciks);
        const results: Citation[] = hits.map((hit) => ({
          n: ++citationCounter,
          ticker: hit.ticker,
          company: hit.companyName,
          form: hit.form,
          period: periodLabel(hit.fiscalYear, hit.fiscalPeriod),
          section: hit.sectionTitle,
          filedAt: hit.filedAt,
          url: deepLink(hit),
          text: hit.content,
        }));
        return { query, results, notIndexed, unknown };
      },
    }),

    getFinancials: tool({
      description:
        "Reported financials from SEC XBRL data for any US public company (no indexing needed). The user sees the result as a chart.",
      inputSchema: z.object({
        tickers: tickerList(4),
        metrics: z
          .array(z.enum(ALL_METRICS))
          .min(1)
          .max(4)
          .describe("Metrics to fetch. Margins are percentages; freeCashFlow = operating cash flow − capex."),
        period: z.enum(["annual", "quarterly"]).default("annual"),
        count: z
          .number()
          .int()
          .min(2)
          .max(12)
          .optional()
          .describe("How many recent periods (default 5 annual / 8 quarterly)."),
      }),
      execute: async ({ tickers, metrics, period, count }) => {
        const periods = count ?? (period === "annual" ? 5 : 8);
        const { series, missing } = await getFinancialSeries(
          tickers,
          metrics as Parameters<typeof getFinancialSeries>[1],
          period === "annual" ? "FY" : "Q",
          periods,
        );
        return { period, series, missing, source: "SEC XBRL (companyfacts)" };
      },
    }),

    compareFilings: tool({
      description:
        "Compares a section between a company's two most recent annual reports and returns added, removed, and reworded passages with highlights.",
      inputSchema: z.object({
        ticker: z.string().min(1).max(10),
        topic: z.enum(["risk_factors", "mdna"]).default("risk_factors"),
      }),
      execute: async ({ ticker, topic }) => compareFilings(ticker.toUpperCase(), topic),
    }),

    listFilings: tool({
      description:
        "Lists a company's recent 10-K/10-Q filings on EDGAR and whether each is indexed for search.",
      inputSchema: z.object({ ticker: z.string().min(1).max(10) }),
      execute: async ({ ticker }) => {
        const company = await findCompany(ticker);
        if (!company) return { error: `Unknown ticker ${ticker}.` };
        const [{ filings: onEdgar }, known] = await Promise.all([
          fetchCompanyFilings(company.cik, 1),
          filingsForCompany(company.cik),
        ]);
        const status = new Map(known.map((f) => [f.accession, f.status]));
        return {
          ticker: company.ticker,
          company: company.name,
          filings: onEdgar.slice(0, 6).map((f) => ({
            form: f.form,
            period: periodLabel(f.fiscalYear, f.fiscalPeriod),
            filedAt: f.filedAt,
            indexed: status.get(f.accession) === "ready",
          })),
        };
      },
    }),

    indexFiling: tool({
      description:
        "Downloads, parses, and indexes a company's latest 10-K or 10-Q so it can be searched. Takes ~10–40 seconds.",
      inputSchema: z.object({
        ticker: z.string().min(1).max(10),
        form: z.enum(["10-K", "10-Q"]).default("10-K"),
      }),
      execute: async function* ({ ticker, form }) {
        const company = await findCompany(ticker);
        if (!company) {
          yield {
            status: "error" as const,
            ticker,
            form,
            progress: 0,
            message: `Unknown ticker ${ticker}.`,
          };
          return;
        }
        const base = { ticker: company.ticker, company: company.name, form };
        const [filing] = await latestFilings(company, form, 1);
        if (!filing) {
          yield {
            ...base,
            status: "error" as const,
            progress: 0,
            message: `${company.ticker} has no ${form} on EDGAR. Foreign issuers file 20-F/40-F, which aren't supported yet.`,
          };
          return;
        }
        const period = periodLabel(filing.fiscalYear, filing.fiscalPeriod);
        if (filing.status === "ready") {
          yield {
            ...base,
            period,
            status: "ready" as const,
            progress: 100,
            chunks: filing.chunkCount,
          };
          return;
        }

        const limit = await consumeLimit("index", context.clientKey);
        if (!limit.ok) {
          yield {
            ...base,
            period,
            status: "error" as const,
            progress: 0,
            message: limitMessage("index", limit.reason),
          };
          return;
        }

        yield { ...base, period, status: "indexing" as const, progress: 5 };
        const job = ingestFiling(filing.accession, "full");
        let done = false;
        job.then(
          () => {
            done = true;
          },
          () => {
            done = true;
          },
        );
        while (!done) {
          await new Promise((resolve) => setTimeout(resolve, 1200));
          if (done) break;
          const current = await getFiling(filing.accession);
          yield {
            ...base,
            period,
            status: "indexing" as const,
            progress: current?.progress ?? 5,
          };
        }
        try {
          const result = await job;
          yield {
            ...base,
            period,
            status: "ready" as const,
            progress: 100,
            chunks: result.chunkCount,
          };
        } catch (error) {
          yield {
            ...base,
            period,
            status: "error" as const,
            progress: 0,
            message: `Indexing failed: ${error instanceof Error ? error.message : "unknown error"}`,
          };
        }
      },
    }),
  };
}

export type ChatTools = ReturnType<typeof createTools>;
