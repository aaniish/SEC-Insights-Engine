import { and, asc, eq, inArray } from "drizzle-orm";
import { findCompany } from "@/lib/companies";
import { db } from "@/lib/db/client";
import { chunks, companies, filings, financialFacts } from "@/lib/db/schema";
import { padCik, SecRequestError, secJson } from "@/lib/sec/client";
import {
  type AnyMetric,
  type CompanyFacts,
  DERIVED_METRICS,
  type DerivedMetric,
  filingFiscalPeriods,
  METRICS,
  type MetricKey,
  normalizeCompanyFacts,
} from "@/lib/sec/xbrl";

const FACTS_TTL_MS = 24 * 60 * 60 * 1000;
const INSERT_BATCH = 500;

/** Replaces date-inferred fiscal labels with the ones the company tagged in XBRL. */
async function relabelFilings(cik: number, facts: CompanyFacts): Promise<void> {
  const tagged = filingFiscalPeriods(facts);
  const known = await db().select().from(filings).where(eq(filings.cik, cik));
  for (const filing of known) {
    const label = tagged.get(filing.accession);
    if (!label || (label.fiscalYear === filing.fiscalYear && label.fiscalPeriod === filing.fiscalPeriod))
      continue;
    await db().update(filings).set(label).where(eq(filings.accession, filing.accession));
    await db().update(chunks).set(label).where(eq(chunks.accession, filing.accession));
  }
}

/**
 * Fetches and caches a company's XBRL facts (refreshed at most once a day, or
 * when forced during ingestion) and corrects its filings' fiscal labels.
 */
export async function ensureFinancials(cik: number, { force = false } = {}): Promise<boolean> {
  const company = await findCompany(cik);
  if (!company) return false;
  const fresh = company.factsFetchedAt && Date.now() - company.factsFetchedAt.getTime() < FACTS_TTL_MS;
  if (fresh && !force) return true;

  let facts: CompanyFacts;
  try {
    facts = await secJson<CompanyFacts>(`https://data.sec.gov/api/xbrl/companyfacts/CIK${padCik(cik)}.json`);
  } catch (error) {
    // Companies that never filed XBRL have no companyfacts document.
    if (error instanceof SecRequestError && error.status === 404) return false;
    throw error;
  }

  const points = normalizeCompanyFacts(facts);
  await db().delete(financialFacts).where(eq(financialFacts.cik, cik));
  for (let i = 0; i < points.length; i += INSERT_BATCH) {
    await db()
      .insert(financialFacts)
      .values(points.slice(i, i + INSERT_BATCH).map((p) => ({ ...p, cik })))
      .onConflictDoNothing();
  }
  await relabelFilings(cik, facts);
  await db().update(companies).set({ factsFetchedAt: new Date() }).where(eq(companies.cik, cik));
  return points.length > 0;
}

export interface SeriesPoint {
  label: string;
  fiscalYear: number;
  fiscalPeriod: string;
  periodEnd: string;
  value: number;
}

export interface FinancialSeries {
  ticker: string;
  company: string;
  metric: MetricKey | DerivedMetric;
  label: string;
  unit: "USD" | "USD/shares" | "percent";
  points: SeriesPoint[];
}

const DERIVATION: Record<
  DerivedMetric,
  { inputs: [MetricKey, MetricKey]; compute: (a: number, b: number) => number }
> = {
  freeCashFlow: {
    inputs: ["operatingCashFlow", "capex"],
    compute: (ocf, capex) => ocf - capex,
  },
  grossMargin: {
    inputs: ["grossProfit", "revenue"],
    compute: (gp, rev) => (gp / rev) * 100,
  },
  operatingMargin: {
    inputs: ["operatingIncome", "revenue"],
    compute: (oi, rev) => (oi / rev) * 100,
  },
  netMargin: {
    inputs: ["netIncome", "revenue"],
    compute: (ni, rev) => (ni / rev) * 100,
  },
};

function periodLabel(fiscalYear: number, fiscalPeriod: string): string {
  return fiscalPeriod === "FY" ? `FY${fiscalYear}` : `${fiscalPeriod} FY${String(fiscalYear).slice(2)}`;
}

/**
 * Returns one series per (company, metric), limited to the most recent
 * `periods` annual or quarterly values. Derived metrics are computed here.
 */
export async function getFinancialSeries(
  tickers: string[],
  metrics: AnyMetric[],
  periodType: "FY" | "Q",
  periods: number,
): Promise<{ series: FinancialSeries[]; missing: string[] }> {
  const series: FinancialSeries[] = [];
  const missing: string[] = [];

  for (const ticker of tickers) {
    const company = await findCompany(ticker);
    if (!company || !(await ensureFinancials(company.cik))) {
      missing.push(ticker);
      continue;
    }

    const baseMetrics = new Set<MetricKey>();
    for (const m of metrics) {
      if (m in DERIVATION) for (const input of DERIVATION[m as DerivedMetric].inputs) baseMetrics.add(input);
      else baseMetrics.add(m as MetricKey);
    }

    const rows = await db()
      .select()
      .from(financialFacts)
      .where(
        and(
          eq(financialFacts.cik, company.cik),
          eq(financialFacts.periodType, periodType),
          inArray(financialFacts.metric, [...baseMetrics]),
        ),
      )
      .orderBy(asc(financialFacts.periodEnd));

    const byMetric = new Map<string, Map<string, (typeof rows)[number]>>();
    for (const row of rows) {
      const key = `${row.fiscalYear}|${row.fiscalPeriod}`;
      if (!byMetric.has(row.metric)) byMetric.set(row.metric, new Map());
      byMetric.get(row.metric)?.set(key, row);
    }

    for (const metric of metrics) {
      let points: SeriesPoint[];
      if (metric in DERIVATION) {
        const { inputs, compute } = DERIVATION[metric as DerivedMetric];
        const a = byMetric.get(inputs[0]) ?? new Map();
        const b = byMetric.get(inputs[1]) ?? new Map();
        points = [...a.entries()]
          .filter(([key]) => b.has(key) && b.get(key).value !== 0)
          .map(([, row]) => {
            const other = b.get(`${row.fiscalYear}|${row.fiscalPeriod}`);
            return {
              label: periodLabel(row.fiscalYear, row.fiscalPeriod),
              fiscalYear: row.fiscalYear,
              fiscalPeriod: row.fiscalPeriod,
              periodEnd: row.periodEnd,
              value: compute(row.value, other.value),
            };
          });
      } else {
        points = [...(byMetric.get(metric)?.values() ?? [])].map((row) => ({
          label: periodLabel(row.fiscalYear, row.fiscalPeriod),
          fiscalYear: row.fiscalYear,
          fiscalPeriod: row.fiscalPeriod,
          periodEnd: row.periodEnd,
          value: row.value,
        }));
      }
      points.sort((x, y) => x.periodEnd.localeCompare(y.periodEnd));
      const def =
        metric in DERIVATION ? DERIVED_METRICS[metric as DerivedMetric] : METRICS[metric as MetricKey];
      series.push({
        ticker: company.ticker,
        company: company.name,
        metric,
        label: def.label,
        unit: def.unit as FinancialSeries["unit"],
        points: points.slice(-periods),
      });
    }
  }
  return { series, missing };
}
