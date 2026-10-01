import { daysBetween } from "./fiscal";

export const METRICS = {
  revenue: {
    label: "Revenue",
    unit: "USD",
    kind: "duration",
    concepts: [
      "RevenueFromContractWithCustomerExcludingAssessedTax",
      "RevenuesNetOfInterestExpense",
      "Revenues",
      "SalesRevenueNet",
      "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
  },
  grossProfit: {
    label: "Gross profit",
    unit: "USD",
    kind: "duration",
    concepts: ["GrossProfit"],
  },
  operatingIncome: {
    label: "Operating income",
    unit: "USD",
    kind: "duration",
    concepts: ["OperatingIncomeLoss"],
  },
  netIncome: {
    label: "Net income",
    unit: "USD",
    kind: "duration",
    concepts: ["NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
  },
  epsDiluted: {
    label: "Diluted EPS",
    unit: "USD/shares",
    kind: "duration",
    concepts: ["EarningsPerShareDiluted"],
    // Share counts change each quarter, so EPS can't be derived by subtracting YTD values.
    nonAdditive: true,
  },
  researchAndDevelopment: {
    label: "R&D expense",
    unit: "USD",
    kind: "duration",
    concepts: [
      "ResearchAndDevelopmentExpense",
      "ResearchAndDevelopmentExpenseExcludingAcquiredInProcessCost",
    ],
  },
  operatingCashFlow: {
    label: "Operating cash flow",
    unit: "USD",
    kind: "duration",
    concepts: [
      "NetCashProvidedByUsedInOperatingActivities",
      "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
  },
  capex: {
    label: "Capital expenditures",
    unit: "USD",
    kind: "duration",
    concepts: ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
  },
  buybacks: {
    label: "Share repurchases",
    unit: "USD",
    kind: "duration",
    concepts: ["PaymentsForRepurchaseOfCommonStock"],
  },
  dividends: {
    label: "Dividends paid",
    unit: "USD",
    kind: "duration",
    concepts: ["PaymentsOfDividends", "PaymentsOfDividendsCommonStock"],
  },
  cash: {
    label: "Cash and equivalents",
    unit: "USD",
    kind: "instant",
    concepts: [
      "CashAndCashEquivalentsAtCarryingValue",
      "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
      "CashAndDueFromBanks",
    ],
  },
  longTermDebt: {
    label: "Long-term debt",
    unit: "USD",
    kind: "instant",
    concepts: ["LongTermDebt", "LongTermDebtNoncurrent"],
  },
  totalAssets: {
    label: "Total assets",
    unit: "USD",
    kind: "instant",
    concepts: ["Assets"],
  },
  stockholdersEquity: {
    label: "Stockholders' equity",
    unit: "USD",
    kind: "instant",
    concepts: ["StockholdersEquity"],
  },
} as const;

export type MetricKey = keyof typeof METRICS;
export const METRIC_KEYS = Object.keys(METRICS) as MetricKey[];

export const DERIVED_METRICS = {
  freeCashFlow: { label: "Free cash flow", unit: "USD" },
  grossMargin: { label: "Gross margin", unit: "percent" },
  operatingMargin: { label: "Operating margin", unit: "percent" },
  netMargin: { label: "Net margin", unit: "percent" },
} as const;

export type DerivedMetric = keyof typeof DERIVED_METRICS;
export type AnyMetric = MetricKey | DerivedMetric;

export function metricLabel(metric: string): string {
  if (metric in METRICS) return METRICS[metric as MetricKey].label;
  if (metric in DERIVED_METRICS) return DERIVED_METRICS[metric as DerivedMetric].label;
  return metric;
}

type Quarter = "Q1" | "Q2" | "Q3" | "Q4";

export interface FinancialPoint {
  metric: MetricKey;
  periodType: "FY" | "Q";
  fiscalYear: number;
  fiscalPeriod: "FY" | Quarter;
  periodStart: string | null;
  periodEnd: string;
  value: number;
  unit: string;
  form: string | null;
  accession: string | null;
}

export interface RawFact {
  start?: string;
  end: string;
  val: number;
  accn: string;
  fy: number | null;
  fp: string | null;
  form: string;
  filed: string;
}

export interface CompanyFacts {
  facts: Record<string, Record<string, { units: Record<string, RawFact[]> }>>;
}

const ACCEPTED_FORMS = new Set(["10-K", "10-Q", "10-K/A", "10-Q/A"]);

interface Period {
  start: string | null;
  end: string;
  months: 0 | 3 | 6 | 9 | 12;
  value: number;
  rank: number;
  filed: string;
  form: string;
  accession: string;
  /** Fiscal year/period of the earliest filing that reported this period, i.e. its "own" filing. */
  label: { fy: number; fp: string; filed: string } | null;
}

function monthsOf(start: string, end: string): Period["months"] | null {
  const days = daysBetween(start, end);
  if (days >= 80 && days <= 100) return 3;
  if (days >= 170 && days <= 195) return 6;
  if (days >= 260 && days <= 285) return 9;
  if (days >= 350 && days <= 380) return 12;
  return null;
}

/**
 * Merges every concept for a metric into one value per period. Earlier concepts
 * in the list win; within a concept the latest filing wins (restatements).
 */
function collectPeriods(facts: CompanyFacts, metric: MetricKey): Period[] {
  const def = METRICS[metric];
  const gaap = facts.facts["us-gaap"] ?? {};
  const periods = new Map<string, Period>();

  def.concepts.forEach((concept, rank) => {
    const points = gaap[concept]?.units[def.unit] ?? [];
    for (const fact of points) {
      if (!ACCEPTED_FORMS.has(fact.form)) continue;
      let months: Period["months"] | null = 0;
      if (def.kind === "duration") {
        if (!fact.start) continue;
        months = monthsOf(fact.start, fact.end);
        if (months === null) continue;
      }
      const key = `${fact.start ?? ""}|${fact.end}`;
      const existing = periods.get(key);
      const label =
        fact.fy && fact.fp && (!existing?.label || fact.filed < existing.label.filed)
          ? { fy: fact.fy, fp: fact.fp, filed: fact.filed }
          : (existing?.label ?? null);
      const replaces =
        !existing || rank < existing.rank || (rank === existing.rank && fact.filed > existing.filed);
      if (replaces) {
        periods.set(key, {
          start: fact.start ?? null,
          end: fact.end,
          months,
          value: fact.val,
          rank,
          filed: fact.filed,
          form: fact.form,
          accession: fact.accn,
          label,
        });
      } else if (existing) {
        existing.label = label;
      }
    }
  });
  return [...periods.values()].filter((p) => p.label);
}

function point(
  metric: MetricKey,
  period: Pick<Period, "start" | "end" | "value" | "form" | "accession">,
  fiscalYear: number,
  fiscalPeriod: FinancialPoint["fiscalPeriod"],
): FinancialPoint {
  return {
    metric,
    periodType: fiscalPeriod === "FY" ? "FY" : "Q",
    fiscalYear,
    fiscalPeriod,
    periodStart: period.start,
    periodEnd: period.end,
    value: period.value,
    unit: METRICS[metric].unit,
    form: period.form,
    accession: period.accession,
  };
}

function normalizeDurations(metric: MetricKey, periods: Period[]): FinancialPoint[] {
  const byYear = new Map<
    number,
    {
      quarters: Partial<Record<Quarter, Period>>;
      ytd6?: Period;
      ytd9?: Period;
      fy?: Period;
    }
  >();
  for (const p of periods) {
    const label = p.label as NonNullable<Period["label"]>;
    const year = byYear.get(label.fy) ?? { quarters: {} };
    byYear.set(label.fy, year);
    if (p.months === 12 && label.fp === "FY") year.fy = p;
    else if (p.months === 3 && /^Q[1-3]$/.test(label.fp)) year.quarters[label.fp as Quarter] = p;
    else if (p.months === 3 && label.fp === "FY") year.quarters.Q4 = p;
    else if (p.months === 6 && label.fp === "Q2") year.ytd6 = p;
    else if (p.months === 9 && label.fp === "Q3") year.ytd9 = p;
  }

  const additive = !("nonAdditive" in METRICS[metric]);
  const out: FinancialPoint[] = [];
  for (const [fy, { quarters, ytd6, ytd9, fy: annual }] of byYear) {
    if (annual) out.push(point(metric, annual, fy, "FY"));
    if (!additive) {
      for (const [quarter, value] of Object.entries(quarters))
        out.push(point(metric, value, fy, quarter as Quarter));
      continue;
    }

    // 10-Q cash flow statements are year-to-date only, so difference the YTD values.
    const q1 = quarters.Q1;
    const q2 =
      quarters.Q2 ?? (ytd6 && q1 ? { ...ytd6, start: q1.end, value: ytd6.value - q1.value } : undefined);
    const q3 =
      quarters.Q3 ??
      (ytd9 && ytd6 ? { ...ytd9, start: ytd6.end, value: ytd9.value - ytd6.value } : undefined) ??
      (ytd9 && q1 && q2 ? { ...ytd9, start: q2.end, value: ytd9.value - q1.value - q2.value } : undefined);
    const nineMonths = ytd9?.value ?? (q1 && q2 && q3 ? q1.value + q2.value + q3.value : undefined);
    const q4 =
      quarters.Q4 ??
      (annual && nineMonths !== undefined
        ? {
            ...annual,
            start: q3?.end ?? null,
            value: annual.value - nineMonths,
          }
        : undefined);

    const derived: [Quarter, typeof q1][] = [
      ["Q1", q1],
      ["Q2", q2],
      ["Q3", q3],
      ["Q4", q4],
    ];
    for (const [quarter, value] of derived) if (value) out.push(point(metric, value, fy, quarter));
  }
  return out;
}

function normalizeInstants(metric: MetricKey, periods: Period[]): FinancialPoint[] {
  const out: FinancialPoint[] = [];
  const seen = new Set<string>();
  for (const p of periods) {
    const label = p.label as NonNullable<Period["label"]>;
    if (label.fp === "FY") {
      out.push(point(metric, p, label.fy, "FY"));
      if (!seen.has(`${label.fy}Q4`)) out.push(point(metric, p, label.fy, "Q4"));
      seen.add(`${label.fy}Q4`);
    } else if (/^Q[1-3]$/.test(label.fp) && !seen.has(`${label.fy}${label.fp}`)) {
      out.push(point(metric, p, label.fy, label.fp as Quarter));
      seen.add(`${label.fy}${label.fp}`);
    }
  }
  return out;
}

/** Instants at a period's start date (equity roll-forwards) also get labels; keep one per fiscal slot. */
function dedupeSlots(points: FinancialPoint[]): FinancialPoint[] {
  const bySlot = new Map<string, FinancialPoint>();
  for (const p of points) {
    const slot = `${p.fiscalYear}|${p.fiscalPeriod}`;
    const existing = bySlot.get(slot);
    if (!existing || p.periodEnd > existing.periodEnd) bySlot.set(slot, p);
  }
  return [...bySlot.values()];
}

/** Normalizes SEC companyfacts JSON into annual and quarterly series for every metric. */
export function normalizeCompanyFacts(facts: CompanyFacts): FinancialPoint[] {
  return METRIC_KEYS.flatMap((metric) => {
    const periods = collectPeriods(facts, metric);
    const points =
      METRICS[metric].kind === "duration"
        ? normalizeDurations(metric, periods)
        : normalizeInstants(metric, periods);
    return dedupeSlots(points).sort((a, b) => a.periodEnd.localeCompare(b.periodEnd));
  });
}

/**
 * Each filing's own fiscal year/period, as the company tagged it (dei:DocumentFiscalYearFocus
 * surfaces as `fy`/`fp` on every fact). Companies name 52/53-week years differently, so this
 * beats inferring the label from dates.
 */
export function filingFiscalPeriods(
  facts: CompanyFacts,
): Map<string, { fiscalYear: number; fiscalPeriod: string }> {
  const periods = new Map<string, { fiscalYear: number; fiscalPeriod: string }>();
  for (const concepts of Object.values(facts.facts)) {
    for (const { units } of Object.values(concepts)) {
      for (const points of Object.values(units)) {
        for (const fact of points) {
          if (!fact.fy || !fact.fp || periods.has(fact.accn) || !ACCEPTED_FORMS.has(fact.form)) continue;
          if (fact.fp === "FY" || /^Q[1-3]$/.test(fact.fp)) {
            periods.set(fact.accn, { fiscalYear: fact.fy, fiscalPeriod: fact.fp });
          }
        }
      }
    }
  }
  return periods;
}
