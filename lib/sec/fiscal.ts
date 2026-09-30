const DAY_MS = 86_400_000;
/** 52/53-week fiscal years end a few days either side of the nominal date. */
const TOLERANCE_DAYS = 14;

function utcDate(year: number, month: number, day: number): Date {
  return new Date(Date.UTC(year, month - 1, day));
}

function parseIsoDate(value: string): Date {
  const [y, m, d] = value.split("-").map(Number);
  return utcDate(y, m, d);
}

export interface FiscalPeriod {
  fiscalYear: number;
  fiscalPeriod: "FY" | "Q1" | "Q2" | "Q3" | "Q4";
}

/**
 * Estimates the fiscal year/quarter a report date falls in, given the company's
 * fiscal year end as "MMDD" (from EDGAR submissions). The fiscal year is named
 * after the calendar year in which it ends (Apple's year ending Sep 2025 is FY2025).
 */
export function fiscalPeriodOf(reportDate: string, fiscalYearEnd: string | null, form: string): FiscalPeriod {
  const report = parseIsoDate(reportDate);
  const mmdd = fiscalYearEnd && /^\d{4}$/.test(fiscalYearEnd) ? fiscalYearEnd : "1231";
  const month = Number(mmdd.slice(0, 2));
  const day = Number(mmdd.slice(2));
  const earliest = report.getTime() - TOLERANCE_DAYS * DAY_MS;

  const year = report.getUTCFullYear();
  const yearEnd = [year - 1, year, year + 1]
    .map((y) => utcDate(y, month, day))
    .find((d) => d.getTime() >= earliest) as Date;
  // A 52/53-week year ending in the first days of January is a calendar year that
  // spilled over (e.g. Domino's), and companies name it after the earlier year.
  const spillsIntoJanuary = month === 1 && day <= 7;
  const fiscalYear = yearEnd.getUTCFullYear() - (spillsIntoJanuary ? 1 : 0);

  if (form.startsWith("10-K")) return { fiscalYear, fiscalPeriod: "FY" };

  const previousYearEnd = utcDate(yearEnd.getUTCFullYear() - 1, month, day);
  const months = (report.getTime() - previousYearEnd.getTime()) / (DAY_MS * 30.44);
  const quarter = Math.min(4, Math.max(1, Math.round(months / 3)));
  return {
    fiscalYear,
    fiscalPeriod: `Q${quarter}` as FiscalPeriod["fiscalPeriod"],
  };
}

export function daysBetween(start: string, end: string): number {
  return Math.round((parseIsoDate(end).getTime() - parseIsoDate(start).getTime()) / DAY_MS);
}
