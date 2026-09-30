import { padCik, secJson } from "./client";
import { fiscalPeriodOf } from "./fiscal";

export interface TickerEntry {
  cik: number;
  ticker: string;
  tickers: string[];
  name: string;
  /** SEC lists companies roughly by market cap; lower is larger. */
  rank: number;
}

interface RawTicker {
  cik_str: number;
  ticker: string;
  title: string;
}

/** All SEC filers with tickers, grouped by CIK (share classes share one CIK). */
export async function fetchTickerList(): Promise<TickerEntry[]> {
  const raw = await secJson<Record<string, RawTicker>>("https://www.sec.gov/files/company_tickers.json");
  const byCik = new Map<number, TickerEntry>();
  Object.values(raw).forEach(({ cik_str: cik, ticker, title }, rank) => {
    const existing = byCik.get(cik);
    if (existing) existing.tickers.push(ticker);
    else byCik.set(cik, { cik, ticker, tickers: [ticker], name: title, rank });
  });
  return [...byCik.values()];
}

export type SupportedForm = "10-K" | "10-Q";

export interface EdgarFiling {
  accession: string;
  form: SupportedForm;
  filedAt: string;
  reportDate: string;
  fiscalYear: number;
  fiscalPeriod: string;
  docUrl: string;
}

interface FilingColumns {
  accessionNumber: string[];
  filingDate: string[];
  reportDate: string[];
  form: string[];
  primaryDocument: string[];
}

interface Submissions {
  name: string;
  fiscalYearEnd: string | null;
  filings: { recent: FilingColumns; files: { name: string }[] };
}

export interface CompanyFilings {
  name: string;
  fiscalYearEnd: string | null;
  filings: EdgarFiling[];
}

export function filingDocUrl(cik: number, accession: string, primaryDocument: string): string {
  return `https://www.sec.gov/Archives/edgar/data/${cik}/${accession.replaceAll("-", "")}/${primaryDocument}`;
}

function toFilings(cik: number, columns: FilingColumns, fiscalYearEnd: string | null): EdgarFiling[] {
  const out: EdgarFiling[] = [];
  columns.form.forEach((form, i) => {
    if (form !== "10-K" && form !== "10-Q") return;
    const reportDate = columns.reportDate[i] || columns.filingDate[i];
    out.push({
      accession: columns.accessionNumber[i],
      form,
      filedAt: columns.filingDate[i],
      reportDate,
      ...fiscalPeriodOf(reportDate, fiscalYearEnd, form),
      docUrl: filingDocUrl(cik, columns.accessionNumber[i], columns.primaryDocument[i]),
    });
  });
  return out;
}

/**
 * Lists a company's 10-K/10-Q filings, newest first. Frequent filers (banks)
 * push older annual reports out of "recent", so older pages are read until
 * `minTenKs` annual reports are found.
 */
export async function fetchCompanyFilings(cik: number, minTenKs = 2): Promise<CompanyFilings> {
  const submissions = await secJson<Submissions>(`https://data.sec.gov/submissions/CIK${padCik(cik)}.json`);
  const { fiscalYearEnd } = submissions;
  const filings = toFilings(cik, submissions.filings.recent, fiscalYearEnd);

  for (const page of submissions.filings.files) {
    if (filings.filter((f) => f.form === "10-K").length >= minTenKs) break;
    const older = await secJson<FilingColumns>(`https://data.sec.gov/submissions/${page.name}`);
    filings.push(...toFilings(cik, older, fiscalYearEnd));
  }

  filings.sort((a, b) => b.filedAt.localeCompare(a.filedAt));
  return { name: submissions.name, fiscalYearEnd, filings };
}
