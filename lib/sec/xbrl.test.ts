import { describe, expect, it } from "vitest";
import { type CompanyFacts, normalizeCompanyFacts, type RawFact } from "./xbrl";

function facts(concepts: Record<string, RawFact[]>, unit = "USD"): CompanyFacts {
  return {
    facts: {
      "us-gaap": Object.fromEntries(
        Object.entries(concepts).map(([name, points]) => [name, { units: { [unit]: points } }]),
      ),
    },
  };
}

const fact = (
  start: string | undefined,
  end: string,
  val: number,
  fy: number,
  fp: string,
  filed: string,
): RawFact => ({
  start,
  end,
  val,
  accn: `acc-${filed}`,
  fy,
  fp,
  form: fp === "FY" ? "10-K" : "10-Q",
  filed,
});

const series = (points: ReturnType<typeof normalizeCompanyFacts>, metric: string) =>
  points.filter((p) => p.metric === metric).map((p) => `${p.fiscalYear}${p.fiscalPeriod}=${p.value}`);

describe("normalizeCompanyFacts", () => {
  it("derives quarters from year-to-date cash flow values", () => {
    const result = normalizeCompanyFacts(
      facts({
        NetCashProvidedByUsedInOperatingActivities: [
          fact("2025-01-01", "2025-03-31", 10, 2025, "Q1", "2025-04-30"),
          fact("2025-01-01", "2025-06-30", 25, 2025, "Q2", "2025-07-30"),
          fact("2025-01-01", "2025-09-30", 45, 2025, "Q3", "2025-10-30"),
          fact("2025-01-01", "2025-12-31", 70, 2025, "FY", "2026-02-01"),
        ],
      }),
    );
    expect(series(result, "operatingCashFlow")).toEqual([
      "2025Q1=10",
      "2025Q2=15",
      "2025Q3=20",
      "2025FY=70",
      "2025Q4=25",
    ]);
  });

  it("labels comparative periods by the filing that first reported them", () => {
    const result = normalizeCompanyFacts(
      facts({
        Revenues: [
          fact("2024-01-01", "2024-12-31", 100, 2024, "FY", "2025-02-01"),
          // The FY2025 10-K repeats FY2024 as a comparative, tagged fy=2025.
          fact("2024-01-01", "2024-12-31", 100, 2025, "FY", "2026-02-01"),
          fact("2025-01-01", "2025-12-31", 120, 2025, "FY", "2026-02-01"),
        ],
      }),
    );
    expect(series(result, "revenue")).toEqual(["2024FY=100", "2025FY=120"]);
  });

  it("prefers earlier concepts and uses later ones only to fill gaps", () => {
    const result = normalizeCompanyFacts(
      facts({
        RevenueFromContractWithCustomerExcludingAssessedTax: [
          fact("2025-01-01", "2025-12-31", 120, 2025, "FY", "2026-02-01"),
        ],
        SalesRevenueNet: [
          fact("2017-01-01", "2017-12-31", 80, 2017, "FY", "2018-02-01"),
          fact("2025-01-01", "2025-12-31", 999, 2025, "FY", "2026-02-01"),
        ],
      }),
    );
    expect(series(result, "revenue")).toEqual(["2017FY=80", "2025FY=120"]);
  });

  it("uses the latest filing when a period is restated", () => {
    const result = normalizeCompanyFacts(
      facts({
        NetIncomeLoss: [
          fact("2024-01-01", "2024-12-31", 50, 2024, "FY", "2025-02-01"),
          fact("2024-01-01", "2024-12-31", 48, 2025, "FY", "2026-02-01"),
        ],
      }),
    );
    expect(series(result, "netIncome")).toEqual(["2024FY=48"]);
  });

  it("never derives per-share quarters by subtraction", () => {
    const result = normalizeCompanyFacts(
      facts(
        {
          EarningsPerShareDiluted: [
            fact("2025-01-01", "2025-03-31", 1.1, 2025, "Q1", "2025-04-30"),
            fact("2025-01-01", "2025-06-30", 2.3, 2025, "Q2", "2025-07-30"),
            fact("2025-01-01", "2025-12-31", 5, 2025, "FY", "2026-02-01"),
          ],
        },
        "USD/shares",
      ),
    );
    expect(series(result, "epsDiluted")).toEqual(["2025Q1=1.1", "2025FY=5"]);
  });

  it("maps balance-sheet instants to fiscal periods", () => {
    const result = normalizeCompanyFacts(
      facts({
        Assets: [
          fact(undefined, "2025-06-30", 900, 2025, "Q2", "2025-07-30"),
          fact(undefined, "2025-12-31", 1000, 2025, "FY", "2026-02-01"),
        ],
      }),
    );
    expect(series(result, "totalAssets")).toEqual(["2025Q2=900", "2025FY=1000", "2025Q4=1000"]);
  });
});
