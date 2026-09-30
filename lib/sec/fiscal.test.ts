import { describe, expect, it } from "vitest";
import { fiscalPeriodOf } from "./fiscal";

describe("fiscalPeriodOf", () => {
  it("handles 52/53-week years ending a day after the nominal date (Apple)", () => {
    expect(fiscalPeriodOf("2025-09-27", "0926", "10-K")).toEqual({
      fiscalYear: 2025,
      fiscalPeriod: "FY",
    });
    expect(fiscalPeriodOf("2025-12-27", "0926", "10-Q")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "Q1",
    });
    expect(fiscalPeriodOf("2026-06-27", "0926", "10-Q")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "Q3",
    });
  });

  it("handles calendar-year filers", () => {
    expect(fiscalPeriodOf("2025-12-31", "1231", "10-K")).toEqual({
      fiscalYear: 2025,
      fiscalPeriod: "FY",
    });
    expect(fiscalPeriodOf("2026-06-30", "1231", "10-Q")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "Q2",
    });
  });

  it("names a January year-end after the calendar year it ends in (Walmart)", () => {
    expect(fiscalPeriodOf("2026-01-31", "0131", "10-K")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "FY",
    });
    expect(fiscalPeriodOf("2025-10-31", "0131", "10-Q")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "Q3",
    });
  });

  it("defaults to a December year-end when unknown", () => {
    expect(fiscalPeriodOf("2026-03-31", null, "10-Q")).toEqual({
      fiscalYear: 2026,
      fiscalPeriod: "Q1",
    });
  });
});
