/**
 * Live tests against SEC EDGAR (network). Run with `pnpm test:live`.
 * They catch format drift in real filings that fixtures can't.
 */
import { describe, expect, it } from "vitest";
import { secText } from "./client";
import { fetchCompanyFilings } from "./edgar";
import { htmlToText } from "./html-to-text";
import { splitSections } from "./sections";

const live = process.env.RUN_LIVE ? describe : describe.skip;

async function latestTenKSections(cik: number) {
  const { filings } = await fetchCompanyFilings(cik, 1);
  const tenK = filings.find((f) => f.form === "10-K");
  if (!tenK) throw new Error(`No 10-K for CIK ${cik}`);
  return splitSections(htmlToText(await secText(tenK.docUrl)), "10-K");
}

live("real 10-K parsing", () => {
  it.each([
    ["Apple", 320193],
    ["Tesla", 1318605],
  ])(
    "finds substantial Risk Factors and MD&A for %s",
    async (_name, cik) => {
      const sections = await latestTenKSections(cik);
      const byCode = new Map(sections.map((s) => [s.itemCode, s]));
      expect(byCode.get("1A")?.content.length).toBeGreaterThan(30_000);
      expect(byCode.get("7")?.content.length).toBeGreaterThan(10_000);
    },
    60_000,
  );

  it("recovers MD&A from JPMorgan's appended annual report", async () => {
    const sections = await latestTenKSections(19617);
    const annualReport = sections.find((s) => s.itemCode === "AR");
    expect(annualReport?.topic).toBe("mdna");
    expect(annualReport?.content.length).toBeGreaterThan(200_000);
    expect(sections.find((s) => s.itemCode === "1A")?.content.length).toBeGreaterThan(50_000);
  }, 90_000);
});
