import { describe, expect, it } from "vitest";
import { splitSections } from "./sections";

const para = (label: string, n = 3) =>
  Array.from(
    { length: n },
    (_, i) => `${label} paragraph ${i + 1}. ${"Lorem ipsum dolor sit amet. ".repeat(12)}`,
  ).join("\n");

const tenK = [
  "UNITED STATES SECURITIES AND EXCHANGE COMMISSION",
  "Part I",
  "Item 1. | Business | 1",
  "Item 1A. | Risk Factors | 5",
  "Item 1B. | Unresolved Staff Comments | 17",
  "Part II",
  "Item 7. | Management’s Discussion and Analysis | 21",
  "Item 15. | Exhibits | 54",
  "PART I",
  "Item 1. Business",
  para("Business"),
  "Item 1A. Risk Factors",
  para("Risk"),
  "Item 1B. Unresolved Staff Comments",
  "None.",
  "PART II",
  "Item 7. Management’s Discussion and Analysis of Financial Condition",
  para("MDNA"),
  "Item 15. Exhibit and Financial Statement Schedules",
  "3.1 | Articles of Incorporation",
].join("\n");

describe("splitSections", () => {
  it("uses body headings, not the table of contents", () => {
    const sections = splitSections(tenK, "10-K");
    expect(sections.map((s) => s.itemCode)).toEqual(["1", "1A", "7"]);
    expect(sections[1]).toMatchObject({
      title: "Risk Factors",
      topic: "risk_factors",
    });
    expect(sections[1].content).toMatch(/^Risk paragraph 1/);
    expect(sections[1].content).not.toMatch(/Item 1B/);
    expect(sections[2].topic).toBe("mdna");
  });

  it("drops trivial and exhibit-only items", () => {
    const codes = splitSections(tenK, "10-K").map((s) => s.itemCode);
    expect(codes).not.toContain("1B");
    expect(codes).not.toContain("15");
  });

  it("ignores a table of contents without page numbers", () => {
    const text = [
      "Item 1. Business",
      "Item 1A. Risk Factors",
      "Item 7. MD&A",
      "Item 1. Business",
      para("Business"),
      "Item 1A. Risk Factors",
      para("Risk"),
      "Item 7. MD&A",
      para("MDNA"),
    ].join("\n");
    const sections = splitSections(text, "10-K");
    expect(sections.map((s) => s.itemCode)).toEqual(["1", "1A", "7"]);
    for (const s of sections) expect(s.content).not.toMatch(/^Item/m);
  });

  it("keys 10-Q items by part", () => {
    const tenQ = [
      "PART I — FINANCIAL INFORMATION",
      "Item 1. Financial Statements",
      para("Statements"),
      "Item 2. Management’s Discussion and Analysis",
      para("MDNA"),
      "PART II — OTHER INFORMATION",
      "Item 1. Legal Proceedings",
      para("Legal"),
      "Item 1A. Risk Factors",
      para("Risk"),
    ].join("\n");
    const sections = splitSections(tenQ, "10-Q");
    expect(sections.map((s) => [s.itemCode, s.topic])).toEqual([
      ["I-1", "financial_statements"],
      ["I-2", "mdna"],
      ["II-1", "legal"],
      ["II-1A", "risk_factors"],
    ]);
  });

  it("splits an annual report appended after Item 15 into its own MD&A section", () => {
    const annualReport = Array.from(
      { length: 40 },
      (_, i) => `Annual report prose ${i}. ${"x".repeat(420)}`,
    ).join("\n");
    const text = [
      "Item 1. Business",
      para("Business"),
      "Item 7. Management’s Discussion and Analysis",
      "Management’s discussion and analysis appears on pages 46–160.",
      "Item 15. Exhibits",
      "3.1 | Restated Certificate",
      "(a) Filed herewith.",
      annualReport,
    ].join("\n");
    const sections = splitSections(text, "10-K");
    const report = sections.find((s) => s.itemCode === "AR");
    expect(report?.topic).toBe("mdna");
    expect(report?.content).toMatch(/^3\.1|^\(a\)|^Annual report prose 0/);
    expect(report?.content).toContain("Annual report prose 39");
  });

  it("falls back to the whole document when there are no item headings", () => {
    const sections = splitSections(para("Plain", 2), "10-K");
    expect(sections).toHaveLength(1);
    expect(sections[0].itemCode).toBe("DOC");
  });
});
