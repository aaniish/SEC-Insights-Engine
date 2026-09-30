import { describe, expect, it } from "vitest";
import { extractFigures, mentionsFigure } from "./figures";

describe("extractFigures", () => {
  it("reads scaled dollar amounts, percentages, and per-share values", () => {
    const figures = extractFigures("Revenue was $416.2B (up 6.4%) and EPS was $7.46 in FY2025.");
    expect(figures).toEqual([
      { value: 416.2e9, kind: "money" },
      { value: 6.4, kind: "percent" },
      { value: 7.46, kind: "money" },
    ]);
  });

  it("handles words, commas, and negatives", () => {
    expect(extractFigures("a loss of −$1.2 billion on $391,035 million")).toEqual([
      { value: -1.2e9, kind: "money" },
      { value: 391_035e6, kind: "money" },
    ]);
  });
});

describe("mentionsFigure", () => {
  it("matches within tolerance after rounding", () => {
    expect(mentionsFigure("Apple reported $416.2 billion of revenue.", 416_161_000_000)).toBe(true);
    expect(mentionsFigure("Apple reported $391.0B of revenue.", 416_161_000_000)).toBe(false);
    expect(mentionsFigure("Gross margin was 71.1%.", 71.08)).toBe(true);
  });
});
