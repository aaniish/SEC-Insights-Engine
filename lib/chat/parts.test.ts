import { describe, expect, it } from "vitest";
import type { Citation } from "@/lib/ai/tools";
import { citedNumbers, linkCitations } from "./parts";

const known = new Map<number, Citation>([1, 2, 3, 5].map((n) => [n, { n } as Citation]));

describe("linkCitations", () => {
  it("links single, grouped, and ranged citations", () => {
    expect(linkCitations("Risk [1]. More [2, 3]. Range [1–3].", known)).toBe(
      "Risk [1](#cite-1). More [2](#cite-2)[3](#cite-3). Range [1](#cite-1)[2](#cite-2)[3](#cite-3).",
    );
  });

  it("leaves unknown numbers and ordinary brackets alone", () => {
    expect(linkCitations("See [9] and [note] and [4].", known)).toBe("See [9] and [note] and [4].");
    expect(linkCitations("Mixed [4, 5]", known)).toBe("Mixed [5](#cite-5)");
  });
});

describe("citedNumbers", () => {
  it("returns first-appearance order without duplicates", () => {
    expect(citedNumbers("a [3] b [1, 3] c [2–3]")).toEqual([3, 1, 2]);
  });
});
