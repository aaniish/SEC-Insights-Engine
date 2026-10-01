import { describe, expect, it } from "vitest";
import { alignParagraphs, splitParagraphs, wordChangeRatio } from "./align";

/** Tiny fake embeddings: similar paragraphs get nearby vectors. */
const unit = (text: string, vector: number[]) => ({ text, vector });

describe("alignParagraphs", () => {
  const supply =
    "We depend on outsourcing partners in Asia to manufacture most of our products and components.";
  const supplyReworded =
    "We depend on outsourcing partners in Asia, India and Vietnam to manufacture substantially all of our products.";
  const tariffs = "New tariffs on imports from China could materially increase our costs and reduce demand.";
  const ai =
    "Our generative AI features may not be adopted by customers and could expose us to new liability.";
  const legacy =
    "The transition to our new ERP system may disrupt financial reporting processes during the year.";

  it("classifies unchanged, reworded, added and removed paragraphs", () => {
    const previous = [unit(supply, [1, 0, 0]), unit(tariffs, [0, 1, 0]), unit(legacy, [0, 0, 1])];
    const current = [
      unit(supplyReworded, [0.95, 0.05, 0]),
      unit(tariffs, [0, 1, 0]),
      unit(ai, [0.1, 0.1, -1]),
    ];
    const result = alignParagraphs(previous, current);
    expect(result.unchanged).toBe(1);
    expect(result.modified).toHaveLength(1);
    expect(result.modified[0]).toMatchObject({
      before: supply,
      after: supplyReworded,
    });
    expect(result.added).toEqual([ai]);
    expect(result.removed).toEqual([legacy]);
  });

  it("treats year-only edits as unchanged", () => {
    const before =
      "As of September 2024, the Company had approximately 164,000 full-time equivalent employees worldwide.";
    const after =
      "As of September 2025, the Company had approximately 164,000 full-time equivalent employees worldwide.";
    const result = alignParagraphs([unit(before, [1, 0])], [unit(after, [0.99, 0.01])]);
    expect(result).toMatchObject({
      unchanged: 1,
      modified: [],
      added: [],
      removed: [],
    });
  });

  it("matches each paragraph at most once, best similarity first", () => {
    const previous = [
      unit("Demand for our products depends on consumer spending in our largest markets", [1, 0]),
    ];
    const current = [
      unit("Demand for our services depends on business spending in our largest markets", [0.9, 0.44]),
      unit("Demand for our products depends on consumer spending in our closer markets", [0.99, 0.14]),
    ];
    const result = alignParagraphs(previous, current, {
      trivialChangeRatio: 0,
    });
    expect(result.modified).toHaveLength(1);
    expect(result.modified[0].after).toContain("closer");
    expect(result.added).toHaveLength(1);
  });

  it("splits a semantically similar but rewritten pair into removed + added", () => {
    const before =
      "We have experienced launch and production ramp delays for new vehicle programs in the past.";
    const after =
      "Our robotaxi service depends on regulators approving unsupervised driving in each new city.";
    const result = alignParagraphs([unit(before, [1, 0])], [unit(after, [0.95, 0.31])]);
    expect(result).toMatchObject({
      modified: [],
      added: [after],
      removed: [before],
    });
  });
});

describe("splitParagraphs", () => {
  it("drops table rows and short lines", () => {
    const text = ["Risk Factors", "Net sales | $100 | $90", "A".repeat(80), "Short line"].join("\n");
    expect(splitParagraphs(text)).toEqual(["A".repeat(80)]);
  });
});

describe("wordChangeRatio", () => {
  it("is zero for identical text and grows with edits", () => {
    expect(wordChangeRatio("a b c d", "a b c d")).toBe(0);
    expect(wordChangeRatio("a b c d", "a x c d")).toBeGreaterThan(0.2);
  });
});
