import { describe, expect, it } from "vitest";
import { anchorTextFor, chunkText } from "./chunk";

const paragraph = (i: number, length = 300) => `Paragraph ${i} ${"word ".repeat(length / 5)}`.trim();

describe("chunkText", () => {
  it("keeps every chunk under the size limit and covers all paragraphs", () => {
    const paragraphs = Array.from({ length: 30 }, (_, i) => paragraph(i));
    const chunks = chunkText(paragraphs.join("\n"), {
      maxChars: 1400,
      overlapChars: 200,
    });
    expect(chunks.length).toBeGreaterThan(5);
    for (const c of chunks) expect(c.content.length).toBeLessThanOrEqual(1400);
    const joined = chunks.map((c) => c.content).join("\n");
    for (const p of paragraphs) expect(joined).toContain(p);
    expect(chunks.map((c) => c.index)).toEqual(chunks.map((_, i) => i));
  });

  it("carries a small overlap between consecutive chunks", () => {
    const text = Array.from({ length: 12 }, (_, i) => paragraph(i, 150)).join("\n");
    const [first, second] = chunkText(text, {
      maxChars: 700,
      overlapChars: 200,
    });
    const lastOfFirst = first.content.split("\n").at(-1) as string;
    expect(second.content.startsWith(lastOfFirst)).toBe(true);
  });

  it("never emits a chunk that is only overlap", () => {
    const text = [paragraph(1, 600), paragraph(2, 150), paragraph(3, 1300), paragraph(4, 1300)].join("\n");
    const chunks = chunkText(text, { maxChars: 1400, overlapChars: 200 });
    const contents = chunks.map((c) => c.content);
    expect(new Set(contents).size).toBe(contents.length);
    for (let i = 1; i < contents.length; i++) expect(contents[i - 1].endsWith(contents[i])).toBe(false);
  });

  it("splits a single oversized paragraph at sentence boundaries", () => {
    const sentence = `This is a sentence about supply chain risk ${"x".repeat(80)}.`;
    const chunks = chunkText(Array.from({ length: 40 }, () => sentence).join(" "), { maxChars: 600 });
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.content.length).toBeLessThanOrEqual(600);
      expect(c.content.startsWith("This is a sentence")).toBe(true);
    }
  });
});

describe("anchorTextFor", () => {
  it("skips table rows and trims to the first words of prose", () => {
    const content =
      "Net sales | $391,035 | $383,285\nThe Company’s net sales increased during 2025 compared to 2024, driven by iPhone.";
    expect(anchorTextFor(content)).toBe("The Company’s net sales increased during 2025 compared");
  });
});
