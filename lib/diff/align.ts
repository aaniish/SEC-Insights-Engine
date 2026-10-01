import { diffWords } from "diff";

export interface Unit {
  text: string;
  vector?: number[];
}

export interface ModifiedPair {
  before: string;
  after: string;
  similarity: number;
  /** Share of words inserted or deleted, 0–1. */
  changeRatio: number;
}

export interface Alignment {
  unchanged: number;
  modified: ModifiedPair[];
  added: string[];
  removed: string[];
}

export interface AlignOptions {
  /** Below this cosine similarity two paragraphs are treated as unrelated. */
  matchThreshold?: number;
  /** Edits smaller than this share of words (e.g. "2024" → "2025") count as unchanged. */
  trivialChangeRatio?: number;
  /** Above this share of changed words, a "match" is two different paragraphs on one topic. */
  rewriteChangeRatio?: number;
}

/** Paragraph-sized units: prose lines, skipping table rows and short headings. */
export function splitParagraphs(section: string, minChars = 60, maxUnits = 700): string[] {
  return section
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length >= minChars && !line.includes(" | "))
    .slice(0, maxUnits);
}

const normalize = (text: string) => text.toLowerCase().replace(/\s+/g, " ").replace(/\d{4}/g, "#").trim();

function cosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

export function wordChangeRatio(before: string, after: string): number {
  let changed = 0;
  let total = 0;
  for (const part of diffWords(before, after)) {
    const words = part.value.split(/\s+/).filter(Boolean).length;
    total += words;
    if (part.added || part.removed) changed += words;
  }
  return total === 0 ? 0 : changed / total;
}

/** Exact matches (ignoring whitespace, case, and year numbers) need no embedding. */
export function exactMatches(
  previous: string[],
  current: string[],
): { prev: Set<number>; curr: Set<number> } {
  const byText = new Map<string, number[]>();
  previous.forEach((text, i) => {
    const key = normalize(text);
    byText.set(key, [...(byText.get(key) ?? []), i]);
  });
  const prev = new Set<number>();
  const curr = new Set<number>();
  current.forEach((text, j) => {
    const candidates = byText.get(normalize(text));
    const i = candidates?.shift();
    if (i !== undefined) {
      prev.add(i);
      curr.add(j);
    }
  });
  return { prev, curr };
}

/**
 * Aligns paragraphs of two versions of a section. Exact matches are paired
 * first; the rest are matched greedily by embedding similarity, best pairs first.
 */
export function alignParagraphs(
  previous: Unit[],
  current: Unit[],
  { matchThreshold = 0.82, trivialChangeRatio = 0.04, rewriteChangeRatio = 0.6 }: AlignOptions = {},
): Alignment {
  const exact = exactMatches(
    previous.map((u) => u.text),
    current.map((u) => u.text),
  );
  const openPrev = previous.map((_, i) => i).filter((i) => !exact.prev.has(i));
  const openCurr = current.map((_, j) => j).filter((j) => !exact.curr.has(j));

  const pairs: { i: number; j: number; similarity: number }[] = [];
  for (const i of openPrev) {
    const a = previous[i].vector;
    if (!a) continue;
    for (const j of openCurr) {
      const b = current[j].vector;
      if (!b) continue;
      const similarity = cosine(a, b);
      if (similarity >= matchThreshold) pairs.push({ i, j, similarity });
    }
  }
  pairs.sort((x, y) => y.similarity - x.similarity);

  const matchedPrev = new Set<number>();
  const matchedCurr = new Set<number>();
  let unchanged = exact.prev.size;
  const modified: ModifiedPair[] = [];

  for (const { i, j, similarity } of pairs) {
    if (matchedPrev.has(i) || matchedCurr.has(j)) continue;
    const changeRatio = wordChangeRatio(previous[i].text, current[j].text);
    if (changeRatio > rewriteChangeRatio) continue; // same topic, different paragraph
    matchedPrev.add(i);
    matchedCurr.add(j);
    if (changeRatio < trivialChangeRatio) unchanged++;
    else
      modified.push({
        before: previous[i].text,
        after: current[j].text,
        similarity,
        changeRatio,
      });
  }

  modified.sort((a, b) => b.changeRatio - a.changeRatio);
  return {
    unchanged,
    modified,
    added: openCurr.filter((j) => !matchedCurr.has(j)).map((j) => current[j].text),
    removed: openPrev.filter((i) => !matchedPrev.has(i)).map((i) => previous[i].text),
  };
}
