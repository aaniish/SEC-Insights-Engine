export interface Figure {
  value: number;
  kind: "money" | "percent" | "number";
}

const SCALE: Record<string, number> = {
  t: 1e12,
  trillion: 1e12,
  b: 1e9,
  bn: 1e9,
  billion: 1e9,
  m: 1e6,
  mm: 1e6,
  million: 1e6,
  k: 1e3,
  thousand: 1e3,
};

const FIGURE =
  /(-|−|\()?\$?\s?(\d[\d,]*(?:\.\d+)?)\s?(trillion|billion|million|thousand|bn|mm|[tbmk])?\b(%)?/gi;

/** Extracts dollar amounts, percentages, and plain numbers ("$416.2B", "71.1%", "$7.46") from prose. */
export function extractFigures(text: string): Figure[] {
  const figures: Figure[] = [];
  for (const match of text.matchAll(FIGURE)) {
    const [whole, negative, digits, scale, percent] = match;
    if (/^(19|20)\d{2}$/.test(digits) && !scale && !percent && !whole.includes("$")) continue; // a year
    let value = Number(digits.replaceAll(",", ""));
    if (Number.isNaN(value)) continue;
    if (scale) value *= SCALE[scale.toLowerCase()];
    if (negative) value = -value;
    const kind = percent ? "percent" : whole.includes("$") || scale ? "money" : "number";
    figures.push({ value, kind });
  }
  return figures;
}

/** True if any figure in the text is within `tolerance` (relative) of the expected value. */
export function mentionsFigure(text: string, expected: number, tolerance = 0.01): boolean {
  return extractFigures(text).some(({ value }) => {
    if (expected === 0) return value === 0;
    return Math.abs(value - expected) / Math.abs(expected) <= tolerance;
  });
}
