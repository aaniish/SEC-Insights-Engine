export interface TextChunk {
  index: number;
  content: string;
  /** A short verbatim phrase for `#:~:text=` deep links into the filing on sec.gov. */
  anchorText: string;
}

export interface ChunkOptions {
  maxChars?: number;
  overlapChars?: number;
}

const SENTENCE_BOUNDARY = /(?<=[.!?])\s+(?=[A-Z(“"])/;

/** Breaks one oversized paragraph at sentence boundaries, hard-splitting run-ons. */
function splitLongParagraph(paragraph: string, maxChars: number): string[] {
  const pieces: string[] = [];
  let current = "";
  for (const sentence of paragraph.split(SENTENCE_BOUNDARY)) {
    if (sentence.length > maxChars) {
      if (current) pieces.push(current);
      current = "";
      for (let i = 0; i < sentence.length; i += maxChars) pieces.push(sentence.slice(i, i + maxChars));
      continue;
    }
    if (current && current.length + sentence.length + 1 > maxChars) {
      pieces.push(current);
      current = sentence;
    } else {
      current = current ? `${current} ${sentence}` : sentence;
    }
  }
  if (current) pieces.push(current);
  return pieces;
}

/** Trailing paragraphs of a chunk, up to `overlapChars`, carried into the next chunk. */
function overlapTail(paragraphs: string[], overlapChars: number): string[] {
  const tail: string[] = [];
  let size = 0;
  for (let i = paragraphs.length - 1; i >= 0; i--) {
    size += paragraphs[i].length + 1;
    if (size > overlapChars) break;
    tail.unshift(paragraphs[i]);
  }
  return tail;
}

/** First ~8 words of the first prose line (table rows don't appear verbatim on the page). */
export function anchorTextFor(content: string): string {
  const line = content.split("\n").find((l) => !l.includes(" | ") && l.length >= 40) ?? "";
  return line
    .split(/\s+/)
    .slice(0, 8)
    .join(" ")
    .replace(/[,;:.]+$/, "");
}

/**
 * Packs paragraphs (one per line) into chunks of at most `maxChars`, with
 * paragraph-level overlap so sentences near a boundary keep their context.
 */
export function chunkText(
  text: string,
  { maxChars = 1400, overlapChars = 200 }: ChunkOptions = {},
): TextChunk[] {
  const paragraphs = text
    .split("\n")
    .map((p) => p.trim())
    .filter(Boolean)
    .flatMap((p) => (p.length > maxChars ? splitLongParagraph(p, maxChars) : [p]));

  const chunks: string[] = [];
  let current: string[] = [];
  let carried = 0; // leading paragraphs of `current` that repeat the previous chunk
  let size = 0;

  for (const paragraph of paragraphs) {
    if (size > 0 && size + paragraph.length + 1 > maxChars) {
      if (current.length > carried) {
        chunks.push(current.join("\n"));
        current = overlapTail(current, overlapChars);
      } else {
        current = []; // only overlap so far; emitting it would duplicate text
      }
      carried = current.length;
      size = current.reduce((total, p) => total + p.length + 1, 0);
      if (size + paragraph.length + 1 > maxChars) {
        current = [];
        carried = 0;
        size = 0;
      }
    }
    current.push(paragraph);
    size += paragraph.length + 1;
  }
  if (current.length > carried) chunks.push(current.join("\n"));

  return chunks.map((content, index) => ({
    index,
    content,
    anchorText: anchorTextFor(content),
  }));
}
