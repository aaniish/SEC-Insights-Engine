export type FilingForm = "10-K" | "10-Q";

export type SectionTopic =
  | "business"
  | "risk_factors"
  | "mdna"
  | "market_risk"
  | "financial_statements"
  | "legal"
  | "other";

export interface FilingSection {
  /** "1A", "7" for 10-Ks; "I-2", "II-1A" for 10-Qs; "AR" for an appended annual report. */
  itemCode: string;
  title: string;
  topic: SectionTopic;
  content: string;
}

const TEN_K_TITLES: Record<string, string> = {
  "1": "Business",
  "1A": "Risk Factors",
  "1B": "Unresolved Staff Comments",
  "1C": "Cybersecurity",
  "2": "Properties",
  "3": "Legal Proceedings",
  "4": "Mine Safety Disclosures",
  "5": "Market for Common Equity",
  "6": "Reserved",
  "7": "Management's Discussion and Analysis",
  "7A": "Market Risk Disclosures",
  "8": "Financial Statements",
  "9": "Changes in Accountants",
  "9A": "Controls and Procedures",
  "9B": "Other Information",
  "9C": "Foreign Jurisdiction Inspections",
  "10": "Directors and Corporate Governance",
  "11": "Executive Compensation",
  "12": "Security Ownership",
  "13": "Related Transactions",
  "14": "Accountant Fees",
  "15": "Exhibits",
  "16": "Form 10-K Summary",
  AR: "Annual Report (MD&A and Financial Statements)",
};

const TEN_Q_TITLES: Record<string, string> = {
  "I-1": "Financial Statements",
  "I-2": "Management's Discussion and Analysis",
  "I-3": "Market Risk Disclosures",
  "I-4": "Controls and Procedures",
  "II-1": "Legal Proceedings",
  "II-1A": "Risk Factors",
  "II-2": "Unregistered Sales of Equity",
  "II-3": "Defaults Upon Senior Securities",
  "II-4": "Mine Safety Disclosures",
  "II-5": "Other Information",
  "II-6": "Exhibits",
};

const TOPICS: Record<string, SectionTopic> = {
  "1": "business",
  "1A": "risk_factors",
  "3": "legal",
  "7": "mdna",
  "7A": "market_risk",
  "8": "financial_statements",
  AR: "mdna",
  "I-1": "financial_statements",
  "I-2": "mdna",
  "I-3": "market_risk",
  "II-1": "legal",
  "II-1A": "risk_factors",
};

/** Items that are boilerplate or exhibit lists; skipped to save vector storage. */
const LOW_VALUE = new Set(["6", "15", "16", "II-6"]);

const ITEM_HEADING = /^item\s*(\d{1,2}[a-c]?)\b\s*[.:\-–—|]?\s*(.*)$/i;
const PART_HEADING = /^part\s+(iv|iii|ii|i)\b/i;
const MAX_HEADING_LENGTH = 220;

export function sectionTitle(form: FilingForm, itemCode: string): string {
  const titles = form === "10-K" ? TEN_K_TITLES : TEN_Q_TITLES;
  return titles[itemCode] ?? `Item ${itemCode}`;
}

export function sectionTopic(itemCode: string): SectionTopic {
  return TOPICS[itemCode] ?? "other";
}

interface Heading {
  code: string;
  line: number;
}

function findHeadings(lines: string[], form: FilingForm): Heading[] {
  const headings: Heading[] = [];
  let part = "I";
  lines.forEach((line, index) => {
    if (line.length > MAX_HEADING_LENGTH) return;
    const partMatch = PART_HEADING.exec(line);
    if (partMatch) {
      part = partMatch[1].toUpperCase();
      return;
    }
    const itemMatch = ITEM_HEADING.exec(line);
    if (!itemMatch) return;
    const item = itemMatch[1].toUpperCase();
    headings.push({
      code: form === "10-Q" ? `${part}-${item}` : item,
      line: index,
    });
  });
  return headings;
}

/** Table-of-contents rows end in a page number or range: "Item 1A. | Risk Factors | 9-31". */
const TOC_PAGE_SUFFIX = /(\||\s)\s*\d{1,3}(\s*[-–]\s*\d{1,3})?\s*$/;

function withoutTableOfContents(headings: Heading[], lines: string[]): Heading[] {
  const body = headings.filter((h) => !TOC_PAGE_SUFFIX.test(lines[h.line]));
  return body.length > 0 ? body : headings;
}

/** Catches tables of contents without page numbers: a "section" that is mostly headings. */
function isMostlyHeadings(content: string): boolean {
  const lines = content.split("\n").filter(Boolean);
  if (lines.length === 0) return true;
  const headingLines = lines.filter((l) => ITEM_HEADING.test(l) || PART_HEADING.test(l)).length;
  return headingLines / lines.length >= 0.5;
}

function segmentLength(lines: string[], from: number, to: number): number {
  let total = 0;
  for (let i = from; i < to; i++) total += lines[i].length + 1;
  return total;
}

/**
 * Splits filing text into Items. Each Item heading appears twice (table of
 * contents + body), so we keep the occurrence that starts the longest segment
 * and use only those chosen starts as boundaries.
 */
export function splitSections(text: string, form: FilingForm): FilingSection[] {
  const lines = text.split("\n");
  const headings = withoutTableOfContents(findHeadings(lines, form), lines);
  if (headings.length === 0) return [fullDocument(text)];

  const best = new Map<string, { line: number; length: number }>();
  headings.forEach((heading, i) => {
    const end = headings[i + 1]?.line ?? lines.length;
    const length = segmentLength(lines, heading.line, end);
    const previous = best.get(heading.code);
    if (!previous || length > previous.length) best.set(heading.code, { line: heading.line, length });
  });

  const starts = [...best.entries()]
    .map(([code, { line }]) => ({ code, line }))
    .sort((a, b) => a.line - b.line);

  let sections = starts.map(({ code, line }, i) => {
    const end = starts[i + 1]?.line ?? lines.length;
    // Drop the heading line itself; the canonical title replaces it.
    const content = lines
      .slice(line + 1, end)
      .join("\n")
      .trim();
    return {
      itemCode: code,
      title: sectionTitle(form, code),
      topic: sectionTopic(code),
      content,
    };
  });

  if (form === "10-K") sections = splitAppendedAnnualReport(sections, text.length);

  const kept = sections.filter(
    (s) => s.content.length >= 200 && !LOW_VALUE.has(s.itemCode) && !isMostlyHeadings(s.content),
  );
  return kept.length > 0 ? kept : [fullDocument(text)];
}

/**
 * Some filers (mostly banks) put MD&A and the financial statements in an annual
 * report appended after Item 15, leaving Item 7 as a one-line cross reference.
 */
function splitAppendedAnnualReport(sections: FilingSection[], totalLength: number): FilingSection[] {
  const mdna = sections.find((s) => s.itemCode === "7");
  const last = sections.at(-1);
  if (!mdna || !last || mdna.content.length > 4000) return sections;
  if (last.content.length < totalLength * 0.4) return sections;

  const lines = last.content.split("\n");
  // Skip exhibit-index footnotes like "(c) Furnished herewith. ..." which can be long.
  const firstProse = lines.findIndex((line) => line.length > 400 && !/^[(\d]/.test(line));
  if (firstProse <= 0) return sections;

  // Include the short heading lines just above the first paragraph.
  const start = Math.max(0, firstProse - 3);
  return [
    ...sections.slice(0, -1),
    { ...last, content: lines.slice(0, start).join("\n").trim() },
    {
      itemCode: "AR",
      title: TEN_K_TITLES.AR,
      topic: "mdna",
      content: lines.slice(start).join("\n").trim(),
    },
  ];
}

function fullDocument(text: string): FilingSection {
  return {
    itemCode: "DOC",
    title: "Full Document",
    topic: "other",
    content: text.trim(),
  };
}
