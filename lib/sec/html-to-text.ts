import * as cheerio from "cheerio";
import type { AnyNode, Element } from "domhandler";

const BLOCK_TAGS = new Set([
  "address",
  "article",
  "blockquote",
  "br",
  "center",
  "dd",
  "div",
  "dl",
  "dt",
  "footer",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "header",
  "hr",
  "li",
  "ol",
  "p",
  "pre",
  "section",
  "table",
  "tbody",
  "thead",
  "tfoot",
  "ul",
]);

const SKIP_TAGS = new Set(["script", "style", "head", "title", "ix:header"]);

/** Currency/paren fragments that SEC tables split into their own cells. */
const DANGLING_PREFIX = /^[$€£(]$/;
const DANGLING_SUFFIX = /^[)%]$/;

function isHidden(el: Element): boolean {
  const style = el.attribs?.style;
  return !!style && /display\s*:\s*none/i.test(style);
}

function collapse(text: string): string {
  return text.replace(/[\s ​]+/g, " ");
}

/** Joins the cells of one table row, gluing "$" / ")" / "%" fragments to their numbers. */
function joinCells(cells: string[]): string {
  const out: string[] = [];
  for (const raw of cells) {
    const cell = raw.trim();
    if (!cell) continue;
    const prev = out.at(-1);
    if (prev !== undefined && DANGLING_PREFIX.test(prev)) {
      out[out.length - 1] = `${prev}${cell}`;
    } else if (prev !== undefined && DANGLING_SUFFIX.test(cell)) {
      out[out.length - 1] = `${prev}${cell}`;
    } else {
      out.push(cell);
    }
  }
  return out.join(" | ");
}

/**
 * Converts an EDGAR filing (HTML or inline-XBRL XHTML) into plain text with one
 * block element per line. Table rows become a single "cell | cell" line.
 */
export function htmlToText(html: string): string {
  const $ = cheerio.load(html);
  const lines: string[] = [];
  let current = "";

  const flush = () => {
    const line = current.trim();
    if (line) lines.push(line);
    current = "";
  };

  const cellText = (node: AnyNode): string => {
    let text = "";
    const visit = (n: AnyNode) => {
      if (n.type === "text") {
        text += (n as unknown as { data: string }).data;
        return;
      }
      if (n.type !== "tag") return;
      const el = n as Element;
      if (SKIP_TAGS.has(el.name) || isHidden(el)) return;
      if (el.name === "br") text += " ";
      for (const child of el.children) visit(child);
    };
    visit(node);
    return collapse(text).trim();
  };

  const walk = (node: AnyNode) => {
    if (node.type === "text") {
      current += collapse((node as unknown as { data: string }).data);
      return;
    }
    if (node.type !== "tag") return;
    const el = node as Element;
    if (SKIP_TAGS.has(el.name) || isHidden(el)) return;

    if (el.name === "tr") {
      flush();
      const cells = el.children
        .filter((c): c is Element => c.type === "tag" && (c.name === "td" || c.name === "th"))
        .map(cellText);
      const row = joinCells(cells);
      if (row) lines.push(row);
      return;
    }

    const isBlock = BLOCK_TAGS.has(el.name);
    if (isBlock) flush();
    for (const child of el.children) walk(child);
    if (isBlock) flush();
  };

  const root = $("body").get(0) ?? $.root().get(0);
  if (root) walk(root);
  flush();
  return removeBoilerplate(lines).join("\n");
}

/** Running headers like "Apple Inc. | 2025 Form 10-K | 12" differ only by page number. */
function boilerplateKey(line: string): string {
  return line.replace(/\d{1,3}\s*$/, "#");
}

/** Drops page numbers and running headers/footers that repeat on every page. */
function removeBoilerplate(lines: string[]): string[] {
  const counts = new Map<string, number>();
  for (const line of lines) {
    if (line.length > 90) continue;
    const key = boilerplateKey(line);
    counts.set(key, (counts.get(key) ?? 0) + 1);
  }
  return lines.filter((line) => {
    if (/^(page\s+)?\d{1,3}$/i.test(line)) return false;
    if (/^table of contents$/i.test(line)) return false;
    if (line.length > 90 || /^(part|item)\s/i.test(line)) return true;
    return (counts.get(boilerplateKey(line)) ?? 0) < 8;
  });
}
