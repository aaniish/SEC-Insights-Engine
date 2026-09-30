import type { Citation } from "@/lib/ai/tools";
import type { ChatMessage } from "@/lib/ai/types";

export type MessagePart = ChatMessage["parts"][number];
export type ToolPart = Extract<MessagePart, { type: `tool-${string}` }>;
export type ToolName = ToolPart["type"] extends `tool-${infer N}` ? N : never;

export function isToolPart(part: MessagePart): part is ToolPart {
  return part.type.startsWith("tool-");
}

export function toolName(part: ToolPart): ToolName {
  return part.type.slice(5) as ToolName;
}

export function answerText(message: ChatMessage): string {
  return message.parts
    .filter((p) => p.type === "text")
    .map((p) => p.text)
    .join("\n\n");
}

/** Every passage returned by searchFilings in this answer, keyed by citation number. */
export function citationsOf(message: ChatMessage): Map<number, Citation> {
  const map = new Map<number, Citation>();
  for (const part of message.parts) {
    if (part.type === "tool-searchFilings" && part.state === "output-available") {
      for (const citation of part.output.results) map.set(citation.n, citation);
    }
  }
  return map;
}

const CITATION_GROUP = /\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]/g;

/** Citation numbers in the order they first appear in the answer. */
export function citedNumbers(text: string): number[] {
  const seen: number[] = [];
  for (const match of text.matchAll(CITATION_GROUP)) {
    for (const n of expand(match[1])) if (!seen.includes(n)) seen.push(n);
  }
  return seen;
}

function expand(group: string): number[] {
  return group.split(",").flatMap((piece) => {
    const [from, to] = piece.split(/[–-]/).map((x) => Number(x.trim()));
    if (!to || to < from || to - from > 20) return [from];
    return Array.from({ length: to - from + 1 }, (_, i) => from + i);
  });
}

/** Turns "[3]", "[3, 5]" and "[3–5]" into markdown links the renderer shows as citation markers. */
export function linkCitations(text: string, known: Map<number, Citation>): string {
  return text.replace(CITATION_GROUP, (whole, group: string) => {
    const numbers = expand(group).filter((n) => known.has(n));
    if (numbers.length === 0) return whole;
    return numbers.map((n) => `[${n}](#cite-${n})`).join("");
  });
}

export function suggestionsOf(message: ChatMessage): string[] {
  const part = message.parts.find((p) => p.type === "data-suggestions");
  return part?.type === "data-suggestions" ? part.data.questions : [];
}

/** Groups a flat message list into question → answer turns. */
export function toTurns(messages: ChatMessage[]): { question: ChatMessage; answer?: ChatMessage }[] {
  const turns: { question: ChatMessage; answer?: ChatMessage }[] = [];
  for (const message of messages) {
    if (message.role === "user") turns.push({ question: message });
    else if (turns.length > 0) turns[turns.length - 1].answer = message;
  }
  return turns;
}
