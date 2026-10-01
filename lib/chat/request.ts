import { z } from "zod";
import type { ChatMessage } from "@/lib/ai/types";

/** The composer allows 1,000 characters; this leaves slack without letting a request carry a novel. */
export const MAX_QUESTION_CHARS = 2000;
/** Earlier answers are resent as context; anything longer is trimmed. */
const MAX_ANSWER_CHARS = 6000;
const MAX_HISTORY = 10;

const messageSchema = z.object({
  id: z.string().max(200),
  // Only the user and assistant speak. A client-sent "system" message would override the instructions.
  role: z.enum(["user", "assistant"]),
  parts: z.array(z.looseObject({ type: z.string(), text: z.unknown().optional() })).max(300),
});

const bodySchema = z.object({
  messages: z.array(messageSchema).min(1).max(500),
  tickers: z
    .array(z.string().regex(/^[A-Za-z0-9.-]{1,10}$/))
    .max(3)
    .default([])
    .transform((t) => t.map((x) => x.toUpperCase())),
  mode: z.enum(["fast", "deep"]).default("fast"),
});

export type ChatRequest = { messages: ChatMessage[]; tickers: string[]; mode: "fast" | "deep" };

function textOf(parts: { type: string; text?: unknown }[]): string {
  return parts.flatMap((p) => (p.type === "text" && typeof p.text === "string" ? [p.text] : [])).join("\n");
}

/**
 * Validates a chat request and reduces the conversation to the last few turns as plain text.
 * Earlier answers' tool calls and results are dropped, so a client can't hand the model
 * fabricated search results, and resending old passages would multiply token costs anyway.
 */
export function parseChatRequest(
  body: unknown,
): { ok: true; request: ChatRequest } | { ok: false; error: string } {
  const parsed = bodySchema.safeParse(body);
  if (!parsed.success) return { ok: false, error: "Invalid request." };

  const messages = parsed.data.messages
    .slice(-MAX_HISTORY)
    .map((m) => {
      const text = textOf(m.parts);
      return {
        id: m.id,
        role: m.role,
        text: m.role === "assistant" ? text.slice(0, MAX_ANSWER_CHARS) : text,
      };
    })
    .filter((m) => m.text.trim());

  if (messages.at(-1)?.role !== "user") return { ok: false, error: "Invalid request." };
  if (messages.some((m) => m.role === "user" && m.text.length > MAX_QUESTION_CHARS))
    return {
      ok: false,
      error: `Questions are limited to ${MAX_QUESTION_CHARS.toLocaleString("en-US")} characters.`,
    };

  return {
    ok: true,
    request: {
      tickers: parsed.data.tickers,
      mode: parsed.data.mode,
      messages: messages.map(({ id, role, text }) => ({ id, role, parts: [{ type: "text", text }] })),
    },
  };
}
