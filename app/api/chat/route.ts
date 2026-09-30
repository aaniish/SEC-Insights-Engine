import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  isStepCount,
  streamText,
} from "ai";
import { z } from "zod";
import { chatModel, reasoningOptions } from "@/lib/ai/models";
import { systemPrompt } from "@/lib/ai/prompts";
import { suggestFollowUps } from "@/lib/ai/suggestions";
import { createTools } from "@/lib/ai/tools";
import type { ChatMessage } from "@/lib/ai/types";
import { clientKey, consumeLimit, limitMessage } from "@/lib/rate-limit";

export const maxDuration = 300;

const MAX_HISTORY = 10;

const bodySchema = z.object({
  messages: z.array(z.custom<ChatMessage>()).min(1),
  tickers: z
    .array(z.string().max(10))
    .max(3)
    .default([])
    .transform((t) => t.map((x) => x.toUpperCase())),
  mode: z.enum(["fast", "deep"]).default("fast"),
});

/**
 * Earlier turns keep only their text: resending old search passages and chart
 * data on every follow-up would multiply token costs for little benefit.
 */
function compactHistory(messages: ChatMessage[]): ChatMessage[] {
  const recent = messages.slice(-MAX_HISTORY);
  return recent
    .map((message, i) =>
      i === recent.length - 1
        ? message
        : { ...message, parts: message.parts.filter((p) => p.type === "text") },
    )
    .filter((message) => message.parts.length > 0);
}

function lastUserText(messages: ChatMessage[]): string {
  const last = messages.findLast((m) => m.role === "user");
  return last?.parts.map((p) => (p.type === "text" ? p.text : "")).join(" ") ?? "";
}

export async function POST(req: Request) {
  const parsed = bodySchema.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: "Invalid request." }, { status: 400 });
  const { tickers, mode } = parsed.data;

  const key = clientKey(req.headers);
  const kind = mode === "deep" ? "deep" : "question";
  const limit = await consumeLimit(kind, key);
  if (!limit.ok) return Response.json({ error: limitMessage(kind, limit.reason) }, { status: 429 });

  const messages = compactHistory(parsed.data.messages);
  const question = lastUserText(messages);
  const today = new Date().toISOString().slice(0, 10);

  const stream = createUIMessageStream<ChatMessage>({
    execute: async ({ writer }) => {
      const result = streamText({
        model: chatModel(mode),
        instructions: systemPrompt({ tickers, today }),
        messages: await convertToModelMessages(messages),
        tools: createTools({ clientKey: key, deep: mode === "deep" }),
        stopWhen: isStepCount(mode === "deep" ? 12 : 8),
        providerOptions: reasoningOptions(mode === "deep" ? "medium" : "low"),
      });

      // Forward chunks in order (not merge) so suggestions land after the answer.
      for await (const chunk of result.toUIMessageStream<ChatMessage>({
        sendFinish: false,
      }))
        writer.write(chunk);

      const questions = await suggestFollowUps({
        question,
        answer: await result.text,
        tickers,
      });
      if (questions.length > 0)
        writer.write({
          type: "data-suggestions",
          id: "suggestions",
          data: { questions },
        });
      writer.write({ type: "finish" });
    },
    onError: (error) => {
      console.error("chat error", error);
      return "Something went wrong while researching that. Please try again.";
    },
  });

  return createUIMessageStreamResponse({ stream });
}
