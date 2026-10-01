import {
  convertToModelMessages,
  createUIMessageStream,
  createUIMessageStreamResponse,
  isStepCount,
  streamText,
} from "ai";
import { chatModel, reasoningOptions } from "@/lib/ai/models";
import { systemPrompt } from "@/lib/ai/prompts";
import { suggestFollowUps } from "@/lib/ai/suggestions";
import { createTools } from "@/lib/ai/tools";
import type { ChatMessage } from "@/lib/ai/types";
import { parseChatRequest } from "@/lib/chat/request";
import { clientKey, consumeLimit, limitMessage } from "@/lib/rate-limit";

export const maxDuration = 300;

function lastUserText(messages: ChatMessage[]): string {
  const last = messages.findLast((m) => m.role === "user");
  return last?.parts.map((p) => (p.type === "text" ? p.text : "")).join(" ") ?? "";
}

export async function POST(req: Request) {
  const parsed = parseChatRequest(await req.json().catch(() => null));
  if (!parsed.ok) return Response.json({ error: parsed.error }, { status: 400 });
  const { messages, tickers, mode } = parsed.request;

  const key = clientKey(req.headers);
  const kind = mode === "deep" ? "deep" : "question";
  const limit = await consumeLimit(kind, key);
  if (!limit.ok) return Response.json({ error: limitMessage(kind, limit.reason) }, { status: 429 });

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

      // No research (an off-topic or clarifying reply) means nothing to follow up on.
      const researched = (await result.steps).some((step) => step.toolCalls.length > 0);
      const questions = researched
        ? await suggestFollowUps({ question, answer: await result.text, tickers })
        : [];
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
