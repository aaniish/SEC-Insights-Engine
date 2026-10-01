import { readUIMessageStream } from "ai";
import { POST } from "@/app/api/chat/route";
import type { ChatMessage } from "@/lib/ai/types";

/**
 * Runs one question through the real chat route handler (same code path as
 * production) and returns the user + final assistant messages.
 */
export async function runChat(
  question: string,
  tickers: readonly string[],
  { mode = "fast", client = "script" }: { mode?: "fast" | "deep"; client?: string } = {},
): Promise<[ChatMessage, ChatMessage]> {
  const userMessage: ChatMessage = {
    id: crypto.randomUUID(),
    role: "user",
    parts: [{ type: "text", text: question }],
  };
  const response = await POST(
    new Request("http://localhost/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json", "x-forwarded-for": client },
      body: JSON.stringify({ messages: [userMessage], tickers, mode }),
    }),
  );
  if (!response.ok || !response.body) {
    throw new Error(`Chat failed: ${response.status} ${await response.text()}`);
  }

  // The response is SSE; reassemble "data: {...}" lines (a line can span network chunks).
  let buffered = "";
  const chunks = response.body.pipeThrough(new TextDecoderStream()).pipeThrough(
    new TransformStream({
      transform(text: string, controller) {
        buffered += text;
        const lines = buffered.split("\n");
        buffered = lines.pop() ?? "";
        for (const line of lines) {
          if (line.startsWith("data: ") && line !== "data: [DONE]")
            controller.enqueue(JSON.parse(line.slice(6)));
        }
      },
    }),
  );

  let assistant: ChatMessage | undefined;
  for await (const message of readUIMessageStream<ChatMessage>({ stream: chunks })) assistant = message;
  if (!assistant) throw new Error("No assistant message produced");
  return [userMessage, assistant];
}
