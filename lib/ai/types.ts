import type { InferUITools, UIMessage } from "ai";
import type { ChatTools } from "./tools";

export type ChatDataParts = {
  suggestions: { questions: string[] };
};

export type ChatMessage = UIMessage<never, ChatDataParts, InferUITools<ChatTools>>;

export interface ChatRequestBody {
  messages: ChatMessage[];
  tickers?: string[];
  mode?: "fast" | "deep";
}
