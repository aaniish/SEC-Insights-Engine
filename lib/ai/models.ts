import { embed, embedMany } from "ai";
import { EMBEDDING_DIMENSIONS } from "@/lib/db/schema";
import { env } from "@/lib/env";

export type ChatMode = "fast" | "deep";

export function chatModel(mode: ChatMode): string {
  return mode === "deep" ? env().CHAT_MODEL_DEEP : env().CHAT_MODEL_FAST;
}

/**
 * Reasoning effort per mode. Reasoning tokens count toward output limits and
 * latency, so fast mode and small helper calls keep it low. Ignored by
 * providers that don't support it.
 */
export function reasoningOptions(effort: "low" | "medium") {
  return { openai: { reasoningEffort: effort } };
}

const embeddingOptions = { openai: { dimensions: EMBEDDING_DIMENSIONS } };

/** halfvec keeps ~3 significant digits, so extra precision only bloats the payload. */
const compact = (vector: number[]) => vector.map((v) => Math.round(v * 1e5) / 1e5);

export async function embedTexts(values: string[]): Promise<number[][]> {
  const { embeddings } = await embedMany({
    model: env().EMBEDDING_MODEL,
    values,
    providerOptions: embeddingOptions,
    maxParallelCalls: 4,
  });
  return embeddings.map(compact);
}

export async function embedQuery(value: string): Promise<number[]> {
  const { embedding } = await embed({
    model: env().EMBEDDING_MODEL,
    value,
    providerOptions: embeddingOptions,
  });
  return compact(embedding);
}
