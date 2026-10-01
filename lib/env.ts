import { z } from "zod";

const envSchema = z.object({
  DATABASE_URL: z.url().optional(),
  SEC_EMAIL_ADDRESS: z.email().optional(),
  CHAT_MODEL_FAST: z.string().default("openai/gpt-6-luna"),
  CHAT_MODEL_DEEP: z.string().default("openai/gpt-6-sol"),
  EMBEDDING_MODEL: z.string().default("openai/text-embedding-3-small"),
  RATE_LIMIT_FAST_PER_DAY: z.coerce.number().int().positive().default(30),
  RATE_LIMIT_DEEP_PER_DAY: z.coerce.number().int().positive().default(5),
  RATE_LIMIT_INDEX_PER_DAY: z.coerce.number().int().positive().default(3),
  GLOBAL_QUESTIONS_PER_DAY: z.coerce.number().int().positive().default(1000),
  GLOBAL_INDEXING_PER_DAY: z.coerce.number().int().positive().default(25),
  CRON_SECRET: z.string().min(16).optional(),
  IP_HASH_SALT: z.string().default("local-dev-salt"),
});

export type Env = z.infer<typeof envSchema>;

let cached: Env | undefined;

/** Validated server env. Parsed lazily so `next build` works without secrets. */
export function env(): Env {
  cached ??= envSchema.parse(process.env);
  return cached;
}

export function requireEnv<K extends "DATABASE_URL" | "SEC_EMAIL_ADDRESS">(key: K): string {
  const value = env()[key];
  if (!value) throw new Error(`Missing required environment variable ${key}. See .env.example.`);
  return value;
}
