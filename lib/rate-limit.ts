import { createHash } from "node:crypto";
import { sql } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { env } from "@/lib/env";

export type UsageKind = "question" | "deep" | "index";

export interface LimitResult {
  ok: boolean;
  reason?: "ip" | "global";
}

/** Vercel sets x-forwarded-for itself, so the first entry is the real client IP. */
export function clientKey(headers: Headers): string {
  const ip = headers.get("x-forwarded-for")?.split(",")[0]?.trim() || headers.get("x-real-ip") || "local";
  return createHash("sha256").update(`${env().IP_HASH_SALT}:${ip}`).digest("hex").slice(0, 32);
}

function limitsFor(kind: UsageKind): { perClient: number; global: number } {
  const e = env();
  if (kind === "index")
    return {
      perClient: e.RATE_LIMIT_INDEX_PER_DAY,
      global: e.GLOBAL_INDEXING_PER_DAY,
    };
  if (kind === "deep")
    return {
      perClient: e.RATE_LIMIT_DEEP_PER_DAY,
      global: e.GLOBAL_QUESTIONS_PER_DAY,
    };
  return {
    perClient: e.RATE_LIMIT_FAST_PER_DAY,
    global: e.GLOBAL_QUESTIONS_PER_DAY,
  };
}

async function increment(key: string, kind: string): Promise<number> {
  const result = await db().execute<{ count: number }>(sql`
    insert into usage_counters (key, day, kind, count)
    values (${key}, (now() at time zone 'utc')::date, ${kind}, 1)
    on conflict (key, day, kind) do update set count = usage_counters.count + 1
    returning count
  `);
  return result.rows[0]?.count ?? 0;
}

/**
 * Counts one use against the per-client and global daily limits. Deep-mode
 * questions also count toward the global question budget.
 */
export async function consumeLimit(kind: UsageKind, key: string): Promise<LimitResult> {
  const { perClient, global } = limitsFor(kind);
  const globalKind = kind === "index" ? "index" : "question";
  const [clientCount, globalCount] = await Promise.all([
    increment(key, kind),
    increment("global", globalKind),
  ]);
  if (clientCount > perClient) return { ok: false, reason: "ip" };
  if (globalCount > global) return { ok: false, reason: "global" };
  return { ok: true };
}

export function limitMessage(kind: UsageKind, reason: LimitResult["reason"]): string {
  if (reason === "global") {
    return "The demo has hit its shared daily budget. It resets at midnight UTC — thanks for trying it!";
  }
  if (kind === "index")
    return "You've indexed the maximum number of new filings for today. Try an already-indexed company.";
  if (kind === "deep")
    return "You've used today's Deep analysis questions. Switch to Fast mode to keep going.";
  return "You've reached today's question limit for this demo. It resets at midnight UTC.";
}
