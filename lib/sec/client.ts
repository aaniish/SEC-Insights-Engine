import { requireEnv } from "@/lib/env";

/** SEC allows 10 requests/second per client; stay comfortably under it. */
const MIN_INTERVAL_MS = 125;
const MAX_ATTEMPTS = 4;
const RETRYABLE = new Set([403, 429, 500, 502, 503, 504]);

let nextSlot = 0;

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

async function throttle() {
  const now = Date.now();
  const wait = Math.max(0, nextSlot - now);
  nextSlot = Math.max(now, nextSlot) + MIN_INTERVAL_MS;
  if (wait > 0) await sleep(wait);
}

export class SecRequestError extends Error {
  constructor(
    readonly url: string,
    readonly status: number,
  ) {
    super(`SEC request failed (${status}): ${url}`);
  }
}

export function secUserAgent(): string {
  return `SEC Insights Engine ${requireEnv("SEC_EMAIL_ADDRESS")}`;
}

/** fetch() for sec.gov with the required User-Agent, throttling, and retry/backoff. */
export async function secFetch(url: string, timeoutMs = 30_000): Promise<Response> {
  let lastStatus = 0;
  for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
    await throttle();
    const res = await fetch(url, {
      headers: {
        "User-Agent": secUserAgent(),
        "Accept-Encoding": "gzip, deflate",
      },
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (res.ok) return res;
    lastStatus = res.status;
    if (!RETRYABLE.has(res.status)) break;
    await sleep(500 * 2 ** attempt + Math.random() * 250);
  }
  throw new SecRequestError(url, lastStatus);
}

export async function secJson<T>(url: string): Promise<T> {
  const res = await secFetch(url);
  return (await res.json()) as T;
}

export async function secText(url: string): Promise<string> {
  const res = await secFetch(url, 60_000);
  return res.text();
}

export function padCik(cik: number): string {
  return String(cik).padStart(10, "0");
}
