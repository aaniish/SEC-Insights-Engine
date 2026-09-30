/**
 * Pre-computes answers for the home page's starter questions by running them
 * through the real chat route, then stores the resulting messages so clicks
 * replay instantly at zero model cost.
 *
 *   pnpm cache:featured                     # all starter questions
 *   pnpm cache:featured tsla-risk-changes   # just one
 */
import { db } from "@/lib/db/client";
import { featuredAnswers } from "@/lib/db/schema";
import { FEATURED_QUESTIONS } from "@/lib/featured";
import { runChat } from "./lib/run-chat";

async function main() {
  const only = process.argv.slice(2);
  const targets = FEATURED_QUESTIONS.filter((q) => only.length === 0 || only.includes(q.id));
  for (const featured of targets) {
    const started = Date.now();
    const messages = await runChat(featured.question, featured.tickers, { client: "featured-cache" });
    await db()
      .insert(featuredAnswers)
      .values({ id: featured.id, question: featured.question, tickers: [...featured.tickers], messages })
      .onConflictDoUpdate({ target: featuredAnswers.id, set: { messages, question: featured.question } });
    const tools = messages[1].parts.map((p) => p.type).filter((t) => t.startsWith("tool-"));
    console.log(
      `${featured.id}: cached in ${((Date.now() - started) / 1000).toFixed(1)}s · ${tools.join(", ")}`,
    );
  }
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
