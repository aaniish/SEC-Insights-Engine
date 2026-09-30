import { eq } from "drizzle-orm";
import { db } from "@/lib/db/client";
import { featuredAnswers } from "@/lib/db/schema";

/** Pre-computed conversation for a starter question (replayed at zero model cost). */
export async function GET(_req: Request, ctx: RouteContext<"/api/featured/[id]">) {
  const { id } = await ctx.params;
  const [row] = await db().select().from(featuredAnswers).where(eq(featuredAnswers.id, id)).limit(1);
  if (!row) return Response.json({ error: "Not found." }, { status: 404 });
  return Response.json({
    question: row.question,
    tickers: row.tickers,
    messages: row.messages,
  });
}
