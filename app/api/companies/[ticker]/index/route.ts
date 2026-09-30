import { after } from "next/server";
import { findCompany } from "@/lib/companies";
import { ingestFiling, latestFilings, readyFilings } from "@/lib/ingest/ingest-filing";
import { clientKey, consumeLimit, limitMessage } from "@/lib/rate-limit";

export const maxDuration = 300;

async function status(ticker: string) {
  const company = await findCompany(ticker);
  if (!company) return null;
  const [latest] = await latestFilings(company, "10-K", 1);
  return { company, latest };
}

/** Current indexing status of a company's latest 10-K (polled by the UI). */
export async function GET(_req: Request, ctx: RouteContext<"/api/companies/[ticker]/index">) {
  const { ticker } = await ctx.params;
  const found = await status(ticker);
  if (!found) return Response.json({ error: "Unknown ticker." }, { status: 404 });
  const ready = await readyFilings(found.company.cik);
  return Response.json({
    ticker: found.company.ticker,
    indexed: ready.length > 0,
    filing: found.latest
      ? {
          form: found.latest.form,
          fiscalYear: found.latest.fiscalYear,
          status: found.latest.status,
          progress: found.latest.progress,
        }
      : null,
  });
}

/** Starts indexing the latest 10-K in the background; the client polls GET. */
export async function POST(req: Request, ctx: RouteContext<"/api/companies/[ticker]/index">) {
  const { ticker } = await ctx.params;
  const found = await status(ticker);
  if (!found) return Response.json({ error: "Unknown ticker." }, { status: 404 });
  if (!found.latest) {
    return Response.json(
      {
        error: `${found.company.ticker} has no 10-K on EDGAR (foreign issuers file 20-F, not supported yet).`,
      },
      { status: 422 },
    );
  }
  if (found.latest.status === "ready") return Response.json({ started: false, status: "ready" });

  const limit = await consumeLimit("index", clientKey(req.headers));
  if (!limit.ok) return Response.json({ error: limitMessage("index", limit.reason) }, { status: 429 });

  const accession = found.latest.accession;
  after(async () => {
    await ingestFiling(accession, "full").catch((error) => console.error("index failed", accession, error));
  });
  return Response.json({ started: true, status: "processing" }, { status: 202 });
}
