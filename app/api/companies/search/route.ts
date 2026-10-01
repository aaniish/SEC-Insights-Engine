import { searchCompanies } from "@/lib/companies";

export async function GET(req: Request) {
  const q = new URL(req.url).searchParams.get("q") ?? "";
  const results = await searchCompanies(q.slice(0, 60));
  return Response.json({ results });
}
