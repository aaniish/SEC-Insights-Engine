/**
 * Evaluation harness. Runs every case in evals/cases.json and reports:
 *   - retrieval hit@8: hybrid search returns a passage from the right company
 *     (and section, when specified) containing an expected keyword
 *   - tool choice: the agent called the expected tool
 *   - citation validity: every [n] in the answer refers to a retrieved passage
 *   - numeric accuracy: financial answers state the XBRL value (±1%)
 *   - faithfulness: an LLM judge checks each cited sentence against its passage
 *
 *   pnpm eval                 # all cases, writes evals/results.md
 *   pnpm eval aapl-eps-fy25   # selected cases
 */
import { readFileSync, writeFileSync } from "node:fs";
import { generateText, Output } from "ai";
import { z } from "zod";
import { chatModel, reasoningOptions } from "@/lib/ai/models";
import type { Citation } from "@/lib/ai/tools";
import { answerText, citationsOf, citedNumbers, isToolPart, toolName } from "@/lib/chat/parts";
import { findCompany } from "@/lib/companies";
import { mentionsFigure } from "@/lib/eval/figures";
import { getFinancialSeries } from "@/lib/financials";
import { hybridSearch } from "@/lib/retrieval/hybrid-search";
import type { SectionTopic } from "@/lib/sec/sections";
import type { AnyMetric } from "@/lib/sec/xbrl";
import { runChat } from "./lib/run-chat";

interface EvalCase {
  id: string;
  question: string;
  tickers: string[];
  tool: "searchFilings" | "getFinancials" | "compareFilings";
  topic?: SectionTopic;
  keywords?: string[];
  facts?: { ticker: string; metric: AnyMetric; fiscalYear: number }[];
}

interface CaseResult {
  id: string;
  retrieval?: boolean;
  tool: boolean;
  citations?: boolean;
  numeric?: boolean;
  faithfulness?: number;
  seconds: number;
  note?: string;
}

async function retrievalHit(c: EvalCase): Promise<boolean> {
  const ciks = (await Promise.all(c.tickers.map((t) => findCompany(t)))).flatMap((co) =>
    co ? [co.cik] : [],
  );
  const hits = await hybridSearch(c.question, { ciks }, 8);
  const keywords = (c.keywords ?? []).map((k) => k.toLowerCase());
  return hits.some(
    (h) =>
      (!c.topic || h.topic === c.topic) &&
      (keywords.length === 0 || keywords.some((k) => h.content.toLowerCase().includes(k))),
  );
}

async function expectedValues(c: EvalCase): Promise<number[]> {
  const values: number[] = [];
  for (const fact of c.facts ?? []) {
    const { series } = await getFinancialSeries([fact.ticker], [fact.metric], "FY", 12);
    const point = series[0]?.points.find((p) => p.fiscalYear === fact.fiscalYear);
    if (point) values.push(point.value);
  }
  return values;
}

const judgeSchema = z.object({
  verdicts: z.array(z.object({ supported: z.boolean() })),
});

/** Share of cited sentences whose claims are supported by the passages they cite. */
async function faithfulness(text: string, citations: Map<number, Citation>): Promise<number | undefined> {
  // Citations often trail the period ("…growth. [2][3] Next sentence"); move them back to their sentence.
  const pieces = text.split(/(?<=[.!?])\s+|\n+/).filter((s) => s.trim());
  const merged: string[] = [];
  for (const piece of pieces) {
    const [, leading = "", rest = piece] = piece.trim().match(/^((?:\[\d[\d,\s–-]*\]\s*)*)(.*)$/) ?? [];
    if (leading && merged.length > 0) merged[merged.length - 1] += ` ${leading.trim()}`;
    else if (leading) merged.push(leading);
    if (rest.trim()) merged.push(rest);
  }
  const sentences = merged.filter((s) => citedNumbers(s).some((n) => citations.has(n))).slice(0, 8);
  if (sentences.length === 0) return undefined;
  const claims = sentences.map((sentence, i) => {
    const passages = citedNumbers(sentence)
      .flatMap((n) => (citations.has(n) ? [`[${n}] ${citations.get(n)?.text}`] : []))
      .join("\n");
    return `CLAIM ${i + 1}: ${sentence}\nPASSAGES:\n${passages}`;
  });
  const { output } = await generateText({
    model: chatModel("fast"),
    output: Output.object({ schema: judgeSchema }),
    providerOptions: reasoningOptions("low"),
    prompt: `For each claim, decide whether its passages support it. Paraphrase and summary count as supported; claims that add facts not in the passages are not supported. Return one verdict per claim, in order.\n\n${claims.join("\n\n")}`,
  });
  const verdicts = output.verdicts.slice(0, sentences.length);
  if (process.env.DEBUG_EVAL) {
    verdicts.forEach((v, i) => {
      if (!v.supported) console.log(`  unsupported: ${claims[i].slice(0, 900)}\n`);
    });
  }
  return verdicts.filter((v) => v.supported).length / Math.max(1, verdicts.length);
}

/** Evals run with the owner's credentials, so each run gets its own rate-limit bucket. */
const runId = Date.now().toString(36);

async function runCase(c: EvalCase): Promise<CaseResult> {
  const started = Date.now();
  const result: Partial<CaseResult> = { id: c.id };
  if (c.tool === "searchFilings") result.retrieval = await retrievalHit(c);

  const [, answer] = await runChat(c.question, c.tickers, { client: `eval-${runId}` });
  const tools = answer.parts.filter(isToolPart).map(toolName);
  result.tool = tools.includes(c.tool);

  const text = answerText(answer);
  const citations = citationsOf(answer);
  const cited = citedNumbers(text);
  if (c.tool === "searchFilings") {
    result.citations = cited.length > 0 && cited.every((n) => citations.has(n));
    result.faithfulness = await faithfulness(text, citations);
  }
  if (c.facts) {
    const expected = await expectedValues(c);
    result.numeric = expected.length > 0 && expected.every((value) => mentionsFigure(text, value));
    if (!result.numeric) result.note = `expected ${expected.join(", ")}`;
  }
  if (!result.tool) result.note = `called ${tools.join(", ") || "no tools"}`;
  return { ...(result as CaseResult), seconds: (Date.now() - started) / 1000 };
}

const mark = (v: boolean | undefined) => (v === undefined ? "–" : v ? "✓" : "✗");
const rate = (values: (boolean | undefined)[]) => {
  const scored = values.filter((v): v is boolean => v !== undefined);
  return scored.length
    ? `${Math.round((100 * scored.filter(Boolean).length) / scored.length)}% (${scored.filter(Boolean).length}/${scored.length})`
    : "–";
};

async function main() {
  const only = process.argv.slice(2);
  const cases = (JSON.parse(readFileSync("evals/cases.json", "utf8")) as EvalCase[]).filter(
    (c) => only.length === 0 || only.includes(c.id),
  );

  const results: CaseResult[] = [];
  for (const c of cases) {
    try {
      const r = await runCase(c);
      results.push(r);
      console.log(
        `${c.id.padEnd(24)} tool ${mark(r.tool)} retrieval ${mark(r.retrieval)} cites ${mark(r.citations)} numbers ${mark(r.numeric)} ${r.note ?? ""}`,
      );
    } catch (error) {
      console.error(`${c.id}: ${error instanceof Error ? error.message : error}`);
      results.push({ id: c.id, tool: false, seconds: 0, note: "error" });
    }
  }

  const faith = results.flatMap((r) => (r.faithfulness === undefined ? [] : [r.faithfulness]));
  const summary = [
    `| Metric | Score |`,
    `|---|---|`,
    `| Retrieval hit@8 | ${rate(results.map((r) => r.retrieval))} |`,
    `| Tool choice | ${rate(results.map((r) => r.tool))} |`,
    `| Citation validity | ${rate(results.map((r) => r.citations))} |`,
    `| Numeric accuracy (vs XBRL) | ${rate(results.map((r) => r.numeric))} |`,
    `| Faithfulness (LLM judge) | ${faith.length ? `${Math.round((100 * faith.reduce((a, b) => a + b, 0)) / faith.length)}%` : "–"} |`,
    `| Median latency | ${results
      .map((r) => r.seconds)
      .sort((a, b) => a - b)
      [Math.floor(results.length / 2)]?.toFixed(1)}s |`,
  ].join("\n");

  const rows = results
    .map(
      (r) =>
        `| ${r.id} | ${mark(r.tool)} | ${mark(r.retrieval)} | ${mark(r.citations)} | ${mark(r.numeric)} | ${r.faithfulness === undefined ? "–" : `${Math.round(r.faithfulness * 100)}%`} | ${r.seconds.toFixed(1)}s |`,
    )
    .join("\n");

  const report = `# Eval results\n\nModel: \`${chatModel("fast")}\` · ${new Date().toISOString().slice(0, 10)} · ${results.length} cases\n\n${summary}\n\n| Case | Tool | Retrieval | Citations | Numbers | Faithful | Time |\n|---|---|---|---|---|---|---|\n${rows}\n`;
  if (only.length === 0) writeFileSync("evals/results.md", report);
  console.log(`\n${summary}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
