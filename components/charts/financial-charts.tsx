"use client";

import { Table2 } from "lucide-react";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, Tooltip, XAxis, YAxis } from "recharts";
import { ChartContainer } from "@/components/ui/chart";
import type { FinancialSeries } from "@/lib/financials";
import { cn } from "@/lib/utils";

const SERIES_COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)", "var(--chart-4)"];

type Unit = FinancialSeries["unit"];

export function formatValue(value: number, unit: Unit, compact = true): string {
  if (unit === "percent") return `${value.toFixed(1)}%`;
  if (unit === "USD/shares") return `$${value.toFixed(2)}`;
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (!compact) return `${sign}$${abs.toLocaleString("en-US")}`;
  if (abs >= 1e12) return `${sign}$${(abs / 1e12).toFixed(2)}T`;
  if (abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(abs >= 1e11 ? 0 : 1)}B`;
  if (abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(abs >= 1e8 ? 0 : 1)}M`;
  return `${sign}$${abs.toLocaleString("en-US")}`;
}

function axisValue(value: number, unit: Unit): string {
  if (unit === "percent") return `${Math.round(value)}%`;
  if (unit === "USD/shares") return `$${value.toFixed(value >= 10 ? 0 : 1)}`;
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  if (abs >= 1e12) return `${sign}$${+(abs / 1e12).toFixed(1)}T`;
  if (abs >= 1e9) return `${sign}$${+(abs / 1e9).toFixed(0)}B`;
  if (abs >= 1e6) return `${sign}$${+(abs / 1e6).toFixed(0)}M`;
  return `${sign}$${abs}`;
}

/** Round axis ticks (1, 2, 2.5, 5 × 10ⁿ steps) that always include zero. */
export function niceTicks(values: number[], target = 4): number[] {
  const min = Math.min(0, ...values);
  const max = Math.max(0, ...values);
  const span = max - min || Math.abs(max) || 1;
  const magnitude = 10 ** Math.floor(Math.log10(span / target));
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => span / s <= target + 0.5) ?? magnitude * 10;
  const ticks: number[] = [];
  for (let t = Math.floor(min / step) * step; t <= Math.ceil(max / step) * step + step / 2; t += step) {
    ticks.push(Number(t.toPrecision(12)));
  }
  return ticks;
}

/** Sort key for period labels like "FY2025" or "Q2 FY26". */
function periodOrder(point: { fiscalYear: number; fiscalPeriod: string }): number {
  const quarter = point.fiscalPeriod === "FY" ? 5 : Number(point.fiscalPeriod.slice(1));
  return point.fiscalYear * 10 + quarter;
}

interface Row {
  label: string;
  order: number;
  [ticker: string]: number | string;
}

/** One row per period, one column per company, aligned on fiscal period labels. */
function toRows(series: FinancialSeries[]): Row[] {
  const rows = new Map<string, Row>();
  for (const s of series) {
    for (const p of s.points) {
      const row = rows.get(p.label) ?? { label: p.label, order: periodOrder(p) };
      row[s.ticker] = p.value;
      rows.set(p.label, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.order - b.order);
}

function ChartTooltipCard({
  active,
  label,
  payload,
  unit,
}: {
  active?: boolean;
  label?: string;
  payload?: { dataKey?: string | number; value?: number; color?: string }[];
  unit: Unit;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="glass glass-dense rounded-xl px-3 py-2">
      <div className="mb-1 text-[0.7rem] font-medium text-graphite">{label}</div>
      <ul className="space-y-0.5">
        {payload.map((entry) => (
          <li key={String(entry.dataKey)} className="flex items-center gap-2 text-xs">
            <span className="h-0.5 w-3 rounded-full" style={{ background: entry.color }} aria-hidden="true" />
            <span className="font-medium tabular">{formatValue(Number(entry.value), unit)}</span>
            <span className="text-[0.7rem] text-graphite">{String(entry.dataKey)}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

function MetricChart({
  series,
  colorFor,
}: {
  series: FinancialSeries[];
  colorFor: (ticker: string) => string;
}) {
  const [showTable, setShowTable] = useState(false);
  const rows = toRows(series);
  const unit = series[0].unit;
  const tickers = series.map((s) => s.ticker);
  const first = rows[0]?.label;
  const last = rows.at(-1)?.label;
  const config = Object.fromEntries(tickers.map((t) => [t, { label: t, color: colorFor(t) }]));
  const ticks = niceTicks(series.flatMap((s) => s.points.map((p) => p.value)));

  if (rows.length === 0) {
    return (
      <div className="glass rounded-2xl p-4 text-sm text-graphite">
        No reported {series[0].label.toLowerCase()} for {tickers.join(", ")} in XBRL data.
      </div>
    );
  }

  return (
    <figure className="glass rounded-2xl p-4">
      <figcaption className="mb-3 flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div>
          <div className="text-sm font-semibold">{series[0].label}</div>
          <div className="text-[0.72rem] text-graphite">
            {first === last ? first : `${first} – ${last}`} · SEC XBRL
          </div>
        </div>
        <div className="flex items-center gap-3">
          {tickers.length > 1 && (
            <ul className="flex flex-wrap gap-x-3 gap-y-1" aria-label="Legend">
              {tickers.map((t) => (
                <li key={t} className="flex items-center gap-1.5 text-[0.72rem] font-medium">
                  <span
                    className="h-0.5 w-3.5 rounded-full"
                    style={{ background: colorFor(t) }}
                    aria-hidden="true"
                  />
                  {t}
                </li>
              ))}
            </ul>
          )}
          <button
            type="button"
            onClick={() => setShowTable((v) => !v)}
            aria-pressed={showTable}
            className={cn(
              "inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.72rem] text-graphite hover:bg-muted hover:text-ink",
              showTable && "bg-muted text-foreground",
            )}
          >
            <Table2 className="size-3" /> Table
          </button>
        </div>
      </figcaption>

      {showTable ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs tabular">
            <thead>
              <tr className="border-b text-graphite">
                <th className="py-1.5 pr-4 text-left font-normal">Period</th>
                {tickers.map((t) => (
                  <th key={t} className="py-1.5 pl-4 text-right font-normal">
                    {t}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.label} className="border-b border-rule/60 last:border-0">
                  <td className="py-1.5 pr-4">{row.label}</td>
                  {tickers.map((t) => (
                    <td key={t} className="py-1.5 pl-4 text-right">
                      {typeof row[t] === "number" ? formatValue(row[t] as number, unit, false) : "—"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <ChartContainer config={config} className="aspect-auto h-52 w-full">
          <LineChart data={rows} margin={{ top: 8, right: 56, bottom: 0, left: 4 }} accessibilityLayer>
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={{ stroke: "var(--chart-axis)" }}
              tickMargin={8}
              interval="preserveStartEnd"
              minTickGap={12}
              tick={{ fontSize: 11, fontFamily: "var(--font-inter)" }}
            />
            <YAxis
              width={52}
              tickLine={false}
              axisLine={false}
              ticks={ticks}
              domain={[ticks[0], ticks.at(-1) ?? "auto"]}
              tickFormatter={(v: number) => axisValue(v, unit)}
              tick={{ fontSize: 11, fontFamily: "var(--font-inter)" }}
            />
            <Tooltip
              cursor={{ stroke: "var(--chart-axis)", strokeWidth: 1 }}
              content={(props) => <ChartTooltipCard {...(props as object)} unit={unit} />}
            />
            {tickers.map((ticker) => {
              const lastIndex = rows.findLastIndex((r) => typeof r[ticker] === "number");
              return (
                <Line
                  key={ticker}
                  dataKey={ticker}
                  type="monotone"
                  stroke={colorFor(ticker)}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  connectNulls
                  isAnimationActive={false}
                  dot={{ r: 4, fill: colorFor(ticker), stroke: "var(--card)", strokeWidth: 2 }}
                  activeDot={{ r: 5, fill: colorFor(ticker), stroke: "var(--card)", strokeWidth: 2 }}
                  label={(props: {
                    index?: number;
                    x?: number | string;
                    y?: number | string;
                    value?: unknown;
                  }) =>
                    props.index === lastIndex && typeof props.value === "number" ? (
                      <text
                        x={Number(props.x) + 9}
                        y={Number(props.y)}
                        dy={4}
                        className="fill-foreground text-[10.5px] font-medium"
                      >
                        {formatValue(props.value, unit)}
                      </text>
                    ) : (
                      <g />
                    )
                  }
                />
              );
            })}
          </LineChart>
        </ChartContainer>
      )}
    </figure>
  );
}

/**
 * Charts for a getFinancials result: small multiples, one per metric (never a
 * dual axis), with each company keeping the same color across charts.
 */
export function FinancialCharts({ series, missing }: { series: FinancialSeries[]; missing: string[] }) {
  const tickers = [...new Set(series.map((s) => s.ticker))];
  const colorFor = (ticker: string) => SERIES_COLORS[tickers.indexOf(ticker) % SERIES_COLORS.length];
  const byMetric = new Map<string, FinancialSeries[]>();
  for (const s of series) byMetric.set(s.metric, [...(byMetric.get(s.metric) ?? []), s]);
  const charts = [...byMetric.values()].filter((group) => group.some((s) => s.points.length > 0));

  if (charts.length === 0) {
    return (
      <p className="glass rounded-2xl p-4 text-sm text-graphite">
        No XBRL financial data was found for {missing.length ? missing.join(", ") : "that request"}.
      </p>
    );
  }

  return (
    <div className={cn("grid gap-3", charts.length > 1 && "sm:grid-cols-2")}>
      {charts.map((group) => (
        <MetricChart key={group[0].metric} series={group} colorFor={colorFor} />
      ))}
    </div>
  );
}
