import { describe, expect, it } from "vitest";
import { formatValue, niceTicks } from "./financial-charts";

describe("niceTicks", () => {
  it("uses round steps that start at zero and cover the max", () => {
    expect(niceTicks([27e9, 215.9e9])).toEqual([0, 50e9, 100e9, 150e9, 200e9, 250e9]);
    expect(niceTicks([44.9, 75])).toEqual([0, 20, 40, 60, 80]);
  });

  it("extends below zero for losses", () => {
    const ticks = niceTicks([-3e9, 7e9]);
    expect(ticks[0]).toBeLessThan(0);
    expect(ticks).toContain(0);
    expect(ticks.at(-1)).toBeGreaterThanOrEqual(7e9);
  });
});

describe("formatValue", () => {
  it("formats money, percentages, and per-share values", () => {
    expect(formatValue(391_035_000_000, "USD")).toBe("$391B");
    expect(formatValue(34_600_000_000, "USD")).toBe("$34.6B");
    expect(formatValue(-1_200_000_000, "USD")).toBe("−$1.2B");
    expect(formatValue(71.08, "percent")).toBe("71.1%");
    expect(formatValue(7.456, "USD/shares")).toBe("$7.46");
  });
});
