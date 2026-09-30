import { describe, expect, it } from "vitest";
import { balanceByGroup, reciprocalRankFusion } from "./fusion";

describe("reciprocalRankFusion", () => {
  it("ranks items found by both retrievers above single-list items", () => {
    const semantic = [{ id: "a" }, { id: "b" }, { id: "c" }];
    const keyword = [{ id: "d" }, { id: "c" }, { id: "a" }];
    const fused = reciprocalRankFusion([semantic, keyword], (x) => x.id).map((x) => x.id);
    expect(fused.slice(0, 2)).toEqual(["a", "c"]);
    expect(fused).toHaveLength(4);
  });

  it("attaches a descending score", () => {
    const fused = reciprocalRankFusion([[{ id: 1 }, { id: 2 }]], (x) => x.id);
    expect(fused[0].score).toBeGreaterThan(fused[1].score);
  });
});

describe("balanceByGroup", () => {
  it("interleaves groups round-robin", () => {
    const items = [
      { g: "AAPL", n: 1 },
      { g: "AAPL", n: 2 },
      { g: "AAPL", n: 3 },
      { g: "MSFT", n: 4 },
    ];
    expect(balanceByGroup(items, (x) => x.g, 3).map((x) => x.n)).toEqual([1, 4, 2]);
  });

  it("keeps rank order for a single group", () => {
    expect(balanceByGroup([{ g: 1 }, { g: 1 }, { g: 1 }], (x) => x.g, 2)).toHaveLength(2);
  });
});
