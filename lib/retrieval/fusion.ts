/** Reciprocal Rank Fusion constant; 60 is the value from the original RRF paper. */
const RRF_K = 60;

/**
 * Fuses several ranked lists (best first) into one, scoring each item by
 * Σ 1 / (k + rank). Items found by multiple retrievers rise to the top.
 */
export function reciprocalRankFusion<T>(
  lists: T[][],
  key: (item: T) => string | number,
): (T & { score: number })[] {
  const scores = new Map<string | number, { item: T; score: number }>();
  for (const list of lists) {
    list.forEach((item, index) => {
      const id = key(item);
      const entry = scores.get(id) ?? { item, score: 0 };
      entry.score += 1 / (RRF_K + index + 1);
      scores.set(id, entry);
    });
  }
  return [...scores.values()]
    .sort((a, b) => b.score - a.score)
    .map(({ item, score }) => ({ ...item, score }));
}

/**
 * Interleaves results round-robin by group (company) so a comparison question
 * doesn't return eight passages about one company and none about the other.
 */
export function balanceByGroup<T>(items: T[], group: (item: T) => string | number, limit: number): T[] {
  const buckets = new Map<string | number, T[]>();
  for (const item of items) {
    const g = group(item);
    if (!buckets.has(g)) buckets.set(g, []);
    buckets.get(g)?.push(item);
  }
  if (buckets.size <= 1) return items.slice(0, limit);

  const out: T[] = [];
  const queues = [...buckets.values()];
  while (out.length < limit && queues.some((q) => q.length > 0)) {
    for (const queue of queues) {
      const next = queue.shift();
      if (next !== undefined && out.length < limit) out.push(next);
    }
  }
  return out;
}
