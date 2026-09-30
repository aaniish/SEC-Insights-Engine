/** Companies indexed ahead of time (the seed script and daily cron keep these fresh). */
export const CURATED_COMPANIES = [
  { ticker: "AAPL", name: "Apple" },
  { ticker: "MSFT", name: "Microsoft" },
  { ticker: "NVDA", name: "NVIDIA" },
  { ticker: "AMZN", name: "Amazon" },
  { ticker: "GOOGL", name: "Alphabet" },
  { ticker: "META", name: "Meta" },
  { ticker: "TSLA", name: "Tesla" },
  { ticker: "JPM", name: "JPMorgan Chase" },
  { ticker: "WMT", name: "Walmart" },
  { ticker: "NFLX", name: "Netflix" },
  { ticker: "COST", name: "Costco" },
  { ticker: "AMD", name: "AMD" },
] as const;

export const CURATED_TICKERS = CURATED_COMPANIES.map((c) => c.ticker);

/** Starter questions. Their answers are pre-computed and replayed for free. */
export const FEATURED_QUESTIONS = [
  {
    id: "tsla-risk-changes",
    question: "What changed in Tesla's risk factors in its latest 10-K?",
    tickers: ["TSLA"],
    shows: "Filing diff",
  },
  {
    id: "nvda-amd-margins",
    question: "Chart NVIDIA vs AMD revenue and gross margin over the last four years",
    tickers: ["NVDA", "AMD"],
    shows: "XBRL chart",
  },
  {
    id: "aapl-china-supply",
    question: "How exposed is Apple's supply chain to China?",
    tickers: ["AAPL"],
    shows: "Cited answer",
  },
  {
    id: "msft-googl-ai-capex",
    question: "How much are Microsoft and Alphabet spending on AI infrastructure?",
    tickers: ["MSFT", "GOOGL"],
    shows: "Chart + filings",
  },
] as const;

export type FeaturedQuestion = (typeof FEATURED_QUESTIONS)[number];
