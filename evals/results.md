# Eval results

Model: `openai/gpt-6-luna` · 2026-09-30 · 17 cases

| Metric | Score |
|---|---|
| Retrieval hit@8 | 100% (10/10) |
| Tool choice | 100% (17/17) |
| Citation validity | 100% (10/10) |
| Numeric accuracy (vs XBRL) | 100% (6/6) |
| Faithfulness (LLM judge) | 91% |
| Median latency | 12.1s |

| Case | Tool | Retrieval | Citations | Numbers | Faithful | Time |
|---|---|---|---|---|---|---|
| aapl-single-source | ✓ | ✓ | ✓ | – | 67% | 18.2s |
| nflx-competition | ✓ | ✓ | ✓ | – | 100% | 13.8s |
| meta-legal | ✓ | ✓ | ✓ | – | 67% | 21.0s |
| cost-membership | ✓ | ✓ | ✓ | – | 100% | 14.1s |
| wmt-tariffs | ✓ | ✓ | ✓ | – | 100% | 12.5s |
| jpm-allowance | ✓ | ✓ | ✓ | – | 100% | 26.5s |
| amzn-aws | ✓ | ✓ | ✓ | – | 100% | 9.7s |
| amd-export | ✓ | ✓ | ✓ | – | 100% | 12.4s |
| googl-revenue-sources | ✓ | ✓ | ✓ | – | 80% | 14.5s |
| msft-cyber | ✓ | ✓ | ✓ | – | 100% | 12.1s |
| aapl-revenue-fy25 | ✓ | – | – | ✓ | – | 5.1s |
| tsla-revenue-fy24 | ✓ | – | – | ✓ | – | 5.3s |
| nvda-revenue-fy25 | ✓ | – | – | ✓ | – | 6.7s |
| jpm-net-income-fy24 | ✓ | – | – | ✓ | – | 6.6s |
| aapl-eps-fy25 | ✓ | – | – | ✓ | – | 4.0s |
| meta-margin-fy25 | ✓ | – | – | ✓ | – | 5.9s |
| tsla-new-risks | ✓ | – | – | – | – | 8.9s |
