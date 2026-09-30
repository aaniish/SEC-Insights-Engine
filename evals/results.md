# Eval results

Model: `openai/gpt-6-luna` · 2026-09-30 · 17 cases

| Metric | Score |
|---|---|
| Retrieval hit@8 | 100% (10/10) |
| Tool choice | 100% (17/17) |
| Citation validity | 100% (10/10) |
| Numeric accuracy (vs XBRL) | 100% (6/6) |
| Faithfulness (LLM judge) | 91% |
| Median latency | 9.6s |

| Case | Tool | Retrieval | Citations | Numbers | Faithful | Time |
|---|---|---|---|---|---|---|
| aapl-single-source | ✓ | ✓ | ✓ | – | 100% | 11.3s |
| nflx-competition | ✓ | ✓ | ✓ | – | 100% | 9.9s |
| meta-legal | ✓ | ✓ | ✓ | – | 100% | 12.4s |
| cost-membership | ✓ | ✓ | ✓ | – | 100% | 11.3s |
| wmt-tariffs | ✓ | ✓ | ✓ | – | 100% | 8.8s |
| jpm-allowance | ✓ | ✓ | ✓ | – | 75% | 11.3s |
| amzn-aws | ✓ | ✓ | ✓ | – | 100% | 8.1s |
| amd-export | ✓ | ✓ | ✓ | – | 100% | 11.8s |
| googl-revenue-sources | ✓ | ✓ | ✓ | – | 50% | 13.9s |
| msft-cyber | ✓ | ✓ | ✓ | – | 86% | 11.0s |
| aapl-revenue-fy25 | ✓ | – | – | ✓ | – | 5.5s |
| tsla-revenue-fy24 | ✓ | – | – | ✓ | – | 5.7s |
| nvda-revenue-fy25 | ✓ | – | – | ✓ | – | 9.6s |
| jpm-net-income-fy24 | ✓ | – | – | ✓ | – | 3.9s |
| aapl-eps-fy25 | ✓ | – | – | ✓ | – | 4.0s |
| meta-margin-fy25 | ✓ | – | – | ✓ | – | 7.7s |
| tsla-new-risks | ✓ | – | – | – | – | 7.9s |
