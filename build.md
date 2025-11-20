target:
| Metric            | Target    |
| ----------------- | --------- |
| **CAGR**          | ≥ 50%     |
| **Max Drawdown**  | ≤ 15–25%  |
| **Sharpe Ratio**  | 1.2 – 1.6 |
| **Sortino Ratio** | 2.0 – 3.0 |
| **Win Rate**      | ≥ 70%     |

pipeline:
Raw Market Data → Feature Engineering → XGBoost Model → Predictions → Trading Rules → Backtest Results → Evaluate Metrics

goals:
| Target         | How to achieve it                                        |
| -------------- | -------------------------------------------------------  |
| High WinRate | model + entry filter (volume/volatility/oi/liquidation)    |
| High Sharpe  | reduce noise, discard low confidence signals               |
| Low MaxDD    | dynamic stop loss / exit on confidence drop                |
| High Sortino | reduce large losses (cut faster, hold longer winners)      |
| High CAGR    | combine small leverage & compounding                       |

pymysql integration for db connection