output target:
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

# 1. Primary price data
            price_query = """
                SELECT time as timestamp, open, high, low, close, volume_usd as volume
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 2. Open Interest data
            oi_query = """
                SELECT time as timestamp, close as open_interest
                FROM cg_open_interest_aggregated_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 3. Liquidation data
            liq_query = """
                SELECT time as timestamp,
                       aggregated_long_liquidation_usd,
                       aggregated_short_liquidation_usd
                FROM cg_liquidation_aggregated_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 4. Taker Volume data
            volume_query = """
                SELECT time as timestamp,
                       aggregated_buy_volume_usd,
                       aggregated_sell_volume_usd
                FROM cg_spot_aggregated_taker_volume_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 5. Funding Rate data
            funding_query = """
                SELECT time as timestamp, close as funding_rate
                FROM cg_funding_rate_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 6. Top Account Ratio data
            top_ratio_query = """
                SELECT time as timestamp, top_account_long_short_ratio
                FROM cg_long_short_top_account_ratio_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

# 7. Global Account Ratio data
            global_ratio_query = """
                SELECT time as timestamp, global_account_long_short_ratio
                FROM cg_long_short_global_account_ratio_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """