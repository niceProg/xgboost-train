#!/usr/bin/env python3
"""
Collect trading signals and store them in cg_train_dataset table.
This script fetches market data, calculates features, and stores them for later labeling.
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, Any

from env_config import get_database_config
from feature_engineering import FeatureEngineer
from database import DatabaseManager

class SignalCollector:
    def __init__(self):
        self.db_config = get_database_config()
        self.feature_engineer = FeatureEngineer()
        self.db_manager = DatabaseManager(self.db_config)

    def _create_table_if_not_exists(self):
        """Create the cg_train_dataset table if it doesn't exist"""
        try:
            table_schema_sql = """
                CREATE TABLE IF NOT EXISTS cg_train_dataset (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    pair VARCHAR(20) NOT NULL,
                    time_interval VARCHAR(10) NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    horizon_minutes INT NOT NULL,
                    price_now DECIMAL(20, 8) NOT NULL,
                    features_payload JSON,
                    signal_rule ENUM('BUY', 'SELL', 'NEUTRAL') DEFAULT NULL,
                    signal_score DECIMAL(10, 6) DEFAULT NULL,
                    price_future DECIMAL(20, 8) DEFAULT NULL,
                    label_direction ENUM('UP', 'DOWN', 'FLAT') DEFAULT NULL,
                    label_magnitude DECIMAL(10, 6) DEFAULT NULL,
                    label_status ENUM('pending', 'labeled') DEFAULT 'pending',
                    labeled_at TIMESTAMP NULL,
                    snapshot_version VARCHAR(20) DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,

                    INDEX idx_symbol_pair_interval (symbol, pair, time_interval),
                    INDEX idx_generated_at (generated_at),
                    INDEX idx_label_status (label_status),
                    INDEX idx_horizon_minutes (horizon_minutes),
                    INDEX idx_labeled_at (labeled_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """
            self.db_manager.execute_update(table_schema_sql)
            print("✅ Table cg_train_dataset ensured")
        except Exception as e:
            print(f"❌ Error creating table: {e}")

    def get_market_data(self, symbol: str, pair: str, interval: str,
                       start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch comprehensive market data from multiple sources"""
        try:
            print(f"📊 Fetching multi-source market data for {symbol} {pair} {interval}")

            # 1. Primary price data
            price_query = """
                SELECT time as timestamp, open, high, low, close, volume_usd as volume
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            # Convert timestamps to integers to avoid scientific notation issues
            start_timestamp = int(start_date.timestamp() * 1000)
            end_timestamp = int(end_date.timestamp() * 1000)

            price_df = self.db_manager.execute_query(price_query, (pair, interval, start_timestamp, end_timestamp))

            if price_df.empty:
                print(f"⚠️  No price data found for {symbol} {pair} {interval}")
                return pd.DataFrame()

            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'], unit='ms')
            price_df.set_index('timestamp', inplace=True)

            # Ensure numeric columns
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                price_df[col] = pd.to_numeric(price_df[col], errors='coerce')

            # 2. Open Interest data
            oi_query = """
                SELECT time as timestamp, close as open_interest
                FROM cg_open_interest_aggregated_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            oi_df = self.db_manager.execute_query(oi_query, (symbol, interval, start_timestamp, end_timestamp))
            if not oi_df.empty:
                oi_df['timestamp'] = pd.to_datetime(oi_df['timestamp'], unit='ms')
                oi_df.set_index('timestamp', inplace=True)
                oi_df['open_interest'] = pd.to_numeric(oi_df['open_interest'], errors='coerce')
                price_df = price_df.join(oi_df, how='left')
                price_df['open_interest'] = price_df['open_interest'].ffill()
                price_df['open_interest'].fillna(0, inplace=True)
                print(f"✅ Added open interest data")
            else:
                price_df['open_interest'] = 0
                print(f"⚠️  No open interest data available")

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

            liq_df = self.db_manager.execute_query(liq_query, (symbol, interval, start_timestamp, end_timestamp))
            if not liq_df.empty:
                liq_df['timestamp'] = pd.to_datetime(liq_df['timestamp'], unit='ms')
                liq_df.set_index('timestamp', inplace=True)
                for col in ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd']:
                    liq_df[col] = pd.to_numeric(liq_df[col], errors='coerce')
                price_df = price_df.join(liq_df, how='left')
                for col in ['aggregated_long_liquidation_usd', 'aggregated_short_liquidation_usd']:
                    price_df[col].fillna(0, inplace=True)
                price_df['total_liquidations'] = price_df['aggregated_long_liquidation_usd'] + price_df['aggregated_short_liquidation_usd']
                print(f"✅ Added liquidation data")
            else:
                price_df['aggregated_long_liquidation_usd'] = 0
                price_df['aggregated_short_liquidation_usd'] = 0
                price_df['total_liquidations'] = 0
                print(f"⚠️  No liquidation data available")

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

            vol_df = self.db_manager.execute_query(volume_query, (symbol, interval, start_timestamp, end_timestamp))
            if not vol_df.empty:
                vol_df['timestamp'] = pd.to_datetime(vol_df['timestamp'], unit='ms')
                vol_df.set_index('timestamp', inplace=True)
                for col in ['aggregated_buy_volume_usd', 'aggregated_sell_volume_usd']:
                    vol_df[col] = pd.to_numeric(vol_df[col], errors='coerce')
                price_df = price_df.join(vol_df, how='left')
                price_df['total_taker_volume'] = price_df['aggregated_buy_volume_usd'] + price_df['aggregated_sell_volume_usd']
                price_df['buy_sell_ratio'] = price_df['aggregated_buy_volume_usd'] / (price_df['total_taker_volume'] + 1e-8)
                for col in ['aggregated_buy_volume_usd', 'aggregated_sell_volume_usd', 'total_taker_volume', 'buy_sell_ratio']:
                    price_df[col].fillna(0, inplace=True)
                print(f"✅ Added taker volume data")
            else:
                price_df['aggregated_buy_volume_usd'] = 0
                price_df['aggregated_sell_volume_usd'] = 0
                price_df['total_taker_volume'] = 0
                price_df['buy_sell_ratio'] = 0.5
                print(f"⚠️  No taker volume data available")

            # 5. Funding Rate data
            funding_query = """
                SELECT time as timestamp, close as funding_rate
                FROM cg_funding_rate_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            funding_df = self.db_manager.execute_query(funding_query, (pair, interval, start_timestamp, end_timestamp))
            if not funding_df.empty:
                funding_df['timestamp'] = pd.to_datetime(funding_df['timestamp'], unit='ms')
                funding_df.set_index('timestamp', inplace=True)
                funding_df['funding_rate'] = pd.to_numeric(funding_df['funding_rate'], errors='coerce')
                price_df = price_df.join(funding_df, how='left')
                price_df['funding_rate'] = price_df['funding_rate'].ffill()
                price_df['funding_rate'].fillna(0, inplace=True)
                print(f"✅ Added funding rate data")
            else:
                price_df['funding_rate'] = 0
                print(f"⚠️  No funding rate data available")

            # 6. Top Account Ratio data
            top_ratio_query = """
                SELECT time as timestamp, top_account_long_short_ratio
                FROM cg_long_short_top_account_ratio_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            top_ratio_df = self.db_manager.execute_query(top_ratio_query, (pair, interval, start_timestamp, end_timestamp))
            if not top_ratio_df.empty:
                top_ratio_df['timestamp'] = pd.to_datetime(top_ratio_df['timestamp'], unit='ms')
                top_ratio_df.set_index('timestamp', inplace=True)
                top_ratio_df['top_account_long_short_ratio'] = pd.to_numeric(top_ratio_df['top_account_long_short_ratio'], errors='coerce')
                price_df = price_df.join(top_ratio_df, how='left')
                price_df['top_account_long_short_ratio'] = price_df['top_account_long_short_ratio'].ffill()
                price_df['top_account_long_short_ratio'].fillna(1.0, inplace=True)  # Neutral ratio
                print(f"✅ Added top account ratio data")
            else:
                price_df['top_account_long_short_ratio'] = 1.0
                print(f"⚠️  No top account ratio data available")

            # 7. Global Account Ratio data
            global_ratio_query = """
                SELECT time as timestamp, global_account_long_short_ratio
                FROM cg_long_short_global_account_ratio_history
                WHERE pair = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            global_ratio_df = self.db_manager.execute_query(global_ratio_query, (pair, interval, start_timestamp, end_timestamp))
            if not global_ratio_df.empty:
                global_ratio_df['timestamp'] = pd.to_datetime(global_ratio_df['timestamp'], unit='ms')
                global_ratio_df.set_index('timestamp', inplace=True)
                global_ratio_df['global_account_long_short_ratio'] = pd.to_numeric(global_ratio_df['global_account_long_short_ratio'], errors='coerce')
                price_df = price_df.join(global_ratio_df, how='left')
                price_df['global_account_long_short_ratio'] = price_df['global_account_long_short_ratio'].ffill()
                price_df['global_account_long_short_ratio'].fillna(1.0, inplace=True)  # Neutral ratio
                print(f"✅ Added global account ratio data")
            else:
                price_df['global_account_long_short_ratio'] = 1.0
                print(f"⚠️  No global account ratio data available")

            # 8. NEW: Spot Orderbook History (Tier 1 - EXTREMELY HIGH impact)
            orderbook_query = """
                SELECT time as timestamp,
                       aggregated_bids_usd, aggregated_bids_quantity,
                       aggregated_asks_usd, aggregated_asks_quantity,
                       range_percent
                FROM cg_spot_aggregated_ask_bids_history
                WHERE symbol = %s AND `interval` = %s AND time BETWEEN %s AND %s
                ORDER BY time
            """

            orderbook_df = self.db_manager.execute_query(orderbook_query, (symbol, interval, start_timestamp, end_timestamp))
            if not orderbook_df.empty:
                orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'], unit='ms')
                orderbook_df.set_index('timestamp', inplace=True)

                # Convert to numeric
                for col in ['aggregated_bids_usd', 'aggregated_bids_quantity', 'aggregated_asks_usd', 'aggregated_asks_quantity']:
                    orderbook_df[col] = pd.to_numeric(orderbook_df[col], errors='coerce')

                # Calculate orderbook microstructure features
                orderbook_df['total_depth'] = orderbook_df['aggregated_bids_usd'] + orderbook_df['aggregated_asks_usd']
                orderbook_df['bid_ask_imbalance'] = (orderbook_df['aggregated_bids_usd'] - orderbook_df['aggregated_asks_usd']) / (orderbook_df['total_depth'] + 1e-8)
                orderbook_df['liquidity_ratio'] = orderbook_df['aggregated_bids_quantity'] / (orderbook_df['aggregated_asks_quantity'] + 1e-8)
                orderbook_df['orderbook_spread'] = (orderbook_df['aggregated_asks_usd'] - orderbook_df['aggregated_bids_usd']) / (orderbook_df['total_depth'] + 1e-8)

                price_df = price_df.join(orderbook_df, how='left')
                # Forward fill microstructure features
                for col in ['total_depth', 'bid_ask_imbalance', 'liquidity_ratio', 'orderbook_spread']:
                    price_df[col] = price_df[col].ffill()
                    price_df[col].fillna(0, inplace=True)

                print(f"✅ Added spot orderbook microstructure data")
            else:
                price_df['total_depth'] = 0
                price_df['bid_ask_imbalance'] = 0
                price_df['liquidity_ratio'] = 1.0
                price_df['orderbook_spread'] = 0.01
                print(f"⚠️  No orderbook data available")

            # 9. NEW: Futures Basis History (Tier 1 - EXTREMELY HIGH impact)
            basis_query = """
                SELECT time as timestamp,
                       open_basis, close_basis, open_change, close_change
                FROM cg_futures_basis_history
                WHERE pair = %s AND `interval` = %s AND time BETWEEN %s AND %s
                ORDER BY time
            """

            basis_df = self.db_manager.execute_query(basis_query, (pair, interval, start_timestamp, end_timestamp))
            if not basis_df.empty:
                basis_df['timestamp'] = pd.to_datetime(basis_df['timestamp'], unit='ms')
                basis_df.set_index('timestamp', inplace=True)

                # Convert to numeric
                for col in ['open_basis', 'close_basis', 'open_change', 'close_change']:
                    basis_df[col] = pd.to_numeric(basis_df[col], errors='coerce')

                # Calculate basis features
                basis_df['basis_momentum'] = basis_df['close_basis'].diff()
                basis_df['basis_volatility'] = basis_df['close_basis'].rolling(window=10, min_periods=1).std()
                basis_df['positive_basis'] = (basis_df['close_basis'] > 0).astype(int)  # Contango
                basis_df['basis_trend'] = basis_df['close_basis'].rolling(window=5, min_periods=1).mean()

                price_df = price_df.join(basis_df, how='left')
                # Forward fill basis features
                for col in ['open_basis', 'close_basis', 'basis_momentum', 'basis_volatility', 'positive_basis', 'basis_trend']:
                    price_df[col] = price_df[col].ffill()
                    price_df[col].fillna(0, inplace=True)

                print(f"✅ Added futures basis data")
            else:
                price_df['open_basis'] = 0
                price_df['close_basis'] = 0
                price_df['basis_momentum'] = 0
                price_df['basis_volatility'] = 0
                price_df['positive_basis'] = 0
                price_df['basis_trend'] = 0
                print(f"⚠️  No futures basis data available")

            # 10. NEW: Futures Footprint History (Tier 2 - VERY HIGH impact)
            footprint_query = """
                SELECT time as timestamp, price_start, price_end,
                       taker_buy_volume, taker_sell_volume,
                       taker_buy_trades, taker_sell_trades
                FROM cg_futures_footprint_history
                WHERE symbol = %s AND `interval` = %s AND time BETWEEN %s AND %s
                ORDER BY time
            """

            footprint_df = self.db_manager.execute_query(footprint_query, (symbol, interval, start_timestamp, end_timestamp))
            if not footprint_df.empty:
                footprint_df['timestamp'] = pd.to_datetime(footprint_df['timestamp'], unit='ms')
                footprint_df.set_index('timestamp', inplace=True)

                # Convert to numeric
                for col in ['price_start', 'price_end', 'taker_buy_volume', 'taker_sell_volume',
                           'taker_buy_trades', 'taker_sell_trades']:
                    footprint_df[col] = pd.to_numeric(footprint_df[col], errors='coerce')

                # Calculate footprint/aggressiveness features
                total_volume = footprint_df['taker_buy_volume'] + footprint_df['taker_sell_volume']
                total_trades = footprint_df['taker_buy_trades'] + footprint_df['taker_sell_trades']

                footprint_df['volume_aggression'] = (footprint_df['taker_buy_volume'] - footprint_df['taker_sell_volume']) / (total_volume + 1e-8)
                footprint_df['trade_aggression'] = (footprint_df['taker_buy_trades'] - footprint_df['taker_sell_trades']) / (total_trades + 1e-8)
                footprint_df['price_impact'] = (footprint_df['price_end'] - footprint_df['price_start']) / footprint_df['price_start'].replace(0, 1e-8)
                footprint_df['aggressive_volume_ratio'] = total_volume / (price_df['volume'] + 1e-8)

                price_df = price_df.join(footprint_df, how='left')
                # Forward fill footprint features
                for col in ['volume_aggression', 'trade_aggression', 'price_impact', 'aggressive_volume_ratio']:
                    price_df[col] = price_df[col].ffill()
                    price_df[col].fillna(0, inplace=True)

                print(f"✅ Added futures footprint data")
            else:
                price_df['volume_aggression'] = 0
                price_df['trade_aggression'] = 0
                price_df['price_impact'] = 0
                price_df['aggressive_volume_ratio'] = 0.1
                print(f"⚠️  No futures footprint data available")

            # 11. NEW: Option Exchange OI History (Tier 2 - VERY HIGH impact)
            options_oi_query = """
                SELECT o.timestamp, o.price,
                       SUM(e.open_interest) as total_oi,
                       COUNT(DISTINCT e.exchange) as exchange_count
                FROM cg_option_exchange_oi_history_time_list o
                JOIN cg_option_exchange_oi_history_exchange_data e ON o.option_exchange_oi_history_id = e.option_exchange_oi_history_id
                JOIN cg_option_exchange_oi_history main ON o.option_exchange_oi_history_id = main.id
                WHERE main.symbol = %s AND o.timestamp BETWEEN %s AND %s
                GROUP BY o.timestamp, o.price
                ORDER BY o.timestamp
            """

            options_df = self.db_manager.execute_query(options_oi_query, (symbol, start_timestamp, end_timestamp))
            if not options_df.empty:
                options_df['timestamp'] = pd.to_datetime(options_df['timestamp'], unit='ms')
                options_df.set_index('timestamp', inplace=True)

                # Convert to numeric
                for col in ['total_oi', 'exchange_count']:
                    options_df[col] = pd.to_numeric(options_df[col], errors='coerce')

                # Calculate options features
                options_df['oi_change'] = options_df['total_oi'].diff()
                options_df['oi_volatility'] = options_df['total_oi'].rolling(window=10, min_periods=1).std()
                options_df['exchange_diversification'] = options_df['exchange_count'] / 5.0  # Normalize by typical exchange count

                price_df = price_df.join(options_df, how='left')
                # Forward fill options features
                for col in ['total_oi', 'oi_change', 'oi_volatility', 'exchange_diversification']:
                    price_df[col] = price_df[col].ffill()
                    price_df[col].fillna(0, inplace=True)

                print(f"✅ Added options OI data")
            else:
                price_df['total_oi'] = 0
                price_df['oi_change'] = 0
                price_df['oi_volatility'] = 0
                price_df['exchange_diversification'] = 0
                print(f"⚠️  No options OI data available")

            # Final cleanup
            price_df = price_df.dropna(subset=['close'])  # Ensure we have price data
            print(f"📊 Loaded comprehensive market data: {len(price_df)} rows with {len(price_df.columns)} features")
            print(f"📊 Data sources: Price, OI, Liquidations, Taker Volume, Funding, Account Ratios, Orderbook, Basis, Footprint, Options OI")

            return price_df

        except Exception as e:
            print(f"❌ Error fetching market data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def calculate_signal_rule(self, df: pd.DataFrame, latest_row: pd.Series) -> Dict[str, Any]:
        """Calculate enhanced signal rule using multiple data sources"""
        try:
            if len(df) < 20:
                return {'signal_rule': 'NEUTRAL', 'signal_score': 0.5}

            # Technical Analysis Component
            df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
            df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
            latest_sma_20 = df['sma_20'].iloc[-1]
            latest_sma_50 = df['sma_50'].iloc[-1]
            current_price = latest_row['close']

            # Price momentum signal
            if current_price > latest_sma_20 > latest_sma_50:
                price_signal = 0.7  # Bullish
            elif current_price < latest_sma_20 < latest_sma_50:
                price_signal = 0.3  # Bearish
            else:
                price_signal = 0.5  # Neutral

            # Volume Analysis Component
            buy_sell_ratio = latest_row.get('buy_sell_ratio', 0.5)
            volume_signal = np.clip(buy_sell_ratio, 0.2, 0.8)

            # Open Interest Analysis
            current_oi = latest_row.get('open_interest', 0)
            if len(df) >= 10:
                oi_ma = df['open_interest'].rolling(window=10, min_periods=1).mean().iloc[-2]
                if oi_ma > 0:
                    oi_change = (current_oi - oi_ma) / oi_ma
                    oi_signal = 0.5 + np.clip(oi_change, -0.3, 0.3)
                else:
                    oi_signal = 0.5
            else:
                oi_signal = 0.5

            # Liquidation Analysis (contrarian indicator)
            total_liq = latest_row.get('total_liquidations', 0)
            long_liq = latest_row.get('aggregated_long_liquidation_usd', 0)
            short_liq = latest_row.get('aggregated_short_liquidation_usd', 0)

            if total_liq > 0:
                # High long liquidations = potential bounce (bullish)
                # High short liquidations = potential pullback (bearish)
                liq_ratio = long_liq / (total_liq + 1e-8)
                liq_signal = 0.5 + (liq_ratio - 0.5) * 0.3  # Dampened effect
            else:
                liq_signal = 0.5

            # Funding Rate Analysis (contrarian indicator)
            funding_rate = latest_row.get('funding_rate', 0)
            if abs(funding_rate) > 0.01:  # 1% threshold
                # High positive funding = overleveraged longs = bearish
                # High negative funding = overleveraged shorts = bullish
                funding_signal = 0.5 - np.clip(funding_rate, -0.2, 0.2)
            else:
                funding_signal = 0.5

            # Smart Money Analysis (top accounts)
            top_ratio = latest_row.get('top_account_long_short_ratio', 1.0)
            if top_ratio > 1.2:  # Top accounts heavily long
                smart_money_signal = 0.4  # Contrarian bearish
            elif top_ratio < 0.8:  # Top accounts heavily short
                smart_money_signal = 0.6  # Contrarian bullish
            else:
                smart_money_signal = 0.5

            # Global Sentiment Analysis
            global_ratio = latest_row.get('global_account_long_short_ratio', 1.0)
            if global_ratio > 1.5:  # Crowd heavily long
                crowd_signal = 0.3  # Contrarian bearish
            elif global_ratio < 0.67:  # Crowd heavily short
                crowd_signal = 0.7  # Contrarian bullish
            else:
                crowd_signal = 0.5

            # NEW: Orderbook Microstructure Analysis (Tier 1 - EXTREMELY HIGH impact)
            bid_ask_imbalance = latest_row.get('bid_ask_imbalance', 0)
            depth_ratio = latest_row.get('depth_ratio', 1.0)
            orderbook_spread = latest_row.get('orderbook_spread', 0.01)

            # Orderbook pressure is VERY predictive for immediate price moves
            if abs(bid_ask_imbalance) > 0.15:  # Significant imbalance
                orderbook_signal = 0.5 + np.clip(bid_ask_imbalance * 1.5, -0.4, 0.4)
            else:
                orderbook_signal = 0.5

            # Liquidity quality adjustment
            if depth_ratio < 0.5 and abs(bid_ask_imbalance) > 0.1:
                # Thin liquidity + imbalance = explosive moves possible
                orderbook_signal = 0.5 + np.sign(bid_ask_imbalance) * 0.2

            # NEW: Futures Basis Analysis (Tier 1 - EXTREMELY HIGH impact)
            basis_momentum = latest_row.get('basis_momentum', 0)
            close_basis = latest_row.get('close_basis', 0)

            if abs(basis_momentum) > 0.0001:  # Significant basis movement
                # Basis momentum leads spot price
                basis_signal = 0.5 + np.clip(basis_momentum * 1000, -0.3, 0.3)
            else:
                basis_signal = 0.5

            # Contango/backwardation regime bias
            if close_basis > 0:  # Contango = normal bullish bias
                basis_signal += 0.05
            elif close_basis < 0:  # Backwardation = bearish pressure
                basis_signal -= 0.05

            basis_signal = np.clip(basis_signal, 0.0, 1.0)

            # NEW: Futures Footprint Aggressiveness Analysis (Tier 2 - VERY HIGH impact)
            volume_aggression = latest_row.get('volume_aggression', 0)
            trade_aggression = latest_row.get('trade_aggression', 0)
            price_impact = latest_row.get('price_impact', 0)

            if abs(volume_aggression) > 0.25:  # High aggressive buying/selling
                aggression_signal = 0.5 + np.clip(volume_aggression * 0.8, -0.3, 0.3)
            else:
                aggression_signal = 0.5

            # Trade size confirmation
            if abs(trade_aggression) > 0.15:  # Large institutional trades
                aggression_signal += np.clip(trade_aggression * 0.3, -0.1, 0.1)

            aggression_signal = np.clip(aggression_signal, 0.0, 1.0)

            # NEW: Options OI Analysis (Tier 2 - VERY HIGH impact)
            options_oi_ratio = latest_row.get('options_oi_ratio', 1.0)
            oi_volatility = latest_row.get('oi_volatility', 0)

            if options_oi_ratio > 1.2:  # High options OI growth
                options_signal = 0.6  # Bullish options flow
            elif options_oi_ratio < 0.8:  # High options OI decline
                options_signal = 0.4  # Bearish options flow
            else:
                options_signal = 0.5

            # Volatility adjustment
            if oi_volatility > oi_volatility * 1.5:  # High options volatility
                options_signal = 0.5  # Neutralize during high volatility

            # ENHANCED Weighted ensemble signal with new microstructure data
            weights = {
                'price': 0.15,          # Technical analysis (reduced weight)
                'volume': 0.10,         # Buy/sell pressure (reduced weight)
                'oi': 0.08,             # Open interest changes (reduced weight)
                'liquidation': 0.10,    # Liquidation pressure (reduced weight)
                'funding': 0.08,        # Funding rate extremes (reduced weight)
                'smart_money': 0.10,    # Top account positioning (reduced weight)
                'crowd': 0.06,          # Global sentiment (reduced weight)
                'orderbook': 0.20,      # NEW: Orderbook microstructure (HIGH weight)
                'basis': 0.18,          # NEW: Futures basis (HIGH weight)
                'aggression': 0.15,     # NEW: Footprint aggressiveness
                'options': 0.10         # NEW: Options OI flow
            }

            ensemble_score = (
                price_signal * weights['price'] +
                volume_signal * weights['volume'] +
                oi_signal * weights['oi'] +
                liq_signal * weights['liquidation'] +
                funding_signal * weights['funding'] +
                smart_money_signal * weights['smart_money'] +
                crowd_signal * weights['crowd'] +
                orderbook_signal * weights['orderbook'] +
                basis_signal * weights['basis'] +
                aggression_signal * weights['aggression'] +
                options_signal * weights['options']
            )

            ensemble_score = np.clip(ensemble_score, 0.0, 1.0)

            # Convert to trading signal
            if ensemble_score >= 0.65:
                signal_rule = 'BUY'
            elif ensemble_score <= 0.35:
                signal_rule = 'SELL'
            else:
                signal_rule = 'NEUTRAL'

            return {
                'signal_rule': signal_rule,
                'signal_score': float(ensemble_score),
                'components': {
                    'price_signal': price_signal,
                    'volume_signal': volume_signal,
                    'oi_signal': oi_signal,
                    'liq_signal': liq_signal,
                    'funding_signal': funding_signal,
                    'smart_money_signal': smart_money_signal,
                    'crowd_signal': crowd_signal,
                    'orderbook_signal': orderbook_signal,
                    'basis_signal': basis_signal,
                    'aggression_signal': aggression_signal,
                    'options_signal': options_signal
                },
                'microstructure_insights': {
                    'bid_ask_imbalance': float(bid_ask_imbalance),
                    'basis_momentum': float(basis_momentum),
                    'volume_aggression': float(volume_aggression),
                    'options_oi_ratio': float(options_oi_ratio),
                    'depth_quality': 'thin' if depth_ratio < 0.5 else 'normal',
                    'basis_regime': 'contango' if close_basis > 0 else 'backwardation' if close_basis < 0 else 'neutral',
                    'aggressive_flow': 'buying' if volume_aggression > 0.25 else 'selling' if volume_aggression < -0.25 else 'balanced',
                    'options_flow': 'bullish' if options_oi_ratio > 1.2 else 'bearish' if options_oi_ratio < 0.8 else 'neutral'
                }
            }

        except Exception as e:
            print(f"⚠️  Error calculating enhanced signal rule: {e}")
            import traceback
            traceback.print_exc()
            return {
                'signal_rule': 'NEUTRAL',
                'signal_score': 0.5
            }

    def store_signal(self, symbol: str, pair: str, interval: str, horizon_minutes: int,
                    generated_at: datetime, latest_row: pd.Series, features: Dict[str, Any]):
        """Store signal in database"""
        try:
            signal_data = self.calculate_signal_rule(features.get('features_df', pd.DataFrame()), latest_row)

            # Prepare features payload with robust error handling
            def safe_float(value, default=0.0):
                """Convert value to float safely"""
                try:
                    if pd.isna(value) or value is None:
                        return default
                    return float(value)
                except (ValueError, TypeError, OverflowError):
                    return default

            def safe_int(value, default=0):
                """Convert value to int safely"""
                try:
                    if pd.isna(value) or value is None:
                        return default
                    return int(value)
                except (ValueError, TypeError, OverflowError):
                    return default

            # Extract values safely from latest_row
            close_price = safe_float(latest_row.get('close', 0.0))

            features_payload = {
                'technical_indicators': {
                    'sma_5': safe_float(latest_row.get('sma_5', close_price)),
                    'sma_20': safe_float(latest_row.get('sma_20', close_price)),
                    'sma_50': safe_float(latest_row.get('sma_50', close_price)),
                    'ema_12': safe_float(latest_row.get('ema_12', close_price)),
                    'ema_26': safe_float(latest_row.get('ema_26', close_price)),
                    'rsi': safe_float(latest_row.get('rsi', 50.0)),
                    'macd': safe_float(latest_row.get('macd', 0.0)),
                    'bb_upper': safe_float(latest_row.get('bb_upper', close_price)),
                    'bb_lower': safe_float(latest_row.get('bb_lower', close_price)),
                    'atr': safe_float(latest_row.get('atr', 0.0)),
                    'volume_ratio': safe_float(latest_row.get('volume_ratio', 1.0)),
                    'volatility_20': safe_float(latest_row.get('volatility_20', 0.01)),
                },
                'price_features': {
                    'returns': safe_float(latest_row.get('returns', 0.0)),
                    'log_returns': safe_float(latest_row.get('log_returns', 0.0)),
                    'high_low_ratio': safe_float(latest_row.get('high_low_ratio', 1.0)),
                    'close_open_ratio': safe_float(latest_row.get('close_open_ratio', 1.0)),
                },
                'time_features': {
                    'hour': safe_int(generated_at.hour),
                    'day_of_week': safe_int(generated_at.weekday()),
                    'month': safe_int(generated_at.month),
                }
            }

            # Additional validation and cleaning
            def validate_and_clean_dict(d):
                """Validate and clean dictionary values to ensure JSON compatibility"""
                cleaned = {}
                for k, v in d.items():
                    if isinstance(v, dict):
                        cleaned[k] = validate_and_clean_dict(v)
                    elif isinstance(v, (list, tuple)):
                        cleaned[k] = [validate_and_clean_dict(item) if isinstance(item, dict) else item for item in v]
                    elif isinstance(v, (int, float)):
                        # Check for special float values
                        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                            cleaned[k] = 0.0
                        else:
                            cleaned[k] = v
                    elif v is None:
                        cleaned[k] = 0
                    else:
                        cleaned[k] = v
                return cleaned

            features_payload = validate_and_clean_dict(features_payload)

            # Serialize to JSON with error handling
            try:
                features_json = json.dumps(features_payload, ensure_ascii=False, separators=(',', ':'))
            except (TypeError, ValueError) as json_error:
                print(f"⚠️  JSON serialization error: {json_error}")
                # Fallback to minimal payload if JSON serialization fails
                features_json = json.dumps({
                    'technical_indicators': {'close': close_price},
                    'price_features': {},
                    'time_features': {'hour': safe_int(generated_at.hour)}
                }, ensure_ascii=False, separators=(',', ':'))

            with self.db_manager.connection.cursor() as cursor:
                insert_query = """
                    INSERT INTO cg_train_dataset (
                        symbol, pair, time_interval, generated_at, horizon_minutes, price_now,
                        features_payload, signal_rule, signal_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                cursor.execute(insert_query, (
                    symbol, pair, interval, generated_at, horizon_minutes,
                    close_price, features_json,
                    signal_data['signal_rule'], signal_data['signal_score']
                ))

            self.db_manager.connection.commit()
            print(f"✅ Stored signal: {symbol} {pair} {interval} at {generated_at}")

        except Exception as e:
            print(f"❌ Error storing signal: {e}")
            import traceback
            traceback.print_exc()
            if hasattr(self.db_manager, 'connection') and self.db_manager.connection:
                try:
                    self.db_manager.connection.rollback()
                except:
                    pass

    def collect_signals(self, symbol: str, pair: str, interval: str, horizon_minutes: int, target_timestamp=None):
        """Main collection function"""
        print(f"🚀 Starting signal collection for {symbol} {pair} {interval}")
        print(f"📅 Horizon: {horizon_minutes} minutes")

        # Ensure table exists
        self._create_table_if_not_exists()

        # Get the most recent data first to check what's available
        print("📊 Checking available data...")
        check_query = """
            SELECT MIN(time) as earliest_time, MAX(time) as latest_time, COUNT(*) as count
            FROM cg_spot_price_history
            WHERE symbol = %s AND `interval` = %s
        """
        check_result = self.db_manager.execute_query(check_query, (pair, interval))

        if not check_result.empty:
            earliest = pd.to_datetime(check_result.iloc[0]['earliest_time'], unit='ms')
            latest = pd.to_datetime(check_result.iloc[0]['latest_time'], unit='ms')
            count = check_result.iloc[0]['count']
            print(f"📅 Available data: {count:,} rows from {earliest} to {latest}")

        # Define time range based on target_timestamp or recent data
        if target_timestamp:
            # Process ALL data from the specified timestamp onwards
            target_time = pd.to_datetime(target_timestamp, unit='ms')
            start_time = target_time

            # Get all data from target_time to the end
            latest_time_query = """
                SELECT MAX(time) as max_time
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
            """
            latest_result = self.db_manager.execute_query(latest_time_query, (pair, interval))

            if latest_result.empty:
                print("❌ No latest timestamp found")
                return

            latest_timestamp = latest_result.iloc[0]['max_time']
            end_time = pd.to_datetime(latest_timestamp, unit='ms')

            print(f"🎯 Processing ALL data from {target_time} to {end_time}")
            batch_mode = True

        else:
            # Default behavior: get recent data only
            latest_time_query = """
                SELECT MAX(time) as max_time
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
            """
            latest_result = self.db_manager.execute_query(latest_time_query, (pair, interval))

            if latest_result.empty:
                print("❌ No latest timestamp found")
                return

            latest_timestamp = latest_result.iloc[0]['max_time']
            end_time = pd.to_datetime(latest_timestamp, unit='ms')
            start_time = end_time - timedelta(hours=48)  # Get 48 hours of data
            print(f"🎯 Processing recent data from {start_time} to {end_time}")
            batch_mode = False

        # Get all market data for the range
        print("📊 Loading market data...")
        df = self.get_market_data(symbol, pair, interval, start_time, end_time)

        if df.empty:
            print("❌ No market data available")
            return

        print(f"📊 Loaded {len(df):,} rows of market data")

        try:
            # Create features for all data
            print("🔧 Engineering features...")
            df_features = self.feature_engineer.create_all_features(df)

            if df_features.empty:
                print("❌ Failed to create features")
                return

            print(f"✅ Feature engineering completed. Final shape: {df_features.shape}")

            if batch_mode:
                # Process multiple timestamps in batch mode
                self._process_batch_signals(
                    symbol=symbol,
                    pair=pair,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    df_features=df_features
                )
            else:
                # Default behavior: process only the latest signal
                latest_timestamp = df_features.index[-1]
                latest_row = df_features.iloc[-1]
                self.store_signal(
                    symbol=symbol,
                    pair=pair,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    generated_at=latest_timestamp,
                    latest_row=latest_row,
                    features={'features_df': df_features}
                )

            print(f"✅ Signal collection completed successfully")

        except Exception as e:
            print(f"❌ Error during signal collection: {e}")
            import traceback
            traceback.print_exc()

    def _process_batch_signals(self, symbol: str, pair: str, interval: str, horizon_minutes: int, df_features: pd.DataFrame):
        """Process signals in batch mode - collect signals for all timestamps"""
        print(f"🔄 Starting batch signal processing for {len(df_features):,} data points")

        signals_processed = 0
        signals_failed = 0

        # Process every row in the features dataframe
        for i, (timestamp, row) in enumerate(df_features.iterrows()):
            try:
                # Skip the first 50 rows to ensure we have enough history for technical indicators
                if i < 50:
                    continue

                # Store signal for this timestamp
                self.store_signal(
                    symbol=symbol,
                    pair=pair,
                    interval=interval,
                    horizon_minutes=horizon_minutes,
                    generated_at=timestamp,
                    latest_row=row,
                    features={'features_df': df_features.iloc[:i+1]}  # Pass data up to current point
                )

                signals_processed += 1

                # Progress indicator
                if signals_processed % 1000 == 0:
                    print(f"   📈 Processed {signals_processed:,} signals... Current time: {timestamp}")

            except Exception as e:
                signals_failed += 1
                if signals_failed <= 5:  # Only show first 5 errors to avoid spam
                    print(f"⚠️  Failed to process signal at {timestamp}: {e}")

        print(f"🎯 Batch processing completed:")
        print(f"   ✅ Successfully processed: {signals_processed:,} signals")
        if signals_failed > 0:
            print(f"   ❌ Failed: {signals_failed:,} signals")

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

def main():
    parser = argparse.ArgumentParser(description='Collect trading signals')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTC)')
    parser.add_argument('--pair', required=True, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--interval', required=True, help='Time interval (e.g., 1h, 4h)')
    parser.add_argument('--horizon', required=True, type=int, help='Horizon in minutes (e.g., 60)')
    parser.add_argument('--timestamp', type=int, help='Target timestamp in milliseconds (e.g., 1672531200000)')

    args = parser.parse_args()

    try:
        collector = SignalCollector()
        collector.collect_signals(
            symbol=args.symbol,
            pair=args.pair,
            interval=args.interval,
            horizon_minutes=args.horizon,
            target_timestamp=args.timestamp
        )
    except KeyboardInterrupt:
        print("\n👋 Signal collection interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()