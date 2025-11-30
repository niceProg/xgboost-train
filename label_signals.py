#!/usr/bin/env python3
"""
Label pending signals with actual price movements.
This script looks at signals that were generated and labels them based on future price movement.
"""

import argparse
import sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Any

from env_config import get_database_config
from database import DatabaseManager

class SignalLabeler:
    def __init__(self):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)

    def get_pending_signals(self, limit: int = None, symbol: str = None, pair: str = None, interval: str = None) -> List[Dict]:
        """Get pending signals that need labeling"""
        try:
            # Build base query
            query = """
                SELECT id, symbol, pair, time_interval, generated_at, horizon_minutes, price_now
                FROM cg_train_dataset
                WHERE label_status = 'pending'
                  AND generated_at <= %s
            """

            # Only label signals where horizon has passed
            cutoff_time = datetime.now() - timedelta(minutes=1)  # 1 minute buffer
            params = [cutoff_time]

            # Add filters if provided
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            if pair:
                query += " AND pair = %s"
                params.append(pair)
            if interval:
                query += " AND time_interval = %s"
                params.append(interval)

            query += " ORDER BY generated_at"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            df = self.db_manager.execute_query(query, tuple(params))
            signals = df.to_dict('records')

            filter_desc = f" for {symbol} {pair} {interval}" if symbol or pair or interval else ""
            print(f"📊 Found {len(signals)} pending signals to label{filter_desc}")
            return signals

        except Exception as e:
            print(f"❌ Error fetching pending signals: {e}")
            return []

    def get_future_price(self, symbol: str, pair: str, interval: str,
                        generated_at: datetime, horizon_minutes: int) -> float:
        """Get price at horizon time"""
        try:
            # Calculate target time
            target_time = generated_at + timedelta(minutes=horizon_minutes)

            # If target time is in the future, can't label yet
            if target_time > datetime.now():
                return None

            # Convert to millisecond timestamp for database query
            target_timestamp = int(target_time.timestamp() * 1000)

            query = """
                SELECT close
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
                  AND time >= %s
                ORDER BY time ASC
                LIMIT 1
            """

            result = self.db_manager.execute_query(query, (pair, interval, target_timestamp))

            if not result.empty:
                return float(result.iloc[0]['close'])
            else:
                # Try to get the closest available price before target time
                query = """
                    SELECT close
                    FROM cg_spot_price_history
                    WHERE symbol = %s AND `interval` = %s
                      AND time <= %s
                    ORDER BY time DESC
                    LIMIT 1
                """

                result = self.db_manager.execute_query(query, (pair, interval, target_timestamp))

                if not result.empty:
                    return float(result.iloc[0]['close'])
                else:
                    print(f"⚠️  No price data found for {pair} around {target_time}")
                    return None

        except Exception as e:
            print(f"❌ Error getting future price: {e}")
            return None

    def calculate_label(self, price_now: float, price_future: float,
                       threshold_pct: float = 0.5) -> Dict[str, Any]:
        """Calculate label based on price movement"""
        if price_future is None:
            return {
                'label_direction': None,
                'label_magnitude': None,
                'label_status': 'pending'  # Keep pending if no future price
            }

        # Calculate magnitude (convert to float to avoid Decimal issues)
        price_now = float(price_now)
        price_future = float(price_future)
        magnitude = (price_future - price_now) / price_now
        magnitude_pct = magnitude * 100

        # Determine direction based on threshold
        if magnitude_pct > threshold_pct:
            label_direction = 'UP'
        elif magnitude_pct < -threshold_pct:
            label_direction = 'DOWN'
        else:
            label_direction = 'FLAT'

        return {
            'label_direction': label_direction,
            'label_magnitude': float(magnitude),
            'label_status': 'labeled'
        }

    def update_signal_label(self, signal_id: int, price_future: float, label_data: Dict[str, Any]):
        """Update signal with label data"""
        try:
            query = """
                UPDATE cg_train_dataset
                SET price_future = %s,
                    label_direction = %s,
                    label_magnitude = %s,
                    label_status = %s,
                    labeled_at = NOW()
                WHERE id = %s
            """

            affected_rows = self.db_manager.execute_update(query, (
                price_future,
                label_data['label_direction'],
                label_data['label_magnitude'],
                label_data['label_status'],
                signal_id
            ))

            if affected_rows == 0:
                print(f"⚠️  No rows updated for signal ID {signal_id}")

        except Exception as e:
            print(f"❌ Error updating signal label: {e}")

    def label_signals(self, limit: int = None, threshold_pct: float = 0.5, symbol: str = None, pair: str = None, interval: str = None):
        """Main labeling function"""
        filter_info = f" for {symbol} {pair} {interval}" if symbol or pair or interval else ""
        print(f"🏷️  Starting signal labeling{filter_info}")
        print(f"📊 Limit: {limit or 'all pending signals'}")
        print(f"📈 Threshold: ±{threshold_pct}%")

        pending_signals = self.get_pending_signals(limit, symbol, pair, interval)

        if not pending_signals:
            print("✅ No pending signals to label")
            return

        labeled_count = 0
        skipped_count = 0
        error_count = 0

        for signal in pending_signals:
            try:
                print(f"\n🔄 Processing signal ID: {signal['id']}")
                print(f"   Symbol: {signal['symbol']} {signal['pair']}")
                print(f"   Generated: {signal['generated_at']}")
                print(f"   Horizon: {signal['horizon_minutes']} minutes")
                print(f"   Price now: {signal['price_now']}")

                # Get future price
                price_future = self.get_future_price(
                    signal['symbol'], signal['pair'], signal['time_interval'],
                    signal['generated_at'], signal['horizon_minutes']
                )

                if price_future is None:
                    print("   ⏳ Future price not available yet, skipping")
                    skipped_count += 1
                    continue

                # Calculate label
                label_data = self.calculate_label(
                    signal['price_now'], price_future, threshold_pct
                )

                # Update database
                self.update_signal_label(signal['id'], price_future, label_data)

                print(f"   ✅ Labeled: {label_data['label_direction']}")
                print(f"   📊 Price change: {label_data['label_magnitude']*100:.2f}%")
                print(f"   💰 Price now → future: {signal['price_now']} → {price_future}")

                labeled_count += 1

            except Exception as e:
                print(f"   ❌ Error processing signal {signal['id']}: {e}")
                error_count += 1
                continue

        print(f"\n📊 Labeling Summary:")
        print(f"   ✅ Successfully labeled: {labeled_count}")
        print(f"   ⏳ Skipped (future not available): {skipped_count}")
        print(f"   ❌ Errors: {error_count}")

    def get_label_statistics(self) -> Dict[str, Any]:
        """Get statistics about labeled signals"""
        try:
            query = """
                SELECT
                    label_status,
                    label_direction,
                    COUNT(*) as count,
                    AVG(label_magnitude) * 100 as avg_magnitude_pct
                FROM cg_train_dataset
                GROUP BY label_status, label_direction
                ORDER BY label_status, label_direction
            """

            df = self.db_manager.execute_query(query)

            if df.empty:
                print("\n📈 Label Statistics: No data available")
                return {}

            print("\n📈 Label Statistics:")
            for index, row in df.iterrows():
                status = row['label_status'] or 'NULL'
                direction = row['label_direction'] or 'NULL'
                count = row['count']
                magnitude = f"{row['avg_magnitude_pct']:.2f}%" if pd.notna(row['avg_magnitude_pct']) else "N/A"
                print(f"   {status:8} | {direction:6} | {count:4} | {magnitude}")

            return {'results': df.to_dict('records')}

        except Exception as e:
            print(f"❌ Error getting statistics: {e}")
            return {}

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

def main():
    parser = argparse.ArgumentParser(description='Label pending signals')
    parser.add_argument('--symbol', help='Symbol filter (e.g., BTC)')
    parser.add_argument('--pair', help='Pair filter (e.g., BTCUSDT)')
    parser.add_argument('--interval', help='Interval filter (e.g., 1h)')
    parser.add_argument('--limit', type=int, help='Limit number of signals to label')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Price change threshold in percentage (default: 0.5%)')
    parser.add_argument('--stats', action='store_true',
                       help='Show label statistics only')

    args = parser.parse_args()

    try:
        labeler = SignalLabeler()

        if args.stats:
            labeler.get_label_statistics()
        else:
            labeler.label_signals(
                limit=args.limit,
                threshold_pct=args.threshold,
                symbol=args.symbol,
                pair=args.pair,
                interval=args.interval
            )
            labeler.get_label_statistics()

    except KeyboardInterrupt:
        print("\n👋 Signal labeling interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()