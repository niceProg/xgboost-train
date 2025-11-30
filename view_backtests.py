#!/usr/bin/env python3
"""
View and analyze stored backtest results from quantconnect_backtests table
Track performance trends and build.md target achievement over time
"""

import sys
import argparse
from datetime import datetime
import pandas as pd
from typing import Dict, Any, List
import json

from env_config import get_database_config
from database import DatabaseManager
from evaluation import PerformanceEvaluator

class BacktestViewer:
    """View and analyze stored backtest results"""

    def __init__(self):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.evaluator = PerformanceEvaluator()

    def view_all_backtests(self, limit: int = None):
        """View all stored backtest results"""
        try:
            query = """
                SELECT
                    backtest_id, name, strategy_type, description,
                    backtest_start, backtest_end, duration_days,
                    total_return, cagr, sharpe_ratio, sortino_ratio,
                    max_drawdown, recovery_days,
                    total_trades, winning_trades, losing_trades,
                    win_rate, profit_factor, avg_win, avg_loss,
                    largest_win, largest_loss,
                    starting_capital, ending_capital,
                    status, created_at, updated_at
                FROM quantconnect_backtests
                ORDER BY created_at DESC
            """

            if limit:
                query += f" LIMIT {limit}"

            df = self.db_manager.execute_query(query)

            if df.empty:
                print("❌ No backtest results found in database")
                return

            print(f"📊 Found {len(df)} backtest results:")
            print("=" * 100)

            for i, row in df.iterrows():
                self.print_backtest_summary(row, i+1)

        except Exception as e:
            print(f"❌ Error retrieving backtests: {e}")

    def print_backtest_summary(self, row: pd.Series, index: int):
        """Print formatted backtest summary"""
        status_emoji = "✅" if row['status'] == 'completed' else "⏳" if row['status'] == 'running' else "❌"

        print(f"{index}. {status_emoji} {row['backtest_id']}")
        print(f"   📈 Strategy: {row['name']}")
        print(f"   📅 Period: {row['backtest_start']} to {row['backtest_end']} ({row['duration_days']} days)")
        print(f"   💰 Return: {row['total_return']:.2%} (CAGR: {row['cagr']:.2%})")
        print(f"   🎯 Win Rate: {row['win_rate']:.2%} ({row['winning_trades']}/{row['total_trades']})")
        print(f"   📊 Sharpe: {row['sharpe_ratio']:.2f} | Sortino: {row['sortino_ratio']:.2f}")
        print(f"   ⚠️  Max Drawdown: {row['max_drawdown']:.2%}")

        # Target achievements
        targets = {
            'cagr_target': row['cagr'] >= 0.50,
            'max_dd_target': row['max_drawdown'] <= 0.25,
            'sharpe_target': 1.2 <= row['sharpe_ratio'] <= 1.6,
            'sortino_target': 2.0 <= row['sortino_ratio'] <= 3.0,
            'win_rate_target': row['win_rate'] >= 0.70
        }

        achieved = sum(targets.values())
        total = len(targets)
        grade = 'A+' if achieved == 5 else 'A' if achieved == 4 else 'B+' if achieved == 3 else 'B' if achieved == 2 else 'C+' if achieved == 1 else 'F'

        target_emoji = "🎯" if achieved == total else f"✅/❌"

        print(f"   🏆 Build.md Targets: {achieved}/{total} achieved ({grade})")

        target_strs = []
        if targets['cagr_target']:
            target_strs.append(f"CAGR ✅")
        if targets['max_dd_target']:
            target_strs.append(f"Drawdown ✅")
        if targets['sharpe_target']:
            target_strs.append(f"Sharpe ✅")
        if targets['sortino_target']:
            target_strs.append(f"Sortino ✅")
        if targets['win_rate_target']:
            target_strs.append(f"WinRate ✅")

        print(f"   {', '.join(target_strs)}")

        print(f"   📅 Created: {row['created_at']}")

    def view_detailed_backtest(self, backtest_id: str):
        """View detailed information for a specific backtest"""
        try:
            # Get basic info
            basic_query = """
                SELECT * FROM quantconnect_backtests
                WHERE backtest_id = %s
            """

            result = self.db_manager.execute_query(basic_query, (backtest_id,))

            if result.empty:
                print(f"❌ Backtest '{backtest_id}' not found")
                return

            row = result.iloc[0]

            print(f"📊 DETAILED BACKTEST REPORT: {backtest_id}")
            print("=" * 100)
            print(f"📋 Strategy: {row['name']}")
            print(f"📝 Description: {row['description']}")
            print(f"📅 Period: {row['backtest_start']} to {row['backtest_end']}")
            print(f"⏱️  Duration: {row['duration_days']} days")

            # Financial Metrics
            print(f"\n💰 FINANCIAL METRICS:")
            print(f"   Starting Capital: ${row['starting_capital']:,.2f}")
            print(f"   Ending Capital: ${row['ending_capital']:,.2f}")
            print(f"   Total Return: {row['total_return']:.2%}")
            print(f"   CAGR: {row['cagr']:.2%}")
            print(f"   Total Fees: ${row['total_fees']:,.2f}")

            # Risk Metrics
            print(f"\n⚠️  RISK METRICS:")
            print(f"   Volatility (Annual): Calculated from equity curve")
            print(f"   Max Drawdown: {row['max_drawdown']:.2%}")
            print(f"   Recovery Days: {row['recovery_days']}")

            # Risk-Adjusted Metrics
            print(f"\n📈 RISK-ADJUSTED METRICS:")
            print(f"   Sharpe Ratio: {row['sharpe_ratio']:.2f}")
            print(f"   Sortino Ratio: {row['sortino_ratio']:.2f}")
            print(f"   Calmar Ratio: {row['calmar_ratio']:.2f}")

            # Trading Metrics
            print(f"\n📊 TRADING METRICS:")
            print(f"   Total Trades: {row['total_trades']}")
            print(f"   Winning Trades: {row['winning_trades']}")
            print(f"   Losing Trades: {row['losing_trades']}")
            print(f"   Win Rate: {row['win_rate']:.2%}")
            print(f"   Profit Factor: {row['profit_factor']:.2f}")
            print(f"   Expectancy: ${row['expectancy']:.2f}")
            print(f"   Average Win: ${row['avg_win']:,.2f}")
            print(f"   Average Loss: ${row['avg_loss']:,.2f}")
            print(f"   Largest Win: ${row['largest_win']:,.2f}")
            print(f"   Largest Loss: ${row['largest_loss']:,.2f}")

            # Streak Metrics
            print(f"\n🎲 STREAK METRICS:")
            print(f"   Longest Win Streak: {row['longest_win_streak']} trades")
            print(f"   Longest Loss Streak: {row['longest_loss_streak']} trades")

            # Build.md Targets Analysis
            print(f"\n🎯 BUILD.MD TARGETS ACHIEVEMENT:")
            targets = {
                'CAGR ≥ 50%': row['cagr'] >= 0.50,
                'Max Drawdown ≤ 25%': row['max_drawdown'] <= 0.25,
                'Sharpe 1.2-1.6': 1.2 <= row['sharpe_ratio'] <= 1.6,
                'Sortino 2.0-3.0': 2.0 <= row['sortino_ratio'] <= 3.0,
                'Win Rate ≥ 70%': row['win_rate'] >= 0.70
            }

            for target, achieved in targets.items():
                status = "✅ ACHIEVED" if achieved else "❌ NOT ACHIEVED"
                print(f"   {target}: {status}")

            achieved_count = sum(targets.values())
            print(f"\n🏆 Overall: {achieved_count}/5 targets achieved")

            # Show trade details if available
            if row['trades']:
                print(f"\n📋 RECENT TRADES (first 5):")
                try:
                    trades_data = json.loads(row['trades'])
                    for i, trade in enumerate(trades_data[:5]):
                        print(f"   {i+1}. {trade['direction']} ${trade['profit_loss']:,.2f} "
                              f"({trade['confidence']:.2f} confidence)")
                        print(f"      Entry: {trade['entry_price']:.2f} → Exit: {trade['exit_price']:.2f}")
                        print(f"      Reason: {trade['exit_reason']}")

                    if len(trades_data) > 5:
                        print(f"   ... and {len(trades_data) - 5} more trades")
                except:
                    print(f"   Trade details available in database")

            print(f"\n💾 Stored in database with ID: {backtest_id}")
            print(f"📅 Last updated: {row['updated_at']}")

        except Exception as e:
            print(f"❌ Error viewing backtest details: {e}")

    def compare_backtests(self, backtest_ids: List[str]):
        """Compare multiple backtest results"""
        try:
            if len(backtest_ids) < 2:
                print("❌ Need at least 2 backtest IDs to compare")
                return

            print(f"📊 COMPARING {len(backtest_ids)} BACKTESTS")
            print("=" * 100)

            # Fetch all backtest data
            backtest_data = []
            for backtest_id in backtest_ids:
                result = self.db_manager.execute_query(
                    "SELECT * FROM quantconnect_backtests WHERE backtest_id = %s",
                    (backtest_id,)
                )
                if not result.empty:
                    backtest_data.append(result.iloc[0])

            if len(backtest_data) < 2:
                print("❌ Some backtest IDs not found")
                return

            # Create comparison table
            comparison_data = []
            for data in backtest_data:
                comparison_data.append({
                    'Backtest ID': data['backtest_id'],
                    'Return': f"{data['total_return']:.2%}",
                    'CAGR': f"{data['cagr']:.2%}",
                    'Sharpe': f"{data['sharpe_ratio']:.2f}",
                    'Win Rate': f"{data['win_rate']:.2%}",
                    'Max DD': f"{data['max_drawdown']:.2%}",
                    'Trades': data['total_trades']
                })

            df = pd.DataFrame(comparison_data)
            print(df.to_string(index=False))

            # Find best performer
            best_cagr = max(backtest_data, key=lambda x: x['cagr'])
            best_sharpe = max(backtest_data, key=lambda x: x['sharpe_ratio'])
            best_return = max(backtest_data, key=lambda x: x['total_return'])

            print(f"\n🏆 BEST PERFORMERS:")
            print(f"   Highest Return: {best_return['backtest_id']} ({best_return['total_return']:.2%})")
            print(f"   Highest CAGR: {best_cagr['backtest_id']} ({best_cagr['cagr']:.2%})")
            print(f"   Highest Sharpe: {best_sharpe['backtest_id']} ({best_sharpe['sharpe_ratio']:.2f})")

        except Exception as e:
            print(f"❌ Error comparing backtests: {e}")

    def analyze_performance_trends(self, days_back: int = 30):
        """Analyze performance trends over time"""
        try:
            query = """
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as backtests,
                    AVG(total_return) as avg_return,
                    AVG(cagr) as avg_cagr,
                    AVG(sharpe_ratio) as avg_sharpe,
                    AVG(win_rate) as avg_win_rate,
                    MAX(max_drawdown) as worst_drawdown
                FROM quantconnect_backtests
                WHERE created_at >= DATE_SUB(NOW(), INTERVAL %s DAY)
                GROUP BY DATE(created_at)
                ORDER BY date DESC
            """ % (days_back)

            result = self.db_manager.execute_query(query)

            if result.empty:
                print(f"❌ No backtest data from the last {days_back} days")
                return

            print(f"📊 PERFORMANCE TRENDS (Last {days_back} days):")
            print("=" * 80)

            for _, row in result.iterrows():
                status = "🟢" if row['avg_return'] > 0 else "🔴"
                print(f"{status} {row['date']}: {row['backtests']} backtests")
                print(f"   Avg Return: {row['avg_return']:.2%} | Avg CAGR: {row['avg_cagr']:.2%}")
                print(f"   Avg Win Rate: {row['avg_win_rate']:.2%} | Avg Sharpe: {row['avg_sharpe']:.2f}")
                print(f"   Worst Drawdown: {row['worst_drawdown']:.2%}")

            print(f"\n📈 Overall Performance Trend:")
            overall_avg = result['avg_return'].mean()
            overall_status = "🟢 POSITIVE" if overall_avg > 0 else "🔴 NEGATIVE"
            print(f"   {overall_status} Average Return: {overall_avg:.2%}")

        except Exception as e:
            print(f"❌ Error analyzing trends: {e}")

def main():
    parser = argparse.ArgumentParser(description='View stored backtest results')
    parser.add_argument('--list', action='store_true', help='List all backtest results')
    parser.add_argument('--details', help='View detailed backtest (requires --id)')
    parser.add_argument('--id', help='Backtest ID for detailed view')
    parser.add_argument('--compare', nargs='+', help='Compare multiple backtest IDs')
    parser.add_argument('--trends', type=int, default=30, help='Analyze performance trends (days back)')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of results')

    args = parser.parse_args()

    try:
        viewer = BacktestViewer()

        if args.list:
            viewer.view_all_backtests(args.limit)
        elif args.details and args.id:
            viewer.view_detailed_backtest(args.id)
        elif args.compare:
            viewer.compare_backtests(args.compare)
        elif args.trends:
            viewer.analyze_performance_trends(args.trends)
        else:
            print("📊 Enhanced XGBoost Backtest Viewer")
            print("=" * 50)
            print("Usage examples:")
            print("  python view_backtests.py --list --limit 5")
            print("  python view_backtests.py --details --id enhanced_xgboost_BTC_1h_20241201_120000")
            print("  python view_backtests.py --compare id1 id2 id3")
            print("  python view_backtests.py --trends 30")
            print()
            print("View current backtest count:")
            viewer.view_all_backtests(5)

    except KeyboardInterrupt:
        print("\n👋 Viewing interrupted")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()