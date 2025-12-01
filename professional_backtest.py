#!/usr/bin/env python3
"""
Professional Backtest Script with Build.md Target Display
Enhanced with professional reporting and target achievement tracking
"""

import argparse
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json
from typing import Dict, Any, List

from env_config import get_database_config
from database import DatabaseManager
from backtest import BacktestEngine

class ProfessionalBacktester:
    """Professional backtester with build.md target achievement tracking"""

    def __init__(self):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.backtest_engine = BacktestEngine()

    def run_professional_backtest(self, model_path: str, symbol: str, pair: str,
                                  interval: str, start_date: str, end_date: str,
                                  initial_capital: float = 100000,
                                  commission: float = 0.001,
                                  slippage: float = 0.0005,
                                  max_position_size: float = 0.95,
                                  stop_loss_pct: float = 0.02,
                                  take_profit_pct: float = 0.05):
        """Run professional backtest with enhanced reporting"""

        print("🚀 PROFESSIONAL BACKTEST EXECUTION")
        print("=" * 80)
        print(f"📊 Model: {os.path.basename(model_path)}")
        print(f"💰 Symbol: {symbol}")
        print(f"📈 Pair: {pair}")
        print(f"⏱️  Interval: {interval}")
        print(f"📅 Period: {start_date} to {end_date}")
        print(f"💵 Initial Capital: ${initial_capital:,.2f}")
        print("=" * 80)

        try:
            # Configure backtest engine
            self.backtest_engine.initial_capital = initial_capital
            self.backtest_engine.commission = commission
            self.backtest_engine.slippage = slippage
            self.backtest_engine.max_position_size = max_position_size
            self.backtest_engine.stop_loss_pct = stop_loss_pct
            self.backtest_engine.take_profit_pct = take_profit_pct

            # Run the backtest
            metrics = self.backtest_engine.run_backtest(
                model_path=model_path,
                symbol=symbol,
                pair=pair,
                interval=interval,
                start_date=start_date,
                end_date=end_date
            )

            # Display professional results
            self.display_professional_results(metrics, model_path, symbol, pair,
                                             interval, start_date, end_date,
                                             initial_capital, commission, slippage)

        except Exception as e:
            print(f"\n❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()

    def display_professional_results(self, metrics: Dict[str, Any], model_path: str,
                                    symbol: str, pair: str, interval: str,
                                    start_date: str, end_date: str,
                                    initial_capital: float, commission: float, slippage: float):
        """Display professional backtest results with build.md targets"""

        print(f"\n📊 PROFESSIONAL BACKTEST RESULTS")
        print("=" * 80)
        print(f"Strategy: Enhanced XGBoost with 11 Data Sources")
        print(f"Timeframe: {start_date} to {end_date}")
        print(f"Configuration: {commission*100:.2f}% commission, {slippage*100:.2f}% slippage")
        print("=" * 80)

        # Build.md Targets Table
        self.display_build_targets_table(metrics)

        # Key Performance Metrics
        self.display_performance_metrics(metrics, initial_capital)

        # Risk Analysis
        self.display_risk_analysis(metrics)

        # Trading Statistics
        self.display_trading_statistics(metrics)

        # Target Achievement Summary
        self.display_target_achievement(metrics)

        # Model Information
        self.display_model_info(model_path, symbol, pair, interval)

        # Recommendations
        self.display_recommendations(metrics)

    def display_build_targets_table(self, metrics: Dict[str, Any]):
        """Display build.md targets achievement table"""
        print(f"\n🎯 BUILD.MD TARGETS ACHIEVEMENT")
        print("=" * 80)
        print(f"| {'Metric':<20} | {'Target':<12} | {'Actual':<12} | {'Achieved':<10} | {'Status':<10} |")
        print(f"|{'-'*20} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*10} |")

        targets = [
            {
                'metric': 'CAGR',
                'target': '≥ 50%',
                'actual': f"{metrics.get('annualized_return', 0):.1%}",
                'achieved': metrics.get('cagr_target_achieved', False),
                'grade': self.get_cagr_grade(metrics.get('annualized_return', 0))
            },
            {
                'metric': 'Max Drawdown',
                'target': '≤ 25%',
                'actual': f"{metrics.get('max_drawdown', 0):.1%}",
                'achieved': metrics.get('max_drawdown_target_achieved', False),
                'grade': self.get_drawdown_grade(metrics.get('max_drawdown', 0))
            },
            {
                'metric': 'Sharpe Ratio',
                'target': '1.2 - 1.6',
                'actual': f"{metrics.get('sharpe_ratio', 0):.2f}",
                'achieved': 1.2 <= metrics.get('sharpe_ratio', 0) <= 1.6,
                'grade': self.get_sharpe_grade(metrics.get('sharpe_ratio', 0))
            },
            {
                'metric': 'Sortino Ratio',
                'target': '2.0 - 3.0',
                'actual': f"{metrics.get('sortino_ratio', 0):.2f}",
                'achieved': 2.0 <= metrics.get('sortino_ratio', 0) <= 3.0,
                'grade': self.get_sortino_grade(metrics.get('sortino_ratio', 0))
            },
            {
                'metric': 'Win Rate',
                'target': '≥ 70%',
                'actual': f"{metrics.get('win_rate', 0):.1%}",
                'achieved': metrics.get('win_rate', 0) >= 0.70,
                'grade': self.get_win_rate_grade(metrics.get('win_rate', 0))
            }
        ]

        for target in targets:
            status_icon = "✅" if target['achieved'] else "❌"
            grade_icon = target['grade'] if target['grade'] else "F"
            print(f"| {target['metric']:<20} | {target['target']:<12} | {target['actual']:<12} | {grade_icon:<10} | {status_icon:<10} |")

    def get_cagr_grade(self, cagr: float) -> str:
        """Get grade for CAGR"""
        if cagr >= 0.50:
            return "A+"
        elif cagr >= 0.40:
            return "A"
        elif cagr >= 0.30:
            return "B+"
        elif cagr >= 0.20:
            return "B"
        elif cagr >= 0.10:
            return "C+"
        elif cagr >= 0:
            return "C"
        else:
            return "F"

    def get_drawdown_grade(self, max_dd: float) -> str:
        """Get grade for max drawdown"""
        if max_dd <= 0.10:
            return "A+"
        elif max_dd <= 0.15:
            return "A"
        elif max_dd <= 0.20:
            return "B+"
        elif max_dd <= 0.25:
            return "B"
        elif max_dd <= 0.35:
            return "C+"
        elif max_dd <= 0.50:
            return "C"
        else:
            return "F"

    def get_sharpe_grade(self, sharpe: float) -> str:
        """Get grade for Sharpe ratio"""
        if 1.2 <= sharpe <= 1.6:
            return "A+"
        elif sharpe >= 1.0:
            return "A"
        elif sharpe >= 0.8:
            return "B+"
        elif sharpe >= 0.6:
            return "B"
        elif sharpe >= 0.4:
            return "C+"
        elif sharpe >= 0.2:
            return "C"
        elif sharpe >= 0:
            return "D"
        else:
            return "F"

    def get_sortino_grade(self, sortino: float) -> str:
        """Get grade for Sortino ratio"""
        if 2.0 <= sortino <= 3.0:
            return "A+"
        elif sortino >= 1.5:
            return "A"
        elif sortino >= 1.2:
            return "B+"
        elif sortino >= 1.0:
            return "B"
        elif sortino >= 0.8:
            return "C+"
        elif sortino >= 0.6:
            return "C"
        elif sortino >= 0.4:
            return "D"
        else:
            return "F"

    def get_win_rate_grade(self, win_rate: float) -> str:
        """Get grade for win rate"""
        if win_rate >= 0.70:
            return "A+"
        elif win_rate >= 0.65:
            return "A"
        elif win_rate >= 0.60:
            return "B+"
        elif win_rate >= 0.55:
            return "B"
        elif win_rate >= 0.50:
            return "C+"
        elif win_rate >= 0.45:
            return "C"
        elif win_rate >= 0.40:
            return "D"
        else:
            return "F"

    def display_performance_metrics(self, metrics: Dict[str, Any], initial_capital: float):
        """Display key performance metrics"""
        print(f"\n💰 PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Capital: ${metrics.get('ending_capital', initial_capital):,.2f}")
        print(f"Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"Annualized Return (CAGR): {metrics.get('annualized_return', 0):.2%}")
        print(f"Volatility: {metrics.get('volatility', 0):.2%}")
        print(f"Information Ratio: {metrics.get('information_ratio', 0):.2f}")

    def display_risk_analysis(self, metrics: Dict[str, Any]):
        """Display risk analysis"""
        print(f"\n⚠️  RISK ANALYSIS")
        print("=" * 50)
        print(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"Max Drawdown Duration: {metrics.get('max_drawdown_duration', 0)} days")
        print(f"Value at Risk (95%): {metrics.get('var_95', 0):.2%}")
        print(f"Conditional VaR (95%): {metrics.get('cvar_95', 0):.2%}")
        print(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"Recovery Factor: {metrics.get('recovery_factor', 0):.2f}")

    def display_trading_statistics(self, metrics: Dict[str, Any]):
        """Display trading statistics"""
        print(f"\n📊 TRADING STATISTICS")
        print("=" * 50)
        print(f"Total Trades: {metrics.get('total_trades', 0)}")
        print(f"Winning Trades: {metrics.get('winning_trades', 0)}")
        print(f"Losing Trades: {metrics.get('losing_trades', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
        print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        print(f"Average Win: ${metrics.get('avg_win', 0):,.2f}")
        print(f"Average Loss: ${metrics.get('avg_loss', 0):,.2f}")
        print(f"Largest Win: ${metrics.get('largest_win', 0):,.2f}")
        print(f"Largest Loss: ${metrics.get('largest_loss', 0):,.2f}")
        print(f"Longest Win Streak: {metrics.get('longest_win_streak', 0)} trades")
        print(f"Longest Loss Streak: {metrics.get('longest_loss_streak', 0)} trades")

    def display_target_achievement(self, metrics: Dict[str, Any]):
        """Display target achievement summary"""
        print(f"\n🏆 TARGET ACHIEVEMENT SUMMARY")
        print("=" * 50)

        # Calculate overall score
        score = 0
        targets = [
            ('CAGR ≥ 50%', metrics.get('cagr_target_achieved', False)),
            ('Max Drawdown ≤ 25%', metrics.get('max_drawdown_target_achieved', False)),
            ('Sharpe 1.2-1.6', 1.2 <= metrics.get('sharpe_ratio', 0) <= 1.6),
            ('Sortino 2.0-3.0', 2.0 <= metrics.get('sortino_ratio', 0) <= 3.0),
            ('Win Rate ≥ 70%', metrics.get('win_rate', 0) >= 0.70)
        ]

        for target_name, achieved in targets:
            if achieved:
                score += 1
                print(f"✅ {target_name}: ACHIEVED")
            else:
                print(f"❌ {target_name}: NOT ACHIEVED")

        print(f"\n📊 Overall Score: {score}/5 targets achieved")

        # Calculate overall grade
        if score == 5:
            grade = "A+ (EXCELLENT)"
        elif score == 4:
            grade = "A (GOOD)"
        elif score == 3:
            grade = "B+ (SATISFACTORY)"
        elif score == 2:
            grade = "B (ACCEPTABLE)"
        elif score == 1:
            grade = "C+ (MINIMUM)"
        else:
            grade = "F (FAILED)"

        print(f"🏆 Overall Grade: {grade}")

        # Investment Recommendation
        if score >= 4:
            print(f"💡 Recommendation: READY FOR LIVE TRADING")
        elif score >= 3:
            print(f"💡 Recommendation: CONSIDER AFTER OPTIMIZATION")
        elif score >= 2:
            print(f"💡 Recommendation: REQUIRES SIGNIFICANT IMPROVEMENT")
        elif score >= 1:
            print(f"💡 Recommendation: MAJOR REFORMULATION NEEDED")
        else:
            print(f"💡 Recommendation: STRATEGY FAILED - RESTART REQUIRED")

    def display_model_info(self, model_path: str, symbol: str, pair: str, interval: str):
        """Display model information"""
        print(f"\n🤖 MODEL INFORMATION")
        print("=" * 50)
        print(f"Model File: {os.path.basename(model_path)}")
        print(f"Symbol: {symbol}")
        print(f"Trading Pair: {pair}")
        print(f"Time Interval: {interval}")
        print(f"Data Sources: 11 total (7 original + 4 microstructure)")
        print(f"Features: 303 enhanced features")

        # Get model file size
        if os.path.exists(model_path):
            model_size = os.path.getsize(model_path)
            print(f"Model Size: {model_size / 1024:.1f} KB")

    def display_recommendations(self, metrics: Dict[str, Any]):
        """Display actionable recommendations"""
        print(f"\n💡 ACTIONABLE RECOMMENDATIONS")
        print("=" * 50)

        recommendations = []

        # Analyze performance and provide recommendations
        cagr = metrics.get('annualized_return', 0)
        max_dd = metrics.get('max_drawdown', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        win_rate = metrics.get('win_rate', 0)

        if cagr < 0.50:
            recommendations.append("🔴 LOW RETURNS - Consider strategy retraining")
        elif cagr < 0.30:
            recommendations.append("🟡 MODERATE RETURNS - Fine-tune parameters")

        if max_dd > 0.30:
            recommendations.append("🔴 HIGH DRAWDOWN - Implement stronger risk management")
        elif max_dd > 0.20:
            recommendations.append("🟡 MODERATE DRAWDOWN - Adjust position sizing")

        if sharpe < 0.5:
            recommendations.append("🔴 POOR RISK-ADJUSTED RETURNS - Reduce volatility")
        elif sharpe < 1.0:
            recommendations.append("🟡 MODERATE SHARPE - Optimize risk/return balance")

        if win_rate < 0.50:
            recommendations.append("🔴 LOW WIN RATE - Improve signal accuracy")
        elif win_rate < 0.60:
            recommendations.append("🟡 MODERATE WIN RATE - Refine entry conditions")

        if metrics.get('total_trades', 0) < 50:
            recommendations.append("🟡 LOW TRADE COUNT - Increase trading opportunities")
        elif metrics.get('total_trades', 0) < 100:
            recommendations.append("✅ ADEQUATE TRADE COUNT - Acceptable sample size")

        # Add positive feedback
        if cagr >= 0.30 and max_dd <= 0.25:
            recommendations.append("✅ SOLID RETURNS WITH MANAGEABLE RISK")

        if sharpe >= 1.0 and win_rate >= 0.55:
            recommendations.append("✅ GOOD RISK-ADJUSTED PERFORMANCE")

        if not recommendations:
            recommendations.append("✅ PERFORMANCE IS OPTIMAL - MAINTAIN CURRENT STRATEGY")

        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")

def main():
    """Main function to run professional backtest"""
    parser = argparse.ArgumentParser(description='Professional Backtest with Build.md Target Display')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTC)')
    parser.add_argument('--pair', required=True, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--interval', default='1h', help='Time interval (default: 1h)')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate (default: 0.001)')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate (default: 0.0005)')
    parser.add_argument('--position-size', type=float, default=0.95, help='Max position size (default: 0.95)')
    parser.add_argument('--stop-loss', type=float, default=0.02, help='Stop loss percentage (default: 0.02)')
    parser.add_argument('--take-profit', type=float, default=0.05, help='Take profit percentage (default: 0.05)')

    args = parser.parse_args()

    try:
        # Create and run professional backtester
        backtester = ProfessionalBacktester()
        backtester.run_professional_backtest(
            model_path=args.model,
            symbol=args.symbol,
            pair=args.pair,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            commission=args.commission,
            slippage=args.slippage,
            max_position_size=args.position_size,
            stop_loss_pct=args.stop_loss,
            take_profit_pct=args.take_profit
        )

    except KeyboardInterrupt:
        print("\n👋 Backtest interrupted")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()