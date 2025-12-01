#!/usr/bin/env python3
"""
Performance Evaluation Module for Enhanced XGBoost Trading System
Calculates real trading performance metrics including CAGR, Sharpe, Sortino, Win Rate, Max Drawdown
Integrates with QuantConnect for backtesting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class PerformanceEvaluator:
    """
    Comprehensive performance evaluator for trading strategies
    Calculates all key metrics from build.md targets
    """

    def __init__(self):
        self.metrics = {}

    def calculate_trading_metrics(self,
                                trades: List[Dict],
                                equity_curve: pd.DataFrame,
                                benchmark_returns: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive trading performance metrics

        Args:
            trades: List of trade dictionaries with entry/exit info
            equity_curve: DataFrame with timestamp and equity_value columns
            benchmark_returns: Optional benchmark returns for comparison

        Returns:
            Dictionary with all performance metrics
        """
        if equity_curve.empty:
            return self._empty_metrics()

        try:
            equity_curve = equity_curve.copy()
            equity_curve['timestamp'] = pd.to_datetime(equity_curve['timestamp'])
            equity_curve.set_index('timestamp', inplace=True)

            # Calculate returns
            equity_curve['returns'] = equity_curve['equity_value'].pct_change()
            equity_curve['cumulative_returns'] = (equity_curve['equity_value'] / equity_curve['equity_value'].iloc[0]) - 1

            # Core Metrics
            metrics = {
                # Return Metrics
                'total_return': self._calculate_total_return(equity_curve),
                'annualized_return': self._calculate_cagr(equity_curve),
                'monthly_return': self._calculate_monthly_return(equity_curve),

                # Risk Metrics
                'volatility': self._calculate_volatility(equity_curve),
                'max_drawdown': self._calculate_max_drawdown(equity_curve),
                'max_drawdown_duration': self._calculate_max_drawdown_duration(equity_curve),
                'var_95': self._calculate_var(equity_curve, 0.95),
                'cvar_95': self._calculate_cvar(equity_curve, 0.95),

                # Risk-Adjusted Metrics
                'sharpe_ratio': self._calculate_sharpe_ratio(equity_curve),
                'sortino_ratio': self._calculate_sortino_ratio(equity_curve),
                'calmar_ratio': self._calculate_calmar_ratio(equity_curve),
                'information_ratio': self._calculate_information_ratio(equity_curve, benchmark_returns),

                # Trade Metrics
                'win_rate': self._calculate_win_rate(trades),
                'profit_factor': self._calculate_profit_factor(trades),
                'avg_win': self._calculate_avg_win(trades),
                'avg_loss': self._calculate_avg_loss(trades),
                'largest_win': self._calculate_largest_win(trades),
                'largest_loss': self._calculate_largest_loss(trades),
                'avg_trade_duration': self._calculate_avg_trade_duration(trades),

                # Trade Frequency
                'trade_count': len(trades),
                'trades_per_month': self._calculate_trades_per_month(trades, equity_curve),

                # Psychological Metrics
                'recovery_factor': self._calculate_recovery_factor(equity_curve, trades),
                'kelly_criterion': self._calculate_kelly_criterion(trades),
                'expectancy': self._calculate_expectancy(trades),

                # Streak Metrics
                'max_consecutive_wins': self._calculate_max_consecutive_wins(trades),
                'max_consecutive_losses': self._calculate_max_consecutive_losses(trades),

                # Comparative Analysis
                'vs_buy_and_hold': self._compare_vs_buy_and_hold(equity_curve, benchmark_returns),

                # Build.md Target Achievement
                'cagr_target_achieved': False,
                'max_drawdown_target_achieved': False,
                'sharpe_target_achieved': False,
                'sortino_target_achieved': False,
                'win_rate_target_achieved': False,
                'overall_grade': 'F'  # Will be calculated below
            }

            # Check target achievements from build.md
            metrics.get('cagr_target_achieved', False) = metrics['annualized_return'] >= 0.50  # 50% target
            metrics['max_drawdown_target_achieved'] = metrics['max_drawdown'] <= 0.25  # 25% max
            metrics['sharpe_target_achieved'] = 1.2 <= metrics.get('sharpe_ratio', 0) <= 1.6
            metrics['sortino_target_achieved'] = 2.0 <= metrics.get('sortino_ratio', 0) <= 3.0
            metrics['win_rate_target_achieved'] = metrics['win_rate'] >= 0.70  # 70% target

            # Calculate overall grade
            metrics['overall_grade'] = self._calculate_overall_grade(metrics)

            # Add metadata
            metrics['evaluation_date'] = datetime.now().isoformat()
            metrics['total_trading_days'] = len(equity_curve)
            metrics['period_start'] = equity_curve.index[0].strftime('%Y-%m-%d')
            metrics['period_end'] = equity_curve.index[-1].strftime('%Y-%m-%d')

            self.metrics = metrics
            return metrics

        except Exception as e:
            print(f"❌ Error calculating metrics: {e}")
            return self._empty_metrics()

    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        now = datetime.now()
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'trade_count': 0, 'overall_grade': 'F',
            'period_start': now.strftime('%Y-%m-%d'),
            'period_end': now.strftime('%Y-%m-%d'),
            'total_trading_days': 0,
            'evaluation_date': now.isoformat(),
            'monthly_return': 0.0, 'volatility': 0.0, 'max_drawdown_duration': 0,
            'var_95': 0.0, 'cvar_95': 0.0, 'calmar_ratio': 0.0, 'information_ratio': 0.0,
            'cagr_target_achieved': False, 'max_drawdown_target_achieved': True,
            'sharpe_target_achieved': False, 'sortino_target_achieved': False,
            'win_rate_target_achieved': False
        }

    def _calculate_total_return(self, equity_curve: pd.DataFrame) -> float:
        """Calculate total return percentage"""
        if len(equity_curve) < 2:
            return 0.0
        return (equity_curve['equity_value'].iloc[-1] / equity_curve['equity_value'].iloc[0]) - 1

    def _calculate_cagr(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Compound Annual Growth Rate"""
        if len(equity_curve) < 2:
            return 0.0

        start_value = equity_curve['equity_value'].iloc[0]
        end_value = equity_curve['equity_value'].iloc[-1]

        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        if days <= 0:
            return 0.0

        years = days / 365.25
        return (end_value / start_value) ** (1 / years) - 1

    def _calculate_monthly_return(self, equity_curve: pd.DataFrame) -> float:
        """Calculate average monthly return"""
        if len(equity_curve) < 30:
            return 0.0

        monthly_returns = equity_curve['equity_value'].resample('M').last().pct_change().dropna()
        return monthly_returns.mean() if not monthly_returns.empty else 0.0

    def _calculate_volatility(self, equity_curve: pd.DataFrame) -> float:
        """Calculate annualized volatility"""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0

        return returns.std() * np.sqrt(252)  # Annualized

    def _calculate_max_drawdown(self, equity_curve: pd.DataFrame) -> float:
        """Calculate maximum drawdown"""
        if len(equity_curve) < 2:
            return 0.0

        equity = equity_curve['equity_value']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()

    def _calculate_max_drawdown_duration(self, equity_curve: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in days"""
        if len(equity_curve) < 2:
            return 0

        equity = equity_curve['equity_value']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max

        # Find drawdown periods
        in_drawdown = drawdown < 0
        drawdown_periods = []

        start_date = None
        for date, is_dd in in_drawdown.items():
            if is_dd and start_date is None:
                start_date = date
            elif not is_dd and start_date is not None:
                drawdown_periods.append((date - start_date).days)
                start_date = None

        return max(drawdown_periods) if drawdown_periods else 0

    def _calculate_var(self, equity_curve: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Value at Risk"""
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        return np.percentile(returns, (1 - confidence) * 100)

    def _calculate_cvar(self, equity_curve: pd.DataFrame, confidence: float = 0.95) -> float:
        """Calculate Conditional Value at Risk"""
        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0

        var_threshold = self._calculate_var(equity_curve, confidence)
        return returns[returns <= var_threshold].mean()

    def _calculate_sharpe_ratio(self, equity_curve: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio"""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['returns'].dropna()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_return = returns.mean() * 252 - risk_free_rate
        return excess_return / (returns.std() * np.sqrt(252))

    def _calculate_sortino_ratio(self, equity_curve: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio"""
        if len(equity_curve) < 2:
            return 0.0

        returns = equity_curve['returns'].dropna()
        if len(returns) == 0:
            return 0.0

        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')

        downside_deviation = downside_returns.std() * np.sqrt(252)
        excess_return = returns.mean() * 252 - risk_free_rate

        return excess_return / downside_deviation if downside_deviation > 0 else 0.0

    def _calculate_calmar_ratio(self, equity_curve: pd.DataFrame) -> float:
        """Calculate Calmar Ratio"""
        cagr = self._calculate_cagr(equity_curve)
        max_dd = abs(self._calculate_max_drawdown(equity_curve))
        return cagr / max_dd if max_dd > 0 else 0.0

    def _calculate_information_ratio(self, equity_curve: pd.DataFrame, benchmark_returns: Optional[pd.Series]) -> float:
        """Calculate Information Ratio"""
        if benchmark_returns is None or len(equity_curve) < 2:
            return 0.0

        strategy_returns = equity_curve['returns'].dropna()
        if len(strategy_returns) == 0:
            return 0.0

        # Align with benchmark
        aligned_returns = pd.concat([strategy_returns, benchmark_returns], axis=1).dropna()
        if len(aligned_returns) < 2:
            return 0.0

        excess_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
        tracking_error = excess_returns.std()

        return excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0

    def _calculate_win_rate(self, trades: List[Dict]) -> float:
        """Calculate win rate percentage"""
        if not trades:
            return 0.0

        winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
        return len(winning_trades) / len(trades)

    def _calculate_profit_factor(self, trades: List[Dict]) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        if not trades:
            return 0.0

        gross_profit = sum(t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0)
        gross_loss = abs(sum(t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) < 0))

        return gross_profit / gross_loss if gross_loss > 0 else float('inf')

    def _calculate_avg_win(self, trades: List[Dict]) -> float:
        """Calculate average winning trade"""
        winning_trades = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) > 0]
        return np.mean(winning_trades) if winning_trades else 0.0

    def _calculate_avg_loss(self, trades: List[Dict]) -> float:
        """Calculate average losing trade"""
        losing_trades = [t.get('profit_loss', 0) for t in trades if t.get('profit_loss', 0) < 0]
        return np.mean(losing_trades) if losing_trades else 0.0

    def _calculate_largest_win(self, trades: List[Dict]) -> float:
        """Calculate largest winning trade"""
        if not trades:
            return 0.0
        return max(t.get('profit_loss', 0) for t in trades)

    def _calculate_largest_loss(self, trades: List[Dict]) -> float:
        """Calculate largest losing trade"""
        if not trades:
            return 0.0
        return min(t.get('profit_loss', 0) for t in trades)

    def _calculate_avg_trade_duration(self, trades: List[Dict]) -> float:
        """Calculate average trade duration in hours"""
        if not trades:
            return 0.0

        durations = []
        for trade in trades:
            entry_time = trade.get('entry_time')
            exit_time = trade.get('exit_time')
            if entry_time and exit_time:
                duration = (exit_time - entry_time).total_seconds() / 3600  # hours
                durations.append(duration)

        return np.mean(durations) if durations else 0.0

    def _calculate_trades_per_month(self, trades: List[Dict], equity_curve: pd.DataFrame) -> float:
        """Calculate average trades per month"""
        if not trades or len(equity_curve) < 2:
            return 0.0

        trading_months = (equity_curve.index[-1] - equity_curve.index[0]).days / 30.44
        return len(trades) / trading_months if trading_months > 0 else 0.0

    def _calculate_recovery_factor(self, equity_curve: pd.DataFrame, trades: List[Dict]) -> float:
        """Calculate recovery factor (net profit / max drawdown)"""
        if not trades:
            return 0.0

        net_profit = sum(t.get('profit_loss', 0) for t in trades)
        max_dd = abs(self._calculate_max_drawdown(equity_curve))

        return net_profit / max_dd if max_dd > 0 else 0.0

    def _calculate_kelly_criterion(self, trades: List[Dict]) -> float:
        """Calculate Kelly Criterion optimal position size"""
        if not trades:
            return 0.0

        win_rate = self._calculate_win_rate(trades)
        avg_win = self._calculate_avg_win(trades)
        avg_loss = abs(self._calculate_avg_loss(trades))

        if avg_loss == 0:
            return 0.0

        # Kelly = W - (1-W)/R where W=win_rate, R=win/loss ratio
        win_loss_ratio = avg_win / avg_loss
        return win_rate - ((1 - win_rate) / win_loss_ratio)

    def _calculate_expectancy(self, trades: List[Dict]) -> float:
        """Calculate expectancy per trade"""
        if not trades:
            return 0.0

        win_rate = self._calculate_win_rate(trades)
        avg_win = self._calculate_avg_win(trades)
        avg_loss = self._calculate_avg_loss(trades)

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def _calculate_max_consecutive_wins(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive wins"""
        if not trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if trade.get('profit_loss', 0) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _calculate_max_consecutive_losses(self, trades: List[Dict]) -> int:
        """Calculate maximum consecutive losses"""
        if not trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if trade.get('profit_loss', 0) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def _compare_vs_buy_and_hold(self, equity_curve: pd.DataFrame, benchmark_returns: Optional[pd.Series]) -> float:
        """Compare strategy performance vs buy and hold"""
        if benchmark_returns is None or len(equity_curve) < 2:
            return 0.0

        strategy_return = self._calculate_total_return(equity_curve)
        benchmark_return = (1 + benchmark_returns).prod() - 1

        return strategy_return - benchmark_return

    def _calculate_overall_grade(self, metrics: Dict[str, Any]) -> str:
        """Calculate overall performance grade"""
        score = 0

        # CAGR (30% weight)
        if metrics['annualized_return'] >= 0.50:
            score += 30
        elif metrics['annualized_return'] >= 0.30:
            score += 20
        elif metrics['annualized_return'] >= 0.10:
            score += 10

        # Sharpe Ratio (25% weight)
        if 1.2 <= metrics.get('sharpe_ratio', 0) <= 1.6:
            score += 25
        elif metrics.get('sharpe_ratio', 0) >= 1.0:
            score += 20
        elif metrics.get('sharpe_ratio', 0) >= 0.5:
            score += 10

        # Max Drawdown (20% weight)
        if metrics['max_drawdown'] <= 0.15:
            score += 20
        elif metrics['max_drawdown'] <= 0.25:
            score += 15
        elif metrics['max_drawdown'] <= 0.35:
            score += 10

        # Win Rate (15% weight)
        if metrics['win_rate'] >= 0.70:
            score += 15
        elif metrics['win_rate'] >= 0.60:
            score += 10
        elif metrics['win_rate'] >= 0.50:
            score += 5

        # Sortino Ratio (10% weight)
        if 2.0 <= metrics.get('sortino_ratio', 0) <= 3.0:
            score += 10
        elif metrics.get('sortino_ratio', 0) >= 1.5:
            score += 7
        elif metrics.get('sortino_ratio', 0) >= 1.0:
            score += 3

        # Grade mapping
        if score >= 90:
            return 'A+'
        elif score >= 85:
            return 'A'
        elif score >= 80:
            return 'A-'
        elif score >= 75:
            return 'B+'
        elif score >= 70:
            return 'B'
        elif score >= 65:
            return 'B-'
        elif score >= 60:
            return 'C+'
        elif score >= 55:
            return 'C'
        elif score >= 50:
            return 'C-'
        elif score >= 45:
            return 'D+'
        elif score >= 40:
            return 'D'
        elif score >= 35:
            return 'D-'
        else:
            return 'F'

    def print_detailed_report(self, metrics: Dict[str, Any]) -> None:
        """Print detailed performance report"""
        print("\n" + "="*80)
        print("📊 ENHANCED XGBOOST TRADING SYSTEM - PERFORMANCE REPORT")
        print("="*80)

        # Overall Performance
        print(f"\n🎯 OVERALL PERFORMANCE: {metrics['overall_grade']}")
        print(f"   Period: {metrics['period_start']} to {metrics['period_end']}")
        print(f"   Trading Days: {metrics['total_trading_days']:,}")

        # Return Metrics
        print(f"\n💰 RETURN METRICS:")
        print(f"   Total Return: {metrics['total_return']:.2%}")
        print(f"   Annualized Return (CAGR): {metrics['annualized_return']:.2%}")
        print(f"   Monthly Average Return: {metrics.get('monthly_return', 0):.2%}")

        # Risk Metrics
        print(f"\n⚠️  RISK METRICS:")
        print(f"   Volatility (Annual): {metrics.get('volatility', 0):.2%}")
        print(f"   Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   Max DD Duration: {metrics.get('max_drawdown_duration', 0)} days")
        print(f"   VaR (95%): {metrics.get('var_95', 0):.2%}")
        print(f"   CVaR (95%): {metrics.get('cvar_95', 0):.2%}")

        # Risk-Adjusted Metrics
        print(f"\n📈 RISK-ADJUSTED METRICS:")
        print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        print(f"   Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        print(f"   Information Ratio: {metrics.get('information_ratio', 0):.2f}")

        # Build.md Target Achievement
        print(f"\n🎯 BUILD.MD TARGETS:")
        print(f"   CAGR ≥ 50%: {'✅ ACHIEVED' if metrics.get('cagr_target_achieved', False) else '❌ NOT ACHIEVED'} ({metrics['annualized_return']:.2%})")
        print(f"   Max Drawdown ≤ 25%: {'✅ ACHIEVED' if metrics['max_drawdown_target_achieved'] else '❌ NOT ACHIEVED'} ({metrics['max_drawdown']:.2%})")
        print(f"   Sharpe 1.2-1.6: {'✅ ACHIEVED' if metrics['sharpe_target_achieved'] else '❌ NOT ACHIEVED'} ({metrics.get('sharpe_ratio', 0):.2f})")
        print(f"   Sortino 2.0-3.0: {'✅ ACHIEVED' if metrics['sortino_target_achieved'] else '❌ NOT ACHIEVED'} ({metrics.get('sortino_ratio', 0):.2f})")
        print(f"   Win Rate ≥ 70%: {'✅ ACHIEVED' if metrics['win_rate_target_achieved'] else '❌ NOT ACHIEVED'} ({metrics['win_rate']:.2%})")

        # Trade Metrics
        print(f"\n📊 TRADE METRICS:")
        print(f"   Total Trades: {metrics['trade_count']}")
        print(f"   Win Rate: {metrics['win_rate']:.2%}")
        print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"   Average Win: ${metrics['avg_win']:,.2f}")
        print(f"   Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"   Largest Win: ${metrics['largest_win']:,.2f}")
        print(f"   Largest Loss: ${metrics['largest_loss']:,.2f}")
        print(f"   Avg Trade Duration: {metrics['avg_trade_duration']:.1f} hours")
        print(f"   Trades per Month: {metrics['trades_per_month']:.1f}")

        # Streak Metrics
        print(f"\n🎲 STREAK METRICS:")
        print(f"   Max Consecutive Wins: {metrics['max_consecutive_wins']}")
        print(f"   Max Consecutive Losses: {metrics['max_consecutive_losses']}")

        # Psychological Metrics
        print(f"\n🧠 PSYCHOLOGICAL METRICS:")
        print(f"   Recovery Factor: {metrics['recovery_factor']:.2f}")
        print(f"   Kelly Criterion: {metrics['kelly_criterion']:.2%}")
        print(f"   Expectancy per Trade: ${metrics['expectancy']:.2f}")

        # Comparative Analysis
        if metrics['vs_buy_and_hold'] != 0:
            print(f"\n🏁 VS BUY & HOLD:")
            print(f"   Alpha: {metrics['vs_buy_and_hold']:.2%}")

        print("\n" + "="*80)

    def save_report(self, metrics: Dict[str, Any], filename: str = None) -> str:
        """Save performance report to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_report_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)

        print(f"💾 Performance report saved to: {filename}")
        return filename