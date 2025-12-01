#!/usr/bin/env python3
"""
Comprehensive Backtesting System for Enhanced XGBoost Trading
Integrates with database, model predictions, and QuantConnect
Provides real performance metrics vs build.md targets
"""

import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

from env_config import get_database_config
from database import DatabaseManager
from evaluation import PerformanceEvaluator
from feature_engineering import FeatureEngineer
from train_model import ModelTrainer

class BacktestEngine:
    """
    Advanced backtesting engine with realistic trade simulation
    Includes slippage, fees, position sizing, and risk management
    """

    def __init__(self,
                 initial_capital: float = 100000.0,
                 commission: float = 0.001,  # 0.1% commission
                 slippage: float = 0.0005,   # 0.05% slippage
                 max_position_size: float = 1.0,  # 100% of capital
                 stop_loss_pct: float = 0.02,    # 2% stop loss
                 take_profit_pct: float = 0.05):  # 5% take profit
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct

        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.evaluator = PerformanceEvaluator()
        self.feature_engineer = FeatureEngineer()

        # Backtracking state
        self.reset()

    def reset(self):
        """Reset backtesting state"""
        self.current_capital = self.initial_capital
        self.current_position = 0.0
        self.current_price = 0.0
        self.trades = []
        self.equity_curve = []
        self.open_orders = []
        self.stop_loss_price = None
        self.take_profit_price = None

    def run_backtest(self,
                    model_path: str,
                    symbol: str = 'BTC',
                    pair: str = 'BTCUSDT',
                    interval: str = '1h',
                    start_date: str = None,
                    end_date: str = None) -> Dict[str, Any]:
        """
        Run comprehensive backtest with enhanced features

        Args:
            model_path: Path to trained XGBoost model
            symbol: Trading symbol
            pair: Trading pair
            interval: Time interval
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
        """
        print(f"🚀 Starting Enhanced Backtest for {symbol} {pair} {interval}")
        print(f"   Initial Capital: ${self.initial_capital:,.2f}")
        print(f"   Commission: {self.commission:.2%}")
        print(f"   Slippage: {self.slippage:.2%}")
        print(f"   Stop Loss: {self.stop_loss_pct:.1%}")
        print(f"   Take Profit: {self.take_profit_pct:.1%}")

        try:
            # Load model
            trainer = ModelTrainer()
            model = trainer.load_model(model_path)
            if model is None:
                raise Exception("Failed to load model")

            # Load historical data
            historical_data = self._load_historical_data(symbol, pair, interval, start_date, end_date)
            if historical_data.empty:
                raise Exception("No historical data available")

            print(f"📊 Loaded {len(historical_data)} historical data points")

            # Generate trading signals
            signals = self._generate_signals(model, historical_data, symbol, pair, interval)
            print(f"🎯 Generated {len(signals)} trading signals")

            # Execute backtest simulation
            self._execute_trades(signals, historical_data)

            # Calculate performance metrics
            equity_curve_df = pd.DataFrame(self.equity_curve)

            # Handle empty equity curve case
            if equity_curve_df.empty and not self.trades:
                # Create a minimal equity curve for empty backtest
                equity_curve_df = pd.DataFrame([
                    {
                        'timestamp': pd.to_datetime(start_date),
                        'equity_value': self.initial_capital,
                        'price': 0,  # No price data available
                        'position': 0,
                        'cash': self.initial_capital
                    },
                    {
                        'timestamp': pd.to_datetime(end_date),
                        'equity_value': self.initial_capital,
                        'price': 0,  # No price data available
                        'position': 0,
                        'cash': self.initial_capital
                    }
                ])
                equity_curve_df.set_index('timestamp', inplace=True)

            metrics = self.evaluator.calculate_trading_metrics(self.trades, equity_curve_df)

            # Add backtest-specific metrics
            metrics.update({
                'backtest_config': {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.current_capital,
                    'commission': self.commission,
                    'slippage': self.slippage,
                    'max_position_size': self.max_position_size,
                    'stop_loss_pct': self.stop_loss_pct,
                    'take_profit_pct': self.take_profit_pct
                },
                'model_info': {
                    'model_path': model_path,
                    'symbol': symbol,
                    'pair': pair,
                    'interval': interval
                }
            })

            # Store results in database
            self._store_backtest_results(metrics, model_path, symbol, pair, interval, start_date, end_date)

            # Print and save results
            self.evaluator.print_detailed_report(metrics)
            report_file = self.evaluator.save_report(metrics, f"backtest_report_{symbol}_{interval}.json")

            return metrics

        except Exception as e:
            print(f"❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _load_historical_data(self, symbol: str, pair: str, interval: str,
                            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Load historical price data for backtesting"""
        try:
            # Default to last 6 months if no dates specified
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if not start_date:
                start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')

            # Convert to timestamps
            start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
            end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

            # Load price data
            query = """
                SELECT time as timestamp, open, high, low, close, volume_usd as volume
                FROM cg_spot_price_history
                WHERE symbol = %s AND `interval` = %s
                  AND time BETWEEN %s AND %s
                ORDER BY time
            """

            df = self.db_manager.execute_query(query, (pair, interval, start_timestamp, end_timestamp))

            if df.empty:
                print(f"⚠️  No price data found for {pair} {interval} in specified range")
                return pd.DataFrame()

            # Convert timestamp and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Load enhanced data sources (same as collect_signals.py)
            df = self._load_enhanced_data_sources(df, symbol, pair, interval, start_timestamp, end_timestamp)

            print(f"📊 Loaded historical data: {len(df)} rows, {len(df.columns)} features")
            return df

        except Exception as e:
            print(f"❌ Error loading historical data: {e}")
            return pd.DataFrame()

    def _load_enhanced_data_sources(self, df: pd.DataFrame, symbol: str, pair: str, interval: str,
                                  start_timestamp: int, end_timestamp: int) -> pd.DataFrame:
        """Load the 4 new microstructure data sources"""
        try:
            # 1. Spot Orderbook History
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
                orderbook_df['total_depth'] = orderbook_df['aggregated_bids_usd'] + orderbook_df['aggregated_asks_usd']
                orderbook_df['bid_ask_imbalance'] = (orderbook_df['aggregated_bids_usd'] - orderbook_df['aggregated_asks_usd']) / (orderbook_df['total_depth'] + 1e-8)
                df = df.join(orderbook_df, how='left').fillna(method='ffill')

            # 2. Futures Basis History
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
                basis_df['basis_momentum'] = basis_df['close_basis'].diff()
                df = df.join(basis_df, how='left').fillna(method='ffill')

            # Forward fill missing microstructure data
            microstructure_cols = ['total_depth', 'bid_ask_imbalance', 'close_basis', 'basis_momentum']
            for col in microstructure_cols:
                if col in df.columns:
                    df[col] = df[col].fillna(0)

            return df

        except Exception as e:
            print(f"⚠️  Error loading enhanced data sources: {e}")
            return df

    def _generate_signals(self, model, data: pd.DataFrame, symbol: str, pair: str, interval: str) -> pd.DataFrame:
        """Generate trading signals using the enhanced model"""
        print(f"🎯 Generating trading signals with enhanced features...")

        signals = []

        for i in range(100, len(data)):  # Start after initial window
            try:
                # Get historical window for feature engineering
                window_data = data.iloc[i-100:i+1].copy()

                # Create enhanced features
                features_df = self.feature_engineer.create_all_features(window_data)
                if features_df.empty:
                    continue

                # Get latest row for prediction
                latest_features = features_df.iloc[-1:]

                # Select features that match model training
                if hasattr(model, 'feature_names_in_'):
                    expected_features = list(model.feature_names_in_)
                else:
                    # Fallback to numeric features
                    expected_features = [col for col in latest_features.columns
                                       if latest_features[col].dtype in ['float64', 'int64', 'bool']]

                prediction_data = latest_features[expected_features].fillna(0)
                X = prediction_data.values

                # Make prediction
                prediction_prob = model.predict_proba(X)[0]
                prediction = model.predict(X)[0]

                # Create signal
                signal = {
                    'timestamp': data.index[i],
                    'price': data.iloc[i]['close'],
                    'prediction': prediction,
                    'confidence': max(prediction_prob),
                    'features_count': len(expected_features)
                }

                signals.append(signal)

            except Exception as e:
                continue

        signals_df = pd.DataFrame(signals)
        if not signals_df.empty:
            signals_df.set_index('timestamp', inplace=True)

        print(f"✅ Generated {len(signals_df)} signals with enhanced features")
        return signals_df

    def _execute_trades(self, signals: pd.DataFrame, historical_data: pd.DataFrame):
        """Execute trades with realistic simulation"""
        print("🔄 Executing backtest simulation...")

        for timestamp, signal in signals.iterrows():
            current_price = signal['price']
            prediction = signal['prediction']
            confidence = signal['confidence']

            # Update equity curve
            portfolio_value = self.current_capital + (self.current_position * current_price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity_value': portfolio_value,
                'price': current_price,
                'position': self.current_position,
                'cash': self.current_capital
            })

            # Check for stop loss or take profit
            if self.current_position > 0:
                if self.stop_loss_price and current_price <= self.stop_loss_price:
                    self._close_position(timestamp, current_price, reason='Stop Loss')
                    continue
                elif self.take_profit_price and current_price >= self.take_profit_price:
                    self._close_position(timestamp, current_price, reason='Take Profit')
                    continue

            # Generate new trading signals based on prediction
            if confidence < 0.6:  # Low confidence - no action
                continue

            if prediction == 1 and self.current_position <= 0:
                # BUY signal - only if not already in position
                self._open_position(timestamp, current_price, 'BUY', confidence)

            elif prediction == 0 and self.current_position > 0:
                # SELL signal - close existing position
                self._close_position(timestamp, current_price, reason='Signal')

        # Close any remaining position at the end
        if self.current_position > 0:
            final_price = historical_data.iloc[-1]['close']
            self._close_position(historical_data.index[-1], final_price, reason='End of Backtest')

        print(f"✅ Executed {len(self.trades)} trades")

    def _open_position(self, timestamp: datetime, price: float, direction: str, confidence: float):
        """Open a new position with risk management"""
        try:
            # Calculate position size based on confidence
            position_size = min(self.max_position_size, confidence)
            position_value = self.current_capital * position_size

            # Apply slippage
            entry_price = price * (1 + self.slippage) if direction == 'BUY' else price * (1 - self.slippage)

            # Calculate commission
            commission_cost = position_value * self.commission

            # Execute trade
            shares = position_value / entry_price
            cost = position_value + commission_cost

            if cost > self.current_capital:
                return  # Not enough capital

            self.current_position = shares
            self.current_capital -= cost
            self.current_price = entry_price

            # Set stop loss and take profit prices
            self.stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            self.take_profit_price = entry_price * (1 + self.take_profit_pct)

            # Record trade
            trade = {
                'entry_time': timestamp,
                'entry_price': entry_price,
                'direction': direction,
                'shares': shares,
                'position_value': position_value,
                'commission': commission_cost,
                'confidence': confidence,
                'stop_loss': self.stop_loss_price,
                'take_profit': self.take_profit_price
            }

            self.trades.append(trade)

        except Exception as e:
            print(f"❌ Error opening position: {e}")

    def _close_position(self, timestamp: datetime, price: float, reason: str = 'Signal'):
        """Close current position"""
        if self.current_position <= 0:
            return

        try:
            # Apply slippage
            exit_price = price * (1 - self.slippage)

            # Calculate proceeds
            proceeds = self.current_position * exit_price
            commission_cost = proceeds * self.commission
            net_proceeds = proceeds - commission_cost

            # Update capital
            self.current_capital += net_proceeds

            # Calculate profit/loss
            entry_cost = self.trades[-1]['position_value'] + self.trades[-1]['commission']
            profit_loss = net_proceeds - entry_cost
            profit_loss_pct = (profit_loss / entry_cost) * 100

            # Update last trade
            self.trades[-1].update({
                'exit_time': timestamp,
                'exit_price': exit_price,
                'exit_reason': reason,
                'profit_loss': profit_loss,
                'profit_loss_pct': profit_loss_pct,
                'exit_commission': commission_cost
            })

            # Reset position
            self.current_position = 0.0
            self.current_price = 0.0
            self.stop_loss_price = None
            self.take_profit_price = None

        except Exception as e:
            print(f"❌ Error closing position: {e}")

    def _store_backtest_results(self, metrics: Dict[str, Any], model_path: str,
                             symbol: str, pair: str, interval: str,
                             start_date: str = None, end_date: str = None):
        """Store backtest results in quantconnect_backtests table"""
        try:
            # Generate unique backtest ID
            backtest_id = f"enhanced_xgboost_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Calculate derived metrics
            if self.trades:
                winning_trades = [t for t in self.trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in self.trades if t.get('profit_loss', 0) < 0]

                # Calculate streaks
                longest_win_streak = self._calculate_longest_streak(self.trades, True)
                longest_loss_streak = self._calculate_longest_streak(self.trades, False)

                # Create trades JSON for storage
                trades_json = json.dumps([
                    {
                        'entry_time': t.get('entry_time').isoformat() if t.get('entry_time') else None,
                        'exit_time': t.get('exit_time').isoformat() if t.get('exit_time') else None,
                        'entry_price': t.get('entry_price'),
                        'exit_price': t.get('exit_price'),
                        'direction': t.get('direction'),
                        'profit_loss': t.get('profit_loss'),
                        'confidence': t.get('confidence'),
                        'shares': t.get('shares'),
                        'exit_reason': t.get('exit_reason')
                    }
                    for t in self.trades
                ], default=str)
            else:
                winning_trades = losing_trades = []
                longest_win_streak = longest_loss_streak = 0
                trades_json = '[]'

            # Create equity curve JSON
            if self.equity_curve:
                equity_curve_json = json.dumps([
                    {
                        'timestamp': e['timestamp'].isoformat() if isinstance(e['timestamp'], pd.Timestamp) else str(e['timestamp']),
                        'equity_value': float(e['equity_value']),
                        'price': float(e['price']),
                        'position': float(e['position']),
                        'cash': float(e['cash'])
                    }
                    for e in self.equity_curve
                ], default=str)
            else:
                equity_curve_json = '[]'

            # Create monthly returns JSON
            monthly_returns_data = {}
            current_year = datetime.now().year
            for month in range(1, 13):  # Placeholder for actual monthly data
                monthly_returns_data[f"{current_year}-{month:02d}"] = {
                    'return': metrics.get('monthly_return', 0),
                    'volatility': metrics.get('volatility', 0)
                }
            monthly_returns_json = json.dumps(monthly_returns_data, default=str)

            # Insert into database
            insert_query = """
                INSERT INTO quantconnect_backtests (
                    backtest_id, project_id, name, strategy_type, description,
                    backtest_start, backtest_end, duration_days, total_return, cagr,
                    sharpe_ratio, sortino_ratio, calmar_ratio, max_drawdown,
                    recovery_days, total_trades, winning_trades, losing_trades,
                    win_rate, profit_factor, expectancy, avg_win, avg_loss,
                    largest_win, largest_loss, longest_win_streak, longest_loss_streak,
                    starting_capital, ending_capital, total_fees, equity_curve,
                    monthly_returns, trades, parameters, raw_result, status,
                    import_source, created_at, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s,  %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, NOW(), NOW()
                )
            """

            params = (
                backtest_id, 'enhanced_xgboost_project',
                f'Enhanced XGBoost {symbol} {interval}',
                'enhanced_microstructure_trading',
                f'Backtest with {metrics.get("backtest_config", {}).get("max_position_size", 0.95)*100:.0f}% position, {metrics.get("backtest_config", {}).get("commission", 0.001)*100:.2f}% commission, {metrics.get("backtest_config", {}).get("slippage", 0.0005)*100:.2f}% slippage',
                start_date or '2024-01-01',
                end_date or '2024-03-31',
                metrics.get('total_trading_days', 0),
                metrics.get('total_return', 0),
                metrics.get('annualized_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('sortino_ratio', 0),
                metrics.get('calmar_ratio', 0),
                abs(metrics.get('max_drawdown', 0)),
                metrics.get('max_drawdown_duration', 0),
                metrics.get('trade_count', 0),
                len(winning_trades),
                len(losing_trades),
                metrics.get('win_rate', 0),
                metrics.get('profit_factor', 0),
                metrics.get('expectancy', 0),
                metrics.get('avg_win', 0),
                metrics.get('avg_loss', 0),
                metrics.get('largest_win', 0),
                metrics.get('largest_loss', 0),
                longest_win_streak,
                longest_loss_streak,
                self.initial_capital,
                metrics.get('backtest_config', {}).get('final_capital', self.current_capital),
                0,  # Total fees calculation
                equity_curve_json,
                monthly_returns_json,
                trades_json,
                json.dumps(self.backtest_config, default=str),
                json.dumps(metrics, default=str),
                'completed',
                'enhanced_xgboost_local_backtest'
            )

            self.db_manager.execute_insert(insert_query, params)

            print(f"✅ Backtest results stored in database:")
            print(f"   🆔 Backtest ID: {backtest_id}")
            print(f"   📊 Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"   🎯 Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"   📈 Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")

        except Exception as e:
            print(f"❌ Error storing backtest results: {e}")
            import traceback
            traceback.print_exc()

    def _calculate_longest_streak(self, trades: List[Dict], winning: bool) -> int:
        """Calculate longest winning or losing streak"""
        if not trades:
            return 0

        max_streak = 0
        current_streak = 0

        for trade in trades:
            is_winning = trade.get('profit_loss', 0) > 0
            if is_winning == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

def main():
    parser = argparse.ArgumentParser(description='Run enhanced backtest with real performance metrics')
    parser.add_argument('--model', required=True, help='Path to trained XGBoost model')
    parser.add_argument('--symbol', default='BTC', help='Trading symbol')
    parser.add_argument('--pair', default='BTCUSDT', help='Trading pair')
    parser.add_argument('--interval', default='1h', help='Time interval')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--slippage', type=float, default=0.0005, help='Slippage rate')

    args = parser.parse_args()

    try:
        # Initialize backtest engine
        engine = BacktestEngine(
            initial_capital=args.capital,
            commission=args.commission,
            slippage=args.slippage
        )

        # Run backtest
        results = engine.run_backtest(
            model_path=args.model,
            symbol=args.symbol,
            pair=args.pair,
            interval=args.interval,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if results:
            print(f"\n🎉 Backtest completed successfully!")
            print(f"📊 Final Capital: ${results['backtest_config']['final_capital']:,.2f}")
            print(f"📈 Total Return: {results['total_return']:.2%}")
            print(f"🎯 Overall Grade: {results['overall_grade']}")
        else:
            print("❌ Backtest failed")

    except KeyboardInterrupt:
        print("\n👋 Backtest interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()