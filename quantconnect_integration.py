#!/usr/bin/env python3
"""
Enhanced QuantConnect Integration for XGBoost Trading System
Integrates microstructure features with QuantConnect backtesting
Provides institutional-grade backtesting with real market conditions
"""

# region imports
from AlgorithmImports import *
# endregion

import numpy as np
import pandas as pd
from collections import deque
import json
import os
import joblib
from datetime import timedelta, datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedXGBoostAlpha(QCAlgorithm):
    """
    Enhanced XGBoost trading algorithm with microstructure features
    Integrates with the enhanced feature engineering and evaluation systems
    """

    def Initialize(self):
        """Initialize the algorithm with enhanced parameters"""
        # Set algorithm settings
        self.SetStartDate(2024, 1, 1)  # Start backtest
        self.SetEndDate(2024, 12, 31)    # End backtest
        self.SetCash(100000)             # Starting capital

        # Add assets
        self.symbol = self.AddCrypto("BTCUSDT", Resolution.Hour).Symbol

        # Enhanced algorithm parameters
        self.SetWarmUp(100)  # Warmup period for indicators
        self.UniverseSettings.Resolution = Resolution.Hour

        # Model and feature settings
        self.model_path = "xgboost_trading_model_latest.joblib"
        self.lookback_period = 100
        self.min_confidence = 0.65
        self.position_size = 0.95  # 95% of capital
        self.stop_loss_pct = 0.02  # 2%
        self.take_profit_pct = 0.05  # 5%

        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.signals_generated = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Risk management
        self.max_drawdown = 0.15  # 15% max drawdown
        self.current_drawdown = 0

        # Feature cache
        self.feature_cache = deque(maxlen=200)
        self.last_feature_update = None

        # Load enhanced model
        self.LoadEnhancedModel()

        # Schedule events
        self.Schedule.On(self.DateRules.EveryDay(), self.TimeRules.EveryHour(), self.GenerateSignal)

        # Charting
        self.SetupCharting()

    def LoadEnhancedModel(self):
        """Load the enhanced XGBoost model with microstructure features"""
        try:
            # Try to load the latest model
            model_files = [f for f in os.listdir('.') if f.startswith('xgboost_trading_model_') and f.endswith('.joblib')]
            if model_files:
                latest_model = sorted(model_files)[-1]
                self.model_path = latest_model
                self.Debug(f"Loading model: {self.model_path}")

            self.model = joblib.load(self.model_path)
            self.Debug("✅ Enhanced XGBoost model loaded successfully")

            # Get model feature names
            if hasattr(self.model, 'feature_names_in_'):
                self.model_features = list(self.model.feature_names_in_)
                self.Debug(f"Model expects {len(self.model_features)} features")
            else:
                self.model_features = None
                self.Debug("⚠️  Model feature names not available")

        except Exception as e:
            self.Debug(f"❌ Failed to load model: {e}")
            self.model = None

    def SetupCharting(self):
        """Setup enhanced charting for performance visualization"""
        # Equity curve
        equityChart = Chart("Equity Curve")
        equityChart.AddSeries(Series("Equity", SeriesType.Line, 0))
        equityChart.AddSeries(Series("Drawdown", SeriesType.Line, 1))
        self.AddChart(equityChart)

        # Performance metrics
        performanceChart = Chart("Performance")
        performanceChart.AddSeries(Series("Win Rate", SeriesType.Line, 0))
        performanceChart.AddSeries(Series("Profit Factor", SeriesType.Line, 1))
        self.AddChart(performanceChart)

        # Feature importance (if available)
        if hasattr(self.model, 'feature_importances_') and self.model_features:
            featureChart = Chart("Top Features")
            for i in range(min(5, len(self.model_features))):
                featureChart.AddSeries(Series(f"Feature_{i}", SeriesType.Line, i))
            self.AddChart(featureChart)

    def OnData(self, data):
        """Main data handling with enhanced market microstructure"""
        if not self.model or not data.ContainsKey(self.symbol):
            return

        # Get current price and volume
        bar = data[self.symbol]
        current_price = bar.Close
        current_volume = bar.Volume

        # Cache market data for feature engineering
        self.CacheMarketData(bar)

        # Update equity curve
        current_equity = self.Portfolio.TotalPortfolioValue
        self.equity_curve.append({
            'time': self.Time,
            'equity': current_equity,
            'price': current_price,
            'position': self.Portfolio[self.symbol].Quantity
        })

        # Calculate and track drawdown
        self.UpdateDrawdown(current_equity)

        # Risk management check
        self.CheckRiskManagement()

        # Update charts
        self.UpdateCharts(current_equity)

    def CacheMarketData(self, bar):
        """Cache market data for enhanced feature engineering"""
        try:
            market_data = {
                'time': bar.EndTime,
                'open': bar.Open,
                'high': bar.High,
                'low': bar.Low,
                'close': bar.Close,
                'volume': bar.Volume,
                'symbol': str(self.symbol)
            }

            # Add enhanced market data if available
            # In a real implementation, this would connect to your microstructure data sources
            market_data.update(self.GetEnhancedMarketData())

            self.feature_cache.append(market_data)
            self.last_feature_update = bar.EndTime

        except Exception as e:
            self.Debug(f"Error caching market data: {e}")

    def GetEnhancedMarketData(self) -> dict:
        """Get enhanced microstructure market data (placeholder for real integration)"""
        # In production, this would pull from:
        # - Order book depth data
        # - Futures basis rates
        # - Trade footprint data
        # - Options OI data

        # Placeholder values - replace with real data integration
        return {
            'bid_ask_imbalance': 0.0,
            'total_depth': 1000000.0,
            'basis_momentum': 0.0,
            'volume_aggression': 0.0,
            'options_oi_ratio': 1.0
        }

    def GenerateSignal(self):
        """Generate enhanced trading signals using microstructure features"""
        if len(self.feature_cache) < self.lookback_period:
            return

        try:
            # Convert cache to DataFrame for feature engineering
            df = pd.DataFrame(list(self.feature_cache))
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)

            # Apply enhanced feature engineering (simplified for QC)
            features = self.CreateEnhancedFeatures(df)
            if features is None:
                return

            # Make prediction
            prediction, confidence = self.MakePrediction(features)
            if prediction is None:
                return

            self.signals_generated += 1

            # Execute trading logic
            self.ExecuteTradingLogic(prediction, confidence, df.iloc[-1]['close'])

        except Exception as e:
            self.Debug(f"Error in GenerateSignal: {e}")

    def CreateEnhancedFeatures(self, df: pd.DataFrame):
        """Create enhanced features compatible with our model"""
        try:
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['sma_5'] = df['close'].rolling(5).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            df['rsi'] = self.CalculateRSI(df['close'])
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Enhanced microstructure features
            if 'bid_ask_imbalance' in df.columns:
                df['bid_ask_imbalance_sma'] = df['bid_ask_imbalance'].rolling(10).mean()
                df['strong_bid_pressure'] = (df['bid_ask_imbalance'] > 0.1).astype(int)

            if 'basis_momentum' in df.columns:
                df['basis_trend'] = np.sign(df['basis_momentum'])
                df['basis_volatility'] = df['basis_momentum'].rolling(10).std()

            if 'volume_aggression' in df.columns:
                df['aggression_signal'] = np.sign(df['volume_aggression'])
                df['high_aggression'] = (abs(df['volume_aggression']) > 0.3).astype(int)

            # Time features
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month

            # Get latest features
            latest_features = df.iloc[-1].to_dict()

            # Select model features
            if self.model_features:
                selected_features = []
                for feature in self.model_features:
                    if feature in latest_features:
                        selected_features.append(latest_features[feature])
                    else:
                        selected_features.append(0.0)  # Default value
                return selected_features
            else:
                # Fallback to numeric features
                numeric_features = [v for k, v in latest_features.items()
                                   if isinstance(v, (int, float)) and not pd.isna(v)]
                return numeric_features

        except Exception as e:
            self.Debug(f"Error creating features: {e}")
            return None

    def CalculateRSI(self, prices, period=14):
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50

    def MakePrediction(self, features):
        """Make prediction using the enhanced model"""
        try:
            if not self.model or not features:
                return None, 0.0

            # Reshape features for prediction
            X = np.array(features).reshape(1, -1)

            # Get prediction probabilities
            prediction_proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]
            confidence = max(prediction_proba)

            return prediction, confidence

        except Exception as e:
            self.Debug(f"Error making prediction: {e}")
            return None, 0.0

    def ExecuteTradingLogic(self, prediction, confidence, current_price):
        """Execute enhanced trading logic with risk management"""
        if confidence < self.min_confidence:
            return

        current_position = self.Portfolio[self.symbol].Quantity
        cash = self.Portfolio.Cash

        # Buy signal
        if prediction == 1 and current_position <= 0:
            position_value = cash * self.position_size
            shares = position_value / current_price

            # Apply risk management
            if self.current_drawdown > self.max_drawdown:
                self.Debug(f"⚠️  Max drawdown exceeded, skipping trade")
                return

            self.MarketOrder(self.symbol, shares, "Enhanced Buy Signal")
            self.SetStopLoss(self.symbol, current_price * (1 - self.stop_loss_pct), stopMarketOrder=True)
            self.SetTakeProfit(self.symbol, current_price * (1 + self.take_profit_pct))

            self.Debug(f"🟢 BUY: {shares:.4f} @ {current_price:.2f} (confidence: {confidence:.2f})")

        # Sell signal
        elif prediction == 0 and current_position > 0:
            self.MarketOrder(self.symbol, -current_position, "Enhanced Sell Signal")
            self.Debug(f"🔴 SELL: {current_position:.4f} @ {current_price:.2f} (confidence: {confidence:.2f})")

    def CheckRiskManagement(self):
        """Enhanced risk management checks"""
        # Drawdown protection
        if self.current_drawdown > self.max_drawdown:
            self.Debug(f"⚠️  Max drawdown exceeded: {self.current_drawdown:.2%}")
            # Close positions if drawdown too high
            if self.Portfolio[self.symbol].Quantity > 0:
                self.Liquidate(self.symbol)
                self.Debug("🛑 Liquidated position due to max drawdown")

    def UpdateDrawdown(self, current_equity):
        """Update drawdown calculation"""
        if len(self.equity_curve) > 1:
            peak_equity = max([e['equity'] for e in self.equity_curve])
            if peak_equity > 0:
                self.current_drawdown = (peak_equity - current_equity) / peak_equity

    def UpdateCharts(self, current_equity):
        """Update performance charts"""
        if len(self.equity_curve) > 1:
            peak_equity = max([e['equity'] for e in self.equity_curve])
            if peak_equity > 0:
                drawdown = (peak_equity - current_equity) / peak_equity
                self.Plot("Equity Curve", "Drawdown", drawdown)

        self.Plot("Equity Curve", "Equity", current_equity)

        # Update performance metrics
        if self.signals_generated > 0:
            win_rate = self.winning_trades / self.signals_generated
            self.Plot("Performance", "Win Rate", win_rate)

    def OnOrderEvent(self, orderEvent):
        """Handle order events with enhanced tracking"""
        order = orderEvent.Order
        if order.Status == OrderStatus.Filled:
            # Track trade for performance evaluation
            if order.Type == OrderType.Market:
                self.Debug(f"✅ Order filled: {order.Symbol} {order.Quantity} @ {order.AverageFilledPrice:.2f}")

                # Update trade counters (simplified)
                if order.Quantity > 0:
                    # This is a buy order
                    pass
                else:
                    # This is a sell order - could evaluate P&L
                    pass

    def OnEndOfAlgorithm(self):
        """Generate comprehensive backtest report"""
        self.Debug("🏁 Enhanced XGBoost Backtest Complete")
        self.Debug(f"📊 Final Equity: ${self.Portfolio.TotalPortfolioValue:,.2f}")
        self.Debug(f"📈 Total Return: {((self.Portfolio.TotalPortfolioValue / 100000) - 1) * 100:.2f}%")
        self.Debug(f"🎯 Signals Generated: {self.signals_generated}")

        # Generate detailed performance report
        self.GeneratePerformanceReport()

    def GeneratePerformanceReport(self):
        """Generate detailed performance report using our evaluation system"""
        try:
            # This would integrate with the PerformanceEvaluator
            # Convert QC data to evaluation format
            if len(self.equity_curve) > 1:
                equity_df = pd.DataFrame(self.equity_curve)
                equity_df.columns = ['timestamp', 'equity_value', 'price', 'position']

                # Calculate basic metrics (simplified)
                total_return = (equity_df['equity_value'].iloc[-1] / equity_df['equity_value'].iloc[0]) - 1
                self.Debug(f"📈 Total Return: {total_return:.2%}")
                self.Debug(f"📊 Max Drawdown: {self.current_drawdown:.2%}")
                self.Debug(f"🎯 Win Rate: {self.winning_trades / max(1, self.signals_generated):.2%}")

                # Save results for detailed analysis
                results = {
                    'total_return': total_return,
                    'max_drawdown': self.current_drawdown,
                    'signals_generated': self.signals_generated,
                    'equity_curve': equity_df.to_dict('records')
                }

                with open('quantconnect_results.json', 'w') as f:
                    json.dump(results, f, indent=2, default=str)

        except Exception as e:
            self.Debug(f"Error generating report: {e}")