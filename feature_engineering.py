import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self):
        self.feature_columns = []

    def create_all_features(self, df):
        """Create all technical features for the model"""
        df = df.copy()

        print(f"📊 Creating features for dataset with {len(df)} rows...")

        # Basic price features
        df = self._create_price_features(df)

        # Technical indicators
        df = self._create_technical_indicators(df)

        # Volume features (only if volume data exists)
        if 'volume' in df.columns and df['volume'].notna().sum() > 0:
            df = self._create_volume_features(df)
        else:
            print("⚠️  No volume data found, skipping volume features")

        # Volatility features
        df = self._create_volatility_features(df)

        # Time-based features
        df = self._create_time_features(df)

        # Lag features (reduce for small datasets)
        if len(df) > 10:
            df = self._create_lag_features(df)
        else:
            print("⚠️  Dataset too small for lag features")

        # Rolling features (adjust windows for small datasets)
        df = self._create_rolling_features(df)

        # Multi-source data features
        df = self._create_multi_source_features(df)

        print(f"✅ Feature engineering completed. Final shape: {df.shape}")
        return df

    def _create_price_features(self, df):
        """Create basic price-based features"""
        # Ensure numeric data types
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate returns (handle edge cases)
        df['returns'] = df['close'].pct_change()

        # Calculate log returns (handle edge cases)
        price_ratio = df['close'] / df['close'].shift(1)
        # Avoid division by zero and log of negative numbers
        price_ratio = price_ratio.fillna(1)
        price_ratio = price_ratio.clip(lower=0.0001)  # Avoid log(0) or negative
        df['log_returns'] = np.log(price_ratio)

        # Handle division by zero in ratios
        df['high_low_ratio'] = df['high'] / df['low'].replace(0, 1e-8)
        df['close_open_ratio'] = df['close'] / df['open'].replace(0, 1e-8)
        df['hl_range'] = (df['high'] - df['low']) / df['low'].replace(0, 1e-8)
        df['oc_range'] = (df['close'] - df['open']) / df['open'].replace(0, 1e-8)

        return df

    def _create_technical_indicators(self, df):
        """Create technical indicator features"""
        df = df.copy()

        # Moving averages - using manual calculation
        periods = [5, 10, 20, 50, 200]
        for period in periods:
            if len(df) >= period:
                df[f'sma_{period}'] = df['close'].rolling(window=period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            else:
                # For small datasets, use smaller windows
                smaller_period = max(2, len(df) // 4)
                df[f'sma_{period}'] = df['close'].rolling(window=smaller_period, min_periods=1).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=smaller_period, adjust=False).mean()

        # Price relative to moving averages
        df['price_above_sma_20'] = (df['close'] > df['sma_20']).astype(int)
        df['price_above_sma_50'] = (df['close'] > df['sma_50']).astype(int)
        df['price_above_ema_20'] = (df['close'] > df['ema_20']).astype(int)

        # MACD - manual calculation
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # RSI - manual calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-8)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rfi_oversold'] = (df['rsi'] < 30).astype(int)

        # Bollinger Bands
        period = 20
        std = df['close'].rolling(window=period, min_periods=1).std()
        df['bb_middle'] = df['close'].rolling(window=period, min_periods=1).mean()
        df['bb_upper'] = df['bb_middle'] + (std * 2)
        df['bb_lower'] = df['bb_middle'] - (std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'].replace(0, 1e-8)
        bb_range = (df['bb_upper'] - df['bb_lower']).replace(0, 1e-8)
        df['bb_position'] = (df['close'] - df['bb_lower']) / bb_range

        # Stochastic Oscillator
        k_period = 14
        d_period = 3
        low_min = df['low'].rolling(window=k_period, min_periods=1).min()
        high_max = df['high'].rolling(window=k_period, min_periods=1).max()
        stoch_range = (high_max - low_min).replace(0, 1e-8)
        df['stoch_k'] = 100 * (df['close'] - low_min) / stoch_range
        df['stoch_d'] = df['stoch_k'].rolling(window=d_period, min_periods=1).mean()

        return df

    def _create_volume_features(self, df):
        """Create volume-based features"""
        # Volume moving averages
        df['volume_sma_20'] = df['volume'].rolling(window=20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20'].replace(0, 1e-8)

        # Volume Price Trend (VPT)
        df['vpt'] = (df['volume'] * df['returns']).cumsum()

        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(df['returns']) * df['volume']).cumsum()

        # VWAP (Volume Weighted Average Price)
        if 'close' in df.columns and 'volume' in df.columns:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            df['price_above_vwap'] = (df['close'] > df['vwap']).astype(int)

        return df

    def _create_volatility_features(self, df):
        """Create volatility-based features"""
        # ATR - Manual calculation
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14, min_periods=1).mean()
        df['atr_ratio'] = df['atr'] / df['close']

        # Rolling volatility
        for period in [5, 10, 20, 50]:
            if len(df) >= period:
                df[f'volatility_{period}'] = df['returns'].rolling(window=period, min_periods=1).std()
            else:
                smaller_period = max(2, len(df) // 2)
                df[f'volatility_{period}'] = df['returns'].rolling(window=smaller_period, min_periods=1).std()

        # Price acceleration
        df['price_acceleration'] = df['returns'].diff()

        return df

    def _create_time_features(self, df):
        """Create time-based features"""
        df.index = pd.to_datetime(df.index)

        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter

        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        # Session indicators
        df['asian_session'] = ((df.index.hour >= 0) & (df.index.hour < 8)).astype(int)
        df['european_session'] = ((df.index.hour >= 8) & (df.index.hour < 16)).astype(int)
        df['us_session'] = ((df.index.hour >= 16) & (df.index.hour < 24)).astype(int)

        return df

    def _create_lag_features(self, df):
        """Create lagged features"""
        # Adjust lag periods based on dataset size
        max_lag = max(1, len(df) // 4)
        base_lags = [1, 2, 3, 5, 10]
        lags_to_use = []

        for lag in base_lags:
            if lag < len(df):
                lags_to_use.append(lag)

        if not lags_to_use:
            lags_to_use = [1]

        for lag in lags_to_use:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)

            if 'volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            else:
                df[f'volume_lag_{lag}'] = 0

            if 'rsi' in df.columns:
                df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
            else:
                df[f'rsi_lag_{lag}'] = 50  # Default RSI value

        return df

    def _create_rolling_features(self, df):
        """Create rolling window features"""
        # Adjust windows based on dataset size
        base_windows = [5, 10, 20, 50]
        windows_to_use = []

        for window in base_windows:
            if len(df) >= window:
                windows_to_use.append(window)
            else:
                if len(df) >= 3:
                    smaller_window = max(2, len(df) // 2)
                    if smaller_window not in windows_to_use:
                        windows_to_use.append(smaller_window)

        if not windows_to_use:
            windows_to_use = [2]

        for window in windows_to_use:
            try:
                # Rolling statistics for returns
                df[f'returns_mean_{window}'] = df['returns'].rolling(window=window, min_periods=1).mean()
                df[f'returns_std_{window}'] = df['returns'].rolling(window=window, min_periods=1).std()

                if window >= 3:
                    df[f'returns_skew_{window}'] = df['returns'].rolling(window=window, min_periods=1).skew()
                    df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window, min_periods=1).kurt()
                else:
                    df[f'returns_skew_{window}'] = 0
                    df[f'returns_kurt_{window}'] = 0

                if 'volume' in df.columns:
                    df[f'volume_mean_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
                    df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std()
                else:
                    df[f'volume_mean_{window}'] = 0
                    df[f'volume_std_{window}'] = 0

                # Rolling max/min for price
                df[f'high_max_{window}'] = df['high'].rolling(window=window, min_periods=1).max()
                df[f'low_min_{window}'] = df['low'].rolling(window=window, min_periods=1).min()

                denominator = (df[f'high_max_{window}'] - df[f'low_min_{window}']).replace(0, 1e-8)
                df[f'close_position_{window}'] = (df['close'] - df[f'low_min_{window}']) / denominator

            except Exception as e:
                print(f"⚠️  Error creating rolling features for window {window}: {e}")
                # Set default values if calculation fails
                df[f'returns_mean_{window}'] = 0
                df[f'returns_std_{window}'] = 0
                df[f'returns_skew_{window}'] = 0
                df[f'returns_kurt_{window}'] = 0
                df[f'volume_mean_{window}'] = 0
                df[f'volume_std_{window}'] = 0
                df[f'high_max_{window}'] = df['close']
                df[f'low_min_{window}'] = df['close']
                df[f'close_position_{window}'] = 0.5

        return df

    def _create_multi_source_features(self, df):
        """Create comprehensive features from multiple data sources including the 4 new critical tables"""
        print("🔧 Creating enhanced multi-source market features with new microstructure data...")

        # 1. Enhanced Open Interest Features
        if 'open_interest' in df.columns:
            window = min(20, max(2, len(df) // 2))
            df['oi_sma'] = df['open_interest'].rolling(window=window, min_periods=1).mean()
            df['oi_ratio'] = df['open_interest'] / df['oi_sma'].replace(0, 1e-8)
            df['oi_change'] = df['open_interest'].pct_change()
            df['oi_volume_ratio'] = df['open_interest'] / (df['volume'] + 1e-8)
            df['oi_price_ratio'] = df['open_interest'] / (df['close'] + 1e-8)

            # Open interest momentum
            for w in [5, 10]:
                df[f'oi_roc_{w}'] = df['open_interest'].pct_change(w)
                df[f'oi_ratio_sma_{w}'] = df['oi_ratio'].rolling(window=w, min_periods=1).mean()

            print("✅ Enhanced open interest features added")

        # 2. Comprehensive Liquidation Features
        if 'total_liquidations' in df.columns:
            # Liquidation intensity indicators
            window = min(20, max(2, len(df) // 2))
            df['liquidation_sma'] = df['total_liquidations'].rolling(window=window, min_periods=1).mean()
            df['liquidation_ratio'] = df['total_liquidations'] / df['liquidation_sma'].replace(0, 1e-8)
            df['liquidation_volume_ratio'] = df['total_liquidations'] / (df['volume'] + 1e-8)
            df['liquidation_price_ratio'] = df['total_liquidations'] / (df['close'] + 1e-8)

            # Liquidation spike detection
            df['liquidation_spike'] = (df['total_liquidations'] > 2 * df['liquidation_sma']).astype(int)
            df['liquidation_extreme'] = (df['total_liquidations'] > 3 * df['liquidation_sma']).astype(int)

            # Long vs Short liquidation analysis
            if 'aggregated_long_liquidation_usd' in df.columns and 'aggregated_short_liquidation_usd' in df.columns:
                df['long_liq_ratio'] = df['aggregated_long_liquidation_usd'] / (df['total_liquidations'] + 1e-8)
                df['short_liq_ratio'] = df['aggregated_short_liquidation_usd'] / (df['total_liquidations'] + 1e-8)
                df['liq_imbalance'] = (df['aggregated_long_liquidation_usd'] - df['aggregated_short_liquidation_usd']) / (df['total_liquidations'] + 1e-8)

                # Contrarian indicators
                df['long_liq_dominated'] = (df['long_liq_ratio'] > 0.7).astype(int)
                df['short_liq_dominated'] = (df['short_liq_ratio'] > 0.7).astype(int)

            print("✅ Enhanced liquidation features added")

        # 3. Taker Volume & Market Microstructure Features
        if 'buy_sell_ratio' in df.columns:
            # Volume pressure indicators
            window = min(10, max(2, len(df) // 3))
            df['buy_ratio_sma'] = df['buy_sell_ratio'].rolling(window=window, min_periods=1).mean()
            df['buy_ratio_change'] = df['buy_sell_ratio'].diff()

            # Market pressure regimes
            df['strong_buying'] = (df['buy_sell_ratio'] > 0.6).astype(int)
            df['strong_selling'] = (df['buy_sell_ratio'] < 0.4).astype(int)
            df['balanced_market'] = ((df['buy_sell_ratio'] >= 0.4) & (df['buy_sell_ratio'] <= 0.6)).astype(int)

            # Volume intensity
            if 'total_taker_volume' in df.columns:
                df['taker_volume_ratio'] = df['total_taker_volume'] / (df['volume'] + 1e-8)
                df['taker_volume_sma'] = df['total_taker_volume'].rolling(window=window, min_periods=1).mean()
                df['taker_volume_intensity'] = df['total_taker_volume'] / df['taker_volume_sma'].replace(0, 1e-8)

            print("✅ Enhanced taker volume features added")

        # 4. Funding Rate Features (Market Sentiment)
        if 'funding_rate' in df.columns:
            # Funding rate analysis
            window = min(12, max(2, len(df) // 3))
            df['funding_sma'] = df['funding_rate'].rolling(window=window, min_periods=1).mean()
            df['funding_change'] = df['funding_rate'].diff()
            df['funding_volatility'] = df['funding_rate'].rolling(window=window, min_periods=1).std()

            # Extreme funding rate indicators
            df['high_funding'] = (df['funding_rate'] > 0.01).astype(int)  # > 1%
            df['low_funding'] = (df['funding_rate'] < -0.01).astype(int)  # < -1%
            df['extreme_high_funding'] = (df['funding_rate'] > 0.02).astype(int)  # > 2%
            df['extreme_low_funding'] = (df['funding_rate'] < -0.02).astype(int)  # < -2%

            # Funding rate momentum
            df['funding_trend'] = np.sign(df['funding_rate'] - df['funding_sma'])
            df['funding_regime'] = np.where(abs(df['funding_rate']) > 0.005, 1, 0)  # Significant funding

            print("✅ Enhanced funding rate features added")

        # 5. Smart Money Features (Top Account Analysis)
        if 'top_account_long_short_ratio' in df.columns:
            # Smart money positioning
            df['top_ratio_sma'] = df['top_account_long_short_ratio'].rolling(window=window, min_periods=1).mean()
            df['top_ratio_change'] = df['top_account_long_short_ratio'].diff()

            # Smart money signals
            df['smart_money_long'] = (df['top_account_long_short_ratio'] > 1.2).astype(int)
            df['smart_money_short'] = (df['top_account_long_short_ratio'] < 0.8).astype(int)
            df['smart_money_neutral'] = ((df['top_account_long_short_ratio'] >= 0.8) &
                                        (df['top_account_long_short_ratio'] <= 1.2)).astype(int)

            # Smart money extremes
            df['smart_money_extreme_long'] = (df['top_account_long_short_ratio'] > 1.5).astype(int)
            df['smart_money_extreme_short'] = (df['top_account_long_short_ratio'] < 0.5).astype(int)

            # Smart money divergence from price
            if 'close' in df.columns:
                df['price_change'] = df['close'].pct_change()
                df['smart_money_divergence'] = np.sign(df['top_account_long_short_ratio'].diff() * -df['price_change'])

            print("✅ Enhanced smart money features added")

        # 6. Crowd Sentiment Features (Global Account Analysis)
        if 'global_account_long_short_ratio' in df.columns:
            # Retail trader positioning
            df['global_ratio_sma'] = df['global_account_long_short_ratio'].rolling(window=window, min_periods=1).mean()
            df['global_ratio_change'] = df['global_account_long_short_ratio'].diff()

            # Crowd sentiment indicators
            df['crowd_long'] = (df['global_account_long_short_ratio'] > 1.5).astype(int)
            df['crowd_short'] = (df['global_account_long_short_ratio'] < 0.67).astype(int)
            df['crowd_neutral'] = ((df['global_account_long_short_ratio'] >= 0.67) &
                                  (df['global_account_long_short_ratio'] <= 1.5)).astype(int)

            # Crowd extremes (contrarian opportunities)
            df['crowd_extreme_long'] = (df['global_account_long_short_ratio'] > 2.0).astype(int)
            df['crowd_extreme_short'] = (df['global_account_long_short_ratio'] < 0.5).astype(int)

            print("✅ Enhanced crowd sentiment features added")

        # 8. NEW: Spot Orderbook Microstructure Features (Tier 1 - EXTREMELY HIGH impact)
        if all(col in df.columns for col in ['total_depth', 'bid_ask_imbalance', 'liquidity_ratio', 'orderbook_spread']):
            window = min(10, max(2, len(df) // 5))

            # Liquidity depth analysis
            df['depth_sma'] = df['total_depth'].rolling(window=window, min_periods=1).mean()
            df['depth_ratio'] = df['total_depth'] / df['depth_sma'].replace(0, 1e-8)
            df['depth_volatility'] = df['total_depth'].rolling(window=window, min_periods=1).std()

            # Order imbalance signals (VERY POWERFUL for entry timing)
            df['imbalance_sma'] = df['bid_ask_imbalance'].rolling(window=window, min_periods=1).mean()
            df['imbchange'] = df['bid_ask_imbalance'].diff()
            df['strong_bid_pressure'] = (df['bid_ask_imbalance'] > 0.2).astype(int)  # Bids dominate
            df['strong_ask_pressure'] = (df['bid_ask_imbalance'] < -0.2).astype(int)  # Asks dominate

            # Liquidity crunch indicators
            df['liquidity_sma'] = df['liquidity_ratio'].rolling(window=window, min_periods=1).mean()
            df['liquidity_spike'] = (abs(df['liquidity_ratio'] - df['liquidity_sma']) > 0.5).astype(int)

            # Spread analysis for market efficiency
            df['spread_sma'] = df['orderbook_spread'].rolling(window=window, min_periods=1).mean()
            df['spread_widening'] = (df['orderbook_spread'] > df['spread_sma'] * 1.5).astype(int)
            df['spread_narrowing'] = (df['orderbook_spread'] < df['spread_sma'] * 0.7).astype(int)

            # Orderbook momentum (leading indicator for price moves)
            df['orderbook_momentum'] = df['bid_ask_imbalance'].rolling(window=3).mean()
            df['orderbook_acceleration'] = df['orderbook_momentum'].diff()

            print("✅ Enhanced orderbook microstructure features added")

        # 9. NEW: Futures Basis Arbitrage Features (Tier 1 - EXTREMELY HIGH impact)
        if all(col in df.columns for col in ['open_basis', 'close_basis', 'basis_momentum', 'basis_volatility']):
            window = min(15, max(2, len(df) // 3))

            # Basis trend analysis (institutional positioning)
            df['basis_sma'] = df['close_basis'].rolling(window=window, min_periods=1).mean()
            df['basis_ratio'] = df['close_basis'] / (abs(df['basis_sma']) + 1e-8)

            # Basis momentum (leading indicator for spot)
            df['basis_momentum_sma'] = df['basis_momentum'].rolling(window=window, min_periods=1).mean()
            df['basis_acceleration'] = df['basis_momentum'].diff()
            df['basis_reversal'] = (df['basis_momentum'] * df['basis_momentum'].shift(1) < 0).astype(int)  # Sign change

            # Volatility regime detection
            df['basis_vol_regime'] = np.where(df['basis_volatility'] > df['basis_volatility'].rolling(window=window).mean() * 1.5, 2, 1)
            df['basis_calm_regime'] = (df['basis_volatility'] < df['basis_volatility'].rolling(window=window).mean() * 0.7).astype(int)

            # Arbitrage opportunity signals
            df['wide_basis'] = (abs(df['close_basis']) > abs(df['basis_sma']) * 2).astype(int)
            df['extreme_basis'] = (abs(df['close_basis']) > abs(df['basis_sma']) * 3).astype(int)

            # Contango/Backwardation regime
            df['contango_regime'] = (df['close_basis'] > 0).astype(int)
            df['backwardation_regime'] = (df['close_basis'] < 0).astype(int)
            df['regime_change'] = df['contango_regime'].diff().fillna(0).abs()  # 1 when regime changes

            # Basis vs price divergence (powerful signal)
            if 'returns' in df.columns:
                df['basis_price_divergence'] = np.sign(df['basis_momentum'] * df['returns'])  # Negative = reversal signal
                df['divergence_strength'] = abs(df['basis_momentum'] * df['returns'])

            print("✅ Enhanced futures basis arbitrage features added")

        # 10. NEW: Futures Footprint Trade Aggressiveness Features (Tier 2 - VERY HIGH impact)
        if all(col in df.columns for col in ['volume_aggression', 'trade_aggression', 'price_impact', 'aggressive_volume_ratio']):
            window = min(8, max(2, len(df) // 6))

            # Volume aggression analysis
            df['vol_aggression_sma'] = df['volume_aggression'].rolling(window=window, min_periods=1).mean()
            df['vol_aggression_volatility'] = df['volume_aggression'].rolling(window=window, min_periods=1).std()
            df['aggressive_buying'] = (df['volume_aggression'] > 0.3).astype(int)
            df['aggressive_selling'] = (df['volume_aggression'] < -0.3).astype(int)

            # Trade aggression (size vs frequency)
            df['trade_aggression_sma'] = df['trade_aggression'].rolling(window=window, min_periods=1).mean()
            df['institutional_buying'] = (df['trade_aggression'] > 0.2).astype(int)  # Large trades dominate
            df['retail_buying'] = (df['trade_aggression'] < -0.2).astype(int)  # Many small trades

            # Price impact analysis (execution quality)
            df['price_impact_sma'] = df['price_impact'].rolling(window=window, min_periods=1).mean()
            df['high_impact_buying'] = (df['price_impact'] > df['price_impact_sma'] * 1.5).astype(int)
            df['high_impact_selling'] = (df['price_impact'] < df['price_impact_sma'] * 1.5).astype(int)

            # Aggressive volume participation
            df['agg_volume_ratio_sma'] = df['aggressive_volume_ratio'].rolling(window=window, min_periods=1).mean()
            df['aggressive_volume_spike'] = (df['aggressive_volume_ratio'] > df['agg_volume_ratio_sma'] * 2).astype(int)

            # Composite aggressiveness score (0-1)
            aggression_components = []
            if 'volume_aggression' in df.columns:
                aggression_components.append(np.clip(df['volume_aggression'] + 0.5, 0, 1))
            if 'trade_aggression' in df.columns:
                aggression_components.append(np.clip(df['trade_aggression'] + 0.5, 0, 1))
            if 'price_impact' in df.columns:
                aggression_components.append(np.clip(abs(df['price_impact']) * 10, 0, 1))

            if aggression_components:
                df['composite_aggressiveness'] = sum(aggression_components) / len(aggression_components)

            # Aggressiveness momentum (predictive of continued moves)
            df['aggressiveness_momentum'] = df['composite_aggressiveness'].rolling(window=3).mean()
            df['aggressiveness_acceleration'] = df['aggressiveness_momentum'].diff()

            print("✅ Enhanced futures footprint aggressiveness features added")

        # 11. NEW: Options Exchange OI Features (Tier 2 - VERY HIGH impact)
        if all(col in df.columns for col in ['total_oi', 'oi_change', 'oi_volatility', 'exchange_diversification']):
            window = min(12, max(2, len(df) // 4))

            # Options OI momentum
            df['options_oi_sma'] = df['total_oi'].rolling(window=window, min_periods=1).mean()
            df['options_oi_ratio'] = df['total_oi'] / df['options_oi_sma'].replace(0, 1e-8)
            df['options_oi_roc'] = df['total_oi'].pct_change(window)

            # Options volatility regime
            df['options_vol_sma'] = df['oi_volatility'].rolling(window=window, min_periods=1).mean()
            df['options_vol_ratio'] = df['oi_volatility'] / df['options_vol_sma'].replace(0, 1e-8)
            df['options_vol_spike'] = (df['oi_volatility'] > df['options_vol_sma'] * 1.5).astype(int)

            # Exchange diversification benefits
            df['exchange_diversification_sma'] = df['exchange_diversification'].rolling(window=window, min_periods=1).mean()
            df['high_diversification'] = (df['exchange_diversification'] > 0.6).astype(int)  # Diversified OI = healthier

            # Options OI change patterns
            df['oi_accumulation'] = (df['oi_change'] > 0).rolling(window=window).sum() / window
            df['oi_distribution'] = (df['oi_change'] < 0).rolling(window=window).sum() / window
            df['oi_trend_strength'] = abs(df['oi_accumulation'] - df['oi_distribution'])

            # Options vs Spot correlation (contrarian signals)
            if 'returns' in df.columns:
                df['options_spot_correlation'] = np.sign(df['oi_change'] * df['returns'])
                df['options_divergence'] = (df['options_spot_correlation'] < 0).astype(int)  # Divergence = reversal

            # Options extreme levels (potential max pain)
            df['extreme_oi'] = (df['total_oi'] > df['options_oi_sma'] * 2).astype(int)
            df['oi_exhaustion'] = (df['total_oi'] < df['options_oi_sma'] * 0.5).astype(int)

            # Options institutional activity
            df['institutional_options'] = (df['exchange_diversification'] > 0.8) & (df['total_oi'] > df['options_oi_sma'])
            df['institutional_options'] = df['institutional_options'].astype(int)

            print("✅ Enhanced options exchange OI features added")

        # 12. Composite Indicators (Now enhanced with new microstructure data)
        df = self._create_enhanced_composite_indicators(df)

        # 13. Cross-Asset Interaction Features (Now with new microstructure interactions)
        df = self._create_enhanced_interaction_features(df)

        print("✅ Enhanced multi-source feature engineering completed")
        print(f"📈 Total features created: {len(df.columns)}")
        print(f"🔬 Includes: Orderbook Microstructure, Futures Basis, Footprint Analysis, Options OI")
        return df

    def _create_enhanced_composite_indicators(self, df):
        """Create enhanced composite indicators including new microstructure data"""
        try:
            # Market Structure Score (0-1, higher = bullish structure)
            structure_score = 0.5  # Base neutral

            if 'buy_sell_ratio' in df.columns:
                structure_score += (df['buy_sell_ratio'] - 0.5) * 0.3  # Volume pressure

            if 'oi_ratio' in df.columns:
                structure_score += (df['oi_ratio'] - 1.0) * 0.2  # Open interest momentum

            if 'top_account_long_short_ratio' in df.columns:
                # Smart money contrarian signal
                smart_adjust = (1.0 - df['top_account_long_short_ratio']) * 0.1
                structure_score += np.clip(smart_adjust, -0.1, 0.1)

            df['market_structure_score'] = np.clip(structure_score, 0, 1)

            # Leverage Cycle Indicator
            if 'funding_rate' in df.columns and 'total_liquidations' in df.columns:
                # High funding + high liquidations = leverage cycle peak
                leverage_cycle = abs(df['funding_rate']) * df['liquidation_ratio']
                df['leverage_cycle_intensity'] = np.clip(leverage_cycle, 0, 1)

            # Market Regime Classifier
            if all(col in df.columns for col in ['buy_sell_ratio', 'top_account_long_short_ratio', 'funding_rate']):
                # Bull regime: Strong buying + smart money contrarian + moderate funding
                bull_condition = (df['buy_sell_ratio'] > 0.6) & \
                               (df['top_account_long_short_ratio'] < 1.2) & \
                               (abs(df['funding_rate']) < 0.02)

                # Bear regime: Strong selling + smart money contrarian + moderate funding
                bear_condition = (df['buy_sell_ratio'] < 0.4) & \
                               (df['top_account_long_short_ratio'] > 0.8) & \
                               (abs(df['funding_rate']) < 0.02)

                df['market_regime'] = np.where(bull_condition, 2, np.where(bear_condition, 0, 1))  # 2=Bull, 1=Neutral, 0=Bear

            # Risk-On/Risk-Off Indicator (ENHANCED with new data)
            if 'total_taker_volume' in df.columns and 'open_interest' in df.columns:
                # Enhanced version with orderbook depth and basis
                volume_oi_trend = (df['taker_volume_intensity'] > 1.0) & (df['oi_ratio'] > 1.0)
                df['risk_on_indicator'] = volume_oi_trend.astype(int)

            # NEW: Enhanced Market Structure Score with microstructure data
            if all(col in df.columns for col in ['buy_sell_ratio', 'bid_ask_imbalance', 'basis_momentum', 'volume_aggression']):
                # Enhanced structure calculation with new microstructure inputs
                structure_components = {
                    'volume_pressure': (df['buy_sell_ratio'] - 0.5) * 0.25,
                    'orderbook_pressure': df['bid_ask_imbalance'] * 0.3,  # Higher weight - very predictive
                    'basis_pressure': np.sign(df['basis_momentum']) * abs(df['basis_momentum']) * 0.2,
                    'aggression_pressure': df['volume_aggression'] * 0.15,
                    'oi_pressure': (df['oi_ratio'] - 1.0) * 0.1
                }

                enhanced_structure_score = 0.5 + sum(structure_components.values())
                df['enhanced_market_structure_score'] = np.clip(enhanced_structure_score, 0, 1)

            # NEW: Microstructure Efficiency Score (0-1, higher = more efficient market)
            if all(col in df.columns for col in ['orderbook_spread', 'price_impact', 'depth_ratio']):
                # Lower spread + lower impact + higher depth = more efficient
                efficiency_components = []

                # Spread efficiency (inverse)
                if 'orderbook_spread' in df.columns:
                    spread_norm = 1 - np.clip(df['orderbook_spread'] / df['orderbook_spread'].rolling(20).mean(), 0, 1)
                    efficiency_components.append(spread_norm)

                # Impact efficiency (inverse)
                if 'price_impact' in df.columns:
                    impact_norm = 1 - np.clip(abs(df['price_impact']) / (abs(df['price_impact']).rolling(20).mean() + 1e-8), 0, 1)
                    efficiency_components.append(impact_norm)

                # Depth efficiency (direct)
                if 'depth_ratio' in df.columns:
                    depth_norm = np.clip(df['depth_ratio'] / 2, 0, 1)  # Cap at 2x average
                    efficiency_components.append(depth_norm)

                if efficiency_components:
                    df['microstructure_efficiency'] = sum(efficiency_components) / len(efficiency_components)

            # NEW: Institutional Activity Composite
            if all(col in df.columns for col in ['top_account_long_short_ratio', 'trade_aggression', 'basis_momentum']):
                # High institutional activity detection
                institutional_signals = []

                # Smart money positioning
                if 'top_account_long_short_ratio' in df.columns:
                    smart_money_extreme = (abs(df['top_account_long_short_ratio'] - 1.0) > 0.3).astype(float)
                    institutional_signals.append(smart_money_extreme)

                # Institutional trade size
                if 'trade_aggression' in df.columns:
                    large_trades = (df['trade_aggression'] > 0.2).astype(float)
                    institutional_signals.append(large_trades)

                # Institutional futures positioning
                if 'basis_momentum' in df.columns:
                    basis_activity = (abs(df['basis_momentum']) > abs(df['basis_momentum']).rolling(20).std()).astype(float)
                    institutional_signals.append(basis_activity)

                if institutional_signals:
                    df['institutional_activity_composite'] = sum(institutional_signals) / len(institutional_signals)

            # NEW: Market Regime Classifier (ENHANCED with microstructure data)
            if all(col in df.columns for col in ['buy_sell_ratio', 'bid_ask_imbalance', 'basis_momentum', 'vol_aggression_sma']):
                # Enhanced regime classification using microstructure data

                # Strong Bull Regime: Strong buying + bid pressure + positive basis + moderate aggression
                strong_bull = (df['buy_sell_ratio'] > 0.6) & \
                             (df['bid_ask_imbalance'] > 0.1) & \
                             (df['basis_momentum'] > 0) & \
                             (abs(df['vol_aggression_sma']) < 0.3)

                # Strong Bear Regime: Strong selling + ask pressure + negative basis + moderate aggression
                strong_bear = (df['buy_sell_ratio'] < 0.4) & \
                             (df['bid_ask_imbalance'] < -0.1) & \
                             (df['basis_momentum'] < 0) & \
                             (abs(df['vol_aggression_sma']) < 0.3)

                # Volatile Regime: High aggression + wide spreads + volatile basis
                volatile = ((abs(df['vol_aggression_sma']) > 0.4) |
                           (df['orderbook_spread'] > df['orderbook_spread'].rolling(20).mean() * 1.5) |
                           (df['basis_volatility'] > df['basis_volatility'].rolling(20).mean() * 1.5))

                df['enhanced_market_regime'] = np.where(strong_bull, 3,  # Strong Bull
                                             np.where(strong_bear, 0,   # Strong Bear
                                             np.where(volatile, 2, 1))) # Volatile/Neutral

            # NEW: Predictive Signal Strength (Composite of leading indicators)
            leading_indicators = []

            # Orderbook pressure (leading)
            if 'bid_ask_imbalance' in df.columns:
                orderbook_signal = np.clip(df['bid_ask_imbalance'] * 2 + 0.5, 0, 1)
                leading_indicators.append(orderbook_signal * 0.35)

            # Basis momentum (leading)
            if 'basis_momentum' in df.columns:
                basis_signal = np.clip(np.sign(df['basis_momentum']) * abs(df['basis_momentum']) * 10 + 0.5, 0, 1)
                leading_indicators.append(basis_signal * 0.30)

            # Aggressiveness (confirming)
            if 'composite_aggressiveness' in df.columns:
                aggression_signal = df['composite_aggressiveness']
                leading_indicators.append(aggression_signal * 0.20)

            # Options OI (confirming)
            if 'options_oi_ratio' in df.columns:
                options_signal = np.clip(df['options_oi_ratio'] / 2 + 0.5, 0, 1)
                leading_indicators.append(options_signal * 0.15)

            if leading_indicators:
                df['predictive_signal_strength'] = sum(leading_indicators)

            print("✅ Enhanced composite indicators created with microstructure data")

        except Exception as e:
            print(f"⚠️  Error creating enhanced composite indicators: {e}")

        return df

    def _create_enhanced_interaction_features(self, df):
        """Create enhanced interaction features including new microstructure interactions"""
        try:
            # Volume vs Open Interest interaction
            if 'buy_sell_ratio' in df.columns and 'oi_ratio' in df.columns:
                df['volume_oi_alignment'] = (df['buy_sell_ratio'] - 0.5) * (df['oi_ratio'] - 1.0)

            # Smart Money vs Crowd divergence
            if 'top_account_long_short_ratio' in df.columns and 'global_account_long_short_ratio' in df.columns:
                smart_crowd_divergence = abs(df['top_account_long_short_ratio'] - df['global_account_long_short_ratio'])
                df['smart_crowd_divergence'] = smart_crowd_divergence

                # High divergence = potential turning point
                df['high_divergence'] = (smart_crowd_divergence > 0.5).astype(int)

            # Funding Rate vs Liquidations interaction
            if 'funding_rate' in df.columns and 'liquidation_ratio' in df.columns:
                # High funding + high liquidations = market stress
                df['market_stress'] = abs(df['funding_rate']) * df['liquidation_ratio']
                df['extreme_stress'] = (df['market_stress'] > 2.0).astype(int)

            # Price Action vs Market Structure alignment
            if 'returns' in df.columns and 'market_structure_score' in df.columns:
                df['structure_alignment'] = np.sign(df['returns']) * (df['market_structure_score'] - 0.5) * 2
                df['structure_conflict'] = (abs(df['structure_alignment']) < 0.1).astype(int)

            # NEW: Orderbook microstructure interactions
            if all(col in df.columns for col in ['bid_ask_imbalance', 'total_depth', 'orderbook_spread']):
                # Depth vs Imbalance (liquidity under pressure)
                df['depth_imbalance_interaction'] = df['bid_ask_imbalance'] * np.log1p(df['total_depth'])

                # Spread vs Imbalance (market maker positioning)
                df['spread_imbalance_interaction'] = abs(df['bid_ask_imbalance']) * df['orderbook_spread']
                df['tight_spread_imbalance'] = (df['orderbook_spread'] < df['orderbook_spread'].rolling(20).mean() * 0.8) & (abs(df['bid_ask_imbalance']) > 0.1)

                # Liquidity crunch indicator
                df['liquidity_crunch'] = (df['depth_ratio'] < 0.5) & (abs(df['bid_ask_imbalance']) > 0.2)

            # NEW: Futures Basis microstructure interactions
            if all(col in df.columns for col in ['basis_momentum', 'bid_ask_imbalance', 'volume_aggression']):
                # Basis vs Orderbook pressure (arbitrage opportunities)
                df['basis_orderbook_alignment'] = np.sign(df['basis_momentum']) * df['bid_ask_imbalance']
                df['basis_orderbook_divergence'] = abs(df['basis_momentum'] * df['bid_ask_imbalance'])  # High = opportunity

                # Basis vs Aggressiveness (institutional confirmation)
                df['basis_aggression_confirmation'] = np.sign(df['basis_momentum']) * df['volume_aggression']
                df['basis_aggression_divergence'] = (np.sign(df['basis_momentum']) != np.sign(df['volume_aggression'])).astype(int)

            # NEW: Footprint vs Orderbook interactions
            if all(col in df.columns for col in ['volume_aggression', 'trade_aggression', 'bid_ask_imbalance']):
                # Aggressive flow vs orderbook pressure
                df['aggression_orderbook_alignment'] = df['volume_aggression'] * df['bid_ask_imbalance']
                df['aggression_orderbook_resistance'] = (abs(df['volume_aggression']) > 0.3) & (abs(df['bid_ask_imbalance']) < 0.05)  # High aggression meets thin orderbook

                # Trade size vs market depth
                df['trade_size_depth_impact'] = abs(df['trade_aggression']) / (df['depth_ratio'] + 1e-8)
                df['large_trade_low_depth'] = (abs(df['trade_aggression']) > 0.2) & (df['depth_ratio'] < 0.8)

            # NEW: Options OI vs microstructure interactions
            if all(col in df.columns for col in ['total_oi', 'options_oi_ratio', 'basis_momentum', 'volume_aggression']):
                # Options OI vs futures basis (institutional positioning)
                df['options_basis_alignment'] = np.sign(df['options_oi_ratio'] - 1.0) * np.sign(df['basis_momentum'])
                df['options_basis_divergence'] = (np.sign(df['options_oi_ratio'] - 1.0) != np.sign(df['basis_momentum'])).astype(int)

                # Options OI vs spot aggressiveness
                df['options_aggression_confirmation'] = np.sign(df['options_oi_ratio'] - 1.0) * df['volume_aggression']
                df['options_aggression_divergence'] = (np.sign(df['options_oi_ratio'] - 1.0) != np.sign(df['volume_aggression'])).astype(int)

            # NEW: Enhanced liquidation microstructure interactions
            if all(col in df.columns for col in ['liquidation_ratio', 'bid_ask_imbalance', 'depth_ratio']):
                # Liquidations vs orderbook imbalance
                df['liquidation_orderbook_catalyst'] = df['liquidation_ratio'] * abs(df['bid_ask_imbalance'])
                df['liquidation_cascade_risk'] = (df['liquidation_ratio'] > 2.0) & (df['depth_ratio'] < 0.5)  # High liquidations + low depth = cascade risk

            # NEW: Enhanced funding rate microstructure interactions
            if all(col in df.columns for col in ['funding_rate', 'basis_momentum', 'depth_ratio']):
                # Funding vs basis (arbitrage efficiency)
                df['funding_basis_arbitrage'] = abs(df['funding_rate'] - df['basis_momentum'])
                df['funding_basis_inefficiency'] = (df['funding_basis_arbitrage'] > df['funding_basis_arbitrage'].rolling(20).std()).astype(int)

            # NEW: Three-way interactions for complex market conditions
            if all(col in df.columns for col in ['bid_ask_imbalance', 'basis_momentum', 'volume_aggression']):
                # Perfect storm: Orderbook + Basis + Aggression alignment
                momentum_alignment = (np.sign(df['bid_ask_imbalance']) == np.sign(df['basis_momentum'])) & \
                                   (np.sign(df['basis_momentum']) == np.sign(df['volume_aggression']))
                df['perfect_storm_alignment'] = momentum_alignment.astype(int) * abs(df['bid_ask_imbalance'])

                # Confluence signals (multiple indicators pointing same direction)
                confluence_strength = abs(df['bid_ask_imbalance']) * abs(df['basis_momentum']) * abs(df['volume_aggression'])
                df['confluence_strength'] = np.clip(confluence_strength * 10, 0, 1)

            # NEW: Market efficiency vs institutional flow interactions
            if all(col in df.columns for col in ['microstructure_efficiency', 'institutional_activity_composite', 'predictive_signal_strength']):
                # Efficiency degradation during high institutional activity
                df['efficiency_institutional_impact'] = df['institutional_activity_composite'] * (1 - df['microstructure_efficiency'])

                # Predictive signal reliability based on market efficiency
                df['signal_reliability_score'] = df['predictive_signal_strength'] * df['microstructure_efficiency']

            # NEW: Cross-market arbitrage opportunity score
            arbitrage_signals = []
            if all(col in df.columns for col in ['funding_rate', 'close_basis']):
                # Futures-Spot arbitrage
                futures_spread_arbitrage = abs(df['close_basis'] - df['funding_rate']) > (abs(df['close_basis']).rolling(20).std())
                arbitrage_signals.append(futures_spread_arbitrage.astype(int))

            if all(col in df.columns for col in ['bid_ask_imbalance', 'volume_aggression']):
                # Spot-Futures execution arbitrage
                execution_arbitrage = (abs(df['bid_ask_imbalance']) > 0.15) & (abs(df['volume_aggression']) > 0.3)
                arbitrage_signals.append(execution_arbitrage.astype(int))

            if arbitrage_signals:
                df['arbitrage_opportunity_score'] = sum(arbitrage_signals) / len(arbitrage_signals)

            print("✅ Enhanced interaction features created with microstructure data")

        except Exception as e:
            print(f"⚠️  Error creating enhanced interaction features: {e}")

        return df

    def _create_open_interest_features(self, df):
        """Create open interest features (legacy method)"""
        if 'open_interest' in df.columns:
            window = min(20, max(2, len(df) // 2))
            df['oi_sma_20'] = df['open_interest'].rolling(window=window, min_periods=1).mean()
            df['oi_ratio'] = df['open_interest'] / df['oi_sma_20'].replace(0, 1e-8)
            df['oi_change'] = df['open_interest'].pct_change()
            df['oi_volume_ratio'] = df['open_interest'] / (df['volume'] + 1e-8)

        return df

    def _create_liquidation_features(self, df):
        """Create liquidation-based features (legacy method)"""
        if 'total_liquidations' in df.columns:
            df['liquidation_ratio'] = df['total_liquidations'] / df['liquidation_sma_20'].replace(0, 1e-8)
            df['liquidation_volume_ratio'] = df['total_liquidations'] / (df['volume'] + 1e-8)

        return df