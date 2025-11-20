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
        """Create comprehensive features from multiple data sources"""
        print("🔧 Creating multi-source market features...")

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

        # 7. Composite Indicators
        df = self._create_composite_indicators(df)

        # 8. Cross-Asset Interaction Features
        df = self._create_interaction_features(df)

        print("✅ Multi-source feature engineering completed")
        return df

    def _create_composite_indicators(self, df):
        """Create composite indicators from multiple data sources"""
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

            # Risk-On/Risk-Off Indicator
            if 'total_taker_volume' in df.columns and 'open_interest' in df.columns:
                # High volume + rising OI = Risk On
                # Low volume + falling OI = Risk Off
                volume_oi_trend = (df['taker_volume_intensity'] > 1.0) & (df['oi_ratio'] > 1.0)
                df['risk_on_indicator'] = volume_oi_trend.astype(int)

            print("✅ Composite indicators created")

        except Exception as e:
            print(f"⚠️  Error creating composite indicators: {e}")

        return df

    def _create_interaction_features(self, df):
        """Create interaction features between different data sources"""
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

            print("✅ Interaction features created")

        except Exception as e:
            print(f"⚠️  Error creating interaction features: {e}")

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