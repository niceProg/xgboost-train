(.xgboostvenv) root@localhost:/www/wwwroot/xgboost-train# python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60 
✅ Database connection established successfully
🚀 Starting signal collection for BTC BTCUSDT 1h
📅 Horizon: 60 minutes
✅ Table cg_train_dataset ensured
📊 Checking available data...
📅 Available data: 33,578 rows from 2024-01-01 00:00:00 to 2025-11-30 12:00:00
🎯 Processing recent data from 2025-11-28 12:00:00 to 2025-11-30 12:00:00
📊 Loading market data...
📊 Fetching multi-source market data for BTC BTCUSDT 1h
✅ Added open interest data
✅ Added liquidation data
✅ Added taker volume data
✅ Added funding rate data
✅ Added top account ratio data
✅ Added global account ratio data
✅ Added spot orderbook microstructure data
✅ Added futures basis data
⚠️  No futures footprint data available
⚠️  No options OI data available
📊 Loaded comprehensive market data: 18048 rows with 41 features
📊 Data sources: Price, OI, Liquidations, Taker Volume, Funding, Account Ratios, Orderbook, Basis, Footprint, Options OI
📊 Loaded 18,048 rows of market data
🔧 Engineering features...
📊 Creating features for dataset with 18048 rows...
/www/wwwroot/xgboost-train/feature_engineering.py:253: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'returns_skew_{window}'] = df['returns'].rolling(window=window, min_periods=1).skew()
/www/wwwroot/xgboost-train/feature_engineering.py:254: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'returns_kurt_{window}'] = df['returns'].rolling(window=window, min_periods=1).kurt()
/www/wwwroot/xgboost-train/feature_engineering.py:260: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'volume_mean_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:261: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std()
/www/wwwroot/xgboost-train/feature_engineering.py:267: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'high_max_{window}'] = df['high'].rolling(window=window, min_periods=1).max()
/www/wwwroot/xgboost-train/feature_engineering.py:268: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'low_min_{window}'] = df['low'].rolling(window=window, min_periods=1).min()
/www/wwwroot/xgboost-train/feature_engineering.py:271: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'close_position_{window}'] = (df['close'] - df[f'low_min_{window}']) / denominator
🔧 Creating enhanced multi-source market features with new microstructure data...
/www/wwwroot/xgboost-train/feature_engineering.py:295: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_sma'] = df['open_interest'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:296: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_ratio'] = df['open_interest'] / df['oi_sma'].replace(0, 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:298: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_volume_ratio'] = df['open_interest'] / (df['volume'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:299: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_price_ratio'] = df['open_interest'] / (df['close'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:303: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'oi_roc_{w}'] = df['open_interest'].pct_change(w)
/www/wwwroot/xgboost-train/feature_engineering.py:304: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'oi_ratio_sma_{w}'] = df['oi_ratio'].rolling(window=w, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:303: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'oi_roc_{w}'] = df['open_interest'].pct_change(w)
/www/wwwroot/xgboost-train/feature_engineering.py:304: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df[f'oi_ratio_sma_{w}'] = df['oi_ratio'].rolling(window=w, min_periods=1).mean()
✅ Enhanced open interest features added
/www/wwwroot/xgboost-train/feature_engineering.py:312: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_sma'] = df['total_liquidations'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:313: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_ratio'] = df['total_liquidations'] / df['liquidation_sma'].replace(0, 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:314: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_volume_ratio'] = df['total_liquidations'] / (df['volume'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:315: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_price_ratio'] = df['total_liquidations'] / (df['close'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:318: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_spike'] = (df['total_liquidations'] > 2 * df['liquidation_sma']).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:319: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_extreme'] = (df['total_liquidations'] > 3 * df['liquidation_sma']).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:323: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['long_liq_ratio'] = df['aggregated_long_liquidation_usd'] / (df['total_liquidations'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:324: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['short_liq_ratio'] = df['aggregated_short_liquidation_usd'] / (df['total_liquidations'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:325: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liq_imbalance'] = (df['aggregated_long_liquidation_usd'] - df['aggregated_short_liquidation_usd']) / (df['total_liquidations'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:328: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['long_liq_dominated'] = (df['long_liq_ratio'] > 0.7).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:329: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['short_liq_dominated'] = (df['short_liq_ratio'] > 0.7).astype(int)
✅ Enhanced liquidation features added
/www/wwwroot/xgboost-train/feature_engineering.py:337: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['buy_ratio_sma'] = df['buy_sell_ratio'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:338: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['buy_ratio_change'] = df['buy_sell_ratio'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:341: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['strong_buying'] = (df['buy_sell_ratio'] > 0.6).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:342: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['strong_selling'] = (df['buy_sell_ratio'] < 0.4).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:343: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['balanced_market'] = ((df['buy_sell_ratio'] >= 0.4) & (df['buy_sell_ratio'] <= 0.6)).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:347: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['taker_volume_ratio'] = df['total_taker_volume'] / (df['volume'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:348: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['taker_volume_sma'] = df['total_taker_volume'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:349: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['taker_volume_intensity'] = df['total_taker_volume'] / df['taker_volume_sma'].replace(0, 1e-8)
✅ Enhanced taker volume features added
/www/wwwroot/xgboost-train/feature_engineering.py:357: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_sma'] = df['funding_rate'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:358: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_change'] = df['funding_rate'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:359: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_volatility'] = df['funding_rate'].rolling(window=window, min_periods=1).std()
/www/wwwroot/xgboost-train/feature_engineering.py:362: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['high_funding'] = (df['funding_rate'] > 0.01).astype(int)  # > 1%
/www/wwwroot/xgboost-train/feature_engineering.py:363: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['low_funding'] = (df['funding_rate'] < -0.01).astype(int)  # < -1%
/www/wwwroot/xgboost-train/feature_engineering.py:364: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['extreme_high_funding'] = (df['funding_rate'] > 0.02).astype(int)  # > 2%
/www/wwwroot/xgboost-train/feature_engineering.py:365: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['extreme_low_funding'] = (df['funding_rate'] < -0.02).astype(int)  # < -2%
/www/wwwroot/xgboost-train/feature_engineering.py:368: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_trend'] = np.sign(df['funding_rate'] - df['funding_sma'])
/www/wwwroot/xgboost-train/feature_engineering.py:369: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_regime'] = np.where(abs(df['funding_rate']) > 0.005, 1, 0)  # Significant funding
✅ Enhanced funding rate features added
/www/wwwroot/xgboost-train/feature_engineering.py:376: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['top_ratio_sma'] = df['top_account_long_short_ratio'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:377: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['top_ratio_change'] = df['top_account_long_short_ratio'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:380: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_long'] = (df['top_account_long_short_ratio'] > 1.2).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:381: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_short'] = (df['top_account_long_short_ratio'] < 0.8).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:382: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_neutral'] = ((df['top_account_long_short_ratio'] >= 0.8) &
/www/wwwroot/xgboost-train/feature_engineering.py:386: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_extreme_long'] = (df['top_account_long_short_ratio'] > 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:387: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_extreme_short'] = (df['top_account_long_short_ratio'] < 0.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:391: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['price_change'] = df['close'].pct_change()
/www/wwwroot/xgboost-train/feature_engineering.py:392: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_money_divergence'] = np.sign(df['top_account_long_short_ratio'].diff() * -df['price_change'])
✅ Enhanced smart money features added
/www/wwwroot/xgboost-train/feature_engineering.py:399: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['global_ratio_sma'] = df['global_account_long_short_ratio'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:400: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['global_ratio_change'] = df['global_account_long_short_ratio'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:403: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['crowd_long'] = (df['global_account_long_short_ratio'] > 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:404: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['crowd_short'] = (df['global_account_long_short_ratio'] < 0.67).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:405: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['crowd_neutral'] = ((df['global_account_long_short_ratio'] >= 0.67) &
/www/wwwroot/xgboost-train/feature_engineering.py:409: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['crowd_extreme_long'] = (df['global_account_long_short_ratio'] > 2.0).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:410: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['crowd_extreme_short'] = (df['global_account_long_short_ratio'] < 0.5).astype(int)
✅ Enhanced crowd sentiment features added
/www/wwwroot/xgboost-train/feature_engineering.py:419: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['depth_sma'] = df['total_depth'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:420: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['depth_ratio'] = df['total_depth'] / df['depth_sma'].replace(0, 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:421: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['depth_volatility'] = df['total_depth'].rolling(window=window, min_periods=1).std()
/www/wwwroot/xgboost-train/feature_engineering.py:424: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['imbalance_sma'] = df['bid_ask_imbalance'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:425: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['imbchange'] = df['bid_ask_imbalance'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:426: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['strong_bid_pressure'] = (df['bid_ask_imbalance'] > 0.2).astype(int)  # Bids dominate
/www/wwwroot/xgboost-train/feature_engineering.py:427: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['strong_ask_pressure'] = (df['bid_ask_imbalance'] < -0.2).astype(int)  # Asks dominate
/www/wwwroot/xgboost-train/feature_engineering.py:430: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidity_sma'] = df['liquidity_ratio'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:431: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidity_spike'] = (abs(df['liquidity_ratio'] - df['liquidity_sma']) > 0.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:434: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['spread_sma'] = df['orderbook_spread'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:435: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['spread_widening'] = (df['orderbook_spread'] > df['spread_sma'] * 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:436: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['spread_narrowing'] = (df['orderbook_spread'] < df['spread_sma'] * 0.7).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:439: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['orderbook_momentum'] = df['bid_ask_imbalance'].rolling(window=3).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:440: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['orderbook_acceleration'] = df['orderbook_momentum'].diff()
✅ Enhanced orderbook microstructure features added
/www/wwwroot/xgboost-train/feature_engineering.py:449: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_sma'] = df['close_basis'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:450: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_ratio'] = df['close_basis'] / (abs(df['basis_sma']) + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:453: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_momentum_sma'] = df['basis_momentum'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:454: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_acceleration'] = df['basis_momentum'].diff()
/www/wwwroot/xgboost-train/feature_engineering.py:455: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_reversal'] = (df['basis_momentum'] * df['basis_momentum'].shift(1) < 0).astype(int)  # Sign change
/www/wwwroot/xgboost-train/feature_engineering.py:458: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_vol_regime'] = np.where(df['basis_volatility'] > df['basis_volatility'].rolling(window=window).mean() * 1.5, 2, 1)
/www/wwwroot/xgboost-train/feature_engineering.py:459: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_calm_regime'] = (df['basis_volatility'] < df['basis_volatility'].rolling(window=window).mean() * 0.7).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:462: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['wide_basis'] = (abs(df['close_basis']) > abs(df['basis_sma']) * 2).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:463: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['extreme_basis'] = (abs(df['close_basis']) > abs(df['basis_sma']) * 3).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:466: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['contango_regime'] = (df['close_basis'] > 0).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:467: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['backwardation_regime'] = (df['close_basis'] < 0).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:468: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['regime_change'] = df['contango_regime'].diff().fillna(0).abs()  # 1 when regime changes
/www/wwwroot/xgboost-train/feature_engineering.py:472: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_price_divergence'] = np.sign(df['basis_momentum'] * df['returns'])  # Negative = reversal signal
/www/wwwroot/xgboost-train/feature_engineering.py:473: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['divergence_strength'] = abs(df['basis_momentum'] * df['returns'])
✅ Enhanced futures basis arbitrage features added
/www/wwwroot/xgboost-train/feature_engineering.py:482: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['vol_aggression_sma'] = df['volume_aggression'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:483: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['vol_aggression_volatility'] = df['volume_aggression'].rolling(window=window, min_periods=1).std()
/www/wwwroot/xgboost-train/feature_engineering.py:484: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggressive_buying'] = (df['volume_aggression'] > 0.3).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:485: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggressive_selling'] = (df['volume_aggression'] < -0.3).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:488: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['trade_aggression_sma'] = df['trade_aggression'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:489: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['institutional_buying'] = (df['trade_aggression'] > 0.2).astype(int)  # Large trades dominate
/www/wwwroot/xgboost-train/feature_engineering.py:490: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['retail_buying'] = (df['trade_aggression'] < -0.2).astype(int)  # Many small trades
/www/wwwroot/xgboost-train/feature_engineering.py:493: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['price_impact_sma'] = df['price_impact'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:494: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['high_impact_buying'] = (df['price_impact'] > df['price_impact_sma'] * 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:495: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['high_impact_selling'] = (df['price_impact'] < df['price_impact_sma'] * 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:498: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['agg_volume_ratio_sma'] = df['aggressive_volume_ratio'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:499: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggressive_volume_spike'] = (df['aggressive_volume_ratio'] > df['agg_volume_ratio_sma'] * 2).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:511: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['composite_aggressiveness'] = sum(aggression_components) / len(aggression_components)
/www/wwwroot/xgboost-train/feature_engineering.py:514: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggressiveness_momentum'] = df['composite_aggressiveness'].rolling(window=3).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:515: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggressiveness_acceleration'] = df['aggressiveness_momentum'].diff()
✅ Enhanced futures footprint aggressiveness features added
/www/wwwroot/xgboost-train/feature_engineering.py:524: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_oi_sma'] = df['total_oi'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:525: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_oi_ratio'] = df['total_oi'] / df['options_oi_sma'].replace(0, 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:526: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_oi_roc'] = df['total_oi'].pct_change(window)
/www/wwwroot/xgboost-train/feature_engineering.py:529: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_vol_sma'] = df['oi_volatility'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:530: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_vol_ratio'] = df['oi_volatility'] / df['options_vol_sma'].replace(0, 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:531: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_vol_spike'] = (df['oi_volatility'] > df['options_vol_sma'] * 1.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:534: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['exchange_diversification_sma'] = df['exchange_diversification'].rolling(window=window, min_periods=1).mean()
/www/wwwroot/xgboost-train/feature_engineering.py:535: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['high_diversification'] = (df['exchange_diversification'] > 0.6).astype(int)  # Diversified OI = healthier
/www/wwwroot/xgboost-train/feature_engineering.py:538: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_accumulation'] = (df['oi_change'] > 0).rolling(window=window).sum() / window
/www/wwwroot/xgboost-train/feature_engineering.py:539: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_distribution'] = (df['oi_change'] < 0).rolling(window=window).sum() / window
/www/wwwroot/xgboost-train/feature_engineering.py:540: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_trend_strength'] = abs(df['oi_accumulation'] - df['oi_distribution'])
/www/wwwroot/xgboost-train/feature_engineering.py:544: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_spot_correlation'] = np.sign(df['oi_change'] * df['returns'])
/www/wwwroot/xgboost-train/feature_engineering.py:545: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_divergence'] = (df['options_spot_correlation'] < 0).astype(int)  # Divergence = reversal
/www/wwwroot/xgboost-train/feature_engineering.py:548: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['extreme_oi'] = (df['total_oi'] > df['options_oi_sma'] * 2).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:549: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['oi_exhaustion'] = (df['total_oi'] < df['options_oi_sma'] * 0.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:552: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['institutional_options'] = (df['exchange_diversification'] > 0.8) & (df['total_oi'] > df['options_oi_sma'])
✅ Enhanced options exchange OI features added
/www/wwwroot/xgboost-train/feature_engineering.py:585: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['market_structure_score'] = np.clip(structure_score, 0, 1)
/www/wwwroot/xgboost-train/feature_engineering.py:591: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['leverage_cycle_intensity'] = np.clip(leverage_cycle, 0, 1)
/www/wwwroot/xgboost-train/feature_engineering.py:605: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['market_regime'] = np.where(bull_condition, 2, np.where(bear_condition, 0, 1))  # 2=Bull, 1=Neutral, 0=Bear
/www/wwwroot/xgboost-train/feature_engineering.py:611: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['risk_on_indicator'] = volume_oi_trend.astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:625: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['enhanced_market_structure_score'] = np.clip(enhanced_structure_score, 0, 1)
/www/wwwroot/xgboost-train/feature_engineering.py:648: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['microstructure_efficiency'] = sum(efficiency_components) / len(efficiency_components)
/www/wwwroot/xgboost-train/feature_engineering.py:671: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['institutional_activity_composite'] = sum(institutional_signals) / len(institutional_signals)
/www/wwwroot/xgboost-train/feature_engineering.py:694: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['enhanced_market_regime'] = np.where(strong_bull, 3,  # Strong Bull
/www/wwwroot/xgboost-train/feature_engineering.py:722: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['predictive_signal_strength'] = sum(leading_indicators)
✅ Enhanced composite indicators created with microstructure data
/www/wwwroot/xgboost-train/feature_engineering.py:736: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['volume_oi_alignment'] = (df['buy_sell_ratio'] - 0.5) * (df['oi_ratio'] - 1.0)
/www/wwwroot/xgboost-train/feature_engineering.py:741: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['smart_crowd_divergence'] = smart_crowd_divergence
/www/wwwroot/xgboost-train/feature_engineering.py:744: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['high_divergence'] = (smart_crowd_divergence > 0.5).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:749: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['market_stress'] = abs(df['funding_rate']) * df['liquidation_ratio']
/www/wwwroot/xgboost-train/feature_engineering.py:750: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['extreme_stress'] = (df['market_stress'] > 2.0).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:754: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['structure_alignment'] = np.sign(df['returns']) * (df['market_structure_score'] - 0.5) * 2
/www/wwwroot/xgboost-train/feature_engineering.py:755: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['structure_conflict'] = (abs(df['structure_alignment']) < 0.1).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:760: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['depth_imbalance_interaction'] = df['bid_ask_imbalance'] * np.log1p(df['total_depth'])
/www/wwwroot/xgboost-train/feature_engineering.py:763: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['spread_imbalance_interaction'] = abs(df['bid_ask_imbalance']) * df['orderbook_spread']
/www/wwwroot/xgboost-train/feature_engineering.py:764: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['tight_spread_imbalance'] = (df['orderbook_spread'] < df['orderbook_spread'].rolling(20).mean() * 0.8) & (abs(df['bid_ask_imbalance']) > 0.1)
/www/wwwroot/xgboost-train/feature_engineering.py:767: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidity_crunch'] = (df['depth_ratio'] < 0.5) & (abs(df['bid_ask_imbalance']) > 0.2)
/www/wwwroot/xgboost-train/feature_engineering.py:772: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_orderbook_alignment'] = np.sign(df['basis_momentum']) * df['bid_ask_imbalance']
/www/wwwroot/xgboost-train/feature_engineering.py:773: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_orderbook_divergence'] = abs(df['basis_momentum'] * df['bid_ask_imbalance'])  # High = opportunity
/www/wwwroot/xgboost-train/feature_engineering.py:776: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_aggression_confirmation'] = np.sign(df['basis_momentum']) * df['volume_aggression']
/www/wwwroot/xgboost-train/feature_engineering.py:777: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['basis_aggression_divergence'] = (np.sign(df['basis_momentum']) != np.sign(df['volume_aggression'])).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:782: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggression_orderbook_alignment'] = df['volume_aggression'] * df['bid_ask_imbalance']
/www/wwwroot/xgboost-train/feature_engineering.py:783: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['aggression_orderbook_resistance'] = (abs(df['volume_aggression']) > 0.3) & (abs(df['bid_ask_imbalance']) < 0.05)  # High aggression meets thin orderbook
/www/wwwroot/xgboost-train/feature_engineering.py:786: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['trade_size_depth_impact'] = abs(df['trade_aggression']) / (df['depth_ratio'] + 1e-8)
/www/wwwroot/xgboost-train/feature_engineering.py:787: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['large_trade_low_depth'] = (abs(df['trade_aggression']) > 0.2) & (df['depth_ratio'] < 0.8)
/www/wwwroot/xgboost-train/feature_engineering.py:792: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_basis_alignment'] = np.sign(df['options_oi_ratio'] - 1.0) * np.sign(df['basis_momentum'])
/www/wwwroot/xgboost-train/feature_engineering.py:793: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_basis_divergence'] = (np.sign(df['options_oi_ratio'] - 1.0) != np.sign(df['basis_momentum'])).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:796: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_aggression_confirmation'] = np.sign(df['options_oi_ratio'] - 1.0) * df['volume_aggression']
/www/wwwroot/xgboost-train/feature_engineering.py:797: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['options_aggression_divergence'] = (np.sign(df['options_oi_ratio'] - 1.0) != np.sign(df['volume_aggression'])).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:802: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_orderbook_catalyst'] = df['liquidation_ratio'] * abs(df['bid_ask_imbalance'])
/www/wwwroot/xgboost-train/feature_engineering.py:803: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['liquidation_cascade_risk'] = (df['liquidation_ratio'] > 2.0) & (df['depth_ratio'] < 0.5)  # High liquidations + low depth = cascade risk
/www/wwwroot/xgboost-train/feature_engineering.py:808: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_basis_arbitrage'] = abs(df['funding_rate'] - df['basis_momentum'])
/www/wwwroot/xgboost-train/feature_engineering.py:809: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['funding_basis_inefficiency'] = (df['funding_basis_arbitrage'] > df['funding_basis_arbitrage'].rolling(20).std()).astype(int)
/www/wwwroot/xgboost-train/feature_engineering.py:816: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['perfect_storm_alignment'] = momentum_alignment.astype(int) * abs(df['bid_ask_imbalance'])
/www/wwwroot/xgboost-train/feature_engineering.py:820: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['confluence_strength'] = np.clip(confluence_strength * 10, 0, 1)
/www/wwwroot/xgboost-train/feature_engineering.py:825: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['efficiency_institutional_impact'] = df['institutional_activity_composite'] * (1 - df['microstructure_efficiency'])
/www/wwwroot/xgboost-train/feature_engineering.py:828: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['signal_reliability_score'] = df['predictive_signal_strength'] * df['microstructure_efficiency']
/www/wwwroot/xgboost-train/feature_engineering.py:843: PerformanceWarning: DataFrame is highly fragmented.  This is usually the result of calling `frame.insert` many times, which has poor performance.  Consider joining all columns at once using pd.concat(axis=1) instead. To get a de-fragmented frame, use `newframe = frame.copy()`
  df['arbitrage_opportunity_score'] = sum(arbitrage_signals) / len(arbitrage_signals)
✅ Enhanced interaction features created with microstructure data
✅ Enhanced multi-source feature engineering completed
📈 Total features created: 303
🔬 Includes: Orderbook Microstructure, Futures Basis, Footprint Analysis, Options OI
✅ Feature engineering completed. Final shape: (18048, 303)
✅ Feature engineering completed. Final shape: (18048, 303)
✅ Stored signal: BTC BTCUSDT 1h at 2025-11-30 12:00:00
✅ Signal collection completed successfully

(.xgboostvenv) root@localhost:/www/wwwroot/xgboost-train# python label_signals.py --symbol BTC --pair BTCUSDT --interval 1h --threshold 0.3
✅ Database connection established successfully
🏷️  Starting signal labeling for BTC BTCUSDT 1h
📊 Limit: all pending signals
📈 Threshold: ±0.3%
📊 Found 1 pending signals to label for BTC BTCUSDT 1h

🔄 Processing signal ID: 1
   Symbol: BTC BTCUSDT
   Generated: 2025-11-30 12:00:00
   Horizon: 60 minutes
   Price now: 91277.80000000
   ✅ Labeled: FLAT
   📊 Price change: 0.00%
   💰 Price now → future: 91277.80000000 → 91277.8

📊 Labeling Summary:
   ✅ Successfully labeled: 1
   ⏳ Skipped (future not available): 0
   ❌ Errors: 0

📈 Label Statistics:
   labeled  | FLAT   |    1 | 0.00%

(.xgboostvenv) root@localhost:/www/wwwroot/xgboost-train# python train_model.py --symbol BTC --pair BTCUSDT
✅ Database connection established successfully
🎯 Starting Model Training Pipeline
📊 Loaded 1 labeled signals
📊 Found unique labels: ['FLAT']
📊 Small dataset detected (1 unique labels), using binary classification
📊 Binary label distribution: {0: 1}
✅ Final dataset shape: (1, 32)
📊 Label distribution:
label_direction
FLAT    1
Name: count, dtype: int64
🔧 Using 19 features for training
🚀 Training XGBoost model...
📊 Very small dataset detected (1 samples), training on all data
📊 Training set: 1 samples, Validation set: 1 samples
📊 Training labels: {0: 1}
📊 Validation labels: {0: 1}
✅ Validation accuracy: 1.0000

📊 Model Evaluation:
   Overall Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

     CLASS_0       1.00      1.00      1.00         1

    accuracy                           1.00         1
   macro avg       1.00      1.00      1.00         1
weighted avg       1.00      1.00      1.00         1

⚠️  Feature count mismatch: 20 features vs 19 importances

Top 10 Most Important Features:
Empty DataFrame
Columns: []
Index: []
💾 Model saved to output_train/xgboost_trading_model_20251130_215921.joblib
💾 Latest model saved to output_train/latest_model.joblib