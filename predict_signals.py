#!/usr/bin/env python3
"""
Use trained XGBoost model to predict trading signals.
This script loads a trained model and generates predictions on current data.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, Any

from env_config import get_database_config
from database import DatabaseManager
from feature_engineering import FeatureEngineer

class SignalPredictor:
    def __init__(self, model_path: str):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.model_path = model_path
        self.model = None
        self.feature_cols = None
        self._load_model()

    def _load_model(self):
        """Load trained model"""
        try:
            model_data = joblib.load(self.model_path)
            self.model = model_data['model']
            self.feature_cols = model_data['feature_cols']
            self.training_date = model_data.get('training_date', 'Unknown')
            print(f"✅ Model loaded from {self.model_path}")
            print(f"📅 Trained on: {self.training_date}")
            print(f"🔧 Features: {len(self.feature_cols)}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)

    def get_current_data(self, symbol: str, pair: str, interval: str,
                        hours_back: int = 48) -> pd.DataFrame:
        """Get comprehensive current market data from multiple sources"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            print(f"📊 Fetching multi-source current data for {symbol} {pair} {interval}")

            # Import SignalCollector to reuse multi-source data collection
            from collect_signals import SignalCollector
            collector = SignalCollector()

            # Use the same multi-source data collection method
            df = collector.get_market_data(symbol, pair, interval, start_time, end_time)

            if df.empty:
                print(f"⚠️  No current data found for {pair} {interval}")
                return pd.DataFrame()

            print(f"📊 Loaded {len(df)} rows of current multi-source data")
            return df

        except Exception as e:
            print(f"❌ Error getting current data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for prediction"""
        try:
            feature_engineer = FeatureEngineer()
            df_features = feature_engineer.create_all_features(df)

            if df_features.empty:
                print("❌ Failed to create features")
                return pd.DataFrame()

            # Use only the latest row for prediction
            latest_features = df_features.iloc[-1:].copy()

            return latest_features

        except Exception as e:
            print(f"❌ Error preparing features: {e}")
            return pd.DataFrame()

    def flatten_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Flatten features to match training format"""
        try:
            # Create flattened feature dict
            feature_dict = {}

            # Technical indicators
            technical_features = ['sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                                 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
                                 'volume_ratio', 'volatility_20']

            for feature in technical_features:
                if feature in features_df.columns:
                    feature_dict[f'technical_indicators_{feature}'] = features_df[feature].iloc[0]
                else:
                    feature_dict[f'technical_indicators_{feature}'] = 0

            # Price features
            price_features = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio']

            for feature in price_features:
                if feature in features_df.columns:
                    feature_dict[f'price_features_{feature}'] = features_df[feature].iloc[0]
                else:
                    feature_dict[f'price_features_{feature}'] = 0

            # Time features (use current timestamp)
            current_time = features_df.index[0]
            time_features = ['hour', 'day_of_week', 'month']

            for feature in time_features:
                if feature == 'hour':
                    value = current_time.hour
                elif feature == 'day_of_week':
                    value = current_time.weekday()
                elif feature == 'month':
                    value = current_time.month
                else:
                    value = 0

                feature_dict[f'time_features_{feature}'] = value

            # Create flattened feature dict with all expected features
            feature_dict = {}

            # Add signal_score first (model expects this as first feature)
            if 'signal_score' in self.feature_cols:
                # Calculate a simple signal score based on current price action
                current_price = features_df['close'].iloc[0]
                prev_price = features_df['close'].iloc[-2] if len(features_df) > 1 else current_price
                price_change = (current_price - prev_price) / prev_price
                # Simple signal score: 0.5 for neutral, adjusted by price change
                signal_score = np.clip(0.5 + price_change * 5, 0.0, 1.0)  # Scale price change to 0-1 range
                feature_dict['signal_score'] = signal_score

            # Technical indicators
            technical_features = ['sma_5', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                                 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr',
                                 'volume_ratio', 'volatility_20']

            for feature in technical_features:
                feature_name = f'technical_indicators_{feature}'
                if feature in features_df.columns:
                    feature_dict[feature_name] = features_df[feature].iloc[0]
                elif feature_name in self.feature_cols:
                    feature_dict[feature_name] = 0

            # Price features
            price_features = ['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio']

            for feature in price_features:
                feature_name = f'price_features_{feature}'
                if feature in features_df.columns:
                    feature_dict[feature_name] = features_df[feature].iloc[0]
                elif feature_name in self.feature_cols:
                    feature_dict[feature_name] = 0

            # Time features (use current timestamp)
            current_time = features_df.index[0]
            time_features = ['hour', 'day_of_week', 'month']

            for feature in time_features:
                feature_name = f'time_features_{feature}'
                if feature == 'hour':
                    value = current_time.hour
                elif feature == 'day_of_week':
                    value = current_time.weekday()
                elif feature == 'month':
                    value = current_time.month
                else:
                    value = 0

                feature_dict[feature_name] = value

            # Convert to DataFrame with exact feature order
            flattened_df = pd.DataFrame([feature_dict])

            # Ensure all expected columns exist and are in correct order
            for col in self.feature_cols:
                if col not in flattened_df.columns:
                    flattened_df[col] = 0

            # Use the actual XGBoost model feature names, not the stored feature_cols
            if hasattr(self.model, 'feature_names_in_'):
                actual_feature_names = list(self.model.feature_names_in_)
            else:
                actual_feature_names = self.feature_cols

            # Select only the columns used during training in the correct order
            flattened_df = flattened_df[actual_feature_names]

            return flattened_df

        except Exception as e:
            print(f"❌ Error flattening features: {e}")
            return pd.DataFrame()

    def predict_signal(self, symbol: str, pair: str, interval: str, threshold: float = 0.6):
        """Generate trading signal prediction"""
        print(f"🎯 Generating signal prediction for {symbol} {pair} {interval}")

        # Get current data
        df = self.get_current_data(symbol, pair, interval)

        if df.empty:
            print("❌ No data available for prediction")
            return None

        # Prepare features
        features_df = self.prepare_features(df)

        if features_df.empty:
            print("❌ Failed to prepare features")
            return None

        # Flatten features
        flattened_features = self.flatten_features(features_df)

        if flattened_features.empty:
            print("❌ Failed to flatten features")
            return None

        try:
            # Make prediction
            probabilities = self.model.predict_proba(flattened_features)
            prediction = self.model.predict(flattened_features)[0]

            # Get class probabilities
            class_probabilities = probabilities[0]

            # Determine if binary or multi-class classification
            if len(class_probabilities) == 2:
                # Binary classification (UP vs NOT_UP)
                class_names = ['NOT_UP', 'UP']
                pred_class = int(prediction)
                pred_label = class_names[pred_class] if 0 <= pred_class < len(class_names) else 'UNKNOWN'
            else:
                # Multi-class classification
                class_names = ['DOWN', 'UP', 'FLAT']
                pred_class = int(prediction)
                pred_label = class_names[pred_class] if 0 <= pred_class < len(class_names) else 'UNKNOWN'

            # Calculate confidence
            max_prob = np.max(class_probabilities)

            print(f"\n📊 Prediction Results:")
            print(f"   🎯 Predicted Signal: {pred_label}")
            print(f"   📈 Confidence: {max_prob:.3f}")
            print(f"   💰 Current Price: {features_df['close'].iloc[0]}")
            print(f"   🕐 Timestamp: {features_df.index[0]}")

            # Show probability distribution
            print(f"\n📊 Class Probabilities:")
            for i, (cls, prob) in enumerate(zip(class_names, class_probabilities)):
                status = "✅" if i == pred_class else "  "
                print(f"   {status} {cls:6}: {prob:.3f}")

            # Generate trading recommendation
            if max_prob >= threshold:
                if pred_label == 'UP':
                    recommendation = "STRONG BUY"
                elif pred_label == 'DOWN':
                    recommendation = "STRONG SELL"
                elif pred_label == 'NOT_UP':
                    recommendation = "HOLD/NO TRADE"
                else:
                    recommendation = "HOLD"
            else:
                recommendation = "WAIT FOR CONFIRMATION"

            print(f"\n🎯 Trading Recommendation: {recommendation}")

            # Create detailed result
            result = {
                'symbol': symbol,
                'pair': pair,
                'interval': interval,
                'timestamp': features_df.index[0].isoformat(),
                'current_price': float(features_df['close'].iloc[0]),
                'prediction': pred_label,
                'confidence': float(max_prob),
                'probabilities': {
                    class_names[i]: float(class_probabilities[i])
                    for i in range(len(class_names))
                },
                'recommendation': recommendation,
                'threshold_used': threshold
            }

            return result

        except Exception as e:
            print(f"❌ Error making prediction: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_prediction(self, prediction: Dict[str, Any], table_name: str = 'signal_predictions'):
        """Save prediction to database"""
        try:
            # Create predictions table if it doesn't exist
            table_schema_sql = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id BIGINT AUTO_INCREMENT PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    pair VARCHAR(20) NOT NULL,
                    time_interval VARCHAR(10) NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    current_price DECIMAL(20, 8),
                    prediction VARCHAR(20),
                    confidence DECIMAL(10, 6),
                    probabilities JSON,
                    recommendation VARCHAR(50),
                    threshold_used DECIMAL(5, 3),
                    model_file VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    INDEX idx_symbol_pair_interval (symbol, pair, time_interval),
                    INDEX idx_timestamp (timestamp)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """

            self.db_manager.execute_update(table_schema_sql)

            # Insert prediction
            insert_query = f"""
                INSERT INTO {table_name} (
                    symbol, pair, time_interval, timestamp, current_price,
                    prediction, confidence, probabilities, recommendation,
                    threshold_used, model_file
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """

            self.db_manager.execute_update(insert_query, (
                prediction['symbol'], prediction['pair'], prediction['interval'],
                prediction['timestamp'], prediction['current_price'],
                prediction['prediction'], prediction['confidence'],
                json.dumps(prediction['probabilities']), prediction['recommendation'],
                prediction['threshold_used'], self.model_path
            ))

            print(f"💾 Prediction saved to database")

        except Exception as e:
            print(f"❌ Error saving prediction: {e}")

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

def main():
    parser = argparse.ArgumentParser(description='Generate trading signal predictions')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTC)')
    parser.add_argument('--pair', required=True, help='Trading pair (e.g., BTCUSDT)')
    parser.add_argument('--interval', required=True, help='Time interval (e.g., 1h, 4h)')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Confidence threshold for recommendations (default: 0.6)')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save prediction to database')

    args = parser.parse_args()

    try:
        predictor = SignalPredictor(args.model)
        prediction = predictor.predict_signal(
            args.symbol, args.pair, args.interval, args.threshold
        )

        if prediction and not args.no_save:
            predictor.save_prediction(prediction)

    except KeyboardInterrupt:
        print("\n👋 Prediction interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()