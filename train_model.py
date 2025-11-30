#!/usr/bin/env python3
"""
Train XGBoost model using the new cg_train_dataset table.
This script loads labeled signals and trains a trading model.
"""

import argparse
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
from datetime import datetime
import joblib
from typing import Dict
import os

from env_config import get_database_config
from database import DatabaseManager
try:
    from evaluation import PerformanceEvaluator
except ImportError:
    print("⚠️  PerformanceEvaluator not found, continuing without it")
    PerformanceEvaluator = None

class ModelTrainer:
    def __init__(self, output_dir="output_train"):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.output_dir = output_dir
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    def load_labeled_signals(self, symbol: str = None, pair: str = None,
                             min_date: str = None, limit: int = None) -> pd.DataFrame:
        """Load labeled signals from database"""
        try:
            query = """
                SELECT
                    id, symbol, pair, time_interval, generated_at, horizon_minutes,
                    price_now, price_future, label_direction, label_magnitude,
                    features_payload, signal_rule, signal_score
                FROM cg_train_dataset
                WHERE label_status = 'labeled'
                  AND label_direction IS NOT NULL
            """

            params = []
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol)
            if pair:
                query += " AND pair = %s"
                params.append(pair)
            if min_date:
                query += " AND generated_at >= %s"
                params.append(min_date)
            query += " ORDER BY generated_at"

            if limit:
                query += " LIMIT %s"
                params.append(limit)

            df = self.db_manager.execute_query(query, tuple(params) if params else None)

            print(f"📊 Loaded {len(df)} labeled signals")

            if df.empty:
                return pd.DataFrame()

            # Extract features from JSON payload
            features_list = []
            for row in df['features_payload']:
                try:
                    features_json = json.loads(row) if isinstance(row, str) else row

                    # Flatten the nested JSON structure
                    flat_features = {}
                    for category, features in features_json.items():
                        if isinstance(features, dict):
                            for key, value in features.items():
                                flat_features[f"{category}_{key}"] = value
                    features_list.append(flat_features)
                except Exception as e:
                    print(f"⚠️  Error parsing features: {e}")
                    features_list.append({})

            # Create features DataFrame
            features_df = pd.DataFrame(features_list)

            # Add metadata
            metadata_df = df[['id', 'symbol', 'pair', 'time_interval', 'generated_at',
                             'horizon_minutes', 'price_now', 'price_future',
                             'label_direction', 'label_magnitude', 'signal_rule', 'signal_score']]

            # Combine features and metadata
            combined_df = pd.concat([metadata_df.reset_index(drop=True),
                                   features_df.reset_index(drop=True)], axis=1)

            # Create numeric labels
            unique_labels = combined_df['label_direction'].unique()
            print(f"📊 Found unique labels: {unique_labels}")

            if len(unique_labels) < 3:
                print(f"📊 Small dataset detected ({len(unique_labels)} unique labels), using binary classification")
                # Convert to binary: UP=1, everything else=0
                combined_df['label_numeric'] = (combined_df['label_direction'] == 'UP').astype(int)
                print(f"📊 Binary label distribution: {combined_df['label_numeric'].value_counts().to_dict()}")
            else:
                # Full 3-class classification
                label_map = {'UP': 1, 'DOWN': 0, 'FLAT': 2}
                combined_df['label_numeric'] = combined_df['label_direction'].map(label_map)

            # Filter out rows with missing labels
            combined_df = combined_df.dropna(subset=['label_numeric'])

            print(f"✅ Final dataset shape: {combined_df.shape}")
            print(f"📊 Label distribution:")
            print(combined_df['label_direction'].value_counts())

            return combined_df

        except Exception as e:
            print(f"❌ Error loading labeled signals: {e}")
            return pd.DataFrame()

    def prepare_training_data(self, df: pd.DataFrame):
        """Prepare features and labels for training"""
        if df.empty:
            print("❌ No data available for training")
            return None, None, None, None

        # Select numeric feature columns
        exclude_cols = [
            'id', 'symbol', 'pair', 'time_interval', 'generated_at', 'horizon_minutes',
            'price_now', 'price_future', 'label_direction', 'label_magnitude',
            'signal_rule', 'label_numeric'
        ]

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Handle missing values
        feature_df = df[feature_cols].fillna(0).select_dtypes(include=[np.number])

        print(f"🔧 Using {len(feature_df.columns)} features for training")

        # Prepare X and y
        X = feature_df
        y = df['label_numeric']

        # Convert to appropriate dtypes
        X = X.astype(np.float32)
        y = y.astype(int)

        return X, y, feature_cols, df

    def train_model(self, X, y, config: Dict):
        """Train XGBoost model"""
        print("🚀 Training XGBoost model...")

        # For very small datasets, train on all data without validation split
        if len(X) < 10:
            print(f"📊 Very small dataset detected ({len(X)} samples), training on all data")
            X_train, y_train = X, y
            X_val, y_val = X, y  # Use same data for evaluation
        elif len(X) < 20:
            print(f"📊 Small dataset detected ({len(X)} samples), using simple train/validation split")
            # Use stratification only if we have enough samples per class
            use_stratify = len(y) >= 6 and y.nunique() >= 2 and (y.value_counts().min() >= 2)
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y if use_stratify else None
            )
        else:
            # Time series split for larger datasets
            n_splits = min(5, len(X) // 10)
            if n_splits < 2:
                n_splits = 2

            tscv = TimeSeriesSplit(n_splits=n_splits)
            train_idx, val_idx = next(tscv.split(X))
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        print(f"📊 Training set: {len(X_train)} samples, Validation set: {len(X_val)} samples")
        print(f"📊 Training labels: {y_train.value_counts().to_dict()}")
        print(f"📊 Validation labels: {y_val.value_counts().to_dict()}")

        # For very small datasets, use smaller model and no early stopping
        if len(X) < 10:
            model = xgb.XGBClassifier(
                n_estimators=min(100, config.get('n_estimators', 1000)),
                max_depth=min(3, config.get('max_depth', 6)),
                learning_rate=config.get('learning_rate', 0.1),  # Higher learning rate for small data
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0,
                reg_lambda=1,
                min_child_weight=1,
                random_state=42,
                n_jobs=-1,
                verbose=False
            )

            model.fit(X_train, y_train)
        else:
            # Determine number of classes dynamically
            n_classes = len(np.unique(y_train))
            print(f"📊 Training {n_classes}-class classifier")

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=n_classes,
                n_estimators=config.get('n_estimators', 1000),
                max_depth=config.get('max_depth', 6),
                learning_rate=config.get('learning_rate', 0.01),
                subsample=config.get('subsample', 0.8),
                colsample_bytree=config.get('colsample_bytree', 0.8),
                gamma=config.get('gamma', 0.1),
                reg_alpha=config.get('reg_alpha', 0.1),
                reg_lambda=config.get('reg_lambda', 1.0),
                min_child_weight=config.get('min_child_weight', 1),
                eval_metric='mlogloss',
                early_stopping_rounds=50,
                random_state=42,
                n_jobs=-1,
                verbose=False
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

        val_pred = model.predict(X_val)
        score = accuracy_score(y_val, val_pred)

        print(f"✅ Validation accuracy: {score:.4f}")
        return model

    def evaluate_model(self, model, X, y, feature_cols):
        """Evaluate model performance"""
        print("\n📊 Model Evaluation:")

        # Predictions
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)

        # Accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"   Overall Accuracy: {accuracy:.4f}")

        # Classification report
        print("\nClassification Report:")
        unique_labels = sorted(y.unique())
        if len(unique_labels) == 3:
            target_names = ['DOWN', 'UP', 'FLAT']
        elif len(unique_labels) == 2:
            if 0 in unique_labels and 1 in unique_labels:
                target_names = ['NOT_UP', 'UP']  # Binary classification
            else:
                target_names = [f'CLASS_{i}' for i in unique_labels]
        else:
            target_names = [f'CLASS_{i}' for i in unique_labels]

        print(classification_report(y, y_pred, target_names=target_names, labels=unique_labels))

        # Feature importance
        if len(feature_cols) == len(model.feature_importances_):
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            print(f"⚠️  Feature count mismatch: {len(feature_cols)} features vs {len(model.feature_importances_)} importances")
            feature_importance = pd.DataFrame()

        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))

        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }

    def save_model(self, model, feature_cols, model_name: str = None):
        """Save trained model to output folder"""
        if model_name is None:
            model_name = f"xgboost_trading_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"

        # Add output directory prefix if not already present
        if not model_name.startswith(self.output_dir + "/") and not os.path.dirname(model_name):
            model_name = os.path.join(self.output_dir, model_name)

        model_data = {
            'model': model,
            'feature_cols': feature_cols,
            'training_date': datetime.now().isoformat(),
            'model_version': '1.0'
        }

        joblib.dump(model_data, model_name)
        print(f"💾 Model saved to {model_name}")

        # Also save a copy with a generic name for easy access
        latest_model = os.path.join(self.output_dir, "latest_model.joblib")
        joblib.dump(model_data, latest_model)
        print(f"💾 Latest model saved to {latest_model}")

    def load_model(self, model_path: str):
        """Load trained model"""
        try:
            # If just "latest" is specified, load from output folder
            if model_path == "latest":
                model_path = os.path.join(self.output_dir, "latest_model.joblib")
            # If model_path doesn't include output directory, check there first
            elif not model_path.startswith(self.output_dir + "/") and not model_path.startswith("./" + self.output_dir + "/") and not os.path.dirname(model_path):
                potential_path = os.path.join(self.output_dir, model_path)
                if os.path.exists(potential_path):
                    model_path = potential_path
                else:
                    # Also try with .joblib extension
                    if not model_path.endswith('.joblib'):
                        potential_path = os.path.join(self.output_dir, model_path + '.joblib')
                        if os.path.exists(potential_path):
                            model_path = potential_path

            model_data = joblib.load(model_path)

            # Handle different model formats
            if isinstance(model_data, dict) and 'model' in model_data:
                # New format with metadata
                model = model_data['model']
                feature_cols = model_data.get('feature_cols', [])
                print(f"✅ Model loaded from {model_path}")
                print(f"📊 Features: {len(feature_cols)}")
                if 'training_date' in model_data:
                    print(f"📅 Training Date: {model_data['training_date']}")
                if 'model_version' in model_data:
                    print(f"🔢 Version: {model_data['model_version']}")
                return model
            else:
                # Direct model format
                print(f"✅ Model loaded from {model_path}")
                return model_data

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def list_available_models(self):
        """List all available trained models in output folder"""
        if not os.path.exists(self.output_dir):
            print(f"❌ Output directory '{self.output_dir}' does not exist")
            return []

        models = []
        for file in os.listdir(self.output_dir):
            if file.endswith('.joblib'):
                model_path = os.path.join(self.output_dir, file)
                try:
                    # Get file modification time
                    mtime = os.path.getmtime(model_path)
                    mod_time = datetime.fromtimestamp(mtime)
                    models.append({
                        'file': file,
                        'path': model_path,
                        'modified': mod_time,
                        'size': os.path.getsize(model_path)
                    })
                except:
                    continue

        # Sort by modification time (newest first)
        models.sort(key=lambda x: x['modified'], reverse=True)

        if models:
            print(f"📁 Available models in {self.output_dir}:")
            for i, model in enumerate(models, 1):
                status = "🔥 LATEST" if model['file'] == 'latest_model.joblib' else f"  {i}."
                print(f"   {status} {model['file']}")
                print(f"       Modified: {model['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"       Size: {model['size']:,} bytes")
        else:
            print(f"❌ No models found in {self.output_dir}")

        return models

    def train_pipeline(self, symbol: str = None, pair: str = None, min_date: str = None,
                      limit: int = None, save_model: bool = True):
        """Complete training pipeline"""
        print("🎯 Starting Model Training Pipeline")

        # Load data
        df = self.load_labeled_signals(symbol, pair, min_date, limit)

        if df.empty:
            print("❌ No labeled data available for training")
            return

        # Prepare training data
        X, y, feature_cols, metadata = self.prepare_training_data(df)

        if X is None:
            print("❌ Failed to prepare training data")
            return

        # Model configuration
        model_config = {
            'n_estimators': 1000,
            'max_depth': 6,
            'learning_rate': 0.01,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_weight': 1
        }

        # Train model
        model = self.train_model(X, y, model_config)

        # Evaluate
        eval_results = self.evaluate_model(model, X, y, feature_cols)

        # Save model
        if save_model:
            self.save_model(model, feature_cols)

        return model, eval_results

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'db_manager') and self.db_manager:
            self.db_manager.close()

def main():
    parser = argparse.ArgumentParser(description='Train XGBoost model on labeled signals')
    parser.add_argument('--symbol', help='Filter by symbol (e.g., BTC)')
    parser.add_argument('--pair', help='Filter by pair (e.g., BTCUSDT)')
    parser.add_argument('--min-date', help='Minimum date (YYYY-MM-DD format)')
    parser.add_argument('--limit', type=int, help='Limit number of signals to use')
    parser.add_argument('--no-save', action='store_true', help='Don\'t save the model')
    parser.add_argument('--list-models', action='store_true', help='List all available models in output_train folder')
    parser.add_argument('--output-dir', default='output_train', help='Output directory for models (default: output_train)')

    args = parser.parse_args()

    try:
        trainer = ModelTrainer(output_dir=args.output_dir)

        # If listing models, just list them and exit
        if args.list_models:
            trainer.list_available_models()
            return

        trainer.train_pipeline(
            symbol=args.symbol,
            pair=args.pair,
            min_date=args.min_date,
            limit=args.limit,
            save_model=not args.no_save
        )

    except KeyboardInterrupt:
        print("\n👋 Training interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()