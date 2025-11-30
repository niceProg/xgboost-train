# Model Output Directory (output_train)

## Overview

All trained XGBoost models are now automatically saved to the `output_train` folder for better organization and easier management.

## Key Features

### ✅ Automatic Organization
- All models save to `output_train/` folder by default
- Timestamped filenames: `xgboost_trading_model_20241130_213745.joblib`
- Latest model always saved as `latest_model.joblib`

### ✅ Easy Model Loading
- Use `latest` keyword to load newest model
- Automatic fallback to `output_train/` folder
- Smart path resolution

### ✅ Model Management
- List all available models
- Custom output directories supported
- Training metadata included

## Usage Examples

### 1. List Available Models
```bash
python train_model.py --list-models
```

### 2. Train Model (auto-saves to output_train)
```bash
python train_model.py --symbol BTC --pair BTCUSDT --limit 1000
```

### 3. Use Latest Model for Prediction
```bash
python predict_signals.py --model latest --symbol BTC --pair BTCUSDT --interval 1h
```

### 4. Backtest with Latest Model
```bash
python backtest.py --model latest --symbol BTC --pair BTCUSDT --interval 1h --start-date 2024-11-01 --end-date 2024-11-30
```

### 5. Use Custom Output Directory
```bash
python train_model.py --output-dir models_2024 --symbol BTC --pair BTCUSDT
```

### 6. Load Specific Model
```bash
python predict_signals.py --model output_train/xgboost_trading_model_20241130_213745.joblib --symbol BTC --pair BTCUSDT --interval 1h
```

## File Structure

```
output_train/
├── latest_model.joblib                              # Always the newest model
├── xgboost_trading_model_20241130_213745.joblib    # Timestamped models
├── xgboost_trading_model_20241130_214512.joblib
└── xgboost_trading_model_20241201_091234.joblib
```

## Model Metadata

Each saved model includes:
- **Model**: Trained XGBoost classifier
- **Feature Columns**: List of all 303 feature names
- **Training Date**: When model was trained
- **Model Version**: Version tracking

## Benefits

1. **Organization**: No more scattered model files
2. **Version Control**: Timestamped filenames track training history
3. **Convenience**: `latest` keyword for easy access
4. **Flexibility**: Custom output directories supported
5. **Management**: Built-in model listing and metadata

## Backward Compatibility

- Existing model files in root directory still work
- Automatic path resolution checks output_train first
- No breaking changes to existing workflows

## Integration

All core files now work seamlessly with output_train:
- `train_model.py`: Saves models to output_train
- `predict_signals.py`: Loads models from output_train
- `backtest.py`: Uses ModelTrainer class (automatically compatible)

## Default Location

```
Default: ./output_train/
Custom:  --output-dir /path/to/custom/folder/
```

The output_train folder is created automatically if it doesn't exist.