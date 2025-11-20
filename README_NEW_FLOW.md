# XGBoost Trading Model - New Data Flow

## 📋 Overview

This trading system follows a new data collection and labeling flow that separates signal generation from model training, making it more robust and scalable.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Collect        │    │     Label        │    │    Train        │
│   Signals        │───▶│    Signals       │───▶│    Model         │
│                 │    │                 │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│cg_signal_dataset │    │cg_signal_dataset │    │  Saved Model    │
│   (pending)      │    │   (labeled)      │    │  (.joblib)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env file
cp .env.example .env
# Edit .env with your database credentials
```

### 2. Create Database Table

```bash
# Execute the schema
mysql -u your_username -p your_database < database_schema.sql
```

### 3. Start Data Collection

```bash
# Collect signals (run every 5 minutes via cron)
python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60
```

### 4. Label Signals

```bash
# Label pending signals (run every 15 minutes via cron)
python label_signals.py --limit 200
```

### 5. Train Model

```bash
# Train model on labeled data
python train_model.py --symbol BTC --pair BTCUSDT --limit 1000
```

### 6. Generate Predictions

```bash
# Generate trading signals
python predict_signals.py --model xgboost_model.joblib --symbol BTC --pair BTCUSDT --interval 1h
```

## 📅 Cron Jobs Setup

### Data Collection (Every 5 minutes)

```bash
# Edit crontab
crontab -e

# Add this line (adjust paths as needed):
*/5 * * * * /path/to/xgboost-training/.venv/bin/python /path/to/xgboost-training/collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60

# For multiple symbols:
*/5 * * * * /path/to/xgboost-training/.venv/bin/python /path/to/xgboost-training/collect_signals.py --symbol ETH --pair ETHUSDT --interval 1h --horizon 60
```

### Signal Labeling (Every 15 minutes)

```bash
# Add to crontab:
*/15 * * * * /path/to/xgboost-training/.venv/bin/python /path/to/xgboost-training/label_signals.py --limit 200
```

## 📊 Database Schema

### cg_signal_dataset Table

| Column | Type | Description |
|--------|------|-------------|
| id | BIGINT | Primary Key |
| symbol | VARCHAR(20) | Trading symbol (BTC, ETH) |
| pair | VARCHAR(20) | Trading pair (BTCUSDT) |
| interval | VARCHAR(10) | Time interval (1h, 4h, 1d) |
| generated_at | TIMESTAMP | Signal generation time |
| horizon_minutes | INT | Future price horizon |
| price_now | DECIMAL(20,8) | Current price |
| features_payload | JSON | All calculated features |
| signal_rule | ENUM | Basic signal (BUY/SELL/NEUTRAL) |
| signal_score | DECIMAL(10,6) | Signal confidence score |
| price_future | DECIMAL(20,8) | Future price (after horizon) |
| label_direction | ENUM | Price direction (UP/DOWN/FLAT) |
| label_magnitude | DECIMAL(10,6) | Price change magnitude |
| label_status | ENUM | Label status (pending/labeled) |
| labeled_at | TIMESTAMP | When signal was labeled |
| created_at | TIMESTAMP | Record creation time |
| updated_at | TIMESTAMP | Record update time |

## 🔧 Scripts Description

### collect_signals.py
- **Purpose**: Collect market data and calculate trading signals
- **Usage**: `python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60`
- **Frequency**: Every 5 minutes
- **Output**: Stores signals in `cg_signal_dataset` table with `label_status='pending'`

### label_signals.py
- **Purpose**: Label pending signals with actual price movements
- **Usage**: `python label_signals.py --limit 200 --threshold 0.5`
- **Frequency**: Every 15 minutes
- **Output**: Updates `label_status='labeled'` with future price and direction

### train_model.py
- **Purpose**: Train XGBoost model on labeled signals
- **Usage**: `python train_model.py --symbol BTC --pair BTCUSDT --limit 1000`
- **Frequency**: As needed (daily/weekly)
- **Output**: Saves trained model as `.joblib` file

### predict_signals.py
- **Purpose**: Generate trading predictions using trained model
- **Usage**: `python predict_signals.py --model model.joblib --symbol BTC --pair BTCUSDT --interval 1h`
- **Frequency**: As needed for trading decisions
- **Output**: Trading signal recommendation

## 📈 Data Flow

1. **Collection Phase**:
   - Fetch market data from database
   - Calculate 250+ technical features
   - Store as `pending` signals

2. **Labeling Phase**:
   - Wait for horizon period to pass
   - Fetch future price
   - Calculate price direction and magnitude
   - Update signal as `labeled`

3. **Training Phase**:
   - Load labeled signals
   - Prepare features and labels
   - Train XGBoost model with cross-validation
   - Save trained model

4. **Prediction Phase**:
   - Load current market data
   - Calculate same features
   - Generate predictions
   - Provide trading recommendations

## 🎯 Features

### Technical Indicators
- Moving Averages (SMA, EMA)
- Momentum (RSI, MACD, Stochastic)
- Volatility (ATR, Bollinger Bands)
- Volume Analysis (VWAP, OBV, VPT)
- Price Patterns (returns, ratios)

### Label Logic
- **UP**: Price moves up > threshold (default 0.5%)
- **DOWN**: Price moves down > threshold
- **FLAT**: Price movement within threshold

### Model Evaluation
- Time series cross-validation
- Accuracy metrics
- Feature importance analysis
- Classification report

## ⚙️ Configuration

### Model Parameters
```python
model_config = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

### Signal Thresholds
- **Collection**: Basic technical rules
- **Labeling**: Price change threshold (default 0.5%)
- **Prediction**: Confidence threshold (default 0.6)

## 🔍 Monitoring

### Check Data Collection
```sql
SELECT symbol, pair, interval, COUNT(*) as pending_count
FROM cg_signal_dataset
WHERE label_status = 'pending'
GROUP BY symbol, pair, interval;
```

### Check Label Statistics
```bash
python label_signals.py --stats
```

### Model Performance
```bash
python train_model.py --symbol BTC --limit 1000 --no-save
```

## 🐛 Troubleshooting

### Common Issues

1. **No data collected**: Check database connection and market_data table
2. **No signals labeled**: Check if horizon period has passed
3. **Model training fails**: Ensure sufficient labeled data (>100 records)
4. **Prediction errors**: Check model file and feature alignment

### Debug Commands

```bash
# Check database connection
python -c "from env_config import test_connection; test_connection()"

# Test data collection
python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60

# Check labeled signals
python label_signals.py --stats

# Verify model loading
python -c "import joblib; model = joblib.load('your_model.joblib'); print('Model loaded successfully')"
```

## 📈 Next Steps

1. **Setup cron jobs** for automated data collection and labeling
2. **Monitor data quality** and ensure consistent labeling
3. **Fine-tune model** parameters for better performance
4. **Add more symbols** and timeframes
5. **Implement backtesting** on trained models
6. **Deploy to production** with real-time trading