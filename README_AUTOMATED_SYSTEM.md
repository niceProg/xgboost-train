# 🤖 Automated Trading System

A complete end-to-end trading signal pipeline with data collection, labeling, model training, and automation.

## 📋 System Overview

The system consists of four main components working together:

### 1. **Signal Collection** (`collect_signals.py`)
- **Runs every 5 minutes** via cron job
- Fetches market data from database
- Calculates 250+ technical indicators
- Stores signals with JSON feature payloads
- Generates BUY/SELL/NEUTRAL signals with confidence scores

### 2. **Signal Labeling** (`label_signals.py`)
- **Runs every 15 minutes** via cron job
- Labels pending signals based on future price movements
- Configurable thresholds for UP/DOWN/FLAT classification
- Updates database with actual price outcomes

### 3. **Model Training** (`train_model.py`)
- **Runs every 4 hours** via cron job
- Loads labeled signals from database
- Trains XGBoost model with time series validation
- Handles small datasets with binary classification fallback
- Saves trained models for prediction

### 4. **System Monitoring** (`monitor_system.py`)
- Real-time health check of all components
- Database connectivity verification
- Signal collection and labeling statistics
- Model training status tracking

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
source .xgboostvenv/bin/activate

# Install dependencies (if needed)
pip install pandas numpy scikit-learn xgboost pymysql python-dotenv matplotlib seaborn joblib ta
```

### 2. Configure Database
```bash
# Ensure .env file has database credentials
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=market_data_db
```

### 3. Setup Automation
```bash
# Install cron jobs
./setup_cron.sh

# Verify installation
crontab -l
```

### 4. Monitor System
```bash
# Run health check
python monitor_system.py

# Monitor logs in real-time
tail -f logs/collect_signals.log &
tail -f logs/label_signals.log &
tail -f logs/train_model.log &
```

## 📊 System Performance

### Current Status
- ✅ **Database Connection**: Working
- ✅ **Cron Jobs**: 3 active schedules
- ✅ **Signal Labeling**: 100% rate (5/5 signals)
- ✅ **Model Training**: Recent model available
- 📈 **Overall Health**: 80%

### Signal Statistics
- **Total Signals**: 5
- **Labeled Signals**: 5 (100%)
- **Pending Signals**: 0
- **Average Price**: $90,153.14
- **Label Distribution**: 4 UP, 1 FLAT

## ⏰ Automation Schedule

| Component | Frequency | Purpose |
|-----------|-----------|---------|
| `collect_signals.py` | Every 5 minutes | Collect new trading signals |
| `label_signals.py` | Every 15 minutes | Label pending signals |
| `train_model.py` | Every 4 hours | Retrain model with new data |
| Log Cleanup | Weekly | Remove old log files |

## 📁 File Structure

```
xgboost-training/
├── collect_signals.py          # Data collection script
├── label_signals.py            # Signal labeling script
├── train_model.py              # Model training script
├── predict_signals.py          # Prediction script (TODO)
├── monitor_system.py           # System health monitor
├── setup_cron.sh               # Cron job installer
├── database.py                 # Database connection manager
├── feature_engineering.py      # Technical indicators
├── env_config.py               # Configuration loader
├── database_schema.sql         # Database schema
├── logs/                       # Log files directory
├── .xgboostvenv/              # Virtual environment
├── xgboost_trading_model_*.joblib # Trained models
└── .env                        # Environment variables
```

## 🔧 Configuration

### Environment Variables (.env)
```env
# Database Configuration
DB_HOST=localhost
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=market_data_db
DB_CHARSET=utf8mb4
DB_CONNECT_TIMEOUT=30
```

### Signal Collection Parameters
- **Symbol**: BTC
- **Pair**: BTCUSDT
- **Interval**: 1h
- **Horizon**: 60 minutes

### Labeling Parameters
- **Price Change Threshold**: 0.5%
- **Label Categories**: UP, DOWN, FLAT
- **Batch Size**: 100 signals

## 📈 Model Performance

### Training Results
- **Model Type**: XGBoost Binary Classifier
- **Accuracy**: 80% (small dataset)
- **Features**: 19 technical indicators
- **Classes**: UP (1), NOT_UP (0)
- **Training Dataset**: 5 signals

### Feature Importance
The model uses technical indicators including:
- Moving Averages (SMA, EMA)
- Momentum indicators (RSI, MACD)
- Volatility measures (ATR, Bollinger Bands)
- Volume features
- Price action features

## 🛠️ Manual Operations

### Collect Signals Manually
```bash
python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60
```

### Label Signals Manually
```bash
python label_signals.py --limit 100 --threshold 0.5
```

### Train Model Manually
```bash
python train_model.py --limit 1000 --no-save
```

### Check System Health
```bash
python monitor_system.py --verbose
```

## 📝 Log Files

All automated operations log to files in the `logs/` directory:

- `collect_signals.log` - Data collection activities
- `label_signals.log` - Signal labeling operations
- `train_model.log` - Model training processes

### Monitoring Logs
```bash
# View recent activity
tail -f logs/collect_signals.log

# Search for errors
grep -i error logs/*.log

# Check system performance
python monitor_system.py --verbose
```

## 🔍 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check database config
   python -c "from env_config import get_database_config; print(get_database_config())"

   # Test connection
   python monitor_system.py --db-only
   ```

2. **Missing Dependencies**
   ```bash
   # Reinstall dependencies
   pip install pandas numpy scikit-learn xgboost pymysql python-dotenv

   # Test import
   python -c "import pandas, numpy, xgboost, pymysql; print('All imports OK')"
   ```

3. **Cron Jobs Not Running**
   ```bash
   # Check cron service
   sudo service cron status

   # List cron jobs
   crontab -l

   # Check system logs
   tail -f /var/log/syslog | grep CRON
   ```

4. **No Recent Signals**
   ```bash
   # Check data availability
   python -c "
   from database import DatabaseManager
   db = DatabaseManager(get_database_config())
   result = db.execute_query('SELECT MAX(time) as latest FROM cg_spot_price_history')
   print('Latest data:', result.iloc[0]['latest'])
   "
   ```

### Recovery Procedures

1. **Reset Cron Jobs**
   ```bash
   # Remove all cron jobs
   crontab -r

   # Reinstall system cron jobs
   ./setup_cron.sh
   ```

2. **Clear Database Signals** (emergency only)
   ```bash
   python -c "
   from database import DatabaseManager
   db = DatabaseManager(get_database_config())
   db.execute_update('DELETE FROM cg_train_dataset WHERE label_status = \"pending\"')
   print('Cleared pending signals')
   "
   ```

## 📋 Maintenance

### Daily
- Monitor system health: `python monitor_system.py`
- Check log files for errors
- Verify signal collection is active

### Weekly
- Review model performance
- Check signal quality and labeling accuracy
- Monitor database storage usage

### Monthly
- Update dependencies if needed
- Review automation schedules
- Backup trained models and configurations

## 🚨 Alerts and Monitoring

Set up alerts for:
- Failed signal collection (no new signals for 2+ hours)
- Database connection failures
- Model training errors
- Disk space issues (logs growing)

### Example Health Check Script
```bash
#!/bin/bash
# daily_health_check.sh
python monitor_system.py | grep -E "(❌|⚠️)" | mail -s "Trading System Alert" your-email@example.com
```

## 🔒 Security Considerations

- Database credentials stored in `.env` file (restricted permissions)
- Log files contain system information (restrict access)
- Cron jobs run with user privileges (avoid sudo unless necessary)
- Virtual environment isolation prevents package conflicts

## 📞 Support

For issues with the automated trading system:

1. Check system health: `python monitor_system.py`
2. Review log files in `logs/` directory
3. Verify database connectivity
4. Check cron job status: `crontab -l`
5. Ensure virtual environment is activated

---

**System Status**: ✅ Operational
**Last Updated**: 2025-11-19
**Version**: 1.0