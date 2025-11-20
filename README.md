# XGBoost Algorithmic Trading Model

A comprehensive XGBoost-based algorithmic trading system with advanced feature engineering, risk management, and performance evaluation designed to achieve high win rates and consistent returns.

## Features

### 🚀 High Performance Targets
- **CAGR**: ≥ 50% annual return
- **Max Drawdown**: ≤ 15–25%
- **Sharpe Ratio**: 1.2 – 1.6
- **Sortino Ratio**: 2.0 – 3.0
- **Win Rate**: ≥ 70%

### 📊 Advanced Feature Engineering
- **Technical Indicators**: 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)
- **Volume Analysis**: Volume ratios, VWAP, OBV, VPT
- **Volatility Metrics**: ATR, rolling volatility, price acceleration
- **Market Microstructure**: Open interest, liquidation analysis
- **Time Features**: Session indicators, cyclical encoding
- **Lag & Rolling Features**: Multi-period lag and rolling statistics

### 🛡️ Risk Management
- **Dynamic Position Sizing**: Based on volatility and signal strength
- **Entry Filters**: Volume, volatility, RSI, session, liquidation filters
- **Exit Conditions**: Stop loss, take profit, confidence drop exits
- **Portfolio Risk**: Correlation limits, maximum exposure controls

### 📈 Comprehensive Backtesting
- **Realistic Simulation**: Transaction costs, slippage, market impact
- **Performance Metrics**: 20+ performance indicators
- **Risk Metrics**: VaR, CVaR, drawdown analysis
- **Trade Analysis**: Win rate, profit factor, consecutive wins/losses

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd xgboost-training
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Database Setup

1. Set up MySQL database:
```sql
CREATE DATABASE market_data_db;
```

2. Update database configuration in `.env` file:
```env
DB_HOST=your_host
DB_PORT=3306
DB_USER=your_username
DB_PASSWORD=your_password
DB_NAME=market_data_db
```

Or use the provided `.env` file and update the credentials.

3. Create tables (automatically done on first run):
```python
from database import DatabaseManager
from config import get_config

db = DatabaseManager(get_config('database'))
db.create_tables()
```

## Quick Start

### 1. Generate Sample Data (Optional)
```python
from data_generator import DataGenerator

generator = DataGenerator()
generator.generate_sample_data(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2024-01-01',
    save_to_db=True
)
```

### 2. Train and Evaluate Model
```python
from main import XGBoostTradingModel
from config import get_config

# Initialize model
model = XGBoostTradingModel(get_config())

# Train and evaluate
results = model.train_and_evaluate(
    symbol='BTCUSDT',
    timeframe='1h',
    start_date='2023-01-01',
    end_date='2024-01-01'
)

trained_model, backtest_results, metrics = results
```

### 3. Generate Performance Report
```python
from evaluation import PerformanceEvaluator

evaluator = PerformanceEvaluator()

# Generate text report
report = evaluator.generate_performance_report(metrics, save_path='performance_report.txt')
print(report)

# Generate plots
evaluator.plot_performance(backtest_results, save_path='performance_plots.png')
```

## Architecture

### 📁 Project Structure
```
xgboost-training/
├── main.py                 # Main training and evaluation pipeline
├── config.py              # Configuration settings
├── database.py            # Database integration with pymysql
├── feature_engineering.py # Comprehensive feature engineering
├── trading_rules.py       # Entry/exit filters and risk management
├── backtesting.py         # Realistic backtesting engine
├── evaluation.py          # Performance metrics and reporting
├── data_generator.py      # Sample data generation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

### 🔄 Trading Pipeline

```
Raw Market Data → Feature Engineering → XGBoost Model → Predictions
        ↓                              ↓              ↓
   Entry Filters ← Trading Rules ← Signal Generation
        ↓                              ↓              ↓
   Backtest Engine ← Risk Management ← Position Sizing
        ↓                              ↓              ↓
   Performance Evaluation ← Report Generation ← Model Validation
```

## Configuration

### Model Parameters
```python
MODEL_CONFIG = {
    'n_estimators': 1000,
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'early_stopping_rounds': 50
}
```

### Trading Parameters
```python
TRADING_CONFIG = {
    'signal_threshold': 0.6,        # Minimum signal probability
    'position_size': 0.1,           # 10% of capital per trade
    'stop_loss': 0.02,              # 2% stop loss
    'take_profit': 0.06,            # 6% take profit
    'transaction_cost': 0.001       # 0.1% transaction cost
}
```

### Entry Filters
```python
ENTRY_FILTERS_CONFIG = {
    'min_volume_ratio': 1.2,        # Volume must be 20% above average
    'min_volatility': 0.01,         # Minimum volatility
    'rsi_entry_range': [30, 70],    # RSI range for entries
    'vwap_threshold': 0.02          # Distance from VWAP threshold
}
```

## Advanced Features

### Dynamic Position Sizing
- Adjusts position size based on:
  - Signal probability/confidence
  - Market volatility
  - Current portfolio exposure
  - Correlation with existing positions

### Multi-Layer Entry Filters
1. **Volume Filter**: Ensures sufficient liquidity
2. **Volatility Filter**: Avoids extremely low/high volatility periods
3. **RSI Filter**: Prevents entries at extreme overbought/oversold levels
4. **VWAP Filter**: Ensures price action aligns with volume-weighted average
5. **Session Filter**: Trades only during high-liquidity sessions
6. **Liquidation Filter**: Avoids extreme liquidation events
7. **Open Interest Filter**: Considers derivatives market sentiment

### Smart Exit Conditions
1. **Stop Loss**: Dynamic based on ATR and volatility
2. **Take Profit**: Adjusted for signal strength and market conditions
3. **Confidence Drop**: Exit when signal probability drops below threshold
4. **Volume Anomaly**: Exit during unusual volume patterns

## Performance Metrics

The system evaluates 30+ performance metrics across 4 categories:

### Return Metrics
- Total Return, CAGR, Monthly/Quarterly Returns
- Risk-adjusted returns (Sharpe, Sortino, Calmar)

### Risk Metrics
- Maximum Drawdown, Average Drawdown
- Value at Risk (VaR), Conditional VaR
- Volatility, Beta, Correlation

### Trade Metrics
- Win Rate, Profit Factor, Average Win/Loss
- Consecutive Wins/Losses, Trade Duration
- Best/Worst Trade, Recovery Factor

### Consistency Metrics
- Monthly/Quarterly Win Rates
- Positive periods ratio
- Sterling Ratio, consistency scores

## Usage Examples

### Custom Configuration
```python
# Update trading parameters
from config import update_config

update_config('trading', {
    'position_size': 0.15,  # Increase position size
    'stop_loss': 0.015,     # Tighter stop loss
    'signal_threshold': 0.7 # Higher signal threshold
})
```

### Feature Selection
```python
# Train model and get feature importance
model.train_model(df)

# Get top features
importance_df = model.feature_engineer.get_feature_importance_ranking(
    model.model, feature_cols
)

# Select top 50 features
df_selected = model.feature_engineer.select_top_features(
    df, importance_df, n_features=50
)
```

### Custom Backtesting
```python
from backtesting import BacktestEngine

# Initialize backtest engine with custom parameters
backtest_config = {
    'initial_capital': 50000,
    'position_size': 0.05,
    'stop_loss': 0.025,
    'take_profit': 0.08
}

engine = BacktestEngine(backtest_config)
results = engine.run_backtest(df_signals)
```

## Monitoring and Alerts

The system includes comprehensive monitoring:

### Performance Alerts
- Maximum drawdown exceeded
- Consecutive losses threshold
- Daily/monthly loss limits
- Performance degradation

### Logging
- Detailed trade logging
- Performance metrics tracking
- Error and warning logs
- Model prediction logging

## Best Practices

### Data Quality
1. Ensure clean, continuous price data
2. Handle missing values appropriately
3. Validate data ranges and outliers
4. Maintain sufficient history (1000+ data points)

### Model Training
1. Use time series cross-validation
2. Avoid look-ahead bias
3. Implement early stopping
4. Regularly retrain with new data

### Risk Management
1. Never risk more than 1-2% per trade
2. Diversify across uncorrelated assets
3. Use dynamic position sizing
4. Implement maximum portfolio exposure limits

### Performance Evaluation
1. Use out-of-sample testing
2. Consider transaction costs and slippage
3. Evaluate multiple metrics, not just returns
4. Monitor consistency over time

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading financial markets involves substantial risk, and you should not trade with money you cannot afford to lose. Past performance is not indicative of future results. Always conduct thorough research and consider consulting with financial professionals before making investment decisions.

## Support

For questions, issues, or contributions:
- Create an issue on GitHub
- Join our Discord community
- Email: support@trading-model.com

---

**Happy Trading! 🚀**