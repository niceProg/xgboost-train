# Manual Fix Instructions for Backtest Errors

## 🚨 You're Getting Multiple Errors That Need Fixing

The errors indicate several issues in your backtest.py and evaluation.py files:

## 📋 **Errors You're Seeing:**

1. **`'BacktestEngine' object has no attribute 'backtest_config'`**
2. **`KeyError: 'monthly_return'`**
3. **`KeyError: 'volatility'`**
4. **`KeyError: 'period_start'`** (this might still be an issue)

## 🔧 **QUICK FIX - Step by Step:**

### **Step 1: Run the Comprehensive Fix Script**

```bash
(.xgboostvenv) python comprehensive_fix.py
```

This should fix all the issues automatically.

### **Step 2: If Auto-Fix Fails, Apply Manual Fixes**

#### **FIX A: Add backtest_config to BacktestEngine class**

In `backtest.py`, find the `__init__` method and add:

```python
def __init__(self):
    # ... existing code ...
    self.db_manager = DatabaseManager(self.db_config)
    self.backtest_config = {}  # ADD THIS LINE
```

#### **FIX B: Update _store_backtest_results method**

In `backtest.py`, find the `_store_backtest_results` method and replace the problematic line:

**Replace:**
```python
f'Backtest with {self.backtest_config["max_position_size"]*100:.0f}% position...'
```

**With:**
```python
f"Backtest with {metrics.get('backtest_config', {}).get('max_position_size', 0.95)*100:.0f}% position..."
```

#### **FIX C: Update _empty_metrics in evaluation.py**

In `evaluation.py`, replace the entire `_empty_metrics` method:

```python
def _empty_metrics(self) -> Dict[str, Any]:
    """Return empty metrics structure"""
    now = datetime.now()
    return {
        'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
        'sortino_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
        'profit_factor': 0.0, 'trade_count': 0, 'overall_grade': 'F',
        'period_start': now.strftime('%Y-%m-%d'),
        'period_end': now.strftime('%Y-%m-%d'),
        'total_trading_days': 0,
        'evaluation_date': now.isoformat(),
        'monthly_return': 0.0,        # ADD THIS
        'volatility': 0.0,            # ADD THIS
        'calmar_ratio': 0.0,          # ADD THIS
        'var_95': 0.0,               # ADD THIS
        'cvar_95': 0.0,              # ADD THIS
        'information_ratio': 0.0,     # ADD THIS
        'recovery_days': 0,          # ADD THIS
        'expectancy': 0.0,            # ADD THIS
        'kelly_criterion': 0.0,       # ADD THIS
        'recovery_factor': 0.0,       # ADD THIS
    }
```

#### **FIX D: Store backtest_config in metrics**

In `backtest.py`, find the `metrics.update` section and ensure it includes:

```python
metrics.update({
    'backtest_config': {
        'initial_capital': self.initial_capital,
        'final_capital': self.current_capital,
        'commission': self.commission,
        'slippage': self.slippage,
        'max_position_size': self.max_position_size,
        'stop_loss_pct': self.stop_loss_pct,
        'take_profit_pct': self.take_profit_pct
    },
    # ... rest of the metrics
})
```

## 🧪 **Step 3: Test the Fixes**

After applying the fixes:

```bash
(.xgboostvenv) python test_fixes.py
```

## 🚀 **Step 4: Run Your Backtest Again**

```bash
(.xgboostvenv) python backtest.py --model xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT --start-date 2024-01-01 --end-date 2024-06-30
```

## 💡 **Root Cause of These Errors:**

1. **Missing Instance Variables**: The backtest_config wasn't properly initialized
2. **Incomplete Error Handling**: When backtests fail, the error handling doesn't provide all expected keys
3. **Missing Metrics**: The evaluation system expects certain metrics that aren't provided when there are no trades

The comprehensive fix script addresses all these issues systematically.

## 🔥 **If All Else Fails:**

1. Copy your working versions from your local machine to the server
2. Or restart with fresh files from the repository
3. The main issue is that your server has older versions of the files

The comprehensive fix script should resolve all the issues automatically! 🎯