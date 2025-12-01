#!/usr/bin/env python3
"""
Apply quick fixes to backtest.py and evaluation.py
Run this script to fix the two critical errors
"""

import re
import os

def fix_backtest_py():
    """Fix backtest.py monthly returns JSON error"""
    print("🔧 Fixing backtest.py...")

    if not os.path.exists("backtest.py"):
        print("❌ backtest.py not found")
        return False

    # Read file
    with open("backtest.py", "r") as f:
        content = f.read()

    # Fix the monthly returns JSON error
    # Find the problematic line and replace it
    content = re.sub(
        r'monthly_returns_json = json\.dumps\(\{.*?f"\{month\.year\}-\{month\.month\}:02d\}"',
        '''monthly_returns_data = {}
            current_year = datetime.now().year
            for month in range(1, 13):
                monthly_returns_data[f"{current_year}-{month:02d}"] = {
                    'return': metrics.get('monthly_return', 0),
                    'volatility': metrics.get('volatility', 0)
                }
            monthly_returns_json = json.dumps(monthly_returns_data, default=str)''',
        content,
        flags=re.DOTALL
    )

    # Add current_year variable if needed
    if "current_year = datetime.now().year" not in content:
        content = re.sub(
            r'(# Create monthly returns JSON)',
            '            current_year = datetime.now().year\n\\1',
            content
        )

    # Write back
    with open("backtest.py", "w") as f:
        f.write(content)

    print("✅ Fixed backtest.py monthly returns JSON error")
    return True

def fix_evaluation_py():
    """Fix evaluation.py missing keys error"""
    print("🔧 Fixing evaluation.py...")

    if not os.path.exists("evaluation.py"):
        print("❌ evaluation.py not found")
        return False

    # Read file
    with open("evaluation.py", "r") as f:
        content = f.read()

    # Fix the _empty_metrics method
    old_pattern = r'def _empty_metrics\(self\) -> Dict\[str, Any\]:\s*"""Return empty metrics structure"""\s*return \{[^}]*\'overall_grade\': \'F\'\s*\}'

    new_pattern = '''def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        now = datetime.now()
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'trade_count': 0, 'overall_grade': 'F',
            'period_start': now.strftime('%Y-%m-%d'),
            'period_end': now.strftime('%Y-%m-%d'),
            'total_trading_days': 0,
            'evaluation_date': now.isoformat()
        }'''

    content = re.sub(old_pattern, new_pattern, content, flags=re.DOTALL)

    # Write back
    with open("evaluation.py", "w") as f:
        f.write(content)

    print("✅ Fixed evaluation.py missing keys error")
    return True

if __name__ == "__main__":
    print("🚀 APPLYING QUICK FIXES")
    print("=" * 50)

    # Import datetime
    import datetime

    success = True

    if fix_backtest_py():
        print("✅ backtest.py fixed")
    else:
        print("❌ Failed to fix backtest.py")
        success = False

    if fix_evaluation_py():
        print("✅ evaluation.py fixed")
    else:
        print("❌ Failed to fix evaluation.py")
        success = False

    if success:
        print("\n🎉 FIXES APPLIED SUCCESSFULLY!")
        print("\nTry your backtest command again:")
        print("python backtest.py --model xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT --start-date 2024-01-01 --end-date 2024-06-30")
    else:
        print("\n❌ Some fixes failed. Please apply manually.")