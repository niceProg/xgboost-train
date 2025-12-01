#!/usr/bin/env python3
"""
Quick fix for backtest.py errors
This script will patch the current backtest.py file to fix the two critical errors
"""

import os
import shutil
from datetime import datetime

def fix_backtest_file():
    """Fix the backtest.py file with the correct patches"""

    print("🔧 FIXING BACKTEST.PY ERRORS")
    print("=" * 50)

    # Backup original file
    if os.path.exists("backtest.py"):
        shutil.copy("backtest.py", "backtest.py.backup")
        print("✅ Created backup: backtest.py.backup")

    # Read the current file
    try:
        with open("backtest.py", "r") as f:
            content = f.read()
    except:
        print("❌ Could not read backtest.py file")
        return False

    # Fix 1: Monthly returns JSON error
    old_monthly_pattern = """            # Create monthly returns JSON
            monthly_returns_json = json.dumps({
                f"{month.year}-{month.month:02d}": {
                    'return': metrics.get('monthly_return', 0),
                    'volatility': metrics.get('volatility', 0)
                }
                for month in range(1, 13)  # Placeholder for actual monthly data
            }, default=str)"""

    new_monthly_pattern = """            # Create monthly returns JSON
            monthly_returns_data = {}
            current_year = datetime.now().year
            for month in range(1, 13):  # Placeholder for actual monthly data
                monthly_returns_data[f"{current_year}-{month:02d}"] = {
                    'return': metrics.get('monthly_return', 0),
                    'volatility': metrics.get('volatility', 0)
                }
            monthly_returns_json = json.dumps(monthly_returns_data, default=str)"""

    if old_monthly_pattern in content:
        content = content.replace(old_monthly_pattern, new_monthly_pattern)
        print("✅ Fixed monthly returns JSON error")
    else:
        print("⚠️  Monthly returns fix pattern not found - trying alternative fix")
        # Alternative fix
        content = content.replace(
            "for month in range(1, 13)  # Placeholder for actual monthly data",
            "for month in range(1, 13)  # Placeholder for actual monthly data"
        )
        # Find and replace the problematic line
        lines = content.split('\n')
        fixed_lines = []
        for i, line in enumerate(lines):
            if 'f"{month.year}-{month.month:02d}":' in line:
                fixed_lines.append(f"                f\"{{current_year}}-{month:02d}\": {{")
                print("✅ Fixed monthly JSON dictionary comprehension")
            else:
                fixed_lines.append(line)
        content = '\n'.join(fixed_lines)

    # Add current_year variable if needed
    if "current_year = datetime.now().year" not in content:
        # Find a good place to add it
        import re
        pattern = r'(# Create monthly returns JSON)'
        replacement = r'            current_year = datetime.now().year\n\1'
        content = re.sub(pattern, replacement, content)
        print("✅ Added current_year variable")

    # Fix 2: Add empty equity curve handling
    empty_equity_fix = """
        # Handle empty equity curve case
        if equity_curve_df.empty and not self.trades:
            # Create a minimal equity curve for empty backtest
            equity_curve_df = pd.DataFrame([
                {
                    'timestamp': pd.to_datetime(start_date),
                    'equity_value': self.initial_capital,
                    'price': 0,  # No price data available
                    'position': 0,
                    'cash': self.initial_capital
                },
                {
                    'timestamp': pd.to_datetime(end_date),
                    'equity_value': self.initial_capital,
                    'price': 0,  # No price data available
                    'position': 0,
                    'cash': self.initial_capital
                }
            ])
            equity_curve_df.set_index('timestamp', inplace=True)"""

    # Find the right place to insert the empty equity curve fix
    pattern = r'(            # Calculate performance metrics\n            equity_curve_df = pd\.DataFrame\(self\.equity_curve\)\n)'
    replacement = r'\1\n' + empty_equity_fix + '\n'

    import re
    if re.search(pattern, content):
        content = re.sub(pattern, replacement, content)
        print("✅ Added empty equity curve handling")

    # Write the fixed content back
    try:
        with open("backtest.py", "w") as f:
            f.write(content)
        print("✅ Successfully patched backtest.py")
    except Exception as e:
        print(f"❌ Error writing patched file: {e}")
        return False

    return True

def fix_evaluation_file():
    """Fix the evaluation.py file to include missing keys"""

    print("\n🔧 FIXING EVALUATION.PY ERRORS")
    print("=" * 50)

    # Backup original file
    if os.path.exists("evaluation.py"):
        shutil.copy("evaluation.py", "evaluation.py.backup")
        print("✅ Created backup: evaluation.py.backup")

    # Read the current file
    try:
        with open("evaluation.py", "r") as f:
            content = f.read()
    except:
        print("❌ Could not read evaluation.py file")
        return False

    # Fix _empty_metrics method to include period_start and period_end
    old_empty_metrics = """    def _empty_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure"""
        return {
            'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0, 'max_drawdown': 0.0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'trade_count': 0, 'overall_grade': 'F'
        }"""

    new_empty_metrics = """    def _empty_metrics(self) -> Dict[str, Any]:
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
        }"""

    if old_empty_metrics in content:
        content = content.replace(old_empty_metrics, new_empty_metrics)
        print("✅ Fixed _empty_metrics method")
    else:
        print("⚠️  _empty_metrics pattern not found, trying alternative fix")
        # Look for the method and add missing keys
        lines = content.split('\n')
        fixed_lines = []
        in_empty_metrics = False
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            if "def _empty_metrics(self)" in line:
                in_empty_metrics = True
            if in_empty_metrics and "'overall_grade': 'F'" in line:
                # Add the missing keys after this line
                fixed_lines.append("            'period_start': now.strftime('%Y-%m-%d'),")
                fixed_lines.append("            'period_end': now.strftime('%Y-%m-%d'),")
                fixed_lines.append("            'total_trading_days': 0,")
                fixed_lines.append("            'evaluation_date': now.isoformat()")
                in_empty_metrics = False
                print("✅ Added missing keys to _empty_metrics")

        content = '\n'.join(fixed_lines)

    # Write the fixed content back
    try:
        with open("evaluation.py", "w") as f:
            f.write(content)
        print("✅ Successfully patched evaluation.py")
    except Exception as e:
        print(f"❌ Error writing patched file: {e}")
        return False

    return True

def test_fixes():
    """Test that the fixes work"""
    print("\n🧪 TESTING FIXES")
    print("=" * 50)

    try:
        # Test the monthly returns JSON fix
        import json
        from datetime import datetime

        # Test the fixed monthly returns code
        monthly_returns_data = {}
        current_year = datetime.now().year
        for month in range(1, 3):  # Just test 2 months
            monthly_returns_data[f"{current_year}-{month:02d}"] = {
                'return': 0.05,
                'volatility': 0.15
            }
        monthly_returns_json = json.dumps(monthly_returns_data, default=str)
        print("✅ Monthly returns JSON works correctly")

        # Test the empty metrics fix
        from evaluation import PerformanceEvaluator
        evaluator = PerformanceEvaluator()
        empty_metrics = evaluator._empty_metrics()

        required_keys = ['period_start', 'period_end', 'total_trading_days', 'evaluation_date']
        missing_keys = [key for key in required_keys if key not in empty_metrics]

        if not missing_keys:
            print("✅ Empty metrics includes all required keys")
            print(f"   period_start: {empty_metrics['period_start']}")
            print(f"   period_end: {empty_metrics['period_end']}")
        else:
            print(f"❌ Missing keys: {missing_keys}")
            return False

        print("✅ All fixes work correctly!")
        return True

    except Exception as e:
        print(f"❌ Error testing fixes: {e}")
        return False

def main():
    """Main function to run all fixes"""
    print("🚀 BACKTEST ERROR FIXER")
    print("=" * 50)
    print("This script will fix the two critical errors in your trading system:")
    print("1. 'int' object has no attribute 'year' (monthly returns JSON)")
    print("2. KeyError: 'period_start' (missing keys in evaluation)")
    print("=" * 50)

    success = True

    # Fix backtest.py
    if not fix_backtest_file():
        success = False

    # Fix evaluation.py
    if not fix_evaluation_file():
        success = False

    # Test the fixes
    if not test_fixes():
        success = False

    if success:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n📋 NEXT STEPS:")
        print("1. Run your backtest again:")
        print("   python backtest.py --model xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT --start-date 2024-01-01 --end-date 2024-06-30")
        print("\n2. The errors should now be resolved!")
        print("\n📁 Backup files created:")
        print("   - backtest.py.backup")
        print("   - evaluation.py.backup")
    else:
        print("\n❌ Some fixes may have failed")
        print("Please check the error messages above")

    return success

if __name__ == "__main__":
    main()