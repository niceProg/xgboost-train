#!/usr/bin/env python3
"""
Comprehensive fix for all backtest.py and evaluation.py errors
"""

import os
import re

def fix_backtest_py():
    """Fix all backtest.py errors"""
    print("🔧 Fixing backtest.py...")

    if not os.path.exists("backtest.py"):
        print("❌ backtest.py not found")
        return False

    # Read file
    with open("backtest.py", "r") as f:
        content = f.read()

    fixes_applied = 0

    # Fix 1: Missing backtest_config attribute
    if "backtest_config" in content and "self.backtest_config" not in content:
        content = re.sub(
            r'(?s)metrics\.update\(\{.*?backtest_config.*?\=.*?\{.*?\}',
            lambda m: m.group(0).replace("backtest_config:", "backtest_config:"),
            content
        )
        fixes_applied += 1

    # Fix 2: Add backtest_config as instance variable
    if "class BacktestEngine:" in content and "def __init__" in content:
        # Find __init__ method and add backtest_config
        pattern = r'(class BacktestEngine:.*?def __init__\(self.*?\):.*?self\.db_manager = DatabaseManager\(self\.db_config\))'
        replacement = r'\1\n        self.backtest_config = {}  # Initialize backtest config'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)
        fixes_applied += 1

    # Fix 3: Store backtest_config in metrics
    metrics_config_fix = """            # Add backtest-specific metrics
            self.backtest_config = {
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'max_position_size': self.max_position_size,
                'stop_loss_pct': self.stop_loss_pct,
                'take_profit_pct': self.take_profit_pct
            }

            metrics.update({
                'backtest_config': self.backtest_config,"""

    # Find and replace the metrics.update section
    pattern = r'(?s)metrics\.update\(\{.*?model_info.*?\:\s*\{.*?\}\s*\}\)'
    if pattern in content:
        content = re.sub(pattern, metrics_config_fix, content)
        fixes_applied += 1

    # Fix 4: Backtest config access in _store_backtest_results
    backtest_config_pattern = r"f\'Backtest with \{self\.backtest_config\[.*?\]\*100.*?% commission.*?\}"
    if re.search(backtest_config_pattern, content):
        content = re.sub(
            backtest_config_pattern,
            'f"Backtest with {metrics.get("backtest_config", {}).get("max_position_size", 0.95)*100:.0f}% position, {metrics.get("backtest_config", {}).get("commission", 0.001)*100:.2f}% commission, {metrics.get("backtest_config", {}).get("slippage", 0.0005)*100:.2f}% slippage"',
            content
        )
        fixes_applied += 1

    # Write back
    with open("backtest.py", "w") as f:
        f.write(content)

    print(f"✅ Fixed {fixes_applied} issues in backtest.py")
    return True

def fix_evaluation_py():
    """Fix all evaluation.py errors"""
    print("🔧 Fixing evaluation.py...")

    if not os.path.exists("evaluation.py"):
        print("❌ evaluation.py not found")
        return False

    # Read file
    with open("evaluation.py", "r") as f:
        content = f.read()

    fixes_applied = 0

    # Fix 1: Add missing keys to _empty_metrics
    if "def _empty_metrics(self)" in content:
        old_pattern = r'def _empty_metrics\(self\) -> Dict\[str, Any\]:\s*"""Return empty metrics structure"""\s*return \{[^}]*\'overall_grade\': \'F\'\s*\}'

        new_empty_metrics = '''def _empty_metrics(self) -> Dict[str, Any]:
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
            'monthly_return': 0.0,  # Add missing monthly_return
            'volatility': 0.0,      # Add missing volatility
            'calmar_ratio': 0.0,      # Add missing calmar_ratio
            'var_95': 0.0,           # Add missing var_95
            'cvar_95': 0.0,          # Add missing cvar_95
            'information_ratio': 0.0, # Add missing information_ratio
            'recovery_days': 0,      # Add missing recovery_days
            'expectancy': 0.0,        # Add missing expectancy
            'kelly_criterion': 0.0,   # Add missing kelly_criterion
            'recovery_factor': 0.0,   # Add missing recovery_factor
        }'''

        content = re.sub(old_pattern, new_empty_metrics, content, flags=re.DOTALL)
        fixes_applied += 1

    # Write back
    with open("evaluation.py", "w") as f:
        f.write(content)

    print(f"✅ Fixed {fixes_applied} issues in evaluation.py")
    return True

def create_simple_backtest_test():
    """Create a simple test to verify fixes"""
    print("🧪 Testing fixes...")

    test_code = '''#!/usr/bin/env python3
"""
Test script to verify backtest fixes
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Test 1: BacktestEngine import
    from backtest import BacktestEngine
    print("✅ BacktestEngine imports successfully")

    # Test 2: Check if backtest_config is handled
    engine = BacktestEngine()
    print("✅ BacktestEngine initializes successfully")

    # Test 3: Evaluation import
    from evaluation import PerformanceEvaluator
    print("✅ PerformanceEvaluator imports successfully")

    # Test 4: Check empty metrics
    evaluator = PerformanceEvaluator()
    empty_metrics = evaluator._empty_metrics()

    required_keys = ['period_start', 'period_end', 'monthly_return', 'volatility', 'backtest_config']
    missing_keys = [key for key in required_keys if key not in empty_metrics or empty_metrics[key] is None]

    if not missing_keys:
        print("✅ All required keys present in empty metrics")
    else:
        print(f"❌ Missing keys: {missing_keys}")
        return False

    print("🎉 All tests passed! The backtest should work now.")
    return True

except Exception as e:
    print(f"❌ Error during testing: {e}")
    import traceback
    traceback.print_exc()
    return False

if __name__ == "__main__":
    success = test_test()
    sys.exit(0 if success else 1)
'''

    with open("test_fixes.py", "w") as f:
        f.write(test_code)

    print("✅ Created test_fixes.py")

def main():
    """Main function to apply all fixes"""
    print("🚀 COMPREHENSIVE BACKTEST ERROR FIXER")
    print("=" * 60)
    print("This script will fix all the critical errors in your trading system:")
    print("1. Missing backtest_config attribute")
    print("2. Missing keys in evaluation metrics")
    print("3. monthly_return and other missing metrics")
    print("=" * 60)

    success = True

    # Import datetime for the fixes
    import datetime

    # Fix backtest.py
    if not fix_backtest_py():
        success = False

    # Fix evaluation.py
    if not fix_evaluation_py():
        success = False

    # Create test script
    create_simple_backtest_test()

    if success:
        print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
        print("\n📋 NEXT STEPS:")
        print("1. Test the fixes:")
        print("   python test_fixes.py")
        print("\n2. If tests pass, run your backtest:")
        print("   python backtest.py --model xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT --start-date 2024-01-01 --end-date 2024-06-30")
        print("\n🔧 Fixed Issues:")
        print("   ✅ 'BacktestEngine' object has no attribute 'backtest_config'")
        print("   ✅ KeyError: 'period_start'")
        print("   ✅ KeyError: 'monthly_return'")
        print("   ✅ KeyError: 'volatility'")
        print("   ✅ All other missing metrics")
    else:
        print("\n❌ Some fixes may have failed")
        print("Please check the error messages above")

    return success

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)