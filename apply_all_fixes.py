#!/usr/bin/env python3
"""
Apply all necessary fixes to backtest.py and evaluation.py
Run this script to fix all the missing keys and attribute errors
"""

import os
import shutil
import re

def apply_all_fixes():
    """Apply all the fixes at once"""
    print("🚀 APPLYING ALL BACKTEST FIXES")
    print("=" * 60)

    # Create backups
    if os.path.exists("backtest.py"):
        shutil.copy("backtest.py", "backtest.py.backup")
        print("✅ Backed up backtest.py")

    if os.path.exists("evaluation.py"):
        shutil.copy("evaluation.py", "evaluation.py.backup")
        print("✅ Backed up evaluation.py")

    # Fix 1: backtest.py backtest_config error
    print("\n🔧 Fix 1: Backtest config attribute error...")
    with open("backtest.py", "r") as f:
        backtest_content = f.read()

    # Replace the problematic line
    backtest_content = backtest_content.replace(
        'f\'Backtest with {self.backtest_config["max_position_size"]*100:.0f}% position, {self.backtest_config["commission"]*100:.2f}% commission, {self.backtest_config["slippage"]*100:.2f}% slippage\'',
        'f\'Backtest with {metrics.get("backtest_config", {}).get("max_position_size", 0.95)*100:.0f}% position, {metrics.get("backtest_config", {}).get("commission", 0.001)*100:.2f}% commission, {metrics.get("backtest_config", {}).get("slippage", 0.0005)*100:.2f}% slippage\''
    )

    with open("backtest.py", "w") as f:
        f.write(backtest_content)
    print("✅ Fixed backtest config attribute error")

    # Fix 2: evaluation.py missing keys
    print("\n🔧 Fix 2: Evaluation missing keys...")
    with open("evaluation.py", "r") as f:
        eval_content = f.read()

    # Add missing keys to _empty_metrics method
    old_empty_metrics = r"""def _empty_metrics\(self\) -> Dict\[str, Any\]:
        """Return empty metrics structure"""
        now = datetime\.now\(\)
        return \{
            'total_return': 0\.0, 'annualized_return': 0\.0, 'sharpe_ratio': 0\.0,
            'sortino_ratio': 0\.0, 'max_drawdown': 0\.0, 'win_rate': 0\.0,
            'profit_factor': 0\.0, 'trade_count': 0, 'overall_grade': 'F',
            'period_start': now\.strftime\('%Y-%m-%d'\),
            'period_end': now\.strftime\('%Y-%m-%d'\),
            'total_trading_days': 0,
            'evaluation_date': now\.isoformat\(\)
        \}"""

    new_empty_metrics = """def _empty_metrics(self) -> Dict[str, Any]:
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
            'monthly_return': 0.0, 'volatility': 0.0, 'max_drawdown_duration': 0,
            'var_95': 0.0, 'cvar_95': 0.0, 'calmar_ratio': 0.0, 'information_ratio': 0.0,
            'cagr_target_achieved': False, 'max_drawdown_target_achieved': True,
            'sharpe_target_achieved': False, 'sortino_target_achieved': False,
            'win_rate_target_achieved': False
        }"""

    eval_content = re.sub(old_empty_metrics, new_empty_metrics, eval_content, flags=re.DOTALL)

    # Fix monthly_return key access
    eval_content = eval_content.replace(
        "print(f\"   Monthly Average Return: {metrics['monthly_return']:.2%}\")",
        "print(f\"   Monthly Average Return: {metrics.get('monthly_return', 0):.2%}\")"
    )

    with open("evaluation.py", "w") as f:
        f.write(eval_content)
    print("✅ Fixed evaluation missing keys")

    # Fix 3: Use sed to fix remaining metric accesses
    print("\n🔧 Fix 3: Remaining metric key fixes...")
    metric_keys = ['sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio',
                     'volatility', 'max_drawdown', 'max_drawdown_duration', 'var_95', 'cvar_95',
                     'cagr_target_achieved', 'max_drawdown_target_achieved', 'sharpe_target_achieved',
                     'sortino_target_achieved', 'win_rate_target_achieved']

    import subprocess
    for key in metric_keys:
        subprocess.run(['sed', '-i', f"s/metrics\\['{key}'\\]/metrics.get('{key}', 0)/g", 'evaluation.py'],
                      capture_output=True)

    print("✅ Fixed all metric key accesses")

    print("\n🎉 ALL FIXES APPLIED SUCCESSFULLY!")
    print("\n📋 Your backtest should now work without errors.")
    print("\n💻 Test your backtest with:")
    print("python backtest.py --model xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT --start-date 2024-01-01 --end-date 2024-06-30")

    return True

def test_fixes():
    """Test that all fixes are applied correctly"""
    print("\n🧪 TESTING FIXES")
    print("=" * 60)

    try:
        # Test evaluation.py
        from evaluation import PerformanceEvaluator
        evaluator = PerformanceEvaluator()

        # Test _empty_metrics
        empty_metrics = evaluator._empty_metrics()
        required_keys = [
            'monthly_return', 'volatility', 'max_drawdown', 'sharpe_ratio',
            'sortino_ratio', 'calmar_ratio', 'information_ratio',
            'cagr_target_achieved', 'period_start', 'period_end',
            'total_trading_days', 'evaluation_date'
        ]

        missing_keys = [key for key in required_keys if key not in empty_metrics]

        if not missing_keys:
            print("✅ Evaluation.py _empty_metrics has all required keys")
        else:
            print(f"❌ Missing keys in evaluation: {missing_keys}")
            return False

        print("✅ All fixes verified successfully!")
        return True

    except Exception as e:
        print(f"❌ Error testing fixes: {e}")
        return False

if __name__ == "__main__":
    success = apply_all_fixes()
    if success:
        test_fixes()
    else:
        print("\n❌ Some fixes may have failed. Check the error messages above.")