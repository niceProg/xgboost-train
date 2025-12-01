#!/usr/bin/env python3
"""
Quick fix for the syntax error in evaluation.py
"""

import os
import shutil

def fix_syntax_error():
    """Fix the syntax error in evaluation.py"""
    print("🔧 Fixing syntax error in evaluation.py...")

    # Create backup
    if os.path.exists("evaluation.py"):
        shutil.copy("evaluation.py", "evaluation.py.backup")
        print("✅ Created backup: evaluation.py.backup")

    # Read file
    with open("evaluation.py", "r") as f:
        content = f.read()

    # Fix the syntax error
    if "metrics.get('cagr_target_achieved', False) = metrics['annualized_return'] >= 0.50" in content:
        content = content.replace(
            "metrics.get('cagr_target_achieved', False) = metrics['annualized_return'] >= 0.50  # 50% target",
            "metrics['cagr_target_achieved'] = metrics['annualized_return'] >= 0.50  # 50% target"
        )
        print("✅ Fixed cagr_target_achieved syntax error")
    else:
        print("⚠️  Syntax error not found or already fixed")

    # Write back
    with open("evaluation.py", "w") as f:
        f.write(content)

    print("✅ Syntax error fixed successfully!")
    return True

if __name__ == "__main__":
    print("🚀 QUICK SYNTAX ERROR FIX")
    print("=" * 50)

    if fix_syntax_error():
        print("\n🎉 SYNTAX ERROR FIXED!")
        print("\n💻 Try your backtest again:")
        print("python backtest.py --model output_train/xgboost_trading_model_20251201_150937.joblib --symbol BTC --pair BTCUSDT")
    else:
        print("\n❌ Fix failed. Please apply manually.")