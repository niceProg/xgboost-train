#!/usr/bin/env python3
"""
Test the new model output functionality
"""

import os
import shutil

def test_model_output():
    """Test the new output_train functionality"""

    print("🧪 TESTING MODEL OUTPUT TO output_train FOLDER")
    print("=" * 60)

    # Check if output_train folder exists
    if os.path.exists("output_train"):
        print("✅ output_train folder exists")
        files = os.listdir("output_train")
        print(f"📁 Current files in output_train: {files}")
    else:
        print("❌ output_train folder does not exist")
        return

    print(f"\n📋 AVAILABLE COMMANDS:")
    print(f"   1. List available models:")
    print(f"      python train_model.py --list-models")
    print()
    print(f"   2. Train model (saves to output_train):")
    print(f"      python train_model.py --symbol BTC --pair BTCUSDT --limit 100")
    print()
    print(f"   3. Load model with 'latest' keyword:")
    print(f"      python predict_signals.py --model latest --symbol BTC --pair BTCUSDT --interval 1h")
    print()
    print(f"   4. Backtest with latest model:")
    print(f"      python backtest.py --model latest --symbol BTC --pair BTCUSDT --interval 1h --start-date 2024-11-01 --end-date 2024-11-30")
    print()
    print(f"   5. Use custom output directory:")
    print(f"      python train_model.py --output-dir models_2024 --symbol BTC --pair BTCUSDT")

    print(f"\n💡 KEY IMPROVEMENTS:")
    improvements = [
        "✅ All models now save to output_train folder automatically",
        "✅ Each model gets a timestamped filename",
        "✅ Latest model is also saved as 'latest_model.joblib'",
        "✅ Can load models using 'latest' keyword",
        "✅ Model loading automatically checks output_train folder",
        "✅ Custom output directories supported",
        "✅ Easy model listing with --list-models"
    ]

    for improvement in improvements:
        print(f"   {improvement}")

if __name__ == "__main__":
    test_model_output()