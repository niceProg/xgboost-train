#!/usr/bin/env python3
"""
Simple analysis of cg_train_dataset and signal_prediction tables
Focus on understanding the database structure for professional backtest
"""

from env_config import get_database_config
from database import DatabaseManager
import pandas as pd
import json

def analyze_tables():
    """Analyze the two key tables for backtesting"""
    print("🔍 DATABASE TABLE ANALYSIS")
    print("=" * 60)

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # 1. Analyze cg_train_dataset
        print("\n📊 cg_train_dataset TABLE")
        print("-" * 40)

        # Get row counts and date ranges
        count_query = "SELECT COUNT(*) as count FROM cg_train_dataset"
        count_result = db_manager.execute_query(count_query)
        if not count_result.empty:
            total_rows = count_result['count'].iloc[0]
            print(f"📈 Total Rows: {total_rows:,}")

        # Get date range and unique symbols
        meta_query = """
        SELECT
            MIN(generated_at) as earliest_data,
            MAX(generated_at) as latest_data,
            COUNT(DISTINCT symbol) as unique_symbols,
            COUNT(DISTINCT pair) as unique_pairs,
            COUNT(DISTINCT time_interval) as intervals
        FROM cg_train_dataset
        """
        meta_result = db_manager.execute_query(meta_query)
        if not meta_result.empty:
            row = meta_result.iloc[0]
            print(f"📅 Date Range: {row['earliest_data']} to {row['latest_data']}")
            print(f"🔤 Symbols: {row['unique_symbols']}")
            print(f"💱 Pairs: {row['unique_pairs']}")
            print(f"⏱️  Intervals: {row['intervals']}")

        # Analyze signal distribution
        signal_query = """
        SELECT
            signal_rule,
            COUNT(*) as count,
            AVG(signal_score) as avg_score,
            MIN(signal_score) as min_score,
            MAX(signal_score) as max_score
        FROM cg_train_dataset
        GROUP BY signal_rule
        """
        signal_result = db_manager.execute_query(signal_query)
        if not signal_result.empty:
            print(f"\n🎯 Signal Distribution:")
            for _, row in signal_result.iterrows():
                print(f"  {row['signal_rule']:8s}: {row['count']:6,} records "
                      f"(score: {row['avg_score']:.3f} [{row['min_score']:.3f}, {row['max_score']:.3f}])")

        # Check label status
        label_query = """
        SELECT
            label_status,
            COUNT(*) as count
        FROM cg_train_dataset
        GROUP BY label_status
        """
        label_result = db_manager.execute_query(label_query)
        if not label_result.empty:
            print(f"\n📝 Label Status:")
            for _, row in label_result.iterrows():
                print(f"  {row['label_status']:8s}: {row['count']:6,} records")

        # Sample some feature payloads
        sample_query = """
        SELECT
            id,
            symbol,
            generated_at,
            signal_rule,
            signal_score,
            LEFT(features_payload, 200) as feature_sample
        FROM cg_train_dataset
        ORDER BY generated_at DESC
        LIMIT 3
        """
        sample_result = db_manager.execute_query(sample_query)
        if not sample_result.empty:
            print(f"\n🔍 Sample Feature Payloads:")
            for _, row in sample_result.iterrows():
                print(f"  ID {row['id']}: {row['symbol']} @ {row['generated_at']}")
                print(f"    Signal: {row['signal_rule']} (score: {row['signal_score']:.3f})")
                print(f"    Features: {row['feature_sample']}...")

        # 2. Analyze signal_prediction
        print(f"\n{'='*60}")
        print("\n📊 signal_prediction TABLE")
        print("-" * 40)

        # Get row counts and date ranges
        count_query = "SELECT COUNT(*) as count FROM signal_prediction"
        count_result = db_manager.execute_query(count_query)
        if not count_result.empty:
            total_rows = count_result['count'].iloc[0]
            print(f"📈 Total Rows: {total_rows:,}")

        # Check if table exists and get structure
        try:
            desc_query = "DESCRIBE signal_prediction"
            desc_result = db_manager.execute_query(desc_query)
            if not desc_result.empty:
                print(f"\n📋 Table Structure:")
                for _, row in desc_result.iterrows():
                    print(f"  - {row['Field']}: {row['Type']} {row['Extra'] if row['Extra'] else ''}")

            # Get date range and prediction stats
            pred_meta_query = """
            SELECT
                MIN(created_at) as earliest,
                MAX(created_at) as latest,
                COUNT(DISTINCT symbol) as symbols,
                COUNT(DISTINCT model_version) as model_versions
            FROM signal_prediction
            """
            pred_meta_result = db_manager.execute_query(pred_meta_query)
            if not pred_meta_result.empty:
                row = pred_meta_result.iloc[0]
                print(f"\n📅 Prediction Period: {row['earliest']} to {row['latest']}")
                print(f"🔤 Symbols: {row['symbols']}")
                print(f"🤖 Model Versions: {row['model_versions']}")

            # Sample recent predictions
            recent_query = """
            SELECT * FROM signal_prediction
            ORDER BY created_at DESC
            LIMIT 5
            """
            recent_result = db_manager.execute_query(recent_query)
            if not recent_result.empty:
                print(f"\n📊 Recent Predictions:")
                for _, row in recent_result.iterrows():
                    print(f"  {row['created_at']}: {row['symbol']} -> {row['prediction']} "
                          f"(confidence: {row['confidence']:.3f})")

        except Exception as e:
            print(f"⚠️  signal_prediction table may not exist or be empty: {e}")

        # 3. Check data relationship
        print(f"\n{'='*60}")
        print("\n🔗 DATA RELATIONSHIP ANALYSIS")
        print("-" * 40)

        # Check latest data availability
        latest_query = """
        SELECT 'cg_train_dataset' as table_name, MAX(generated_at) as latest_data
        FROM cg_train_dataset
        UNION ALL
        SELECT 'signal_prediction' as table_name, MAX(created_at) as latest_data
        FROM signal_prediction
        """
        latest_result = db_manager.execute_query(latest_query)
        if not latest_result.empty:
            print("📅 Latest Data Availability:")
            for _, row in latest_result.iterrows():
                status = "✅" if row['latest_data'] else "❌"
                print(f"  {row['table_name']:20s}: {row['latest_data']} {status}")

        # Check if we have enough data for backtesting
        data_quality_query = """
        SELECT
            COUNT(*) as total_records,
            COUNT(DISTINCT DATE(generated_at)) as unique_days,
            COUNT(DISTINCT symbol) as unique_symbols
        FROM cg_train_dataset
        WHERE signal_rule IS NOT NULL
        AND signal_score IS NOT NULL
        AND label_status = 'labeled'
        """
        quality_result = db_manager.execute_query(data_quality_query)
        if not quality_result.empty:
            row = quality_result.iloc[0]
            print(f"\n📊 Data Quality for Backtesting:")
            print(f"  Labeled Records: {row['total_records']:,}")
            print(f"  Trading Days: {row['unique_days']}")
            print(f"  Symbols: {row['unique_symbols']}")

            # Backtesting readiness assessment
            if row['total_records'] >= 1000 and row['unique_days'] >= 30:
                print(f"  ✅ READY for professional backtesting")
            elif row['total_records'] >= 500 and row['unique_days'] >= 15:
                print(f"  🟡 LIMITED backtesting possible")
            else:
                print(f"  ❌ INSUFFICIENT data for reliable backtesting")

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db_manager.close()

def check_feature_payload_structure():
    """Check the structure of the JSON feature payloads"""
    print(f"\n{'='*60}")
    print("\n🧬 FEATURE PAYLOAD STRUCTURE")
    print("-" * 40)

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # Get a few feature payloads to analyze structure
        query = """
        SELECT id, symbol, features_payload
        FROM cg_train_dataset
        WHERE features_payload IS NOT NULL
        AND LENGTH(features_payload) > 50
        LIMIT 5
        """
        result = db_manager.execute_query(query)

        if not result.empty:
            for i, row in result.iterrows():
                print(f"\n📋 Record {row['id']} - {row['symbol']}:")
                try:
                    features = json.loads(row['features_payload'])
                    print(f"  🔢 Total Features: {len(features)}")

                    # Show first 10 features
                    feature_keys = list(features.keys())[:10]
                    print(f"  📊 Sample Features:")
                    for key in feature_keys:
                        value = features[key]
                        value_type = type(value).__name__
                        print(f"    - {key:30s}: {value_type:10s} = {str(value)[:20]}...")

                    if len(features) > 10:
                        print(f"    ... and {len(features) - 10} more features")

                except json.JSONDecodeError as e:
                    print(f"  ❌ Invalid JSON: {e}")

        else:
            print("⚠️  No feature payloads found or database error")

    except Exception as e:
        print(f"❌ Feature analysis failed: {e}")

    finally:
        db_manager.close()

if __name__ == "__main__":
    analyze_tables()
    check_feature_payload_structure()