#!/usr/bin/env python3
"""
Analyze cg_train_dataset and signal_prediction tables
Focus on understanding the database structure for professional backtest
"""

from env_config import get_database_config
from database import DatabaseManager
import pandas as pd

def analyze_table_structure():
    """Analyze the structure and content of key database tables"""
    print("🔍 ANALYZING DATABASE TABLES")
    print("=" * 80)

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # Table 1: cg_train_dataset
        print("\n📊 TABLE 1: cg_train_dataset")
        print("-" * 50)

        # Get table structure
        structure_query = """
        DESCRIBE cg_train_dataset
        """
        structure_result = db_manager.execute_query(structure_query)

        if not structure_result.empty:
            print("Columns:")
            for _, row in structure_result.iterrows():
                print(f"  - {row['Field']}: {row['Type']} {row['Extra'] if row['Extra'] else ''}")

        # Get sample data
        sample_query = """
        SELECT * FROM cg_train_dataset
        LIMIT 5
        """
        sample_data = db_manager.execute_query(sample_query)

        if not sample_data.empty:
            print(f"\nSample Data (5 rows, {len(sample_data.columns)} columns):")
            print(sample_data.head())

            # Get row count
            count_query = "SELECT COUNT(*) as total_rows FROM cg_train_dataset"
            count_result = db_manager.execute_query(count_query)
            if count_result:
                total_rows = count_result[0][0]
                print(f"\n📈 Total Rows: {total_rows:,}")

                # Get date range
                if 'timestamp' in sample_data.columns:
                    date_query = """
                    SELECT
                        MIN(timestamp) as start_date,
                        MAX(timestamp) as end_date,
                        COUNT(DISTINCT DATE(timestamp)) as unique_dates
                    FROM cg_train_dataset
                    """
                    date_result = db_manager.execute_query(date_query)
                    if date_result:
                        print(f"📅 Date Range: {date_result[0][0]} to {date_result[0][1]}")
                        print(f"📆 Unique Dates: {date_result[0][2]:,}")

        # Table 2: signal_prediction
        print(f"\n{'='*80}")
        print("\n📊 TABLE 2: signal_prediction")
        print("-" * 50)

        # Get table structure
        structure_query = """
        DESCRIBE signal_prediction
        """
        structure_result = db_manager.execute_query(structure_query)

        if not structure_result.empty:
            print("Columns:")
            for _, row in structure_result.iterrows():
                print(f"  - {row['Field']}: {row['Type']} {row['Extra'] if row['Extra'] else ''}")

        # Get sample data
        sample_query = """
        SELECT * FROM signal_prediction
        LIMIT 5
        """
        sample_data = db_manager.execute_query(sample_query)

        if not sample_data.empty:
            print(f"\nSample Data (5 rows, {len(sample_data.columns)} columns):")
            print(sample_data.head())

            # Get row count
            count_query = "SELECT COUNT(*) as total_rows FROM signal_prediction"
            count_result = db_manager.execute_query(count_query)
            if count_result:
                total_rows = count_result[0][0]
                print(f"\n📈 Total Rows: {total_rows:,}")

                # Get prediction statistics
                if 'prediction' in sample_data.columns:
                    pred_query = """
                    SELECT
                        prediction,
                        COUNT(*) as count,
                        AVG(confidence) as avg_confidence,
                        MIN(timestamp) as first_prediction,
                        MAX(timestamp) as last_prediction
                    FROM signal_prediction
                    GROUP BY prediction
                    """
                    pred_result = db_manager.execute_query(pred_query)
                    if pred_result:
                        print("\n🎯 Prediction Statistics:")
                        for row in pred_result:
                            pred_label = "BUY" if row[0] == 1 else "SELL" if row[0] == 0 else "HOLD"
                            print(f"  {pred_label} ( {row[0]} ): {row[1]:,} predictions")
                            if row[2]:
                                print(f"    Avg Confidence: {row[2]:.3f}")
                        print(f"  Period: {pred_result[0][3]} to {pred_result[0][4]}")

        # Check for relationship between tables
        print(f"\n{'='*80}")
        print("\n🔗 TABLE RELATIONSHIPS")
        print("-" * 50)

        # Check if there's a common timestamp or ID field
        cg_columns_query = """
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'newera' AND TABLE_NAME = 'cg_train_dataset'
        """
        cg_columns = db_manager.execute_query(cg_columns_query)
        cg_column_names = [row[0] for row in cg_columns] if cg_columns else []

        signal_columns_query = """
        SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'newera' AND TABLE_NAME = 'signal_prediction'
        """
        signal_columns = db_manager.execute_query(signal_columns_query)
        signal_column_names = [row[0] for row in signal_columns] if signal_columns else []

        common_columns = set(cg_column_names) & set(signal_column_names)
        if common_columns:
            print(f"🔗 Common Columns: {', '.join(common_columns)}")
        else:
            print("⚠️  No direct common columns found")

        print(f"\ncg_train_dataset has {len(cg_column_names)} columns")
        print(f"signal_prediction has {len(signal_column_names)} columns")

        # Check for recent data availability
        recent_query = """
        SELECT
            'cg_train_dataset' as table_name,
            MAX(timestamp) as latest_data
        FROM cg_train_dataset
        UNION ALL
        SELECT
            'signal_prediction' as table_name,
            MAX(timestamp) as latest_data
        FROM signal_prediction
        """
        recent_result = db_manager.execute_query(recent_query)
        if recent_result:
            print(f"\n📅 Latest Data Availability:")
            for row in recent_result:
                print(f"  {row[0]}: {row[1]}")

    except Exception as e:
        print(f"❌ Error analyzing tables: {e}")
        import traceback
        traceback.print_exc()

    finally:
        db_manager.close()

def check_feature_columns():
    """Check for feature columns in cg_train_dataset"""
    print(f"\n{'='*80}")
    print("\n🧬 FEATURE COLUMN ANALYSIS")
    print("-" * 50)

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # Get all columns from cg_train_dataset
        columns_query = """
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = 'newera' AND TABLE_NAME = 'cg_train_dataset'
        ORDER BY ORDINAL_POSITION
        """
        columns_result = db_manager.execute_query(columns_query)

        if not columns_result.empty:
            print("📋 All Columns in cg_train_dataset:")

            feature_columns = []
            price_volume_columns = []
            metadata_columns = []

            for row in columns_result:
                col_name = row[0]
                data_type = row[1]

                # Categorize columns
                if col_name in ['timestamp', 'symbol', 'pair', 'interval', 'id', 'created_at']:
                    metadata_columns.append((col_name, data_type))
                elif col_name in ['open', 'high', 'low', 'close', 'volume', 'price']:
                    price_volume_columns.append((col_name, data_type))
                else:
                    feature_columns.append((col_name, data_type))

            print(f"\n🕐 Metadata Columns ({len(metadata_columns)}):")
            for col, dtype in metadata_columns:
                print(f"  - {col}: {dtype}")

            print(f"\n💰 Price/Volume Columns ({len(price_volume_columns)}):")
            for col, dtype in price_volume_columns:
                print(f"  - {col}: {dtype}")

            print(f"\n🧬 Feature Columns ({len(feature_columns)}):")
            for i, (col, dtype) in enumerate(feature_columns[:20]):  # Show first 20
                print(f"  {i+1:2d}. {col}: {dtype}")

            if len(feature_columns) > 20:
                print(f"  ... and {len(feature_columns) - 20} more feature columns")

            # Analyze feature data quality
            if feature_columns:
                print(f"\n📊 Feature Data Quality Analysis:")
                sample_features = feature_columns[:5]  # Check first 5 features

                for col, _ in sample_features:
                    quality_query = f"""
                    SELECT
                        COUNT(*) as total_rows,
                        COUNT({col}) as non_null_rows,
                        MIN({col}) as min_value,
                        MAX({col}) as max_value,
                        AVG({col}) as avg_value
                    FROM cg_train_dataset
                    """
                    try:
                        quality_result = db_manager.execute_query(quality_query)
                        if quality_result and quality_result[0]:
                            total, non_null, min_val, max_val, avg_val = quality_result[0]
                            null_pct = ((total - non_null) / total * 100) if total > 0 else 100
                            print(f"  {col}: {non_null:,}/{total:,} non-null ({100-null_pct:.1f}%), "
                                  f"range: [{min_val:.4f}, {max_val:.4f}], avg: {avg_val:.4f}")
                    except Exception as e:
                        print(f"  {col}: ❌ Error analyzing: {e}")

                if len(feature_columns) > 5:
                    print(f"  ... and {len(feature_columns) - 5} more features")

    except Exception as e:
        print(f"❌ Error analyzing feature columns: {e}")

    finally:
        db_manager.close()

if __name__ == "__main__":
    analyze_table_structure()
    check_feature_columns()