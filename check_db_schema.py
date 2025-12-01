#!/usr/bin/env python3
"""
Check database schema for all 9 tables to find correct column names
"""

from env_config import get_database_config
from database import DatabaseManager

def check_table_schemas():
    """Check schema of all 9 database tables"""
    print("🔍 CHECKING DATABASE TABLE SCHEMAS")
    print("=" * 60)

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    # List of 9 tables we need to check
    tables = [
        'cg_spot_price_history',
        'cg_open_interest_aggregated_history',
        'cg_liquidation_aggregated_history',
        'cg_spot_aggregated_taker_volume_history',
        'cg_funding_rate_history',
        'cg_long_short_top_account_ratio_history',
        'cg_long_short_global_account_ratio_history',
        'cg_spot_aggregated_ask_bids_history',
        'cg_futures_basis_history'
    ]

    try:
        for table in tables:
            print(f"\n📊 Table: {table}")
            print("-" * 40)

            # Check if table exists
            try:
                # Get table structure
                desc_query = f"DESCRIBE {table}"
                result = db_manager.execute_query(desc_query)

                if not result.empty:
                    print("Columns:")
                    for _, row in result.iterrows():
                        print(f"  - {row['Field']}: {row['Type']}")

                    # Look for timestamp/date columns
                    time_columns = []
                    for _, row in result.iterrows():
                        field_name = row['Field'].lower()
                        if any(keyword in field_name for keyword in ['time', 'date', 'created', 'updated', 'timestamp']):
                            time_columns.append(row['Field'])

                    if time_columns:
                        print(f"🕐 Time-related columns: {', '.join(time_columns)}")

                    # Get sample data to understand structure
                    sample_query = f"SELECT * FROM {table} LIMIT 1"
                    sample_result = db_manager.execute_query(sample_query)

                    if not sample_result.empty:
                        print(f"📋 Sample data keys: {list(sample_result.columns)}")

                    # Get row count
                    count_query = f"SELECT COUNT(*) as count FROM {table}"
                    count_result = db_manager.execute_query(count_query)
                    if not count_result.empty:
                        print(f"📈 Total rows: {count_result['count'].iloc[0]:,}")

                else:
                    print("❌ Table structure not found")

            except Exception as e:
                print(f"❌ Error checking table {table}: {e}")

    except Exception as e:
        print(f"❌ Database error: {e}")

    finally:
        db_manager.close()

def find_timestamp_column(table_name):
    """Find the correct timestamp column for a specific table"""
    print(f"\n🔍 Finding timestamp column for {table_name}")

    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # Get table structure
        desc_query = f"DESCRIBE {table_name}"
        result = db_manager.execute_query(desc_query)

        if not result.empty:
            print("Available columns:")
            for _, row in result.iterrows():
                print(f"  - {row['Field']}: {row['Type']}")

            # Look for potential timestamp columns
            time_candidates = []
            for _, row in result.iterrows():
                field_name = row['Field'].lower()
                if any(keyword in field_name for keyword in ['time', 'date', 'created', 'updated', 'timestamp']):
                    time_candidates.append(row['Field'])

            if time_candidates:
                print(f"\n🕐 Potential timestamp columns: {time_candidates}")

                # Test each candidate to see which has data
                for candidate in time_candidates:
                    try:
                        test_query = f"SELECT {candidate}, COUNT(*) as count FROM {table_name} GROUP BY {candidate} LIMIT 5"
                        test_result = db_manager.execute_query(test_query)
                        if not test_result.empty:
                            print(f"✅ {candidate}: Has {len(test_result)} distinct values")
                            print(f"   Sample values: {test_result[candidate].tolist()}")
                        else:
                            print(f"❌ {candidate}: No data")
                    except:
                        print(f"❌ {candidate}: Query failed")
            else:
                print("❌ No timestamp-like columns found")

    except Exception as e:
        print(f"❌ Error: {e}")

    finally:
        db_manager.close()

if __name__ == "__main__":
    check_table_schemas()

    # Check specific tables in detail
    print(f"\n{'='*80}")
    print("DETAILED TIMESTAMP COLUMN ANALYSIS")
    print("=" * 80)

    find_timestamp_column('cg_spot_price_history')
    find_timestamp_column('cg_open_interest_aggregated_history')
    find_timestamp_column('cg_liquidation_aggregated_history')