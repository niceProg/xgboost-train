#!/usr/bin/env python3
"""
Check database contents directly
"""

from env_config import get_database_config
from database import DatabaseManager

def check_database():
    """Check what's in the quantconnect_backtests table"""
    db_config = get_database_config()
    db_manager = DatabaseManager(db_config)

    try:
        # Check table structure
        structure_query = "DESCRIBE quantconnect_backtests"
        structure = db_manager.execute_query(structure_query)
        print("📊 Table structure:")
        print(structure)

        # Check if any data exists
        count_query = "SELECT COUNT(*) as count FROM quantconnect_backtests"
        count_result = db_manager.execute_query(count_query)
        print(f"\n📊 Total records: {count_result.iloc[0]['count']}")

        # Show all data
        if count_result.iloc[0]['count'] > 0:
            data_query = "SELECT backtest_id, name, total_return, cagr, win_rate, created_at FROM quantconnect_backtests"
            data = db_manager.execute_query(data_query)
            print(f"\n📊 All records:")
            print(data)
        else:
            print("\n❌ No data found in table")

    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db_manager.close()

if __name__ == "__main__":
    check_database()