import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_database_config():
    """Get database configuration from environment variables"""
    return {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'market_data_db'),
        'charset': 'utf8mb4',
        'connect_timeout': 30
    }

# Test database connection
def test_connection():
    """Test database connection"""
    try:
        from database import DatabaseManager
        db_config = get_database_config()
        db = DatabaseManager(db_config)

        # Test query
        result = db.execute_query("SELECT 1 as test")
        print("✓ Database connection successful!")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()