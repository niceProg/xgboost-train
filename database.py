import pymysql
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

class DatabaseManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self._connect()

    def _connect(self) -> None:
        """Establish database connection"""
        try:
            # Extract connection parameters from config, excluding connect_timeout
            connection_params = {
                'host': self.config['host'],
                'user': self.config['user'],
                'password': self.config['password'],
                'database': self.config['database'],
                'charset': self.config['charset'],
                'cursorclass': pymysql.cursors.DictCursor,
            }

            # Add connect_timeout separately if it exists in config
            if 'connect_timeout' in self.config:
                connection_params['connect_timeout'] = self.config['connect_timeout']

            self.connection = pymysql.connect(**connection_params)
            print("✅ Database connection established successfully")
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            raise

    def _ensure_connection(self) -> None:
        """Ensure database connection is alive"""
        try:
            if not self.connection or not self.connection.open:
                self._connect()
            else:
                self.connection.ping(reconnect=True)
        except Exception as e:
            print(f"Connection check failed: {e}")
            self._connect()

    def execute_query(self, query: str, params: Optional[Tuple] = None) -> pd.DataFrame:
        """Execute SQL query and return results as DataFrame"""
        self._ensure_connection()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                result = cursor.fetchall()
                return pd.DataFrame(result) if result else pd.DataFrame()
        except Exception as e:
            print(f"Query execution failed: {e}")
            raise

    def execute_insert(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute INSERT query and return affected rows"""
        self._ensure_connection()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Insert execution failed: {e}")
            self.connection.rollback()
            raise

    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """Execute UPDATE query and return affected rows"""
        self._ensure_connection()

        try:
            with self.connection.cursor() as cursor:
                cursor.execute(query, params)
                self.connection.commit()
                return cursor.rowcount
        except Exception as e:
            print(f"Update execution failed: {e}")
            self.connection.rollback()
            raise

    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()