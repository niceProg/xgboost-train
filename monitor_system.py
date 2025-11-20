#!/usr/bin/env python3
"""
Monitor the automated trading system status.
This script checks the health of data collection, labeling, and model training.
"""

import argparse
import sys
import os
import subprocess
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

from env_config import get_database_config
from database import DatabaseManager

class SystemMonitor:
    def __init__(self):
        self.db_config = get_database_config()
        self.db_manager = DatabaseManager(self.db_config)
        self.log_dir = Path("logs")

    def check_cron_jobs(self):
        """Check if cron jobs are running"""
        print("🔍 Checking cron jobs...")
        try:
            result = subprocess.run(['crontab', '-l'], capture_output=True, text=True)
            if result.returncode == 0:
                cron_content = result.stdout
                trading_jobs = [line for line in cron_content.split('\n')
                              if any(keyword in line for keyword in ['collect_signals', 'label_signals', 'train_model'])]

                print(f"   ✅ Found {len(trading_jobs)} trading system cron jobs")
                for job in trading_jobs:
                    print(f"   📅 {job}")
                return len(trading_jobs) > 0
            else:
                print("   ❌ No cron jobs found")
                return False
        except Exception as e:
            print(f"   ❌ Error checking cron jobs: {e}")
            return False

    def check_database_connectivity(self):
        """Check database connection"""
        print("🔍 Checking database connectivity...")
        try:
            result = self.db_manager.execute_query("SELECT 1 as test")
            if not result.empty:
                print("   ✅ Database connection successful")
                return True
            else:
                print("   ❌ Database query failed")
                return False
        except Exception as e:
            print(f"   ❌ Database connection failed: {e}")
            return False

    def check_signal_collection(self):
        """Check recent signal collection"""
        print("🔍 Checking signal collection...")
        try:
            # Check signals collected in the last hour
            one_hour_ago = datetime.now() - timedelta(hours=1)
            query = """
                SELECT COUNT(*) as count,
                       MAX(generated_at) as latest,
                       AVG(signal_score) as avg_score
                FROM cg_train_dataset
                WHERE generated_at >= %s
            """

            result = self.db_manager.execute_query(query, (one_hour_ago,))

            if not result.empty:
                count = result.iloc[0]['count']
                latest = result.iloc[0]['latest']
                avg_score = result.iloc[0]['avg_score']

                if count > 0:
                    print(f"   ✅ {count} signals collected in the last hour")
                    print(f"   📅 Latest signal: {latest}")
                    print(f"   📊 Average signal score: {avg_score:.3f}" if avg_score else "")
                    return True
                else:
                    print("   ⚠️  No signals collected in the last hour")
                    return False
            else:
                print("   ❌ Unable to check signal collection")
                return False

        except Exception as e:
            print(f"   ❌ Error checking signal collection: {e}")
            return False

    def check_signal_labeling(self):
        """Check signal labeling status"""
        print("🔍 Checking signal labeling...")
        try:
            # Check label statistics
            query = """
                SELECT label_status, label_direction, COUNT(*) as count
                FROM cg_train_dataset
                GROUP BY label_status, label_direction
                ORDER BY label_status, label_direction
            """

            result = self.db_manager.execute_query(query)

            if not result.empty:
                total_signals = 0
                labeled_signals = 0
                pending_signals = 0

                print("   📊 Label Status:")
                for index, row in result.iterrows():
                    status = row['label_status'] or 'NULL'
                    direction = row['label_direction'] or 'NULL'
                    count = row['count']
                    total_signals += count

                    if status == 'labeled':
                        labeled_signals += count
                        print(f"      ✅ {status:8} | {direction:6} | {count:4}")
                    elif status == 'pending':
                        pending_signals += count
                        print(f"      ⏳ {status:8} | {direction:6} | {count:4}")

                if total_signals > 0:
                    label_rate = (labeled_signals / total_signals) * 100
                    print(f"   📈 Overall labeling rate: {label_rate:.1f}%")

                    if labeled_signals > 0:
                        print(f"   ✅ System is actively labeling signals")
                        return True
                    else:
                        print(f"   ⚠️  No signals labeled yet")
                        return False
                else:
                    print("   ⚠️  No signals in database yet")
                    return False
            else:
                print("   ❌ Unable to check signal labeling")
                return False

        except Exception as e:
            print(f"   ❌ Error checking signal labeling: {e}")
            return False

    def check_model_training(self):
        """Check model training status"""
        print("🔍 Checking model training...")

        # Check for trained model files
        model_dir = Path(".")
        model_files = list(model_dir.glob("xgboost_trading_model_*.joblib"))

        if model_files:
            # Get the most recent model
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model_time = datetime.fromtimestamp(latest_model.stat().st_mtime)
            model_age = datetime.now() - model_time

            print(f"   ✅ Found trained model: {latest_model.name}")
            print(f"   📅 Model trained: {model_time} ({model_age} ago)")

            if model_age < timedelta(hours=6):
                print(f"   ✅ Model is recent (less than 6 hours old)")
                return True
            else:
                print(f"   ⚠️  Model is old ({model_age} ago)")
                return False
        else:
            print("   ❌ No trained model found")
            return False

    def check_log_files(self):
        """Check log file health"""
        print("🔍 Checking log files...")

        if not self.log_dir.exists():
            print("   ⚠️  Log directory does not exist")
            return False

        log_files = {
            'Signal Collection': self.log_dir / 'collect_signals.log',
            'Signal Labeling': self.log_dir / 'label_signals.log',
            'Model Training': self.log_dir / 'train_model.log'
        }

        all_good = True
        for name, log_file in log_files.items():
            if log_file.exists():
                size = log_file.stat().st_size
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                age = datetime.now() - mtime

                print(f"   ✅ {name}: {size:,} bytes, updated {age} ago")

                # Check if log file was updated recently (within 30 minutes)
                if age > timedelta(minutes=30):
                    print(f"      ⚠️  Log file hasn't been updated recently")
                    all_good = False
            else:
                print(f"   ❌ {name}: Log file not found")
                all_good = False

        return all_good

    def get_system_summary(self):
        """Get overall system summary"""
        print("\n📊 SYSTEM SUMMARY")
        print("=" * 50)

        try:
            # Get total counts
            query = """
                SELECT
                    COUNT(*) as total_signals,
                    COUNT(CASE WHEN label_status = 'labeled' THEN 1 END) as labeled_signals,
                    COUNT(CASE WHEN label_status = 'pending' THEN 1 END) as pending_signals,
                    MAX(generated_at) as latest_signal,
                    AVG(price_now) as avg_price
                FROM cg_train_dataset
            """

            result = self.db_manager.execute_query(query)

            if not result.empty:
                row = result.iloc[0]
                total = row['total_signals']
                labeled = row['labeled_signals']
                pending = row['pending_signals']
                latest = row['latest_signal']
                avg_price = row['avg_price']

                print(f"📈 Total Signals: {total:,}")
                print(f"✅ Labeled Signals: {labeled:,}")
                print(f"⏳ Pending Signals: {pending:,}")
                print(f"📅 Latest Signal: {latest}")
                print(f"💰 Average Price: ${avg_price:,.2f}" if avg_price else "")

                if total > 0:
                    label_rate = (labeled / total) * 100
                    print(f"🏷️  Labeling Rate: {label_rate:.1f}%")
        except Exception as e:
            print(f"❌ Error getting system summary: {e}")

    def run_health_check(self, verbose=False):
        """Run complete health check"""
        print("🏥 TRADING SYSTEM HEALTH CHECK")
        print("=" * 50)
        print(f"📅 Check time: {datetime.now()}")
        print()

        checks = [
            ("Cron Jobs", self.check_cron_jobs),
            ("Database", self.check_database_connectivity),
            ("Signal Collection", self.check_signal_collection),
            ("Signal Labeling", self.check_signal_labeling),
            ("Model Training", self.check_model_training),
        ]

        if verbose:
            checks.append(("Log Files", self.check_log_files))

        results = []
        for name, check_func in checks:
            try:
                result = check_func()
                results.append((name, result))
            except Exception as e:
                print(f"❌ Error in {name}: {e}")
                results.append((name, False))
            print()

        # Summary
        passed = sum(1 for _, result in results if result)
        total = len(results)

        print("🎯 HEALTH CHECK SUMMARY")
        print("=" * 30)
        print(f"✅ Passed: {passed}/{total}")

        for name, result in results:
            status = "✅" if result else "❌"
            print(f"{status} {name}")

        print(f"\n🏆 Overall System Health: {(passed/total)*100:.0f}%")

        # Show detailed summary
        self.get_system_summary()

        return passed == total

def main():
    parser = argparse.ArgumentParser(description='Monitor automated trading system')
    parser.add_argument('--verbose', action='store_true', help='Show detailed log file information')
    parser.add_argument('--cron-only', action='store_true', help='Check only cron jobs')
    parser.add_argument('--db-only', action='store_true', help='Check only database')

    args = parser.parse_args()

    try:
        monitor = SystemMonitor()

        if args.cron_only:
            monitor.check_cron_jobs()
        elif args.db_only:
            monitor.check_database_connectivity()
        else:
            monitor.run_health_check(verbose=args.verbose)

    except KeyboardInterrupt:
        print("\n👋 System monitoring interrupted")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()