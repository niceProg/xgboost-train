#!/bin/bash
"""
Setup cron jobs for automated trading data collection and labeling.
This script configures the complete automated pipeline.
"""

# Get the current directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.xgboostvenv"
LOG_DIR="$SCRIPT_DIR/logs"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "🚀 Setting up cron jobs for automated trading system..."
echo "📁 Script directory: $SCRIPT_DIR"
echo "🐍 Virtual environment: $VENV_PATH"
echo "📝 Log directory: $LOG_DIR"

# Create a temporary crontab file
TEMP_CRON=$(mktemp)

# Get existing crontab (if any)
crontab -l > "$TEMP_CRON" 2>/dev/null || echo "# Trading system cron jobs" > "$TEMP_CRON"

# Add comment header for trading system
cat >> "$TEMP_CRON" << 'EOF'

# ===== AUTOMATED TRADING SYSTEM =====
# Data collection every 5 minutes
# Signal labeling every 15 minutes
# Model training every 4 hours
# Cleanup logs weekly

EOF

# Add signal collection job (every 5 minutes)
cat >> "$TEMP_CRON" << EOF
# Collect trading signals every 5 minutes
*/5 * * * * cd $SCRIPT_DIR && source $VENV_PATH/bin/activate && python collect_signals.py --symbol BTC --pair BTCUSDT --interval 1h --horizon 60 >> $LOG_DIR/collect_signals.log 2>&1

EOF

# Add signal labeling job (every 15 minutes)
cat >> "$TEMP_CRON" << EOF
# Label pending signals every 15 minutes
*/15 * * * * cd $SCRIPT_DIR && source $VENV_PATH/bin/activate && python label_signals.py --limit 100 >> $LOG_DIR/label_signals.log 2>&1

EOF

# Add model training job (every 4 hours)
cat >> "$TEMP_CRON" << EOF
# Train new model every 4 hours
0 */4 * * * cd $SCRIPT_DIR && source $VENV_PATH/bin/activate && python train_model.py --limit 1000 >> $LOG_DIR/train_model.log 2>&1

EOF

# Add log cleanup job (weekly)
cat >> "$TEMP_CRON" << EOF
# Clean up old logs (keep last 7 days)
0 2 * * 0 find $LOG_DIR -name "*.log" -type f -mtime +6 -delete

EOF

# Show what will be installed
echo "📋 Cron jobs to be installed:"
echo "================================"
cat "$TEMP_CRON" | grep -A 50 "AUTOMATED TRADING SYSTEM"

# Install the crontab
echo ""
echo "⏰ Installing cron jobs..."
crontab "$TEMP_CRON"

# Remove temporary file
rm "$TEMP_CRON"

echo ""
echo "✅ Cron jobs installed successfully!"
echo ""
echo "📊 Active cron jobs:"
crontab -l | grep -A 20 "AUTOMATED TRADING SYSTEM"
echo ""
echo "📝 Log files will be created in: $LOG_DIR"
echo "🔍 Monitor logs with: tail -f $LOG_DIR/collect_signals.log"
echo "🔍 Monitor labels with: tail -f $LOG_DIR/label_signals.log"
echo "🔍 Monitor training with: tail -f $LOG_DIR/train_model.log"
echo ""
echo "⚠️  To remove cron jobs, run: crontab -e and delete the trading system section"
echo "⚠️  To list all cron jobs, run: crontab -l"
echo "⚠️  To stop cron jobs temporarily, run: crontab -r (this removes all cron jobs)"