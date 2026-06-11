#!/bin/bash
# setup_launchd.sh — install a launchd agent that runs the pipeline daily.
#
# IMPORTANT: run this from a terminal where your project venv is ACTIVE,
# e.g.  (india_jobs) $ bash setup_launchd.sh
# The script bakes the venv's python path into the plist — the previous
# launchd job failed with ModuleNotFoundError because it used system python.
#
# Behaviour:
#   • Runs scheduler.py --now every day at 11:00
#   • If the Mac was asleep at 11:00, launchd runs the job on next wake
#     (StartCalendarInterval jobs are queued, unlike cron)
#   • Output → logs/launchd.log, errors → logs/launchd_error.log
#
# Uninstall:  launchctl unload ~/Library/LaunchAgents/com.aditya.indiajobs.plist
#             rm ~/Library/LaunchAgents/com.aditya.indiajobs.plist

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="$(command -v python3)"
LABEL="com.aditya.indiajobs"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
HOUR="${1:-11}"   # optional: bash setup_launchd.sh 9  → run at 09:00

# Sanity check: the chosen python must have the project deps
if ! "$PYTHON_BIN" -c "import schedule, pandas, sklearn, xgboost" 2>/dev/null; then
    echo "✗ $PYTHON_BIN is missing project dependencies."
    echo "  Activate your venv first (the prompt should show its name), then re-run."
    exit 1
fi

mkdir -p "$PROJECT_DIR/logs" "$HOME/Library/LaunchAgents"

cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>$LABEL</string>

    <key>ProgramArguments</key>
    <array>
        <string>$PYTHON_BIN</string>
        <string>-u</string>
        <string>$PROJECT_DIR/scheduler.py</string>
        <string>--now</string>
    </array>

    <key>WorkingDirectory</key>
    <string>$PROJECT_DIR</string>

    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>$HOUR</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>

    <key>StandardOutPath</key>
    <string>$PROJECT_DIR/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>$PROJECT_DIR/logs/launchd_error.log</string>

    <key>RunAtLoad</key>
    <false/>
</dict>
</plist>
EOF

# Reload cleanly if a previous version is loaded
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load "$PLIST"

echo "✓ Installed: $PLIST"
echo "✓ Python   : $PYTHON_BIN"
echo "✓ Schedule : daily at $(printf '%02d' "$HOUR"):00 (queued until wake if asleep)"
echo
echo "Verify with : launchctl list | grep $LABEL"
echo "Test run now: launchctl start $LABEL   (then tail -f logs/launchd.log)"
