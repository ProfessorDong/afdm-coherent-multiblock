#!/usr/bin/env bash
# Launch the publication training campaign with nohup + timestamped logs.
#
# Usage:
#   bash scripts/launch_publication.sh                        # default: seeds 0 1 2, 500 epochs
#   bash scripts/launch_publication.sh --n_epochs 300 --seeds 0
#   bash scripts/launch_publication.sh --resume               # resume from last checkpoint
#
# After launch:
#   Monitor:  python scripts/monitor_training.py --run-dir runs/pub_v1 --watch
#   Log tail: tail -f runs/pub_v1/campaign.log
#   Stop:     kill $(cat runs/pub_v1/pid.txt)   # sends SIGTERM; trainer saves state + exits

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

RUN_DIR="runs/pub_v1"

# 1. Pre-flight check
echo ">>> Running pre-flight verification..."
python3 scripts/preflight.py
if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED. Aborting launch."
    exit 1
fi

# 2. Create run dir
mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/nohup.out.$(date +%Y%m%d_%H%M%S)"

# 3. Launch with nohup, unbuffered Python output
echo ""
echo ">>> Launching publication training in background..."
echo "    log:      $LOG_FILE"
echo "    campaign: $RUN_DIR/campaign.log"
echo ""

nohup python3 -u scripts/publication_train.py \
    --run-dir "$RUN_DIR" \
    "$@" \
    > "$LOG_FILE" 2>&1 &

PID=$!
echo "$PID" > "$RUN_DIR/pid.txt"
echo ">>> Started PID $PID (saved to $RUN_DIR/pid.txt)"
echo ""
echo "Monitor commands:"
echo "  python scripts/monitor_training.py --run-dir $RUN_DIR"
echo "  python scripts/monitor_training.py --run-dir $RUN_DIR --watch"
echo "  tail -f $LOG_FILE"
echo "  tail -f $RUN_DIR/campaign.log"
echo ""
echo "Stop cleanly (saves state):"
echo "  kill \$(cat $RUN_DIR/pid.txt)"
