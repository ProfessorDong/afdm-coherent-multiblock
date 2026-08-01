#!/usr/bin/env bash
# Launch the v2 publication training campaign.
#
# Key differences from v1:
#   * Workable config: N_p=32, P=3, P_max=3 (not N_p=16, P=5).
#     Classical CG in this regime reaches ~22% SER at 15 dB, giving the
#     receiver real headroom above the baseline.
#   * v2 init recipe: zero-delta + closed-gate + tiny-LM. Reduces to
#     classical CG at t=0; training only ADDS learned corrections.
#   * mu_ce = 5.0 (up from 0.5): symbol cross-entropy dominates the loss
#     since SER is our metric of interest.
#
# Usage:
#   bash scripts/launch_publication_v2.sh
#   bash scripts/launch_publication_v2.sh --resume
#
# Estimated wall-clock: ~25 h on RTX 4090 (smaller P=3 config = ~40% faster than v1).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

RUN_DIR="runs/pub_v2"

echo ">>> Running pre-flight verification..."
python3 scripts/preflight.py
if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED. Aborting launch."
    exit 1
fi

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/nohup.out.$(date +%Y%m%d_%H%M%S)"

echo ""
echo ">>> Launching v2 publication training in background..."
echo "    run-dir:  $RUN_DIR"
echo "    log:      $LOG_FILE"
echo "    config:   N=128, N_p=32, P=3 (workable regime)"
echo "    recipe:   v2 (zero-delta + closed-gate + tiny-LM), mu_ce=5.0"
echo ""

nohup python3 -u scripts/publication_train.py \
    --run-dir "$RUN_DIR" \
    --N 128 --T 8 --P 3 --N_p 32 --K_cg 10 \
    --d_model 64 --n_heads 4 --n_blocks 3 \
    --init-recipe v2 \
    --mu-ce 5.0 \
    --seeds 0 1 2 \
    --n_epochs 500 --steps_per_epoch 100 \
    --val_every 10 --val_batches 3 \
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
