#!/usr/bin/env bash
# Launch the v3 publication training campaign.
#
# Key differences from v2 (which failed cold-start):
#   * Default init (random Set-Transformer): pre-training SER at 15dB is already
#     ~13% vs classical CG 22.95% — the random Set-Transformer helps by accident,
#     and training refines this rather than starting from a plateau.
#   * mu_ce = 5.0 (up from v1's 0.5): symbol cross-entropy dominates, matching
#     the actual metric of interest (SER).
#   * Workable config N_p=32, P=3, P_max=3.
#
# Estimated wall-clock: ~25 h on RTX 4090.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$CODE_DIR"

RUN_DIR="runs/pub_v3"

echo ">>> Running pre-flight verification..."
python3 scripts/preflight.py
if [ $? -ne 0 ]; then
    echo "PRE-FLIGHT FAILED. Aborting launch."
    exit 1
fi

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/nohup.out.$(date +%Y%m%d_%H%M%S)"

echo ""
echo ">>> Launching v3 publication training in background..."
echo "    run-dir:  $RUN_DIR"
echo "    log:      $LOG_FILE"
echo "    config:   N=128, N_p=32, P=3 (workable), mu_ce=5.0, default init"
echo ""

nohup python3 -u scripts/publication_train.py \
    --run-dir "$RUN_DIR" \
    --N 128 --T 8 --P 3 --N_p 32 --K_cg 10 \
    --d_model 64 --n_heads 4 --n_blocks 3 \
    --init-recipe default \
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
