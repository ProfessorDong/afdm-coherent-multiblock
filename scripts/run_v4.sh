#!/usr/bin/env bash
# v4: 100 epochs, 5x position loss weight, easy then hard.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p runs/pathset_v4

echo ">>> v4 EASY (100 epochs, w_ell=w_kap=5, ~60 min)"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_pathset.py \
    --config easy --epochs 100 --w_ell 5.0 --w_kap 5.0 \
    --out_dir runs/pathset_v4 2>&1 | tee /tmp/train_v4_easy.log

echo ">>> v4 EASY eval"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval_pathset.py \
    --checkpoint runs/pathset_v4/pathset_easy.pt --config easy --n_batches 4 \
    2>&1 | tee /tmp/eval_v4_easy.log

echo ">>> v4 HARD (100 epochs, w_ell=w_kap=5, ~70 min)"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_pathset.py \
    --config hard --epochs 100 --w_ell 5.0 --w_kap 5.0 \
    --out_dir runs/pathset_v4 2>&1 | tee /tmp/train_v4_hard.log

echo ">>> v4 HARD eval"
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/eval_pathset.py \
    --checkpoint runs/pathset_v4/pathset_hard.pt --config hard --n_batches 4 \
    2>&1 | tee /tmp/eval_v4_hard.log

echo ">>> ALL DONE"
