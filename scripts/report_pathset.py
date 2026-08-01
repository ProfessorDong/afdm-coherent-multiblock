"""Consolidate PathSet checkpoints and produce a Day-5 report.

Runs eval on all checkpoints in a run dir, aggregates SER + path metrics for
both easy and hard configs, and prints a Markdown-ready table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, default="runs/pathset_v3")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"Run dir does not exist: {run_dir}")
        sys.exit(1)

    print(f"# PathSet Report — {run_dir}\n")

    # Find checkpoints per config.
    configs = ("easy", "hard", "hard32")
    for cfg in configs:
        ckpt = run_dir / f"pathset_{cfg}.pt"
        hist_path = run_dir / f"pathset_{cfg}_history.json"
        if not ckpt.exists():
            print(f"## {cfg}: no checkpoint found — skipping\n")
            continue

        print(f"## Config: {cfg}\n")
        if hist_path.exists():
            with open(hist_path) as f:
                h = json.load(f)
            trajectory = h["history"]
            print(f"Training: {h['args']['epochs']} epochs, "
                  f"loss {trajectory[0]:.2f} -> {trajectory[-1]:.2f} "
                  f"({(trajectory[0]-trajectory[-1])/trajectory[0]:.0%} reduction)\n")

        print("Eval:")
        result = subprocess.run(
            ["python3", "scripts/eval_pathset.py",
             "--checkpoint", str(ckpt), "--config", cfg, "--n_batches", "4"],
            capture_output=True, text=True, timeout=180,
        )
        for line in result.stdout.splitlines():
            if line.strip().startswith(("SNR", "0", "1", "2", "5.0", "15.0", "25.0")):
                print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
