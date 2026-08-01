"""Read the JSON state files from a publication training run and print status.

Usage:
  python scripts/monitor_training.py --run-dir runs/pub_v1
  python scripts/monitor_training.py --run-dir runs/pub_v1 --watch    # refresh every 30s
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def read_state(run_root: Path) -> list[dict]:
    """Collect the state.json from every per-variant-seed subdirectory."""
    states = []
    for d in sorted(run_root.iterdir()):
        if d.is_dir():
            sf = d / "state.json"
            if sf.exists():
                try:
                    with open(sf) as f:
                        st = json.load(f)
                    st["_dir"] = d.name
                    states.append(st)
                except Exception:
                    continue
    return states


def read_summary(run_root: Path) -> dict | None:
    p = run_root / "summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}m"
    return f"{seconds/3600:.2f}h"


def print_status(run_root: Path):
    print("=" * 90)
    print(f"Run dir: {run_root.resolve()}")
    print(f"Time now: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    summary = read_summary(run_root)
    if summary is not None:
        print(f"CAMPAIGN COMPLETE ({summary['campaign_elapsed_h']:.2f}h)")
    print("=" * 90)

    states = read_state(run_root)
    if not states:
        print("No per-variant state files found yet.")
        return
    # Column header
    print(f"{'Run':<28s}  {'Epoch':>10s}  {'Loss':>10s}  {'Best SER':>12s}  {'Elapsed':>8s}  {'ETA':>8s}")
    print("-" * 90)
    for st in states:
        name = st["_dir"]
        ep = st.get("epoch", 0)
        n_ep = st.get("training_config", {}).get("n_epochs", "?")
        loss = st.get("avg_loss", float("nan"))
        best = st.get("best_metric", float("inf"))
        elapsed = st.get("elapsed_h", 0.0) * 3600
        eta = st.get("eta_seconds_remaining", 0.0)
        best_str = f"{best:.4e}" if best != float("inf") else "n/a"
        print(f"{name:<28s}  {ep}/{n_ep}".ljust(52) +
              f"  {loss:>10.4f}  {best_str:>12s}  {format_time(elapsed):>8s}  {format_time(eta):>8s}")

    # Overall campaign progress
    n_done = sum(1 for st in states if st.get("epoch", 0) >= st.get("training_config", {}).get("n_epochs", 1))
    print("-" * 90)
    print(f"Variants completed: {n_done} / {len(states)} (per-run state files seen)")
    total_elapsed = sum(st.get("elapsed_h", 0.0) for st in states)
    print(f"Total elapsed (sum across variants): {total_elapsed:.2f}h")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/pub_v1")
    ap.add_argument("--watch", action="store_true", help="refresh every 30 seconds")
    args = ap.parse_args()
    run_root = Path(args.run_dir)
    if not run_root.exists():
        print(f"Run dir {run_root} does not exist yet.")
        sys.exit(1)
    if args.watch:
        try:
            while True:
                print("\n" * 3)  # clear a bit
                print_status(run_root)
                time.sleep(30)
        except KeyboardInterrupt:
            print("Monitor stopped.")
    else:
        print_status(run_root)


if __name__ == "__main__":
    main()
