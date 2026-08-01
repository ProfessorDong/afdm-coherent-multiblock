"""Figure 4: Ablation — SER vs SNR for each independently-trained variant."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from afdm.experiments import evaluate_receiver_sweep, load_receiver, save_results_json
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


def main(ckpt_dir: str = "checkpoints", seed: int = 0, n_batches: int = 4, batch_size: int = 32):
    set_paper_style()
    snr_dbs = [0, 5, 10, 15, 20, 25]
    variants = ["scalars", "attention", "gate", "proposed"]
    results = {}
    for v in variants:
        ckpt_path = Path(ckpt_dir) / f"{v}_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  [skip] {ckpt_path} missing")
            continue
        rx, cfg = load_receiver(str(ckpt_path))
        print(f"Evaluating {v}...")
        results[v] = evaluate_receiver_sweep(rx, cfg, snr_dbs, n_batches, batch_size, seed=42)
    save_results_json(results, str(results_dir() / "ablation.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for v in variants:
        if v not in results: continue
        y = [results[v][snr]["ser"] for snr in snr_dbs]
        ax.semilogy(snr_dbs, y, color=COLORS[v], marker=MARKERS[v], label=LABELS[v])
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Symbol Error Rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    save_figure(fig, "ablation")
    print("Saved figures/ablation.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.ckpt_dir, args.seed, args.n_batches, args.batch_size)
