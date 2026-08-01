"""Figure 3: Delay and Doppler RMSE vs SNR."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    load_receiver, save_results_json,
)
from afdm.classical import ClassicalCGDetector
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


def main(checkpoint: str = "checkpoints/proposed_seed0.pt", n_batches: int = 4, batch_size: int = 32):
    set_paper_style()
    snr_dbs = [0, 5, 10, 15, 20, 25, 30]
    rx, cfg = load_receiver(checkpoint)
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(), constellation=cfg.constellation(),
        pilot_positions=pp, pilot_values=pv, T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    print("Evaluating classical...")
    r_c = evaluate_classical_sweep(classical, cfg, snr_dbs, n_batches, batch_size, seed=42)
    print("Evaluating proposed...")
    r_p = evaluate_receiver_sweep(rx, cfg, snr_dbs, n_batches, batch_size, seed=42)
    save_results_json({"classical": r_c, "proposed": r_p},
                      str(results_dir() / "delay_doppler_rmse.json"))

    fig, (ax_d, ax_k) = plt.subplots(1, 2, figsize=(6.0, 2.7))
    for name, res in [("classical", r_c), ("proposed", r_p)]:
        y_d = [res[snr]["delay_rmse"] for snr in snr_dbs]
        y_k = [res[snr]["doppler_rmse"] for snr in snr_dbs]
        ax_d.semilogy(snr_dbs, y_d, color=COLORS[name], marker=MARKERS[name], label=LABELS[name])
        ax_k.semilogy(snr_dbs, y_k, color=COLORS[name], marker=MARKERS[name], label=LABELS[name])
    ax_d.set_xlabel("SNR (dB)"); ax_d.set_ylabel("Delay RMSE (samples)"); ax_d.set_title("(a) Delay")
    ax_k.set_xlabel("SNR (dB)"); ax_k.set_ylabel("Doppler RMSE (indices)"); ax_k.set_title("(b) Doppler")
    for ax in (ax_d, ax_k):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, "delay_doppler_rmse")
    print("Saved figures/delay_doppler_rmse.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.n_batches, args.batch_size)
