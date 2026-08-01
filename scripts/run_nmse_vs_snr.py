"""Figure 2: Channel-estimation NMSE vs SNR.

Compares:
  * Classical ridge-LS (from Algorithm 1)
  * Ungated learned receiver (variant with g=1) — should show high-SNR floor
  * Proposed uncertainty-gated receiver — should NOT floor
  * Cramér-Rao slope (analytical lower bound reference)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import matplotlib.pyplot as plt

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    load_receiver, save_results_json,
)
from afdm.classical import ClassicalCGDetector
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


def main(proposed_ckpt: str = "checkpoints/proposed_seed0.pt",
         ungated_ckpt: str = "checkpoints/attention_seed0.pt",
         n_batches: int = 4, batch_size: int = 32):
    set_paper_style()
    snr_dbs = [0, 5, 10, 15, 20, 25, 30]

    rx_prop, cfg = load_receiver(proposed_ckpt)
    try:
        rx_ung, _ = load_receiver(ungated_ckpt)
    except FileNotFoundError:
        rx_ung = None

    pp, pv = cfg.pilots()
    constellation = cfg.constellation()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(), constellation=constellation,
        pilot_positions=pp, pilot_values=pv, T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )

    results = {}
    print("Evaluating classical...")
    results["classical"] = evaluate_classical_sweep(classical, cfg, snr_dbs, n_batches, batch_size, seed=42)
    print("Evaluating proposed...")
    results["proposed"] = evaluate_receiver_sweep(rx_prop, cfg, snr_dbs, n_batches, batch_size, seed=42)
    if rx_ung is not None:
        print("Evaluating ungated (attention-only ablation)...")
        results["ungated"] = evaluate_receiver_sweep(rx_ung, cfg, snr_dbs, n_batches, batch_size, seed=42)

    # CRB reference (analytical): sigma^2 / N * P for a well-separated pilot regressor.
    crb_slope = [10 * np.log10(cfg.P / (cfg.N_p * 10 ** (snr / 10))) for snr in snr_dbs]

    save_results_json(results, str(results_dir() / "nmse_vs_snr.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    # NMSE in dB
    for name in ["classical", "ungated", "proposed"]:
        if name not in results: continue
        y = [10 * np.log10(max(results[name][snr]["nmse_h"], 1e-10)) for snr in snr_dbs]
        ax.plot(snr_dbs, y, color=COLORS[name], marker=MARKERS[name],
                label=LABELS[name])
    ax.plot(snr_dbs, crb_slope, "k--", alpha=0.6, label="CRB slope (ref.)")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Channel NMSE (dB)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    save_figure(fig, "nmse_vs_snr")
    print("Saved figures/nmse_vs_snr.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--proposed_ckpt", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--ungated_ckpt", default="checkpoints/attention_seed0.pt")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.proposed_ckpt, args.ungated_ckpt, args.n_batches, args.batch_size)
