"""Figure 1: BER vs SNR — proposed receiver vs classical baselines vs genie."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    genie_mmse_sweep, load_receiver, save_results_json,
)
from afdm.classical import ClassicalCGDetector
from afdm.pbigabp import PBiGaBPDetector
from afdm.jpnce_sbl import JPNCESBLDetector
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


def main(checkpoint: str = "checkpoints/proposed_seed0.pt", n_batches: int = 4, batch_size: int = 32):
    set_paper_style()
    snr_dbs = [0, 5, 10, 15, 20, 25]

    # Load trained proposed receiver
    rx, cfg = load_receiver(checkpoint)
    pp, pv = cfg.pilots()
    constellation = cfg.constellation()
    sup = cfg.support_recovery()

    # Baselines (untrained model-based detectors, use same config)
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=sup, constellation=constellation,
        pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    pbigabp = PBiGaBPDetector(
        system=cfg.system(), support_recovery=sup, constellation=constellation,
        pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, lambda_h=1e-2, gamma_lr=0.5, gamma_iters=2, omega=20.0, refine_theta=False,
    )
    jpnce_sbl = JPNCESBLDetector(
        system=cfg.system(), constellation=constellation,
        pilot_positions=pp, pilot_values=pv, support_recovery=sup,
        T_em=15, T_grid=2, grid_lr=0.05, magnitude_ratio=0.05, K_cg=10,
    )

    print("Evaluating...")
    results = {}
    print("  genie MMSE...")
    results["genie"] = genie_mmse_sweep(cfg, snr_dbs, n_batches_per_snr=n_batches, batch_size=batch_size, seed=42)
    print("  classical...")
    results["classical"] = evaluate_classical_sweep(classical, cfg, snr_dbs, n_batches_per_snr=n_batches, batch_size=batch_size, seed=42)
    print("  PBiGaBP...")
    results["pbigabp"] = evaluate_classical_sweep(pbigabp, cfg, snr_dbs, n_batches_per_snr=n_batches, batch_size=batch_size, seed=42)
    print("  JPNCE-SBL...")
    results["jpnce_sbl"] = evaluate_classical_sweep(jpnce_sbl, cfg, snr_dbs, n_batches_per_snr=n_batches, batch_size=batch_size, seed=42)
    print("  proposed...")
    results["proposed"] = evaluate_receiver_sweep(rx, cfg, snr_dbs, n_batches_per_snr=n_batches, batch_size=batch_size, seed=42)

    save_results_json(results, str(results_dir() / "ber_vs_snr.json"))

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for name in ["genie", "classical", "pbigabp", "jpnce_sbl", "proposed"]:
        y = [results[name][snr]["ser"] for snr in snr_dbs]
        ax.semilogy(snr_dbs, y, color=COLORS[name], marker=MARKERS[name],
                    linestyle="--" if name == "genie" else "-", label=LABELS[name])
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Symbol Error Rate")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best", framealpha=0.9)
    ax.set_ylim(bottom=1e-5)
    save_figure(fig, "ber_vs_snr")
    print(f"Saved figures/ber_vs_snr.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.n_batches, args.batch_size)
