"""Figure 9: BER vs SNR at multiple pilot counts (uniform deterministic pattern)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    load_receiver, save_results_json,
)
from afdm.classical import ClassicalCGDetector
from _figure_utils import set_paper_style, save_figure, results_dir


def main(checkpoint: str = "checkpoints/proposed_seed0.pt", n_batches: int = 3, batch_size: int = 32):
    """Note: this uses a single trained receiver at N_p=16. Truly comparing N_p=8, 16, 32
    would require training three separate receivers. As a first-order study, we evaluate
    the classical baseline at each N_p and the proposed receiver at its trained N_p only.
    """
    set_paper_style()
    snr_dbs = [0, 5, 10, 15, 20, 25]
    rx, cfg = load_receiver(checkpoint)

    from afdm.pilots import uniform_daft_pilots
    from afdm.experiments import qpsk_constellation
    from afdm.training import sample_batch

    results = {}
    for N_p in [8, 16, 32]:
        pp = uniform_daft_pilots(N=cfg.N, N_p=N_p, device=cfg.device)
        gen = torch.Generator(device=cfg.device); gen.manual_seed(cfg.seed + 999)
        pv = cfg.constellation()[torch.randint(0, cfg.constellation().numel(), (N_p,), device=cfg.device, generator=gen)]
        sup = cfg.support_recovery()
        classical = ClassicalCGDetector(
            system=cfg.system(), support_recovery=sup, constellation=cfg.constellation(),
            pilot_positions=pp, pilot_values=pv, T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
        )
        r_c = evaluate_classical_sweep(classical, cfg, snr_dbs, n_batches, batch_size, seed=42)
        results[f"classical_Np{N_p}"] = r_c
        print(f"  classical N_p={N_p} done")

    # Proposed at its trained N_p
    r_p = evaluate_receiver_sweep(rx, cfg, snr_dbs, n_batches, batch_size, seed=42)
    results[f"proposed_Np{cfg.N_p}"] = r_p
    print(f"  proposed N_p={cfg.N_p} done")

    save_results_json(results, str(results_dir() / "pilot_overhead.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    palette = ["tab:blue", "tab:orange", "tab:green"]
    for i, N_p in enumerate([8, 16, 32]):
        r = results[f"classical_Np{N_p}"]
        y = [r[snr]["ser"] for snr in snr_dbs]
        ax.semilogy(snr_dbs, y, color=palette[i], marker="s", linestyle="--",
                    label=f"Classical, $N_p={N_p}$")
    r = results[f"proposed_Np{cfg.N_p}"]
    y = [r[snr]["ser"] for snr in snr_dbs]
    ax.semilogy(snr_dbs, y, color="tab:red", marker="o", label=f"Proposed, $N_p={cfg.N_p}$")
    ax.set_xlabel("SNR (dB)"); ax.set_ylabel("Symbol Error Rate")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
    save_figure(fig, "pilot_overhead")
    print("Saved figures/pilot_overhead.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.n_batches, args.batch_size)
