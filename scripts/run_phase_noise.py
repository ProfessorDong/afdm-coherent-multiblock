"""Figure 8: BER vs Wiener phase-noise std at fixed SNR."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import torch

from afdm.experiments import ExperimentConfig, load_receiver, save_results_json
from afdm.classical import ClassicalCGDetector
from afdm.operators import FastAFDMOperator
from afdm.training import sample_batch
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


def apply_wiener_phase_noise(y_time: torch.Tensor, sigma_phi: float) -> torch.Tensor:
    """Multiply the time-domain signal by a Wiener phase-noise process."""
    B, L = y_time.shape
    inc = sigma_phi * torch.randn(B, L, device=y_time.device)
    phi = torch.cumsum(inc, dim=-1)
    return y_time * torch.exp(1j * phi).to(y_time.dtype)


@torch.no_grad()
def evaluate_at_pn(rx, cfg, sigma_phi, snr_db, n_batches, batch_size, seed=42, detector=None):
    system = cfg.system()
    channel = cfg.channel()
    constellation = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, constellation, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        r_pn = apply_wiener_phase_noise(batch["r"], sigma_phi)
        if detector is None:
            out = rx(r_pn, sigma_w2_block=batch["sigma_w2_block"])
            hard = out["p_ms"].argmax(dim=-1)
        else:
            out = detector.detect(r_pn, sigma_w2=batch["sigma_w2_block"])
            hard = out["hard_x"]
        ser_b = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
        ser += ser_b.item()
    return ser / n_batches


def main(checkpoint: str = "checkpoints/proposed_seed0.pt", snr_db: float = 20.0,
         n_batches: int = 3, batch_size: int = 32):
    set_paper_style()
    rx, cfg = load_receiver(checkpoint)
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(), constellation=cfg.constellation(),
        pilot_positions=pp, pilot_values=pv, T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    sigma_phis = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]
    print(f"Evaluating BER vs phase-noise std at SNR={snr_db} dB...")
    results = {"classical": {}, "proposed": {}}
    for s in sigma_phis:
        results["classical"][s] = evaluate_at_pn(rx, cfg, s, snr_db, n_batches, batch_size, detector=classical)
        results["proposed"][s] = evaluate_at_pn(rx, cfg, s, snr_db, n_batches, batch_size)
        print(f"  sigma_phi={s}: classical={results['classical'][s]:.4e}, proposed={results['proposed'][s]:.4e}")
    save_results_json(results, str(results_dir() / "phase_noise.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for name in ["classical", "proposed"]:
        y = [results[name][s] for s in sigma_phis]
        ax.semilogy(sigma_phis, y, color=COLORS[name], marker=MARKERS[name], label=LABELS[name])
    ax.set_xlabel(r"Wiener phase-noise std $\sigma_\phi$")
    ax.set_ylabel(f"SER at SNR={int(snr_db)} dB")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    save_figure(fig, "phase_noise")
    print("Saved figures/phase_noise.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--snr_db", type=float, default=20.0)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.snr_db, args.n_batches, args.batch_size)
