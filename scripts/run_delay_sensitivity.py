"""Figure 7: BER vs ell_max (delay-spread sensitivity with fixed c_1 rule)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from afdm.experiments import ExperimentConfig, load_receiver, save_results_json
from afdm.channels import UniformFractionalChannel
from afdm.classical import ClassicalCGDetector
from afdm.training import sample_batch
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


@torch.no_grad()
def evaluate_at_ell(rx, cfg, ell_max_test, snr_db, n_batches, batch_size, seed=42, detector=None):
    system = cfg.system()
    # Note: fix c_1 from training config (does not update at inference), only vary the channel.
    channel = UniformFractionalChannel(
        P=cfg.P, ell_max=ell_max_test, kappa_max=cfg.kappa_max,
        decay_db_per_path=cfg.decay_db_per_path, device=cfg.device,
    )
    constellation = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, constellation, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        if detector is None:
            out = rx(batch["r"], sigma_w2_block=batch["sigma_w2_block"])
            hard = out["p_ms"].argmax(dim=-1)
        else:
            out = detector.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
            hard = out["hard_x"]
        ser_b = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
        ser += ser_b.item()
    return ser / n_batches


def main(checkpoint: str = "checkpoints/proposed_seed0.pt", snr_db: float = 15.0,
         n_batches: int = 3, batch_size: int = 32):
    set_paper_style()
    rx, cfg = load_receiver(checkpoint)
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(), constellation=cfg.constellation(),
        pilot_positions=pp, pilot_values=pv, T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    ell_grid = [2, 4, 6, 8, 10]
    print(f"Evaluating BER vs ell_max at SNR={snr_db} dB...")
    results = {"classical": {}, "proposed": {}}
    for e in ell_grid:
        results["classical"][e] = evaluate_at_ell(rx, cfg, e, snr_db, n_batches, batch_size, detector=classical)
        results["proposed"][e] = evaluate_at_ell(rx, cfg, e, snr_db, n_batches, batch_size)
        print(f"  ell_max={e}: classical={results['classical'][e]:.4e}, proposed={results['proposed'][e]:.4e}")
    save_results_json(results, str(results_dir() / "delay_sensitivity.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    for name in ["classical", "proposed"]:
        y = [results[name][e] for e in ell_grid]
        ax.semilogy(ell_grid, y, color=COLORS[name], marker=MARKERS[name], label=LABELS[name])
    ax.set_xlabel(r"$\ell_{\max}$ (samples)"); ax.set_ylabel(f"SER at SNR={int(snr_db)} dB")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    save_figure(fig, "delay_sensitivity")
    print("Saved figures/delay_sensitivity.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--snr_db", type=float, default=15.0)
    ap.add_argument("--n_batches", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.snr_db, args.n_batches, args.batch_size)
