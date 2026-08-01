"""Figure 5: Per-layer BER (convergence with depth) at a fixed SNR."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import torch

from afdm.experiments import load_receiver, save_results_json
from afdm.training import sample_batch
from _figure_utils import set_paper_style, COLORS, MARKERS, LABELS, save_figure, results_dir


@torch.no_grad()
def per_layer_ser(rx, cfg, snr_db: float, n_batches: int, batch_size: int, seed: int = 42):
    channel = cfg.channel()
    constellation = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    all_sers = []
    for _ in range(n_batches):
        batch = sample_batch(cfg.system(), channel, constellation, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        out = rx(batch["r"], sigma_w2_block=batch["sigma_w2_block"], return_layer_states=True)
        sers = []
        for state in out["layer_states"]:
            hard = state["p_ms"].argmax(dim=-1)
            ser = ((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum()
            sers.append(ser.item())
        all_sers.append(sers)
    # Average across batches
    T = len(all_sers[0])
    avg = [sum(s[t] for s in all_sers) / len(all_sers) for t in range(T)]
    return avg


def main(checkpoint: str = "checkpoints/proposed_seed0.pt",
         snr_db: float = 15.0, n_batches: int = 4, batch_size: int = 32):
    set_paper_style()
    rx, cfg = load_receiver(checkpoint)
    print(f"Evaluating per-layer BER at SNR={snr_db} dB...")
    sers = per_layer_ser(rx, cfg, snr_db, n_batches, batch_size)
    save_results_json({"snr_db": snr_db, "per_layer_ser": sers},
                      str(results_dir() / "convergence.json"))

    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    xs = list(range(1, len(sers) + 1))
    ax.semilogy(xs, sers, color=COLORS["proposed"], marker=MARKERS["proposed"], label=f"Proposed ({snr_db:.0f} dB)")
    ax.set_xlabel("Layer index t"); ax.set_ylabel("Symbol Error Rate")
    ax.set_xticks(xs)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    save_figure(fig, "convergence")
    print("Saved figures/convergence.pdf")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/proposed_seed0.pt")
    ap.add_argument("--snr_db", type=float, default=15.0)
    ap.add_argument("--n_batches", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()
    main(args.checkpoint, args.snr_db, args.n_batches, args.batch_size)
