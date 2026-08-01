"""Train the proposed receiver + 3 ablation variants, save checkpoints.

Runs on cuda:0 (RTX 4090). Trains each variant with the same optimizer, schedule,
and training-channel distribution so the ablation comparison is fair (each variant
is INDEPENDENTLY trained — no jointly-trained-then-deleted variants, per the
reviewer's methodology requirement).

Usage:
  python scripts/train_all.py                 # default: 100 epochs × 40 steps
  python scripts/train_all.py --quick         # smoke: 20 epochs × 20 steps
  python scripts/train_all.py --publication   # 500 epochs × 100 steps
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import ExperimentConfig, build_ablation, train_receiver
from afdm.training import TrainingConfig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smoke training (20 epochs)")
    ap.add_argument("--publication", action="store_true", help="publication training (500 epochs)")
    ap.add_argument("--variants", nargs="+",
                    default=["proposed", "gate", "attention", "scalars"],
                    help="ablation variants to train")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0],
                    help="training seeds (defaults to single seed 0)")
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--P", type=int, default=5)
    args = ap.parse_args()

    if args.quick:
        n_epochs, steps_per_epoch = 20, 20
    elif args.publication:
        n_epochs, steps_per_epoch = 500, 100
    else:
        n_epochs, steps_per_epoch = 100, 40

    config = ExperimentConfig(
        N=args.N, kappa_max=5.0, ell_max=10.0,
        P=args.P, N_p=16,
        T=args.T, K_cg=10, d_model=64, n_heads=4, n_blocks=3,
        P_max=args.P,  # perfect cardinality assumption
    )
    tc = TrainingConfig(
        lr=5e-4, n_epochs=n_epochs, steps_per_epoch=steps_per_epoch,
        batch_size=32, snr_db_min=5.0, snr_db_max=25.0,
        grad_clip=1.0, val_every=max(n_epochs // 5, 1), val_batches=3,
        val_snr_dbs=(5.0, 15.0, 25.0),
        layer_gamma=0.7, mu_ce=0.5, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=max(steps_per_epoch, 1),  # once per epoch
    )
    checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        config.seed = seed
        for variant in args.variants:
            print(f"\n{'='*66}\n>>> Training variant: {variant} (seed={seed})\n{'='*66}")
            torch.manual_seed(seed)
            rx = build_ablation(variant, config)
            t0 = time.time()
            hist = train_receiver(
                rx, config, tc,
                checkpoint_path=str(checkpoint_dir / f"{variant}_seed{seed}.pt"),
                verbose=True,
            )
            print(f">>> variant={variant} seed={seed} done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
