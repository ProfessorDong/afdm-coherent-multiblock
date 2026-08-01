"""v4 recipe: default init + mu_ce=10.0 + set-loss weight on gain reduced to zero.

Hypothesis: set-loss pushing h_hat toward true h fights CG-MMSE's implicit
loss which wants h_hat to best fit the DAFT operator. Weighting h in set-loss
to zero (w_h=0) makes the set loss ONLY about (tau, nu) refinement, letting CE
drive h without conflict.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.experiments import (
    ExperimentConfig, evaluate_receiver_sweep, evaluate_classical_sweep,
    genie_mmse_sweep,
)
from afdm.classical import ClassicalCGDetector
from afdm.training import TrainingConfig, train


def main():
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=3, seed=0,
    )
    snrs = [5.0, 15.0, 25.0]
    n_batches = 3; batch_size = 32

    print("=" * 80)
    print("V4 SMOKE: mu_ce=10.0, w_h=0 (drop h from set loss)")
    print(f"Config: N={cfg.N}, N_p={cfg.N_p}, P={cfg.P}")
    print("=" * 80)

    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    r_class = evaluate_classical_sweep(classical, cfg, snrs, n_batches, batch_size, seed=42)
    r_genie = genie_mmse_sweep(cfg, snrs, n_batches, batch_size, seed=42)

    torch.manual_seed(0)
    receiver = cfg.receiver()
    print("Pre:", {snr: f"{r['ser']:.3e}" for snr, r in
                   evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42).items()})

    tc = TrainingConfig(
        lr=3e-4, n_epochs=30, steps_per_epoch=50, batch_size=32,
        snr_db_min=5.0, snr_db_max=25.0, grad_clip=1.0,
        val_every=10, val_batches=2, val_snr_dbs=(15.0,),
        layer_gamma=0.7,
        mu_ce=10.0,       # v4: CE dominates strongly
        eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=0.0, w_ell=0.3, w_kap=0.3, mu_fa=0.05, mu_md=0.05),  # v4: no h in set loss
        log_every=50,
    )
    print(f"Training with mu_ce={tc.mu_ce}, w_h=0.0 (theta-only set loss)...")
    t0 = time.time()
    history = train(receiver, cfg.system(), cfg.channel(), cfg.constellation(),
                    pp, pv, tc, seed=0, verbose=False)
    print(f"  {time.time()-t0:.1f}s; loss init->end: {history['train_loss'][0]:.3f} -> {history['train_loss'][-1]:.3f}")

    r_post = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    print("\nResults:")
    print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'Post':>10s}  {'vs Classical':>14s}")
    for snr in snrs:
        cls = r_class[snr]["ser"]; post = r_post[snr]["ser"]; gen = r_genie[snr]["ser"]
        ratio = post / cls if cls > 0 else float("nan")
        print(f"{snr:>4.1f}dB  {gen:>10.3e}  {cls:>10.3e}  {post:>10.3e}  {ratio:>13.0%}")

    post_15 = r_post[15.0]["ser"]; cls_15 = r_class[15.0]["ser"]
    print(f"\n{'PASS' if post_15 < 0.5 * cls_15 else 'MARGINAL' if post_15 < 0.9 * cls_15 else 'FAIL'}: "
          f"post 15dB = {post_15:.3e}  ({post_15/cls_15:.0%} of classical)")


if __name__ == "__main__":
    main()
