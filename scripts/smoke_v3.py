"""v3 recipe smoke: default init + mu_ce=5.0 + workable config.

Hypothesis: v1 training failed because mu_ce=0.5 down-weighted the SER-relevant
CE loss and set-loss (5x weight) dominated gradient signal — pushing the model
toward accurate h/theta but away from good symbol posterior.

v3 recipe:
  * DEFAULT init (random SetTransformer, gate open ~0.3-0.8) — healthy gradients.
  * mu_ce = 5.0 (dominates set loss).
  * Same workable config (N_p=32, P=3).

If this trains monotonically, launch full campaign.
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
    print("V3 TRAINING SMOKE: default init + mu_ce=5.0")
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
    receiver = cfg.receiver()  # default init — no fixes applied

    print("Pre-training:")
    r_pre = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    for snr in snrs:
        print(f"  {snr:>4.1f}dB: pre {r_pre[snr]['ser']:.3e}  vs classical {r_class[snr]['ser']:.3e}  vs genie {r_genie[snr]['ser']:.3e}")

    tc = TrainingConfig(
        lr=5e-4, n_epochs=30, steps_per_epoch=50, batch_size=32,
        snr_db_min=5.0, snr_db_max=25.0, grad_clip=1.0,
        val_every=10, val_batches=2, val_snr_dbs=(15.0,),
        layer_gamma=0.7, mu_ce=5.0, eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=50,
    )
    print(f"\nTraining {tc.n_epochs} epochs × {tc.steps_per_epoch} steps (mu_ce={tc.mu_ce})...")
    t0 = time.time()
    history = train(receiver, cfg.system(), cfg.channel(), cfg.constellation(),
                    pp, pv, tc, seed=0, verbose=False)
    print(f"Training done in {time.time()-t0:.1f}s")
    losses = history["train_loss"]
    print(f"  Loss: init {losses[0]:.3f} -> ep10 {losses[9]:.3f} -> ep20 {losses[19]:.3f} -> ep30 {losses[-1]:.3f}")

    print("\nPost-training:")
    r_post = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'Pre':>10s}  {'Post':>10s}  {'Delta':>10s}")
    for snr in snrs:
        pre = r_pre[snr]["ser"]; post = r_post[snr]["ser"]
        cls = r_class[snr]["ser"]; gen = r_genie[snr]["ser"]
        print(f"{snr:>4.1f}dB  {gen:>10.3e}  {cls:>10.3e}  {pre:>10.3e}  {post:>10.3e}  {pre-post:+.3e}")

    print("\n" + "=" * 80)
    post_15 = r_post[15.0]["ser"]; cls_15 = r_class[15.0]["ser"]
    ratio = post_15 / cls_15
    if ratio < 0.5:
        print(f"PASS: post-train SER at 15dB ({post_15:.3e}) is <50% of classical ({cls_15:.3e}). Launch v3 campaign.")
    elif ratio < 0.9:
        print(f"MARGINAL: post-train SER ({post_15:.3e}) is ~{ratio:.0%} of classical.")
    else:
        print(f"FAIL: post-train SER ({post_15:.3e}) is ~{ratio:.0%} of classical. Iterate.")


if __name__ == "__main__":
    main()
