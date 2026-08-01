"""Short training smoke of the v2 recipe.

If v2 fixes work, we should see:
  * Initial (untrained v2) SER at 15dB ≈ classical CG (~22%).
  * After ~30 epochs, SER at 15dB should drop below 15% (visible improvement).
  * Loss decreases monotonically over epochs.

If any of these fail, iterate on the recipe before launching the full campaign.
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


def apply_v2_init(rx, gate_b=-8.0, gamma_raw=-5.0):
    for layer in rx.layers:
        layer.set_transformer.output_proj.weight.data.zero_()
        layer.set_transformer.output_proj.bias.data.zero_()
        layer.gate.b.data.fill_(gate_b)
        layer.gamma_raw.data.fill_(gamma_raw)
    return rx


def main():
    device = "cuda:0"
    cfg = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, d_model=64, n_heads=4, n_blocks=3, P_max=3, seed=0,
    )
    snrs = [5.0, 15.0, 25.0]
    n_batches = 3; batch_size = 32

    print("=" * 80)
    print("V2 TRAINING SMOKE")
    print(f"Config: N={cfg.N}, N_p={cfg.N_p}, P={cfg.P}, mu_ce=5.0 (up from 0.5)")
    print("=" * 80)

    # Baselines
    pp, pv = cfg.pilots()
    classical = ClassicalCGDetector(
        system=cfg.system(), support_recovery=cfg.support_recovery(),
        constellation=cfg.constellation(), pilot_positions=pp, pilot_values=pv,
        T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
    )
    r_class = evaluate_classical_sweep(classical, cfg, snrs, n_batches, batch_size, seed=42)
    r_genie = genie_mmse_sweep(cfg, snrs, n_batches, batch_size, seed=42)

    # Build receiver with v2 init
    torch.manual_seed(0)
    receiver = apply_v2_init(cfg.receiver())

    # Pre-training evaluation
    print("Pre-training evaluation:")
    r_pre = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    for snr in snrs:
        print(f"  SNR {snr:>4.1f}dB: pre {r_pre[snr]['ser']:.3e}  "
              f"vs classical {r_class[snr]['ser']:.3e} vs genie {r_genie[snr]['ser']:.3e}")

    # Short training run
    tc = TrainingConfig(
        lr=5e-4, n_epochs=30, steps_per_epoch=50, batch_size=32,
        snr_db_min=5.0, snr_db_max=25.0, grad_clip=1.0,
        val_every=10, val_batches=2, val_snr_dbs=(15.0,),
        layer_gamma=0.7,
        mu_ce=5.0,       # v2: much higher CE weight
        eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=50,
    )
    print(f"\nTraining {tc.n_epochs} epochs × {tc.steps_per_epoch} steps (mu_ce={tc.mu_ce})...")
    t0 = time.time()
    history = train(receiver, cfg.system(), cfg.channel(), cfg.constellation(),
                    pp, pv, tc, seed=0, verbose=False)
    train_time = time.time() - t0

    # Report loss trajectory
    losses = history["train_loss"]
    print(f"Training done in {train_time:.1f}s ({tc.n_epochs} epochs)")
    print(f"  Loss trajectory: init {losses[0]:.3f} -> ep10 {losses[9]:.3f} -> ep20 {losses[19]:.3f} -> ep30 {losses[-1]:.3f}")

    # Post-training evaluation
    print("\nPost-training evaluation:")
    r_post = evaluate_receiver_sweep(receiver, cfg, snrs, n_batches, batch_size, seed=42)
    print(f"{'SNR':<6s}  {'Genie':>10s}  {'Classical':>10s}  {'Pre-train':>10s}  {'Post-train':>10s}  {'Delta':>10s}")
    for snr in snrs:
        pre = r_pre[snr]["ser"]; post = r_post[snr]["ser"]
        cls = r_class[snr]["ser"]; gen = r_genie[snr]["ser"]
        delta = pre - post
        print(f"{snr:>4.1f}dB  {gen:>10.3e}  {cls:>10.3e}  {pre:>10.3e}  {post:>10.3e}  {delta:+.3e}")

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)
    post_15 = r_post[15.0]["ser"]
    class_15 = r_class[15.0]["ser"]
    ratio = post_15 / class_15
    if ratio < 0.5:
        print(f"PASS: post-train SER at 15dB ({post_15:.3e}) is <50% of classical ({class_15:.3e}).")
        print("      -> The v2 recipe learns effectively. Safe to launch full campaign.")
    elif ratio < 0.9:
        print(f"MARGINAL: post-train SER at 15dB ({post_15:.3e}) is only ~{ratio:.0%} of classical.")
        print("          Longer training may or may not help. Consider more hyperparameter tuning.")
    else:
        print(f"FAIL: post-train SER at 15dB ({post_15:.3e}) is ~{ratio:.0%} of classical ({class_15:.3e}).")
        print("      The v2 recipe is NOT learning. Diagnose further before launching campaign.")


if __name__ == "__main__":
    main()
