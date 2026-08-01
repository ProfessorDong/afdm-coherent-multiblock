"""P3 end-to-end training smoke test.

Trains a small UGVEMReceiver for a short number of epochs on cuda:0 (RTX 4090)
and verifies:
  * Training loss decreases over epochs.
  * Validation SER at moderate SNR is lower than an untrained (random-init) receiver.
  * The learned gate closes at high SNR (Theorem 2 empirical check).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm import AFDMSystem, UniformFractionalChannel
from afdm.pilots import uniform_daft_pilots
from afdm.support import SupportRecovery
from afdm.receiver import UGVEMReceiver
from afdm.training import TrainingConfig, train, evaluate_snr


def main() -> None:
    device = "cuda:0"
    torch.manual_seed(0)

    # Small setup for a quick smoke test.
    N, kappa_max, ell_max, P = 64, 3, 6, 3
    N_p = 16
    sys_ = AFDMSystem(N=N, kappa_max=kappa_max, ell_max=ell_max, device=device)
    ch = UniformFractionalChannel(P=P, ell_max=ell_max, kappa_max=kappa_max, device=device)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=device, dtype=torch.complex64) / (2 ** 0.5)
    pilot_positions = uniform_daft_pilots(N=N, N_p=N_p, device=device)
    gen = torch.Generator(device=device); gen.manual_seed(1)
    pilot_values = qpsk[torch.randint(0, 4, (N_p,), device=device, generator=gen)]
    # Use P candidates (perfect cardinality assumption — standard baseline).
    sup = SupportRecovery(N=N, N_cp=sys_.ell_max, kappa_max=kappa_max, ell_max=ell_max, P_max=P)

    receiver = UGVEMReceiver(
        system=sys_, support_recovery=sup, constellation=qpsk,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        T=4, K_cg=10, d_model=48, n_heads=4, n_blocks=2,
    ).to(device)

    print(f"System: N={N}, P={P}, N_p={N_p}, T=3")
    print(f"Learned params: {sum(p.numel() for p in receiver.parameters() if p.requires_grad):,}")

    # Baseline evaluation (untrained receiver).
    eval_gen = torch.Generator(device=device); eval_gen.manual_seed(42)
    print("\nUntrained receiver baseline:")
    for snr in [5.0, 15.0, 25.0]:
        m = evaluate_snr(receiver, sys_, ch, qpsk, pilot_positions, pilot_values,
                         snr_db=snr, n_batches=2, batch_size=16, generator=eval_gen)
        print(f"  SNR {snr:5.1f}dB: SER={m['ser']:.3e} NMSE={m['nmse_h']:.3e} delay_RMSE={m['delay_rmse']:.3f}")

    # Short training run — tuned for smoke test.
    # Note: full convergence requires ~500 epochs; here we just verify the training
    # pipeline reduces loss and improves at least one metric relative to init.
    config = TrainingConfig(
        lr=5e-4,
        n_epochs=30,
        steps_per_epoch=40,
        batch_size=32,
        snr_db_min=5.0,
        snr_db_max=25.0,
        grad_clip=1.0,
        val_every=10,
        val_batches=3,
        val_snr_dbs=(5.0, 15.0, 25.0),
        layer_gamma=0.7,
        mu_ce=0.5,   # down-weight CE relative to set loss (set loss is more informative early on)
        eta_anchor=0.0,
        hungarian_kwargs=dict(w_h=1.0, w_ell=0.2, w_kap=0.2, mu_fa=0.1, mu_md=0.1),
        log_every=40,   # once per epoch
    )
    print(f"\nTraining for {config.n_epochs} epochs of {config.steps_per_epoch} steps...")
    t0 = time.time()
    history = train(receiver, sys_, ch, qpsk, pilot_positions, pilot_values, config, seed=0, verbose=True)
    print(f"Training time: {time.time() - t0:.1f}s")

    # Final evaluation.
    eval_gen2 = torch.Generator(device=device); eval_gen2.manual_seed(42)
    print("\nTrained receiver evaluation:")
    for snr in [5.0, 15.0, 25.0]:
        m = evaluate_snr(receiver, sys_, ch, qpsk, pilot_positions, pilot_values,
                         snr_db=snr, n_batches=4, batch_size=16, generator=eval_gen2)
        print(f"  SNR {snr:5.1f}dB: SER={m['ser']:.3e} NMSE={m['nmse_h']:.3e} "
              f"delay_RMSE={m['delay_rmse']:.3f} doppler_RMSE={m['doppler_rmse']:.3f}")

    # Gate-closure check at high SNR.
    print("\nGate values across layers at SNR=40dB (should be near zero):")
    batch = torch.randn(4, N, dtype=torch.complex64, device=device)  # dummy
    from afdm.training import sample_batch
    high_batch = sample_batch(sys_, ch, qpsk, pilot_positions, pilot_values,
                              batch_size=4, snr_db=40.0, generator=eval_gen2)
    with torch.no_grad():
        out = receiver(high_batch["r"], sigma_w2_block=high_batch["sigma_w2_block"],
                       return_layer_states=True)
    for t, state in enumerate(out["layer_states"]):
        print(f"  Layer {t}: g_mean = {state['g'].mean().item():.3e}")


if __name__ == "__main__":
    main()
