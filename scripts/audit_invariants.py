"""Day-1/2 implementation audit — verify 4 invariants before drawing conclusions.

Motivation: v3/v4 results showed SER GETTING WORSE from 15 dB to 25 dB
(easy 16.2->19.2, hard 45.0->47.3). Even oracle-ladder R2 (true positions +
LS h) shows this regression (R2 easy 15dB=3.0%, 25dB=4.5%). Before
interpreting this as fundamental physics, run four numerical sanity checks:

  1. Zero-noise operator identity: sigma_w=0, does H CG-MMSE reproduce x_true?
  2. True-support, true-data gain recovery: does h_LS error decrease monotonically?
  3. Exact vs truncated fractional operator agreement.
  4. Regularization scaling: does the LS ridge cause a high-SNR bias?

Any failure narrows down whether the SNR regression is a code bug vs a
physical/model issue.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np

from afdm.channels import UniformFractionalChannel
from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator, slow_afdm_operator
from afdm.training import sample_batch


def line(t):
    print("-" * 78); print(t); print("-" * 78)


# =============================================================================
# Test 1: Zero-noise operator identity
# =============================================================================
def test_zero_noise_identity():
    line("TEST 1: Zero-noise operator identity")
    print("At SNR -> inf, x_soft from CG-MMSE should equal x_true.")
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    for snr_db in (30, 50, 80):
        gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
        batch = sample_batch(system, channel, const, pp, pv, batch_size=4,
                             snr_db=snr_db, generator=gen)
        # Genie CG-MMSE
        op = FastAFDMOperator(system=system, ell=batch["theta_true"][..., 0],
                              kappa=batch["theta_true"][..., 1], h=batch["h_true"])
        sigma_w2 = batch["sigma_w2_block"]
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=60)
        # Compare to x_true directly (not to hard decisions)
        err_l2 = ((x_soft - batch["x_true"]).abs() ** 2).sum().sqrt()
        rel_err = float(err_l2 / batch["x_true"].abs().norm())
        # Also compare hard decoding
        hard = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        print(f"  SNR {snr_db:>3d}dB: sigma_w2={sigma_w2:.2e}  ||x_soft - x_true||/||x_true|| = {rel_err:.2e}  SER = {ser:.2e}")
    print()


# =============================================================================
# Test 2: True-support, true-data gain recovery
# =============================================================================
def test_h_recovery_monotonic():
    line("TEST 2: True support + true data (x_true) LS gain error vs SNR")
    print("With true theta AND true x_true, ||h_ls - h_true|| should decrease monotonically with SNR.")
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    for snr_db in (5, 15, 25, 35, 50):
        gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
        # 20 batches for stable NMSE
        n_batches = 20; nmse_acc = 0.0
        for _ in range(n_batches):
            batch = sample_batch(system, channel, const, pp, pv, batch_size=8,
                                 snr_db=snr_db, generator=gen)
            ell = batch["theta_true"][..., 0]; kap = batch["theta_true"][..., 1]
            # Use TRUE x for regression, not pilot-only.
            A = build_regression_matrix(system, ell, kap, batch["x_true"])
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell.shape[1]
            # Try small ridge to avoid rank issues.
            ridge = 1e-8 * torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
            h_ls = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
            nmse = ((h_ls - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum().clamp(min=1e-12)
            nmse_acc += float(nmse)
        nmse_acc /= n_batches
        print(f"  SNR {snr_db:>3d}dB: NMSE(h_ls, h_true) = {nmse_acc:.3e}")
    print()


def test_h_recovery_pilotonly():
    line("TEST 2b: True support + pilot-only x LS gain error vs SNR")
    print("Same but with x_pilot (data replaced by zeros). This should also decrease")
    print("monotonically UNLESS data self-interference biases the LS at high SNR.")
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    for snr_db in (5, 15, 25, 35, 50):
        gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
        n_batches = 20; nmse_acc = 0.0
        for _ in range(n_batches):
            batch = sample_batch(system, channel, const, pp, pv, batch_size=8,
                                 snr_db=snr_db, generator=gen)
            ell = batch["theta_true"][..., 0]; kap = batch["theta_true"][..., 1]
            B, N = batch["r"].shape
            x_pilot = torch.zeros(B, N, dtype=batch["r"].dtype, device=batch["r"].device)
            x_pilot[:, pp] = pv.unsqueeze(0)
            A = build_regression_matrix(system, ell, kap, x_pilot)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell.shape[1]
            ridge = 1e-3 * torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
            h_ls = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
            nmse = ((h_ls - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum().clamp(min=1e-12)
            nmse_acc += float(nmse)
        nmse_acc /= n_batches
        print(f"  SNR {snr_db:>3d}dB: NMSE(h_ls, h_true) = {nmse_acc:.3e}")
    print()


# =============================================================================
# Test 3: Fast vs slow operator agreement
# =============================================================================
def test_fast_vs_slow_operator():
    line("TEST 3: FastAFDMOperator vs slow_afdm_operator agreement")
    print("Test on a small case where slow (dense) is affordable.")
    device = "cuda:0"
    from afdm.system import AFDMSystem
    system = AFDMSystem(N=32, kappa_max=3, ell_max=4, device=device)
    ch = UniformFractionalChannel(P=3, ell_max=4.0, kappa_max=3.0, device=device)
    gen = torch.Generator(device=device); gen.manual_seed(0)
    d = ch.sample(2, generator=gen)

    op = FastAFDMOperator(system=system, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    H_dense = slow_afdm_operator(system, d["ell"], d["kappa"], d["h"])  # (B, N, N)

    x = torch.randn(2, 32, dtype=torch.complex64, device=device)
    y_fast = op.matvec(x)
    y_slow = (H_dense @ x.unsqueeze(-1)).squeeze(-1)
    err = (y_fast - y_slow).abs().max()
    rel = float(err / y_slow.abs().max())
    print(f"  matvec: max |fast - slow| = {float(err):.3e}  relative = {rel:.3e}")

    # Adjoint
    x_fast = op.rmatvec(y_slow)
    x_slow = (H_dense.conj().transpose(-1, -2) @ y_slow.unsqueeze(-1)).squeeze(-1)
    err_a = (x_fast - x_slow).abs().max()
    rel_a = float(err_a / x_slow.abs().max())
    print(f"  rmatvec (adjoint): max |fast - slow| = {float(err_a):.3e}  relative = {rel_a:.3e}")
    print()


# =============================================================================
# Test 4: Ridge regularization scaling
# =============================================================================
def test_ridge_bias():
    line("TEST 4: Does the fixed lambda_ridge=1e-3 bias LS at high SNR?")
    print("Compare NMSE with lambda_ridge in {1e-6, 1e-4, 1e-3, 1e-2} at SNR=25dB.")
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()

    for ridge_val in (1e-6, 1e-4, 1e-3, 1e-2, 1e-1):
        gen = torch.Generator(device=cfg.device); gen.manual_seed(0)
        n_batches = 20; nmse_acc = 0.0; ser_acc = 0.0
        for _ in range(n_batches):
            batch = sample_batch(system, channel, const, pp, pv, batch_size=8,
                                 snr_db=25.0, generator=gen)
            ell = batch["theta_true"][..., 0]; kap = batch["theta_true"][..., 1]
            B, N = batch["r"].shape
            x_pilot = torch.zeros(B, N, dtype=batch["r"].dtype, device=batch["r"].device)
            x_pilot[:, pp] = pv.unsqueeze(0)
            A = build_regression_matrix(system, ell, kap, x_pilot)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell.shape[1]
            ridge = ridge_val * torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
            h_ls = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
            nmse = ((h_ls - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum().clamp(min=1e-12)
            nmse_acc += float(nmse)
            # Run CG-MMSE with this h and measure SER
            op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h_ls)
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
            hard = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
            ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
            ser_acc += ser
        nmse_acc /= n_batches; ser_acc /= n_batches
        print(f"  lambda={ridge_val:>1.0e}: NMSE(h_ls, h_true) = {nmse_acc:.3e}, SER at 25dB = {ser_acc:.3e}")
    print()


def main():
    print("=" * 78)
    print("IMPLEMENTATION AUDIT — 4 invariant tests")
    print("=" * 78)
    print()
    test_zero_noise_identity()
    test_h_recovery_monotonic()
    test_h_recovery_pilotonly()
    test_fast_vs_slow_operator()
    test_ridge_bias()
    print("=" * 78)
    print("AUDIT COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
