"""Unit tests for the classical semi-blind AFDM detector."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.support import SupportRecovery
from afdm.classical import ClassicalCGDetector, build_regression_matrix, cg_solve


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _test_setup(N=128, N_p=16, P=5, seed=0):
    """Build a common test setup: system, channel realization, pilots."""
    torch.manual_seed(seed)
    sys_ = AFDMSystem(N=N, kappa_max=5, ell_max=10, device=DEVICE)
    ch = UniformFractionalChannel(P=P, ell_max=10.0, kappa_max=5.0, device=DEVICE)
    d = ch.sample(batch=4)
    pilot_positions = uniform_daft_pilots(N=N, N_p=N_p, device=DEVICE)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    gen = torch.Generator(device=DEVICE); gen.manual_seed(seed + 1)
    pilot_idx = torch.randint(0, 4, (N_p,), device=DEVICE, generator=gen)
    pilot_values = qpsk[pilot_idx]
    return sys_, ch, d, pilot_positions, pilot_values, qpsk


def _forward(sys_, d, x_true, sigma_w2):
    """Physical forward: FastAFDMOperator + AWGN in DAFT domain, then IDAFT to time."""
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y_clean = op.matvec(x_true)
    signal_pow = (y_clean.abs() ** 2).mean()
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    w = torch.randn_like(y_clean) * noise_std
    y = y_clean + w
    # Time-domain observation
    r = sys_.idaft(y)
    return r, y


def test_regression_matrix_matches_fast_operator():
    """A(theta, x) built by build_regression_matrix should satisfy A h = time-domain
    IDAFT of the DAFT-domain matvec H^D x for the same theta, h, x."""
    sys_, ch, d, pilot_positions, pilot_values, qpsk = _test_setup()
    # Random symbols
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    # Method 1: build A then A @ h (time-domain)
    A = build_regression_matrix(sys_, d["ell"], d["kappa"], x)  # (B, N, P)
    r_from_A = (A @ d["h"].unsqueeze(-1)).squeeze(-1)  # (B, N)
    # Method 2: fast operator, then IDAFT to time
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y = op.matvec(x)
    r_from_op = sys_.idaft(y)
    err = (r_from_A - r_from_op).norm() / r_from_op.norm()
    assert err < 1e-4, f"regression matrix A vs fast operator disagree: rel err = {err}"


def test_cg_converges_on_hermitian_pd():
    """CG should converge to the correct solution of a small PD system."""
    torch.manual_seed(0)
    B, N = 4, 32
    # Random PD matrix
    A_dense = torch.randn(B, N, N, dtype=torch.complex64, device=DEVICE)
    A_dense = A_dense @ A_dense.conj().transpose(-1, -2) + 0.1 * torch.eye(N, dtype=torch.complex64, device=DEVICE)
    x_true = torch.randn(B, N, dtype=torch.complex64, device=DEVICE)
    b = (A_dense @ x_true.unsqueeze(-1)).squeeze(-1)
    x_cg = cg_solve(lambda v: (A_dense @ v.unsqueeze(-1)).squeeze(-1), b, max_iter=100, tol=1e-8)
    err = (x_cg - x_true).norm() / x_true.norm()
    assert err < 1e-4, f"CG error: {err}"


def test_classical_recovers_at_high_snr():
    """Classical detector should reach low BER at high SNR."""
    sys_, ch, d, pilot_positions, pilot_values, qpsk = _test_setup(N=128, N_p=32, P=3)
    # Symbols with pilots at pilot positions
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    x[:, pilot_positions] = pilot_values.unsqueeze(0)
    sigma_w2 = 10 ** (-25.0 / 10)  # -25 dB noise -> 25 dB SNR
    r, y = _forward(sys_, d, x, sigma_w2)
    sup = SupportRecovery(N=sys_.N, N_cp=sys_.ell_max, kappa_max=5, ell_max=10, P_max=8)
    det = ClassicalCGDetector(
        system=sys_,
        support_recovery=sup,
        constellation=qpsk,
        pilot_positions=pilot_positions,
        pilot_values=pilot_values,
        T=8, K_cg=15, alpha=1.0, lambda_ridge=1e-3,
    )
    out = det.detect(r, sigma_w2=sigma_w2)
    # SER on non-pilot positions
    mask = torch.ones(sys_.N, dtype=torch.bool, device=DEVICE)
    mask[pilot_positions] = False
    true_idx = idx[:, mask]
    hard_idx = out["hard_x"][:, mask]
    ser = (true_idx != hard_idx).float().mean().item()
    print(f"SER at 25 dB SNR (P={3}, N_p=32): {ser:.4f}, p_hat mean = {out['p_hat'].float().mean():.1f}")
    # At 25 dB SNR with support recovery, some errors from support miss possible, but should be low.
    assert ser < 0.15, f"Classical detector SER too high: {ser}"


def test_classical_converges_over_iterations():
    """As T grows, BER should be non-increasing (roughly)."""
    sys_, ch, d, pilot_positions, pilot_values, qpsk = _test_setup(N=128, N_p=32, P=3)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    x[:, pilot_positions] = pilot_values.unsqueeze(0)
    sigma_w2 = 10 ** (-15.0 / 10)  # 15 dB SNR
    r, y = _forward(sys_, d, x, sigma_w2)
    sup = SupportRecovery(N=sys_.N, N_cp=sys_.ell_max, kappa_max=5, ell_max=10, P_max=6)
    mask = torch.ones(sys_.N, dtype=torch.bool, device=DEVICE); mask[pilot_positions] = False
    true_idx = idx[:, mask]

    sers = []
    for T in [1, 3, 5, 8]:
        det = ClassicalCGDetector(
            system=sys_, support_recovery=sup, constellation=qpsk,
            pilot_positions=pilot_positions, pilot_values=pilot_values,
            T=T, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
        )
        out = det.detect(r, sigma_w2=sigma_w2)
        hard_idx = out["hard_x"][:, mask]
        ser = (true_idx != hard_idx).float().mean().item()
        sers.append(ser)
    print(f"SER vs T=1,3,5,8: {sers}")
    # Loose monotonicity: SER at T=8 should be at least as low as at T=1.
    assert sers[-1] <= sers[0] + 0.02, f"detector did not improve with iterations: {sers}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
