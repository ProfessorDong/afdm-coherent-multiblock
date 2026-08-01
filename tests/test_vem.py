"""Tests for V-EM primitives."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest

from afdm import AFDMSystem, UniformFractionalChannel, FastAFDMOperator
from afdm.pilots import uniform_daft_pilots
from afdm.classical import build_regression_matrix
from afdm.vem import (
    h_step_damped_ridge,
    posterior_covariance,
    safeguarded_lm_theta_step,
    symbol_step_soft_posterior,
    _log_likelihood_theta,
)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _setup(N=64, P=3, seed=0):
    torch.manual_seed(seed)
    sys_ = AFDMSystem(N=N, kappa_max=3, ell_max=6, device=DEVICE)
    ch = UniformFractionalChannel(P=P, ell_max=6.0, kappa_max=3.0, device=DEVICE)
    d = ch.sample(batch=4)
    return sys_, ch, d


def test_h_step_recovers_ridge_at_alpha1():
    """At alpha=1, h_step_damped_ridge should equal the closed-form ridge solution."""
    sys_, _, d = _setup()
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    A = build_regression_matrix(sys_, d["ell"], d["kappa"], x)
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y = op.matvec(x)
    r = sys_.idaft(y)
    lam = 1e-3
    eta_old = torch.zeros(4, d["ell"].shape[1], dtype=torch.complex64, device=DEVICE)
    eta_new = h_step_damped_ridge(A, r, lam=lam, alpha=1.0, eta_old=eta_old)
    # Compare to closed-form
    AH = A.conj().transpose(-1, -2)
    P_size = A.shape[-1]
    M = AH @ A + lam * torch.eye(P_size, dtype=A.dtype, device=DEVICE).unsqueeze(0)
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
    eta_ref = torch.linalg.solve(M, Ahr.unsqueeze(-1)).squeeze(-1)
    err = (eta_new - eta_ref).norm() / eta_ref.norm()
    assert err < 1e-5, f"alpha=1 damped ridge != closed-form ridge, rel err = {err}"


def test_h_step_damping_interpolates():
    """At alpha=0, output should equal eta_old (no update)."""
    sys_, _, d = _setup()
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    A = build_regression_matrix(sys_, d["ell"], d["kappa"], x)
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    r = sys_.idaft(op.matvec(x))
    eta_old = torch.randn(4, d["ell"].shape[1], dtype=torch.complex64, device=DEVICE)
    eta_zero = h_step_damped_ridge(A, r, lam=1e-3, alpha=0.0, eta_old=eta_old)
    assert torch.allclose(eta_zero, eta_old)


def test_posterior_covariance_positive_definite():
    """Posterior covariance V_h should be positive-definite (all eigenvalues > 0)."""
    sys_, _, d = _setup()
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    A = build_regression_matrix(sys_, d["ell"], d["kappa"], x)
    V_h, v = posterior_covariance(A, lam=1e-3, sigma_w2=1e-2)
    # Eigenvalues of V_h should all be positive
    eigs = torch.linalg.eigvalsh(V_h)  # (B, P)
    assert (eigs > 0).all(), f"V_h has non-positive eigenvalue: min = {eigs.min()}"
    # Diagonal v should also be positive.
    assert (v > 0).all()


def test_posterior_covariance_scales_with_sigma_w2():
    """V_h should be proportional to sigma_w^2."""
    sys_, _, d = _setup()
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    A = build_regression_matrix(sys_, d["ell"], d["kappa"], x)
    _, v1 = posterior_covariance(A, lam=1e-3, sigma_w2=1.0)
    _, v2 = posterior_covariance(A, lam=1e-3, sigma_w2=0.01)
    ratio = (v1 / v2).mean().item()
    assert abs(ratio - 100.0) < 1e-2, f"expected 100x, got {ratio}x"


def test_safeguarded_lm_never_decreases_objective():
    """The safeguarded LM step must not decrease Q by more than 'slack'."""
    sys_, _, d = _setup(P=3)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    r = sys_.idaft(op.matvec(x))
    eta_h = d["h"]  # oracle
    # Perturb theta away from truth
    ell_init = d["ell"] + 0.3 * torch.randn_like(d["ell"])
    kap_init = d["kappa"] + 0.2 * torch.randn_like(d["kappa"])
    ell_init = ell_init.clamp(min=0, max=sys_.ell_max)
    kap_init = kap_init.clamp(min=-sys_.kappa_max, max=sys_.kappa_max)

    Q_before = _log_likelihood_theta(sys_, r, eta_h, x, ell_init, kap_init, sigma_w2=1e-3)
    slack = 1e-4
    ell_new, kap_new, accepted = safeguarded_lm_theta_step(
        sys_, r, eta_h, x, ell_init, kap_init, sigma_w2=1e-3,
        gamma_lr=0.5, max_step=0.15, slack=slack,
    )
    Q_after = _log_likelihood_theta(sys_, r, eta_h, x, ell_new, kap_new, sigma_w2=1e-3)
    # Q should not decrease by more than slack.
    decrease = (Q_before - Q_after).max().item()
    print(f"Max Q decrease across batch: {decrease:.4e} (slack={slack})")
    assert decrease <= slack + 1e-6, f"Q decreased by {decrease}, more than slack {slack}"
    # At least some batch elements should have improved (accepted a step).
    assert accepted.any(), "no batch element accepted a step"


def test_safeguarded_lm_moves_toward_truth():
    """When perturbed away from truth, safeguarded LM should refine toward truth on average."""
    sys_, _, d = _setup(P=3)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    r = sys_.idaft(op.matvec(x))
    ell_init = d["ell"] + 0.2 * torch.randn_like(d["ell"])
    kap_init = d["kappa"] + 0.15 * torch.randn_like(d["kappa"])
    ell_init = ell_init.clamp(min=0, max=sys_.ell_max)
    kap_init = kap_init.clamp(min=-sys_.kappa_max, max=sys_.kappa_max)
    err_before = ((ell_init - d["ell"]).abs() + (kap_init - d["kappa"]).abs()).mean().item()
    # Iterate a few times
    ell_curr, kap_curr = ell_init.clone(), kap_init.clone()
    for _ in range(5):
        ell_curr, kap_curr, _ = safeguarded_lm_theta_step(
            sys_, r, d["h"], x, ell_curr, kap_curr, sigma_w2=1e-3,
            gamma_lr=0.5, max_step=0.1, slack=1e-4,
        )
    err_after = ((ell_curr - d["ell"]).abs() + (kap_curr - d["kappa"]).abs()).mean().item()
    print(f"Support error before: {err_before:.4f}, after 5 LM iters: {err_after:.4f}")
    assert err_after < err_before, f"LM did not reduce support error: {err_before} -> {err_after}"


def test_symbol_step_returns_valid_posterior():
    """Categorical posterior rows should sum to 1 and pilots must be one-hot at correct class."""
    sys_, _, d = _setup(N=128, P=3)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    pilot_pos = uniform_daft_pilots(N=128, N_p=16, device=DEVICE)
    pilot_val = qpsk[torch.tensor([0]*16, device=DEVICE)]  # all first symbol
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pilot_pos] = pilot_val.unsqueeze(0)
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y = op.matvec(x)
    x_mean, p_ms, z = symbol_step_soft_posterior(
        sys_, y, d["ell"], d["kappa"], d["h"], sigma_w2=1e-3,
        omega=torch.tensor(20.0), constellation=qpsk,
        pilot_positions=pilot_pos, pilot_values=pilot_val, K_cg=15,
    )
    # p_ms sums to 1 per (b, m)
    sums = p_ms.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    # Pilots are one-hot at class 0
    pilot_ps = p_ms[:, pilot_pos, :]
    assert torch.allclose(pilot_ps[..., 0], torch.ones_like(pilot_ps[..., 0])), \
        "pilot posterior not one-hot"
    # Posterior mean at pilot positions equals pilot value
    assert torch.allclose(x_mean[:, pilot_pos], pilot_val.unsqueeze(0))


def test_symbol_step_recovers_at_high_snr():
    """With true h, symbol step should have very low SER at high SNR."""
    sys_, _, d = _setup(N=128, P=3)
    qpsk = torch.tensor([1+1j, 1-1j, -1+1j, -1-1j], device=DEVICE, dtype=torch.complex64) / (2 ** 0.5)
    pilot_pos = uniform_daft_pilots(N=128, N_p=16, device=DEVICE)
    pilot_val = qpsk[torch.randint(0, 4, (16,), device=DEVICE)]
    idx = torch.randint(0, 4, (4, sys_.N), device=DEVICE)
    x = qpsk[idx]; x[:, pilot_pos] = pilot_val.unsqueeze(0)
    op = FastAFDMOperator(system=sys_, ell=d["ell"], kappa=d["kappa"], h=d["h"])
    y_clean = op.matvec(x)
    signal_pow = (y_clean.abs() ** 2).mean()
    sigma_w2 = 10 ** (-30 / 10)
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    y = y_clean + torch.randn_like(y_clean) * noise_std
    abs_noise = (signal_pow * sigma_w2).item()
    x_mean, p_ms, _ = symbol_step_soft_posterior(
        sys_, y, d["ell"], d["kappa"], d["h"], sigma_w2=abs_noise,
        omega=torch.tensor(20.0), constellation=qpsk,
        pilot_positions=pilot_pos, pilot_values=pilot_val, K_cg=30,
    )
    hard = p_ms.argmax(dim=-1)
    mask = torch.ones(sys_.N, dtype=torch.bool, device=DEVICE); mask[pilot_pos] = False
    ser = (hard[:, mask] != idx[:, mask]).float().mean().item()
    print(f"Symbol step SER at 30 dB (genie CSI): {ser:.4e}")
    assert ser < 0.01, f"SER too high: {ser}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
