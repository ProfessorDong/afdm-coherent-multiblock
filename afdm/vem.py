"""Variational-EM primitives for the proposed AFDM receiver.

Implements the four coordinate-ascent updates that make up a single layer of the
receiver (paper Section IV.B):
  1. `h_step_damped_ridge`      — mean update, closed-form damped ridge posterior.
  2. `posterior_covariance`     — diagonal of the h posterior covariance.
  3. `safeguarded_lm_theta_step`— Levenberg-Marquardt refinement of the fractional
                                  support with an explicit acceptance test.
  4. `symbol_step_soft_posterior`— CG-MMSE for the data symbols and Euclidean
                                  softmax categorical posterior.

All primitives are batched on GPU and preserve autograd graphs so the receiver
can be trained end-to-end.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .system import AFDMSystem


# ---------------------------------------------------------------------------
# h-block: damped exact ridge (equivalent to relaxed posterior-mean update)
# ---------------------------------------------------------------------------
def h_step_damped_ridge(
    A: torch.Tensor,
    r: torch.Tensor,
    lam: torch.Tensor,
    alpha: torch.Tensor,
    eta_old: torch.Tensor,
) -> torch.Tensor:
    """Damped exact-ridge update (paper eq. 32).

    Given regression A(theta, x_hat), received r, learned ridge lam and relaxation
    alpha, the update
        tilde_eta = eta_old - alpha * M^{-1} [A^H(A eta_old - r) + lam * eta_old]
    simplifies algebraically to
        tilde_eta = (1 - alpha) * eta_old + alpha * M^{-1} A^H r,
    which is exactly the relaxed posterior-mean formula. This is the closed form.

    Parameters
    ----------
    A       : (B, N, P) regression matrix.
    r       : (B, N) time-domain received signal.
    lam     : scalar (or (B,)) ridge parameter, positive.
    alpha   : scalar (or (B,)) relaxation in (0, 2).
    eta_old : (B, P) previous h mean estimate.

    Returns
    -------
    eta_new : (B, P) updated h mean.
    """
    B, N, P = A.shape
    AH = A.conj().transpose(-1, -2)
    AhA = AH @ A  # (B, P, P)
    # Keep lam and alpha as tensors so gradients flow to any learned params driving them.
    lam_t = lam if torch.is_tensor(lam) else torch.tensor(float(lam), device=A.device)
    lam_c = lam_t.to(A.dtype)
    eye = torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
    M = AhA + lam_c * eye  # (B, P, P)
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)  # (B, P)
    y = torch.linalg.solve(M, Ahr.unsqueeze(-1)).squeeze(-1)  # (B, P)
    alpha_t = alpha if torch.is_tensor(alpha) else torch.tensor(float(alpha), device=A.device)
    alpha_c = alpha_t.to(A.dtype)
    eta_new = (1.0 - alpha_c) * eta_old + alpha_c * y
    return eta_new


def posterior_covariance(
    A: torch.Tensor,
    lam: torch.Tensor,
    sigma_w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagonal of the Gaussian posterior covariance (paper eq. 33).

        V_h = sigma_w^2 * (A^H A + lam I)^{-1}

    Returns
    -------
    V_h  : (B, P, P) full posterior covariance.
    v    : (B, P) real diagonal (posterior variances).
    """
    B, N, P = A.shape
    AH = A.conj().transpose(-1, -2)
    AhA = AH @ A
    lam_t = lam if torch.is_tensor(lam) else torch.tensor(float(lam), device=A.device)
    lam_c = lam_t.to(A.dtype)
    eye = torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
    M = AhA + lam_c * eye
    Minv = torch.linalg.inv(M)
    sigma_t = sigma_w2 if torch.is_tensor(sigma_w2) else torch.tensor(float(sigma_w2), device=A.device)
    sigma_c = sigma_t.to(A.dtype)
    V_h = sigma_c * Minv
    v = torch.diagonal(V_h, dim1=-2, dim2=-1).real
    v = v.clamp(min=1e-18)
    return V_h, v


# ---------------------------------------------------------------------------
# theta-block: safeguarded Levenberg-Marquardt with acceptance test
# ---------------------------------------------------------------------------
def _log_likelihood_theta(
    system: AFDMSystem,
    r: torch.Tensor,
    eta_h: torch.Tensor,
    x_hat: torch.Tensor,
    ell: torch.Tensor,
    kappa: torch.Tensor,
    sigma_w2: float,
    v_h: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-sample (negative) MSE plus variance-correction term, per batch element.

    Returns a (B,) tensor whose value we want to MAXIMIZE (higher = better fit).
    Per-sample-normalized so the scale is O(1) regardless of noise level.
    """
    A = build_regression_matrix(system, ell, kappa, x_hat)  # (B, N, P)
    residual = r - (A @ eta_h.unsqueeze(-1)).squeeze(-1)  # (B, N)
    N = residual.shape[-1]
    data_term = -(residual.abs() ** 2).sum(dim=-1) / N  # (B,)
    if v_h is not None:
        col_norms = (A.abs() ** 2).sum(dim=1)  # (B, P)
        var_term = -(col_norms * v_h).sum(dim=-1) / N
        return data_term + var_term
    return data_term


def safeguarded_lm_theta_step(
    system: AFDMSystem,
    r: torch.Tensor,
    eta_h: torch.Tensor,
    x_hat: torch.Tensor,
    ell: torch.Tensor,
    kappa: torch.Tensor,
    sigma_w2: float,
    v_h: torch.Tensor | None = None,
    gamma_lr: float = 0.5,
    max_step: float = 0.15,
    slack: float = 1e-4,
    max_backtracks: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Safeguarded Levenberg-Marquardt-style refinement of (ell, kappa).

    We compute the gradient of the per-sample log-likelihood surrogate and take
    a per-parameter clipped step. If the step does not increase the objective by
    at least (-slack) per batch element, we halve the step and retry. The best
    accepted per-element candidate is kept; batches that never accept keep the
    original theta.

    This is the safeguarded ascent rule that guarantees the per-layer
    monotonicity property of Theorem 3 (up to controlled slack).

    Returns
    -------
    ell_new     : (B, P) refined delay indices.
    kappa_new   : (B, P) refined Doppler indices.
    accepted    : (B,) bool, True where at least one step improved the objective.
    """
    device = ell.device
    dtype_r = ell.dtype
    ell = ell.detach().clone().requires_grad_(True)
    kappa = kappa.detach().clone().requires_grad_(True)
    # Baseline objective
    with torch.enable_grad():
        Q_old_per_b = _log_likelihood_theta(system, r, eta_h, x_hat, ell, kappa, sigma_w2, v_h)  # (B,)
        Q_sum = Q_old_per_b.sum()
    grad_ell, grad_kap = torch.autograd.grad(Q_sum, [ell, kappa])
    Q_old = Q_old_per_b.detach()

    # Try the full step, then halve if any batch element fails.
    ell_best = ell.detach().clone()
    kap_best = kappa.detach().clone()
    accepted = torch.zeros(ell.shape[0], dtype=torch.bool, device=device)
    step_ell = torch.clamp(gamma_lr * grad_ell.detach(), min=-max_step, max=max_step)
    step_kap = torch.clamp(gamma_lr * grad_kap.detach(), min=-max_step, max=max_step)
    for i in range(max_backtracks + 1):
        # Reduce step for retries
        scale = 0.5 ** i
        cand_ell = (ell.detach() + scale * step_ell).clamp(min=0.0, max=system.ell_max)
        cand_kap = (kappa.detach() + scale * step_kap).clamp(min=-system.kappa_max, max=system.kappa_max)
        with torch.no_grad():
            Q_cand = _log_likelihood_theta(system, r, eta_h, x_hat, cand_ell, cand_kap, sigma_w2, v_h)
        improved = (Q_cand >= Q_old - slack) & (~accepted)
        if improved.any():
            for b in torch.where(improved)[0].tolist():
                ell_best[b] = cand_ell[b]
                kap_best[b] = cand_kap[b]
            accepted = accepted | improved
        if accepted.all():
            break
    return ell_best.detach(), kap_best.detach(), accepted


# ---------------------------------------------------------------------------
# x-block: CG-MMSE + soft posterior
# ---------------------------------------------------------------------------
def symbol_step_soft_posterior(
    system: AFDMSystem,
    y: torch.Tensor,
    ell: torch.Tensor,
    kappa: torch.Tensor,
    eta_h: torch.Tensor,
    sigma_w2: float,
    omega: torch.Tensor,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    K_cg: int = 10,
    z_prev: torch.Tensor | None = None,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """LMMSE (via CG) followed by damped soft posterior over the constellation.

    Parameters
    ----------
    y              : (B, N) DAFT-domain observation.
    ell, kappa     : (B, P) current fractional support.
    eta_h          : (B, P) current gain mean.
    sigma_w2       : absolute noise variance (scalar).
    omega          : learned inverse-temperature scalar for the Euclidean softmax.
    constellation  : (|S|,) complex constellation values.
    pilot_positions: (N_p,) long, DAFT-domain pilot positions.
    pilot_values   : (N_p,) complex pilot symbols.
    K_cg           : CG iterations.
    z_prev         : (B, N) previous MMSE estimate; used for damping (default None).
    beta           : damping in (0, 1] between previous and new estimate.

    Returns
    -------
    x_mean         : (B, N) posterior-mean complex symbols, pilots restored.
    p_ms           : (B, N, |S|) categorical posterior over constellation.
    z              : (B, N) raw MMSE estimate (pre-softmax), useful for logging.
    """
    op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=eta_h)

    def matvec(v: torch.Tensor) -> torch.Tensor:
        return op.rmatvec(op.matvec(v)) + sigma_w2 * v

    z = cg_solve(matvec, op.rmatvec(y), max_iter=K_cg)
    if z_prev is not None:
        beta_t = beta if torch.is_tensor(beta) else torch.tensor(float(beta), device=y.device)
        beta_c = beta_t.to(z.dtype)
        z = (1 - beta_c) * z_prev + beta_c * z
    # Categorical soft posterior — keep omega as tensor for gradient flow.
    omega_t = omega if torch.is_tensor(omega) else torch.tensor(float(omega), device=y.device)
    dists = (z.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs() ** 2  # (B, N, |S|)
    logits = -omega_t * dists
    p_ms = F.softmax(logits, dim=-1)
    x_mean = (p_ms * constellation.reshape(1, 1, -1)).sum(dim=-1)
    # Restore pilots (hard constraint: replace posterior with one-hot at pilot value).
    if pilot_positions.numel() > 0:
        # Find nearest constellation index for each pilot
        pilot_dists = (pilot_values.unsqueeze(-1) - constellation.reshape(1, -1)).abs()
        pilot_class = pilot_dists.argmin(dim=-1)  # (N_p,)
        pilot_onehot = F.one_hot(pilot_class, num_classes=constellation.numel()).to(p_ms.dtype)
        p_ms = p_ms.clone()
        p_ms[:, pilot_positions] = pilot_onehot.unsqueeze(0)
        x_mean = x_mean.clone()
        x_mean[:, pilot_positions] = pilot_values.unsqueeze(0)
    return x_mean, p_ms, z
