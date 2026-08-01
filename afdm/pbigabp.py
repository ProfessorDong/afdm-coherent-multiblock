"""PBiGaBP-style joint channel-data-radar receiver for AFDM.

Faithful to the message-passing structure of the parametric bidirectional Gaussian
belief propagation receiver of Ranasinghe et al. (IEEE TWC, Feb. 2025). The
essential features preserved here are:

  * Bayesian Gaussian posterior for the path gains with a fixed complex-normal
    prior (rather than a plug-in LS estimate).
  * A factorized categorical symbol posterior with per-symbol extrinsic variance,
    updated by a full LMMSE step and a Euclidean-softmax categorical projection.
  * A gradient-based support-parameter refinement that treats delay and Doppler
    as continuous unknowns of a smooth expected-complete-data log-likelihood.
  * Alternation of these three updates for a fixed number of iterations.

We do not reproduce every implementation detail of Ranasinghe et al. — in
particular, we adapt to our operator conventions (non-negative-k fractional
shifts, complex Dirichlet kernel, `FastAFDMOperator` for CG-MMSE). What we
implement is the *Bayesian message-passing structure* with support-refinement
that PBiGaBP represents. This makes for a clean baseline that isolates the
contribution of our proposed learned modules (uncertainty gate, Set-Transformer,
safeguarded acceptance) which sit on top of this backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .support import SupportRecovery
from .system import AFDMSystem


@dataclass
class PBiGaBPDetector:
    """PBiGaBP-inspired Bayesian message-passing receiver for AFDM.

    Configuration:
      T              : outer iterations (default 8).
      K_cg           : CG iterations per x-step.
      lambda_h       : precision of the complex-normal gain prior (h ~ CN(0, lambda_h^{-1})).
      gamma_lr       : learning rate for gradient-based support refinement.
      gamma_iters    : number of gradient steps for support refinement per outer iter.
      omega          : inverse temperature for symbol categorical projection.
    """

    system: AFDMSystem
    support_recovery: SupportRecovery
    constellation: torch.Tensor
    pilot_positions: torch.Tensor
    pilot_values: torch.Tensor
    T: int = 8
    K_cg: int = 10
    lambda_h: float = 1e-2
    gamma_lr: float = 1e-3
    gamma_iters: int = 2
    omega: float = 10.0
    refine_theta: bool = True  # if False, keep initial CFAR+Newton support fixed

    # ------------------------------------------------------------------
    # Update rules
    # ------------------------------------------------------------------
    def _h_update(
        self,
        r: torch.Tensor,
        A: torch.Tensor,
        sigma_w2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian posterior update for h given data-aided regression A.

        Returns (h_mean, h_var_diag) both of shape (B, P).
        """
        B, N, P = A.shape
        AH = A.conj().transpose(-1, -2)  # (B, P, N)
        AhA = AH @ A  # (B, P, P)
        Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)  # (B, P)
        prior_precision = self.lambda_h * torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
        posterior_precision = AhA / sigma_w2 + prior_precision
        posterior_cov = torch.linalg.inv(posterior_precision)  # (B, P, P)
        h_mean = (posterior_cov @ Ahr.unsqueeze(-1) / sigma_w2).squeeze(-1)
        h_var_diag = torch.diagonal(posterior_cov, dim1=-2, dim2=-1).real  # (B, P)
        return h_mean, h_var_diag

    def _x_update(
        self,
        y: torch.Tensor,
        ell: torch.Tensor,
        kappa: torch.Tensor,
        h_mean: torch.Tensor,
        sigma_w2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LMMSE symbol update + categorical soft posterior.

        Returns (x_soft, categorical_posterior).
        The categorical_posterior has shape (B, N, |S|) with rows summing to 1.
        """
        op = FastAFDMOperator(system=self.system, ell=ell, kappa=kappa, h=h_mean)

        def matvec(v):
            return op.rmatvec(op.matvec(v)) + sigma_w2 * v

        Hty = op.rmatvec(y)
        z = cg_solve(matvec, Hty, max_iter=self.K_cg)
        # Categorical soft posterior via Euclidean softmax.
        dists = (z.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs() ** 2
        logits = -self.omega * dists  # (B, N, |S|)
        posterior = torch.softmax(logits, dim=-1)
        # Posterior mean symbol
        x_soft = (posterior * self.constellation.reshape(1, 1, -1)).sum(dim=-1)
        # Restore pilots.
        x_soft[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)
        pilot_onehot = torch.zeros_like(posterior)
        pilot_class = (self.pilot_values.unsqueeze(-1) - self.constellation.reshape(1, -1)).abs().argmin(dim=-1)
        pilot_onehot[:, self.pilot_positions] = 0.0
        pilot_onehot[:, self.pilot_positions] = torch.nn.functional.one_hot(pilot_class, num_classes=self.constellation.numel()).float().unsqueeze(0).expand(y.shape[0], -1, -1).to(posterior.dtype)
        posterior[:, self.pilot_positions] = pilot_onehot[:, self.pilot_positions].to(posterior.dtype)
        return x_soft, posterior

    def _theta_update(
        self,
        r: torch.Tensor,
        x_hat: torch.Tensor,
        h_mean: torch.Tensor,
        h_var_diag: torch.Tensor,
        ell: torch.Tensor,
        kappa: torch.Tensor,
        sigma_w2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gradient-based refinement of support parameters (theta = (ell, kappa)).

        Objective: per-sample MSE loss, -||r - A h||^2 / N (not divided by sigma_w2 to keep
        gradient scale O(1) regardless of noise level). Per-step gradient is clipped to
        `max_step` in each parameter to prevent divergence from imperfect local
        quadratic surfaces.
        """
        max_step = 0.2  # max update in one iteration, in units of samples (ell) / Doppler indices (kappa)
        ell = ell.detach().clone().requires_grad_(True)
        kappa = kappa.detach().clone().requires_grad_(True)
        for _ in range(self.gamma_iters):
            with torch.enable_grad():
                A = build_regression_matrix(self.system, ell, kappa, x_hat)  # (B, N, P)
                residual = r - (A @ h_mean.unsqueeze(-1)).squeeze(-1)  # (B, N)
                N = residual.shape[-1]
                data_term = -(residual.abs() ** 2).sum(dim=-1) / N  # (B,), per-sample MSE
                col_norms = (A.abs() ** 2).sum(dim=1)  # (B, P)
                var_term = -(col_norms * h_var_diag).sum(dim=-1) / N
                obj = (data_term + var_term).sum()
                grad_ell, grad_kappa = torch.autograd.grad(obj, [ell, kappa], create_graph=False)
            with torch.no_grad():
                # Clip per-parameter step magnitude.
                step_ell = torch.clamp(self.gamma_lr * grad_ell, min=-max_step, max=max_step)
                step_kappa = torch.clamp(self.gamma_lr * grad_kappa, min=-max_step, max=max_step)
                ell = (ell + step_ell).clamp(min=0.0, max=self.system.ell_max)
                kappa = (kappa + step_kappa).clamp(min=-self.system.kappa_max, max=self.system.kappa_max)
            ell = ell.detach().requires_grad_(True)
            kappa = kappa.detach().requires_grad_(True)
        return ell.detach(), kappa.detach()

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def detect(self, r: torch.Tensor, sigma_w2: float) -> dict[str, torch.Tensor]:
        B, N = r.shape
        device = r.device
        dtype = r.dtype

        # Pilot-only transmit signal for support recovery.
        x_pilot = torch.zeros(N, dtype=dtype, device=device)
        x_pilot[self.pilot_positions] = self.pilot_values
        s_pilot = self.system.idaft(x_pilot.unsqueeze(0))[0]

        # 1. Support recovery (initial).
        ell_hat, kappa_hat, p_hat = self.support_recovery(r, s_pilot)

        # 2. Initialize x_hat (pilots + zero data), h_hat, y.
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)
        y = self.system.daft(r)

        # 3. Initial h_mean and h_var from pilot-only regression.
        A = build_regression_matrix(self.system, ell_hat, kappa_hat, x_hat)
        h_mean, h_var_diag = self._h_update(r, A, sigma_w2)

        # 4. Outer iterations.
        for t in range(self.T):
            # x-step: LMMSE + soft posterior
            x_hat, posterior = self._x_update(y, ell_hat, kappa_hat, h_mean, sigma_w2)
            # theta-step: gradient-based support refinement (optional; skip first two outer
            # iterations to let x_hat stabilize before refining support against noisy data).
            if self.refine_theta and t >= 2:
                ell_hat, kappa_hat = self._theta_update(
                    r, x_hat, h_mean, h_var_diag, ell_hat, kappa_hat, sigma_w2
                )
            # h-step: Gaussian posterior update
            A = build_regression_matrix(self.system, ell_hat, kappa_hat, x_hat)
            h_mean, h_var_diag = self._h_update(r, A, sigma_w2)

        # Final CG-MMSE + hard demap
        op = FastAFDMOperator(system=self.system, ell=ell_hat, kappa=kappa_hat, h=h_mean)
        def matvec_final(v):
            return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        Hty = op.rmatvec(y)
        x_soft = cg_solve(matvec_final, Hty, max_iter=self.K_cg)
        dists = (x_soft.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs()
        hard_idx = dists.argmin(dim=-1)
        x_hard = self.constellation[hard_idx]
        x_hard[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)

        return {
            "x_hat": x_soft,
            "hard_x": hard_idx,
            "h_hat": h_mean,
            "h_var": h_var_diag,
            "ell_hat": ell_hat,
            "kappa_hat": kappa_hat,
            "p_hat": p_hat,
        }
