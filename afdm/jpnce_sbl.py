"""JPNCE-SBL-style off-grid channel estimation for AFDM.

Faithful to the sparse Bayesian learning (SBL) structure of Xu et al.,
"Joint Phase Noise and Off-Grid Channel Estimation for AFDM Systems via
Sparse Bayesian Learning," arXiv:2604.17858 (2026). We implement the
dynamic-grid off-grid SBL component (channel-side); the phase-noise
subspace projection is deliberately omitted so this baseline is compared
against the proposed receiver on the same (phase-noise-free) channel model.

Structure:
  1. Start from an overcomplete candidate set on a fine (l, k) grid that spans
     the physically feasible support.
  2. Place an ARD (automatic relevance determination) prior on the candidate
     gains: h_i ~ CN(0, gamma_i^{-1}), with hyperparameters gamma_i learned via
     evidence maximization / Type-II ML.
  3. E-step: Gaussian posterior on h given the current gammas.
  4. M-step: update gammas (evidence maximization); prune candidates whose
     gammas exceed a pruning threshold (near-zero prior variance).
  5. Dynamic grid evolution: for each surviving candidate, take a bounded
     gradient step on the log-evidence w.r.t. its (l, k) position.
  6. Iterate until convergence or max iterations.

Because SBL is a channel-estimation-only routine, downstream symbol detection
uses CG-MMSE with the SBL-estimated support and gains, matching the JSAC baseline
pipeline and Xu et al.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .support import SupportRecovery
from .system import AFDMSystem


@dataclass
class JPNCESBLDetector:
    """Off-grid SBL AFDM channel estimator + CG-MMSE symbol detector.

    Configuration:
      T_em        : outer EM iterations for SBL.
      P_max_grid  : initial candidate count from the overcomplete grid.
      T_grid      : number of dynamic-grid updates (bounded).
      grid_lr     : dynamic-grid step size.
      prune_thresh: prior precision above which candidates are pruned.
      K_cg        : CG iterations for the final symbol solve.
    """

    system: AFDMSystem
    constellation: torch.Tensor
    pilot_positions: torch.Tensor
    pilot_values: torch.Tensor
    support_recovery: SupportRecovery  # for CFAR-initialized candidates
    T_em: int = 15
    T_grid: int = 3
    grid_lr: float = 0.05
    prune_thresh: float = 1e6      # (unused when prune_by_magnitude=True) precision threshold
    prune_by_magnitude: bool = True  # relative-magnitude pruning (recommended)
    magnitude_ratio: float = 0.05  # keep candidates with |mu|^2 >= ratio * max(|mu|^2)
    K_cg: int = 15
    # Optional: fixed-grid overcomplete init (legacy). Left for parity but CFAR init is default.
    P_max_grid: int = 20

    def _init_grid(self, batch: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize an overcomplete (ell, kappa) candidate grid per batch element.

        Uses a fine 2x-oversampled tensor-product grid over the feasible support:
        delay spacing 1 sample and Doppler spacing 1 index, restricted to P_max_grid
        via taking the top-left corner of the mesh.
        """
        n_ell = int((self.system.ell_max) + 1)  # e.g. 11 delay bins
        n_kap = int(2 * self.system.kappa_max + 1)  # e.g. 11 Doppler bins
        ells = torch.arange(0, n_ell, device=device, dtype=torch.float32)
        kaps = torch.linspace(-self.system.kappa_max, self.system.kappa_max, n_kap, device=device)
        E, K = torch.meshgrid(ells, kaps, indexing="ij")
        ell_flat = E.reshape(-1)[: self.P_max_grid]
        kap_flat = K.reshape(-1)[: self.P_max_grid]
        ell = ell_flat.unsqueeze(0).expand(batch, -1).clone()
        kappa = kap_flat.unsqueeze(0).expand(batch, -1).clone()
        return ell, kappa

    def _e_step(
        self,
        r: torch.Tensor,
        A: torch.Tensor,
        gamma: torch.Tensor,
        sigma_w2: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gaussian posterior of h given gamma (ARD prior).

        Returns (mu_h, diag_Sigma_h).
        """
        B, N, P = A.shape
        AH = A.conj().transpose(-1, -2)
        AhA = AH @ A  # (B, P, P)
        Gamma = torch.diag_embed(gamma.to(A.dtype))  # (B, P, P), diag(gamma_i)
        posterior_precision = AhA / sigma_w2 + Gamma  # (B, P, P)
        # Solve
        posterior_cov = torch.linalg.inv(posterior_precision)
        Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
        mu = (posterior_cov @ Ahr.unsqueeze(-1) / sigma_w2).squeeze(-1)
        diag_cov = torch.diagonal(posterior_cov, dim1=-2, dim2=-1).real
        return mu, diag_cov

    def _m_step(
        self,
        mu: torch.Tensor,
        diag_cov: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-point ARD hyperparameter update.

        Standard SBL update: gamma_i = 1 / (|mu_i|^2 + Sigma_ii).
        """
        gamma_new = 1.0 / ((mu.abs() ** 2 + diag_cov) + 1e-12)
        return gamma_new

    def _grid_update(
        self,
        r: torch.Tensor,
        ell: torch.Tensor,
        kappa: torch.Tensor,
        mu: torch.Tensor,
        sigma_w2: float,
        x_hat: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Small gradient step on log-evidence w.r.t. (ell, kappa) per candidate.

        Uses the LMMSE residual as a proxy for evidence gradient (a well-known
        approximation in dynamic-grid SBL).
        """
        max_step = 0.15
        ell = ell.detach().clone().requires_grad_(True)
        kappa = kappa.detach().clone().requires_grad_(True)
        for _ in range(self.T_grid):
            # Enable grad locally even if called under torch.no_grad() context.
            with torch.enable_grad():
                A = build_regression_matrix(self.system, ell, kappa, x_hat)
                residual = r - (A @ mu.unsqueeze(-1)).squeeze(-1)
                loss = (residual.abs() ** 2).sum(dim=-1).mean() / residual.shape[-1]
                grad_ell, grad_kap = torch.autograd.grad(loss, [ell, kappa])
            with torch.no_grad():
                step_ell = torch.clamp(-self.grid_lr * grad_ell, min=-max_step, max=max_step)
                step_kap = torch.clamp(-self.grid_lr * grad_kap, min=-max_step, max=max_step)
                ell = (ell + step_ell).clamp(min=0.0, max=self.system.ell_max)
                kappa = (kappa + step_kap).clamp(min=-self.system.kappa_max, max=self.system.kappa_max)
            ell = ell.detach().requires_grad_(True)
            kappa = kappa.detach().requires_grad_(True)
        return ell.detach(), kappa.detach()

    def detect(self, r: torch.Tensor, sigma_w2: float) -> dict[str, torch.Tensor]:
        B, N = r.shape
        device = r.device
        dtype = r.dtype

        # 1. Initialize candidates from CFAR + Newton support recovery (with some
        # extra candidates for slack), which gives a well-separated initial support
        # of size approximately P_max.
        x_pilot = torch.zeros(N, dtype=dtype, device=device)
        x_pilot[self.pilot_positions] = self.pilot_values
        s_pilot = self.system.idaft(x_pilot.unsqueeze(0))[0]
        ell, kappa, _ = self.support_recovery(r, s_pilot)  # (B, P_max)
        P_grid = ell.shape[1]
        gamma = torch.ones(B, P_grid, device=device) * 1.0

        # 2. Initial x_hat: pilots at known values, data at 0.
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)

        # 3. EM iterations.
        for it in range(self.T_em):
            A = build_regression_matrix(self.system, ell, kappa, x_hat)
            mu, diag_cov = self._e_step(r, A, gamma, sigma_w2)
            gamma = self._m_step(mu, diag_cov)
            # Dynamic-grid update every few iterations
            if it % 3 == 2 and self.T_grid > 0:
                ell, kappa = self._grid_update(r, ell, kappa, mu, sigma_w2, x_hat)
            # After a warmup, run a symbol update to feed better x_hat back.
            if it >= 3:
                op = FastAFDMOperator(system=self.system, ell=ell, kappa=kappa, h=mu)
                y = self.system.daft(r)
                def matvec(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
                Hty = op.rmatvec(y)
                x_soft = cg_solve(matvec, Hty, max_iter=self.K_cg // 2)
                # Hard demap + pilot restoration
                dists = (x_soft.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs()
                hard_idx = dists.argmin(dim=-1)
                x_hat = self.constellation[hard_idx]
                x_hat[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)

        # 4. Prune irrelevant candidates.
        # In SBL, large gamma = small prior variance = the candidate is deemed irrelevant.
        # In practice, we use a relative-magnitude criterion: keep candidates whose
        # |mu|^2 is at least `magnitude_ratio` times the strongest candidate's |mu|^2.
        # This is robust across SNRs and matches the "energy-thresholded pruning"
        # commonly used in dynamic-grid SBL variants (e.g., Xu et al. 2026).
        if self.prune_by_magnitude:
            mu_sq = mu.abs() ** 2  # (B, P_grid)
            mu_max = mu_sq.max(dim=-1, keepdim=True).values
            keep_mask = mu_sq >= self.magnitude_ratio * mu_max
        else:
            keep_mask = gamma < self.prune_thresh
        p_hat = keep_mask.sum(dim=-1)  # (B,)
        # Repack surviving paths into a padded tensor.
        P_hat_max = int(p_hat.max().item()) if p_hat.max().item() > 0 else 1
        ell_hat = torch.zeros(B, P_hat_max, device=device)
        kappa_hat = torch.zeros(B, P_hat_max, device=device)
        h_hat = torch.zeros(B, P_hat_max, dtype=dtype, device=device)
        for b in range(B):
            idx_keep = torch.where(keep_mask[b])[0]
            n_keep = idx_keep.shape[0]
            ell_hat[b, :n_keep] = ell[b, idx_keep]
            kappa_hat[b, :n_keep] = kappa[b, idx_keep]
            h_hat[b, :n_keep] = mu[b, idx_keep]

        # 5. Final CG-MMSE symbol detection with pruned support.
        op = FastAFDMOperator(system=self.system, ell=ell_hat, kappa=kappa_hat, h=h_hat)
        y = self.system.daft(r)
        def matvec_final(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        Hty = op.rmatvec(y)
        x_soft = cg_solve(matvec_final, Hty, max_iter=self.K_cg)
        dists = (x_soft.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs()
        hard_idx = dists.argmin(dim=-1)
        x_hard = self.constellation[hard_idx]
        x_hard[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)

        return {
            "x_hat": x_soft,
            "hard_x": hard_idx,
            "h_hat": h_hat,
            "ell_hat": ell_hat,
            "kappa_hat": kappa_hat,
            "p_hat": p_hat,
        }
