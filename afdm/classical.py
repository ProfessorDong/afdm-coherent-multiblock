"""Classical semi-blind AFDM detector (Algorithm 1 of the paper).

Alternates between two blocks:
  * h-step: ridge-regularized least squares for the path gains given the current
    symbol estimate, using the (data-aided) regression matrix
        A[n, i] = exp(j 2 pi kappa_hat_i (n + N_cp)/N) * s_shifted_i[n],
    where s_shifted_i is the FFT-based fractional shift of s = IDAFT(x_hat) by
    ell_hat_i (non-negative-k convention, matching FastAFDMOperator).
  * x-step: DAFT-domain CG-MMSE using FastAFDMOperator, followed by hard
    demapping to the constellation and restoration of pilots.

This is the JSAC-era detector, kept here as the "classical" comparison baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from .operators import FastAFDMOperator
from .system import AFDMSystem
from .support import SupportRecovery


def build_regression_matrix(
    system: AFDMSystem,
    ell: torch.Tensor,   # (B, P) fractional delays
    kappa: torch.Tensor, # (B, P) fractional Doppler
    x_hat: torch.Tensor, # (B, N) DAFT-domain symbol estimate (pilots + data estimates)
) -> torch.Tensor:
    """Build the time-domain regression matrix A(theta, x) of shape (B, N, P).

    Column i of A is the time-domain response of a unit-gain path (ell_i, kappa_i)
    to the transmit signal s = IDAFT(x_hat):
        A[b, n, i] = exp(j 2 pi kappa[b, i] (n + N_cp) / N)
                     * FFT-fractional-shift(s, ell[b, i])[n],
    using the same k = 0..N-1 phase convention as FastAFDMOperator.
    """
    B, P = ell.shape
    N = system.N
    device = system.device
    dtype = system.dtype
    N_cp = system.ell_max
    # 1. Transmit signal
    s = system.idaft(x_hat)  # (B, N)
    S = torch.fft.fft(s, dim=-1)  # (B, N)
    # 2. Fractional-shift phase for each path (non-negative-k convention).
    k = torch.arange(N, device=device, dtype=torch.float32)
    shift_phase = torch.exp(
        -1j * 2 * torch.pi * ell.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0) / N
    ).to(dtype)  # (B, P, N)
    S_shifted = S.unsqueeze(1) * shift_phase  # (B, P, N)
    s_shifted = torch.fft.ifft(S_shifted, dim=-1)  # (B, P, N)
    # 3. Doppler modulation
    n = torch.arange(N, device=device, dtype=torch.float32)
    doppler_phase = torch.exp(
        1j * 2 * torch.pi * kappa.unsqueeze(-1) * (n.unsqueeze(0).unsqueeze(0) + N_cp) / N
    ).to(dtype)  # (B, P, N)
    A = s_shifted * doppler_phase  # (B, P, N)
    # Transpose to (B, N, P)
    return A.transpose(-1, -2)


def cg_solve(
    matvec: callable,
    b: torch.Tensor,
    max_iter: int = 20,
    tol: float = 1e-6,
    x0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Conjugate-gradient solver for a Hermitian positive-definite system Ax = b.

    `matvec(v) -> Av` is a user-supplied callable that respects the leading batch
    dimensions. Solves independently for each batch element with a common iteration
    budget.

    Returns
    -------
    x : same shape as b
    """
    if x0 is None:
        x = torch.zeros_like(b)
    else:
        x = x0.clone()
    r = b - matvec(x)
    p = r.clone()
    r_norm = (torch.conj(r) * r).sum(dim=-1).real
    b_norm = (torch.conj(b) * b).sum(dim=-1).real.clamp(min=1e-30)
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = (torch.conj(p) * Ap).sum(dim=-1).real.clamp(min=1e-30)
        alpha = (r_norm / denom).unsqueeze(-1).to(x.dtype)
        x = x + alpha * p
        r_new = r - alpha * Ap
        r_norm_new = (torch.conj(r_new) * r_new).sum(dim=-1).real
        if (r_norm_new / b_norm).max() < tol:
            break
        beta = (r_norm_new / r_norm.clamp(min=1e-30)).unsqueeze(-1).to(p.dtype)
        p = r_new + beta * p
        r = r_new
        r_norm = r_norm_new
    return x


@dataclass
class ClassicalCGDetector:
    """JSAC-era classical semi-blind AFDM detector.

    Configuration:
      T           : number of outer iterations (default 8).
      K_cg        : CG iterations per x-step (default 10).
      alpha       : h-step relaxation in [0, 1] (default 1.0 = exact ridge posterior).
      lambda_ridge: h-step ridge regularization strength (default 1e-2).
    """

    system: AFDMSystem
    support_recovery: SupportRecovery
    constellation: torch.Tensor
    pilot_positions: torch.Tensor  # (N_p,) long
    pilot_values: torch.Tensor     # (N_p,) complex
    T: int = 8
    K_cg: int = 10
    alpha: float = 1.0
    lambda_ridge: float = 1e-2
    noise_var: Optional[float] = None  # if None, uses a small default

    def detect(self, r: torch.Tensor, sigma_w2: float) -> dict[str, torch.Tensor]:
        """Run the classical detector on a batch of received signals.

        Parameters
        ----------
        r         : (B, N) complex received time-domain signal (post CP strip).
        sigma_w2  : noise variance (assumed known).

        Returns
        -------
        Dict with keys:
          x_hat   : (B, N) posterior-mean DAFT-domain symbol estimate
          hard_x  : (B, N) hard demapped constellation-index tensor (long)
          h_hat   : (B, P_hat) complex gain estimate (padded to support size)
          ell_hat : (B, P_hat)
          kappa_hat : (B, P_hat)
          p_hat   : (B,) number of detected paths
        """
        B, N = r.shape
        device = r.device
        dtype = r.dtype
        # Pilot-only transmit signal (identical across batch).
        x_pilot = torch.zeros(N, dtype=dtype, device=device)
        x_pilot[self.pilot_positions] = self.pilot_values
        s_pilot = self.system.idaft(x_pilot.unsqueeze(0))[0]  # (N,)

        # 1. Support recovery on the received time-domain signal.
        ell_hat, kappa_hat, p_hat = self.support_recovery(r, s_pilot)
        P_max = ell_hat.shape[1]

        # 2. Initialize x_hat: pilots at known values, data at 0.
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)

        # 3. Initialize h_hat: LS with initial x_hat.
        A = build_regression_matrix(self.system, ell_hat, kappa_hat, x_hat)  # (B, N, P)
        AH = A.conj().transpose(-1, -2)  # (B, P, N)
        AhA = AH @ A  # (B, P, P)
        Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)  # (B, P)
        ridge = self.lambda_ridge * torch.eye(P_max, dtype=dtype, device=device).unsqueeze(0)
        h_hat = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)  # (B, P)

        # 4. Compute DAFT-domain observation y = DAFT(r).
        y = self.system.daft(r)

        # 5. Outer iterations.
        for t in range(self.T):
            # x-step: CG-MMSE using the fast operator with current h_hat.
            op = FastAFDMOperator(system=self.system, ell=ell_hat, kappa=kappa_hat, h=h_hat)
            def matvec(v):
                return op.rmatvec(op.matvec(v)) + sigma_w2 * v
            Hty = op.rmatvec(y)
            x_soft = cg_solve(matvec, Hty, max_iter=self.K_cg)
            # Hard demap (nearest constellation), preserve pilots.
            dists = (x_soft.unsqueeze(-1) - self.constellation.reshape(1, 1, -1)).abs()
            hard_idx = dists.argmin(dim=-1)
            x_hard = self.constellation[hard_idx]
            x_hard[:, self.pilot_positions] = self.pilot_values.unsqueeze(0)
            x_hat = x_hard
            # h-step: LS with updated x_hat.
            A = build_regression_matrix(self.system, ell_hat, kappa_hat, x_hat)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
            h_hat = (1 - self.alpha) * h_hat + self.alpha * torch.linalg.solve(
                AhA + ridge, Ahr.unsqueeze(-1)
            ).squeeze(-1)

        # Final CG-MMSE for soft output
        op = FastAFDMOperator(system=self.system, ell=ell_hat, kappa=kappa_hat, h=h_hat)
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
            "h_hat": h_hat,
            "ell_hat": ell_hat,
            "kappa_hat": kappa_hat,
            "p_hat": p_hat,
        }
