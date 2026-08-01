"""Fast quasi-banded DAFT-domain channel operator.

The DAFT-domain channel factors as
    H^D(theta, h) = sum_i h_i * Phi_i(tau_i, nu_i),
where each Phi_i is a chirp-modulated circular shift with Doppler phase. Its action
on a DAFT-domain vector x can be evaluated in O(NP + N log N) via:
    y = DAFT( sum_i h_i * shift_{ell_i}( doppler_{kappa_i}( IDAFT(x) ) ) )
which is the "fast" operator used by CG-MMSE inside the receiver.

For verification, we also provide slow_afdm_operator that builds the dense N x N
DAFT-domain matrix from the time-domain channel matrix, which is O(N^2) but exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .system import AFDMSystem


@dataclass
class FastAFDMOperator:
    """Callable DAFT-domain channel operator implementing y = H^D x.

    Parameters are supplied at construction. Batching is supported: the leading
    dimension of x, ell, kappa, h is a common batch dimension.
    """

    system: AFDMSystem
    ell: torch.Tensor    # (batch, P)
    kappa: torch.Tensor  # (batch, P)
    h: torch.Tensor      # (batch, P) complex

    def __post_init__(self) -> None:
        assert self.ell.shape == self.kappa.shape
        assert self.h.shape == self.ell.shape
        self.batch, self.P = self.ell.shape
        self.N = self.system.N
        # Precompute per-batch, per-path Doppler-phase vector at the CP-adjusted grid.
        n = torch.arange(self.N, device=self.system.device, dtype=torch.float32)
        # phase[b, p, n] = exp(j 2 pi kappa[b, p] * (n + N_cp) / N)
        phase = torch.exp(
            1j * 2 * torch.pi * self.kappa.unsqueeze(-1) * (n.unsqueeze(0).unsqueeze(0) + self.system.ell_max) / self.N
        )
        self._doppler_phase = phase.to(self.system.dtype)
        # Precompute per-batch, per-path fractional-shift phases (frequency-domain multipliers).
        # We use k = 0, 1, ..., N-1 (non-negative convention) so that the resulting
        # fractional-shift kernel matches the [0, N-1] Dirichlet kernel with its
        # exp(j pi (N-1) (m - delta) / N) phase factor, consistent with
        # DoublyDispersiveChannel.periodic_sinc.
        k = torch.arange(self.N, device=self.system.device, dtype=torch.float32)
        # shift_phase[b, p, k] = exp(-j 2 pi ell[b, p] * k / N)
        shift_phase = torch.exp(
            -1j * 2 * torch.pi * self.ell.unsqueeze(-1) * k.unsqueeze(0).unsqueeze(0) / self.N
        )
        self._shift_phase = shift_phase.to(self.system.dtype)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Compute y = H^D x for x of shape (batch, N)."""
        assert x.shape == (self.batch, self.N)
        # 1) IDAFT: time-domain transmit signal
        s = self.system.idaft(x)  # (batch, N)
        # 2) Per-path fractional shift and Doppler modulation
        # Build s_shift[b, p, n] via FFT-based fractional shift of s along last axis.
        S = torch.fft.fft(s, dim=-1)  # (batch, N)
        # Broadcast: (batch, 1, N) * (batch, P, N) -> (batch, P, N)
        S_shifted = S.unsqueeze(1) * self._shift_phase
        s_shifted = torch.fft.ifft(S_shifted, dim=-1)  # (batch, P, N)
        # Doppler modulation
        s_modulated = s_shifted * self._doppler_phase  # (batch, P, N)
        # 3) Weighted sum over paths
        r = (self.h.unsqueeze(-1) * s_modulated).sum(dim=1)  # (batch, N)
        # 4) DAFT
        y = self.system.daft(r)
        return y

    def rmatvec(self, y: torch.Tensor) -> torch.Tensor:
        """Compute x = H^D^H y (adjoint), needed by CG-MMSE.

        By the unitarity of DAFT and the structure of H^D,
            H^D^H y = IDAFT( sum_i conj(h_i) * doppler_{-kappa_i}(shift_{-ell_i}( DAFT(y) )) ).
        The order of operations mirrors matvec but with negated shifts/phases and conjugated gains.
        """
        assert y.shape == (self.batch, self.N)
        r = self.system.idaft(y)  # (batch, N)  <-- IDAFT because DAFT is unitary and its adjoint is IDAFT
        # Doppler demodulation with conjugated phase
        r_expanded = r.unsqueeze(1)  # (batch, 1, N)
        r_demod = r_expanded * torch.conj(self._doppler_phase)  # (batch, P, N)
        # Inverse fractional shift = multiply by conjugate shift_phase in frequency
        R = torch.fft.fft(r_demod, dim=-1)  # (batch, P, N)
        R_shifted = R * torch.conj(self._shift_phase)
        r_shifted = torch.fft.ifft(R_shifted, dim=-1)  # (batch, P, N)
        # Weighted sum with conjugated gains
        r_sum = (torch.conj(self.h).unsqueeze(-1) * r_shifted).sum(dim=1)  # (batch, N)
        x = self.system.daft(r_sum)
        return x


def slow_afdm_operator(
    system: AFDMSystem,
    ell: torch.Tensor,
    kappa: torch.Tensor,
    h: torch.Tensor,
) -> torch.Tensor:
    """Build the dense N x N DAFT-domain channel matrix (batched).

    Returns H^D of shape (batch, N, N), useful only for correctness verification
    against FastAFDMOperator on small N.

    Constructs the time-domain channel matrix H^td first (with periodic-sinc
    fractional-delay taps and Doppler phase) and then applies unitary similarity:
        H^D = F H^td F^H, using our DAFT/IDAFT operations column-by-column.
    """
    from .channels import DoublyDispersiveChannel

    batch, P = ell.shape
    N = system.N
    device = system.device
    dtype = system.dtype
    # Build H^td dense per batch.
    n = torch.arange(N, device=device, dtype=torch.float32)
    m = torch.arange(N, device=device, dtype=torch.float32)
    # For each batch and path, add h_i * g_{delta_i}[(n - m - ell_int_i) mod N] * exp(j 2 pi kappa_i (n + N_cp)/N)
    Htd = torch.zeros(batch, N, N, device=device, dtype=dtype)
    N_cp = system.ell_max
    for b in range(batch):
        for p in range(P):
            ell_p = ell[b, p]
            # Build the length-N complex Dirichlet kernel evaluated at the TOTAL delay,
            # which gives the impulse response of a circular fractional shift by ell_p.
            m_range = torch.arange(N, device=device, dtype=torch.float32)
            g = DoublyDispersiveChannel.periodic_sinc(m_range, ell_p.unsqueeze(0), N).squeeze(0)  # (N,) complex
            # Doppler phase along rows (n index)
            doppler = torch.exp(1j * 2 * torch.pi * kappa[b, p] * (n + N_cp) / N).to(dtype)
            # Circular convolution: G[n, m] = g[(n - m) mod N]
            n_minus_m = (n.unsqueeze(1) - m.unsqueeze(0)).long() % N
            G = g[n_minus_m]  # (N, N) complex
            Htd[b] += h[b, p] * doppler.unsqueeze(1) * G
    # Apply DAFT similarity: H^D = F H^td F^H, using the explicit DAFT matrix
    # (this is a verification-only path, so O(N^3) is acceptable).
    F = system.daft_matrix()  # (N, N), F @ v applies DAFT to column vector v
    Fh = F.conj().transpose(0, 1)  # F^H
    HD = torch.einsum("ij,bjk,kl->bil", F, Htd, Fh)
    return HD
