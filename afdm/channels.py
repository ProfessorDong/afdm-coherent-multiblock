"""Doubly-dispersive channel generation.

Each channel realization consists of P paths with complex Gaussian gains, fractional
delay indices ell_i (samples), and fractional Doppler indices kappa_i (units of
subcarrier spacing).

Provided channel families:
  - UniformFractionalChannel: ell_i ~ U[0, ell_max], kappa_i ~ U[-kappa_max, kappa_max],
    gains ~ CN(0, sigma_i^2) with a linearly decaying dB power profile.
  - TDLProfile: 3GPP TS 38.901 Tapped Delay Line profiles TDL-A/B/C/D/E, converted
    to per-subcarrier normalized indices given a desired RMS delay spread and Doppler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch


TDL_PROFILE_TABLE = {
    # (relative delay in units of tau_rms, path power in dB) from TS 38.901 Table 7.7.2-X
    "TDL-A": [
        (0.0000, -13.4), (0.3819, 0.0), (0.4025, -2.2), (0.5868, -4.0), (0.4610, -6.0),
        (0.5375, -8.2), (0.6708, -9.9), (0.5750, -10.5), (0.7618, -7.5), (0.1522, -15.9),
        (2.1848, -6.6), (2.4577, -16.7), (2.9018, -12.4), (3.0868, -15.2), (3.3986, -10.8),
        (4.0961, -11.3), (4.2226, -12.7), (4.9633, -16.2), (5.0630, -18.3), (5.4560, -18.9),
        (5.5794, -16.6), (5.6934, -19.9), (6.2400, -29.7),
    ],
    "TDL-C": [
        (0.0000, -4.4), (0.2099, -1.2), (0.2219, -3.5), (0.2329, -5.2), (0.2176, -2.5),
        (0.6366, 0.0), (0.6448, -2.2), (0.6560, -3.9), (0.6584, -7.4), (0.7935, -7.1),
        (0.8213, -10.7), (0.9336, -11.1), (1.2285, -5.1), (1.3083, -6.8), (2.1704, -8.7),
        (2.7105, -13.2), (4.2589, -13.9), (4.6003, -13.9), (5.4902, -15.8), (5.6077, -17.1),
        (6.3065, -16.0), (6.6374, -15.7), (7.0427, -21.6), (8.6523, -22.8),
    ],
}


@dataclass
class UniformFractionalChannel:
    """Uniform-random fractional delay/Doppler channel.

    Draws P per block; per-path gains are CN(0, sigma_i^2) with a linear dB decay.
    """

    P: int
    ell_max: float
    kappa_max: float
    decay_db_per_path: float = 2.0  # power decay per path
    device: str | torch.device = "cuda:0"

    def sample(self, batch: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Return a batch of channel realizations.

        Returns dict with:
          ell  : (batch, P) fractional delay indices in [0, ell_max]
          kappa: (batch, P) fractional Doppler indices in [-kappa_max, kappa_max]
          h    : (batch, P) complex path gains, unit total power on average
        """
        u_ell = torch.rand(batch, self.P, device=self.device, generator=generator)
        u_kap = torch.rand(batch, self.P, device=self.device, generator=generator)
        ell = u_ell * self.ell_max
        kappa = (2.0 * u_kap - 1.0) * self.kappa_max
        # Per-path variances: linearly decaying dB profile, normalized to sum = 1.
        db = -self.decay_db_per_path * torch.arange(self.P, device=self.device, dtype=torch.float32)
        sigma2 = 10.0 ** (db / 10.0)
        sigma2 = sigma2 / sigma2.sum()
        sigma = torch.sqrt(sigma2 / 2.0)  # per real/imag component
        h_real = torch.randn(batch, self.P, device=self.device, generator=generator) * sigma
        h_imag = torch.randn(batch, self.P, device=self.device, generator=generator) * sigma
        h = torch.complex(h_real, h_imag)
        return {"ell": ell, "kappa": kappa, "h": h}


@dataclass
class TDLProfile:
    """3GPP TS 38.901 Tapped Delay Line profile with Doppler.

    Parameters
    ----------
    profile        : one of the TDL_PROFILE_TABLE keys, e.g., "TDL-C".
    delay_spread_ns: desired RMS delay spread in nanoseconds (e.g. 300).
    delta_f_hz     : subcarrier spacing (Hz), used to convert delay/Doppler to indices.
    doppler_hz     : maximum Doppler shift in Hz (Jakes / Clarke spectrum).
    P_use          : if not None, use only the first P_use paths (truncate for tractability).
    seed_gains     : if True, gains are Rayleigh; if False, deterministic (line-of-sight-like).
    device         : torch device.
    """

    profile: Literal["TDL-A", "TDL-C"] = "TDL-C"
    delay_spread_ns: float = 300.0
    delta_f_hz: float = 15e3
    doppler_hz: float = 500.0
    P_use: int | None = None
    device: str | torch.device = "cuda:0"

    _ell_deterministic: torch.Tensor = field(default=None, init=False, repr=False)
    _sigma_deterministic: torch.Tensor = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        table = TDL_PROFILE_TABLE[self.profile]
        rel_delay = np.array([d for d, _ in table], dtype=np.float32)
        pow_db = np.array([p for _, p in table], dtype=np.float32)
        if self.P_use is not None:
            rel_delay = rel_delay[: self.P_use]
            pow_db = pow_db[: self.P_use]
        # Convert to samples: sample period = 1 / (N * Delta_f); ell = tau * N * Delta_f.
        tau_seconds = rel_delay * (self.delay_spread_ns * 1e-9)
        ell_samples = tau_seconds * self.delta_f_hz  # per-subcarrier index units
        # Note: this is ell / N in the paper's convention; we return ell_samples * N later
        # (the ell in the paper is normalized to N * Delta_f, meaning ell = tau * N * Delta_f).
        # For clarity, expose ell in the same convention: ell_i = tau_i * N * Delta_f.
        # We'll multiply by N when producing per-block samples.
        self._rel_ell = torch.tensor(ell_samples, device=self.device, dtype=torch.float32)
        # Normalize powers so they sum to 1.
        sigma2 = 10.0 ** (pow_db / 10.0)
        sigma2 = sigma2 / sigma2.sum()
        self._sigma = torch.tensor(np.sqrt(sigma2 / 2.0), device=self.device, dtype=torch.float32)
        self.P = int(self._rel_ell.shape[0])

    def sample(self, batch: int, N: int, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        """Return a batch of channel realizations for a given N-subcarrier system.

        Returns dict with:
          ell  : (batch, P) delay indices, ell_i = tau_i * N * Delta_f
          kappa: (batch, P) Doppler indices, kappa_i = nu_i / Delta_f
          h    : (batch, P) complex Rayleigh gains
        """
        ell = self._rel_ell.unsqueeze(0).expand(batch, -1) * N
        # Doppler for each path: sample from Jakes spectrum by cos(angle) with uniform angle.
        angles = torch.rand(batch, self.P, device=self.device, generator=generator) * (2 * np.pi)
        kappa = (self.doppler_hz / self.delta_f_hz) * torch.cos(angles)
        h_real = torch.randn(batch, self.P, device=self.device, generator=generator) * self._sigma.unsqueeze(0)
        h_imag = torch.randn(batch, self.P, device=self.device, generator=generator) * self._sigma.unsqueeze(0)
        h = torch.complex(h_real, h_imag)
        return {"ell": ell, "kappa": kappa, "h": h}


class DoublyDispersiveChannel:
    """Apply a doubly-dispersive channel to a time-domain signal.

    Uses the periodic-sinc (Dirichlet) kernel for fractional delays and complex
    exponential for fractional Doppler. This is the "slow" implementation used
    for correctness verification and for training data generation (once, per block).
    """

    def __init__(self, N: int, N_cp: int, device: str | torch.device = "cuda:0", dtype: torch.dtype = torch.complex64):
        self.N = N
        self.N_cp = N_cp
        self.device = device
        self.dtype = dtype
        self._n_grid = torch.arange(N + N_cp, device=device, dtype=torch.float32)

    @staticmethod
    def periodic_sinc(m: torch.Tensor, delta_ell: torch.Tensor, N: int) -> torch.Tensor:
        """Length-N complex Dirichlet kernel g_{delta_ell}[m], the impulse response
        of a bandlimited circular fractional-delay filter.

        Derived from the DFT-based fractional shift: for a length-N signal, shifting
        by delta samples via S -> S * exp(-j 2 pi k delta / N) followed by IFFT is
        equivalent to circular convolution with the kernel
            g[m] = (1/N) sum_{k=0}^{N-1} exp(j 2 pi k (m - delta) / N)
                 = (1/N) exp(j pi (N-1) (m - delta) / N)
                   * sin(pi (m - delta)) / sin(pi (m - delta) / N).
        This complex kernel (with the phase factor) is the physically correct
        response of a bandlimited circular channel to a fractional-delay impulse.
        The magnitude-only version g[m] = sin(pi(m-delta)) / (N sin(pi(m-delta)/N))
        is the classical Dirichlet kernel but is INconsistent with FFT-based
        fractional shifts because it omits the phase factor.

        Returns a complex-valued tensor of shape (..., N).
        """
        # Broadcast: delta_ell has shape (...,), m has shape (N,).
        # arg has shape (..., N).
        arg = torch.pi * (m - delta_ell.unsqueeze(-1))
        # Magnitude part
        num = torch.sin(arg)
        den = N * torch.sin(arg / N)
        eps = 1e-12
        mag = torch.where(torch.abs(den) < eps, torch.ones_like(num), num / den)
        # Phase part exp(j pi (N-1) (m - delta) / N)
        phase = torch.exp(1j * torch.pi * (N - 1) * (m - delta_ell.unsqueeze(-1)) / N)
        # Convert to complex dtype matching phase
        return (mag.to(phase.dtype)) * phase

    def apply(self, s: torch.Tensor, ell: torch.Tensor, kappa: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Convolve a transmitted signal s (with prefix) by the doubly-dispersive channel.

        Parameters
        ----------
        s     : (batch, N + N_cp) complex, time-domain transmit signal with CP.
        ell   : (batch, P) fractional delay indices in samples.
        kappa : (batch, P) fractional Doppler indices (normalized to Delta_f).
        h     : (batch, P) complex path gains.

        Returns
        -------
        r     : (batch, N + N_cp) complex, received signal (before CP strip).
                (The paper uses r as the length-N post-CP signal; we return length
                N + N_cp so the caller can decide when to strip.)
        """
        batch, L = s.shape
        assert L == self.N + self.N_cp
        assert ell.shape == (batch, ell.shape[1])
        P = ell.shape[1]
        # For each path, apply Doppler modulation and fractional delay:
        # r[n] = sum_i h_i exp(j 2 pi kappa_i n / N) * (fractional-delay convolution of s by ell_i)
        # Fractional delay is implemented via periodic-sinc interpolation along the length-L axis.
        m = torch.arange(L, device=self.device, dtype=torch.float32)
        # Broadcast: (batch, P, L, L) is too large; instead we compute per path.
        r = torch.zeros(batch, L, device=self.device, dtype=self.dtype)
        # Precompute Doppler phase for each path over all times n.
        # phase[b, p, n] = exp(j 2 pi kappa[b,p] * n / N)
        n_range = m.unsqueeze(0).unsqueeze(0)  # (1, 1, L)
        doppler_phase = torch.exp(1j * 2 * torch.pi * kappa.unsqueeze(-1) * n_range / self.N).to(self.dtype)
        # For each path, build the fractional-shift matrix G_p acting on s.
        # (r[n] contribution from path p) = h_p * doppler_phase[n] * sum_m g_{ell_p}[(n - m) mod L] * s[m]
        # For tractability we implement this per path using FFT-based fractional shifts.
        # A fractional shift by delta in a length-L circular buffer is:
        #   IFFT( DFT(s) * exp(-j 2 pi delta k / L) )
        S = torch.fft.fft(s, dim=-1)  # (batch, L)
        k = torch.fft.fftfreq(L, d=1.0, device=self.device) * L  # (L,) integer bins
        for p in range(P):
            delta_p = ell[:, p]  # (batch,)
            phase_k = torch.exp(-1j * 2 * torch.pi * delta_p.unsqueeze(-1) * k.unsqueeze(0) / L).to(self.dtype)
            S_shift = S * phase_k
            s_shift = torch.fft.ifft(S_shift, dim=-1)  # (batch, L)
            r = r + h[:, p:p + 1] * doppler_phase[:, p, :] * s_shift
        return r
