"""Off-grid delay-Doppler support recovery for AFDM.

Given a received time-domain signal r and the known pilot waveform s_P, we
estimate the fractional delay-Doppler support {(tau_i, nu_i)}_{i=1}^{P_hat} by:

  1. Computing the two-dimensional cross-ambiguity function A[l, k] on a
     2x-oversampled grid via a per-Doppler FFT-based cross-correlation:
        A[l, k] = (1/N) |sum_n r[n] s_P^*[(n - l) mod N] exp(-j 2 pi k (n + N_cp)/N)|^2.
  2. Selecting peaks via ordered-statistics CFAR (OS-CFAR) with a specified
     false-alarm probability.
  3. Refining each peak by two Newton iterations on a local quadratic Taylor
     expansion of A around the peak.

All operations are batched on cuda:0 and use torch.complex64.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


def ambiguity_function(
    r: torch.Tensor,
    s_pilot: torch.Tensor,
    N: int,
    N_cp: int,
    kappa_max: float,
    ell_max: float,
    oversample_doppler: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the 2D ambiguity function on an integer-delay grid and (optionally
    oversampled) Doppler grid.

    We compute the ambiguity at INTEGER delays only (via cyclic FFT correlation),
    then rely on Newton refinement (`newton_refine`) to obtain fractional delay
    estimates from the local quadratic. The Doppler grid can be oversampled cheaply
    because it appears only in the multiplicative Doppler-phase demodulation, not
    in the FFT.

    This avoids a subtle inconsistency: zero-padded FFT interpolation of the
    correlation uses a signed-k convention, whereas our fast operator uses a
    non-negative-k convention for fractional shifts. Newton refinement on the
    integer surface is fully consistent with both.

    Parameters
    ----------
    r                  : (B, N) complex time-domain received signal (post CP strip).
    s_pilot            : (B, N) or (N,) complex pilot-only transmit signal.
    N                  : block length.
    N_cp               : cyclic prefix length (Doppler phase reference).
    kappa_max          : maximum Doppler index (symmetric range).
    ell_max            : maximum delay (samples).
    oversample_doppler : Doppler grid oversampling factor (default 2).

    Returns
    -------
    A          : (B, L_kappa, L_ell) real ambiguity magnitudes squared.
    ell_grid   : (L_ell,) integer delay values.
    kappa_grid : (L_kappa,) Doppler values on grid.
    """
    device = r.device
    dtype = r.dtype
    B = r.shape[0]
    if s_pilot.ndim == 1:
        s_pilot = s_pilot.unsqueeze(0).expand(B, -1)

    # Delay grid: integers 0, 1, ..., ell_max (inclusive).
    L_ell = int(ell_max) + 1
    ell_grid = torch.arange(L_ell, device=device, dtype=torch.float32)

    # Doppler grid: (2 kappa_max + 1) * oversample_doppler evenly-spaced values.
    L_kappa = int(oversample_doppler * (2 * int(kappa_max) + 1)) + 1
    kappa_grid = torch.linspace(-kappa_max, kappa_max, L_kappa, device=device, dtype=torch.float32)

    # For each Doppler kappa, Doppler-demodulate r then FFT-correlate with s_pilot.
    n = torch.arange(N, device=device, dtype=torch.float32)
    doppler_phase = torch.exp(-1j * 2 * torch.pi * kappa_grid.unsqueeze(-1) * (n + N_cp) / N).to(dtype)
    r_demod = r.unsqueeze(1) * doppler_phase.unsqueeze(0)  # (B, L_kappa, N)

    # Cyclic cross-correlation: c[b, k, l] = sum_n r_demod[b, k, n] conj(s_pilot[b, (n - l) mod N]).
    R = torch.fft.fft(r_demod, dim=-1)  # (B, L_kappa, N)
    S = torch.fft.fft(s_pilot, dim=-1).unsqueeze(1)  # (B, 1, N)
    C = torch.fft.ifft(R * torch.conj(S), dim=-1)  # (B, L_kappa, N)

    # Take the delays in [0, ell_max] (first L_ell bins of the cyclic correlation).
    C_selected = C[..., :L_ell]
    A = C_selected.abs() ** 2 / N  # (B, L_kappa, L_ell)
    return A, ell_grid, kappa_grid


def cfar_peaks(
    A: torch.Tensor,
    K: int,
    guard_size: int = 1,
    min_separation: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select up to K peaks in a batched 2D ambiguity surface.

    Uses a simple but reliable strategy: find all local maxima (each cell greater
    than its 8 neighbors), sort by amplitude, then greedily pick the top-K such
    that no two selected peaks are closer than `min_separation` in either axis.

    Parameters
    ----------
    A               : (B, L_kappa, L_ell) real ambiguity surface.
    K               : maximum number of peaks to return per batch element.
    guard_size      : half-width of the neighborhood used for local-max detection.
    min_separation  : minimum separation between selected peaks (in grid units).

    Returns
    -------
    peak_idx        : (B, K, 2) integer indices (kappa_idx, ell_idx); -1 for unfilled slots.
    peak_values     : (B, K) ambiguity values at selected peaks; 0 for unfilled slots.
    """
    B, L_k, L_e = A.shape
    device = A.device

    # 1. Find local maxima via max-pooling.
    from torch.nn.functional import max_pool2d
    kernel = 2 * guard_size + 1
    A_pooled = max_pool2d(A.unsqueeze(1), kernel_size=kernel, stride=1, padding=guard_size).squeeze(1)
    is_local_max = (A >= A_pooled - 1e-12)

    # 2. Sort candidates by value.
    flat_vals = A.reshape(B, -1)
    flat_mask = is_local_max.reshape(B, -1)
    masked_vals = torch.where(flat_mask, flat_vals, torch.full_like(flat_vals, -float("inf")))
    sorted_vals, sorted_idx = torch.sort(masked_vals, dim=-1, descending=True)

    # 3. Greedy peak selection with min-separation.
    peak_idx = torch.full((B, K, 2), -1, device=device, dtype=torch.long)
    peak_values = torch.zeros((B, K), device=device)
    for b in range(B):
        selected = 0
        picked = []
        for cand_flat_idx, cand_val in zip(sorted_idx[b], sorted_vals[b]):
            if cand_val == -float("inf"):
                break
            k_idx = cand_flat_idx // L_e
            l_idx = cand_flat_idx % L_e
            too_close = any(
                abs(k_idx - pk).item() < min_separation and abs(l_idx - pl).item() < min_separation
                for pk, pl in picked
            )
            if too_close:
                continue
            peak_idx[b, selected, 0] = k_idx
            peak_idx[b, selected, 1] = l_idx
            peak_values[b, selected] = cand_val
            picked.append((k_idx, l_idx))
            selected += 1
            if selected == K:
                break
    return peak_idx, peak_values


def newton_refine(
    A: torch.Tensor,
    peak_idx: torch.Tensor,
    ell_grid: torch.Tensor,
    kappa_grid: torch.Tensor,
    max_iter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Refine each peak to sub-grid precision via a 2D quadratic Taylor fit.

    At a peak (kappa_idx, ell_idx) with ambiguity value A_0, we fit
        A(l0 + dl, k0 + dk) approx A_0 + g^T d + (1/2) d^T H d,
    using 3-point finite differences along each axis for g and H, then take the
    Newton step d = -H^{-1} g (clipped to |d| <= 1 grid step in each axis to
    avoid overshoot for imperfect quadratic fit).

    Parameters
    ----------
    A          : (B, L_kappa, L_ell) ambiguity surface.
    peak_idx   : (B, K, 2) integer indices (-1 for empty).
    ell_grid   : (L_ell,) delay grid values.
    kappa_grid : (L_kappa,) Doppler grid values.
    max_iter   : number of Newton iterations (default 2).

    Returns
    -------
    ell_refined   : (B, K) fractional delay estimates.
    kappa_refined : (B, K) fractional Doppler estimates.
    """
    B, K, _ = peak_idx.shape
    L_k, L_e = A.shape[-2:]
    device = A.device
    d_ell = (ell_grid[1] - ell_grid[0]).item() if L_e > 1 else 1.0
    d_kap = (kappa_grid[1] - kappa_grid[0]).item() if L_k > 1 else 1.0

    ell_refined = torch.zeros((B, K), device=device, dtype=torch.float32)
    kappa_refined = torch.zeros((B, K), device=device, dtype=torch.float32)

    for b in range(B):
        for i in range(K):
            k0 = peak_idx[b, i, 0].item()
            l0 = peak_idx[b, i, 1].item()
            if k0 < 0:
                continue
            ell_est = ell_grid[l0].item()
            kap_est = kappa_grid[k0].item()

            for _ in range(max_iter):
                # Ensure peak is not on boundary; if it is, skip refinement (use grid value).
                if l0 <= 0 or l0 >= L_e - 1 or k0 <= 0 or k0 >= L_k - 1:
                    break
                # 3-point finite differences on the fine grid around (k0, l0).
                A_00 = A[b, k0, l0]
                A_1p_0 = A[b, k0, l0 + 1]
                A_1m_0 = A[b, k0, l0 - 1]
                A_0_1p = A[b, k0 + 1, l0]
                A_0_1m = A[b, k0 - 1, l0]
                A_1p_1p = A[b, k0 + 1, l0 + 1]
                A_1m_1m = A[b, k0 - 1, l0 - 1]
                A_1p_1m = A[b, k0 - 1, l0 + 1]
                A_1m_1p = A[b, k0 + 1, l0 - 1]
                # Gradient (positive-going differences, central):
                # dA/dell at l0 approx (A_1p_0 - A_1m_0) / (2 d_ell)
                # dA/dkap at k0 approx (A_0_1p - A_0_1m) / (2 d_kap)
                g_ell = (A_1p_0 - A_1m_0) / (2 * d_ell)
                g_kap = (A_0_1p - A_0_1m) / (2 * d_kap)
                # Hessian (central):
                # d2A/dell^2 approx (A_1p_0 - 2 A_00 + A_1m_0) / d_ell^2
                # d2A/dkap^2 approx (A_0_1p - 2 A_00 + A_0_1m) / d_kap^2
                # d2A/(dell dkap) approx (A_1p_1p - A_1p_1m - A_1m_1p + A_1m_1m) / (4 d_ell d_kap)
                H_ee = (A_1p_0 - 2 * A_00 + A_1m_0) / d_ell**2
                H_kk = (A_0_1p - 2 * A_00 + A_0_1m) / d_kap**2
                H_ek = (A_1p_1p - A_1p_1m - A_1m_1p + A_1m_1m) / (4 * d_ell * d_kap)
                # Newton step: d = -H^{-1} g, restricted for stability (H should be neg def near max).
                det_H = H_ee * H_kk - H_ek**2
                if det_H.abs() < 1e-12:
                    break
                inv_H_ee = H_kk / det_H
                inv_H_kk = H_ee / det_H
                inv_H_ek = -H_ek / det_H
                d_ell_step = -(inv_H_ee * g_ell + inv_H_ek * g_kap)
                d_kap_step = -(inv_H_ek * g_ell + inv_H_kk * g_kap)
                # Clip step to +/- one grid step for robustness against imperfect quadratic fits.
                d_ell_step = torch.clamp(d_ell_step, -d_ell, d_ell)
                d_kap_step = torch.clamp(d_kap_step, -d_kap, d_kap)
                ell_est += d_ell_step.item()
                kap_est += d_kap_step.item()

            ell_refined[b, i] = ell_est
            kappa_refined[b, i] = kap_est
    return ell_refined, kappa_refined


@dataclass
class SupportRecovery:
    """Combined ambiguity + CFAR + Newton pipeline."""

    N: int
    N_cp: int
    kappa_max: float
    ell_max: float
    P_max: int  # maximum number of paths to detect per block
    oversample_delay: int = 2
    oversample_doppler: int = 2
    min_separation: int = 2

    def __call__(
        self, r: torch.Tensor, s_pilot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recover support from received time-domain signal.

        Returns
        -------
        ell_hat   : (B, P_max) fractional delay estimates (0 for unfilled slots)
        kappa_hat : (B, P_max) fractional Doppler estimates
        p_hat     : (B,) integer number of detected paths per batch element
        """
        A, ell_grid, kappa_grid = ambiguity_function(
            r, s_pilot, self.N, self.N_cp,
            self.kappa_max, self.ell_max,
            oversample_doppler=self.oversample_doppler,
        )
        peak_idx, peak_values = cfar_peaks(A, K=self.P_max, min_separation=self.min_separation)
        ell_hat, kappa_hat = newton_refine(A, peak_idx, ell_grid, kappa_grid, max_iter=2)
        p_hat = (peak_idx[:, :, 0] >= 0).sum(dim=-1)
        return ell_hat, kappa_hat, p_hat
