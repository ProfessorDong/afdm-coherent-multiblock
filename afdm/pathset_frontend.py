"""Front-end feature builder for the PathSetEstimator.

Turns a batch of received time-domain signals into (scalar_feats, patch_feats,
valid, ell_cfar, kap_cfar) which the estimator consumes. Also returns the raw
complex ambiguity map so downstream code can re-use it for patch extraction
without recomputing.

Distinct from afdm/support.py:
  * support.py computes only the magnitude-squared ambiguity surface, which loses
    the phase information that a learned network can exploit.
  * this module keeps the COMPLEX correlation (Re, Im, log|A|) as patch channels.
  * this module returns K=24 candidates (overcomplete) rather than P_max=6.
"""

from __future__ import annotations

import torch

from .classical import build_regression_matrix
from .pathset_estimator import extract_ambiguity_patches, PathSetEstimatorConfig
from .support import cfar_peaks, newton_refine
from .system import AFDMSystem


def complex_ambiguity(
    r: torch.Tensor, s_pilot: torch.Tensor,
    system: AFDMSystem, kappa_max: float,
    oversample_doppler: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the COMPLEX cross-ambiguity map (unlike support.ambiguity_function
    which returns |A|^2 only).

    Returns
    -------
    A_complex : (B, L_kappa, L_ell) complex
    ell_grid  : (L_ell,) real
    kap_grid  : (L_kappa,) real
    """
    device = r.device; dtype = r.dtype
    B = r.shape[0]; N = system.N; N_cp = system.ell_max
    if s_pilot.ndim == 1:
        s_pilot = s_pilot.unsqueeze(0).expand(B, -1)

    L_ell = int(system.ell_max) + 1
    ell_grid = torch.arange(L_ell, device=device, dtype=torch.float32)
    L_kappa = int(oversample_doppler * (2 * int(kappa_max) + 1)) + 1
    kap_grid = torch.linspace(-kappa_max, kappa_max, L_kappa, device=device, dtype=torch.float32)

    n = torch.arange(N, device=device, dtype=torch.float32)
    doppler_phase = torch.exp(-1j * 2 * torch.pi * kap_grid.unsqueeze(-1) * (n + N_cp) / N).to(dtype)
    r_demod = r.unsqueeze(1) * doppler_phase.unsqueeze(0)   # (B, L_k, N)
    R = torch.fft.fft(r_demod, dim=-1)
    S = torch.fft.fft(s_pilot, dim=-1).unsqueeze(1)
    C = torch.fft.ifft(R * torch.conj(S), dim=-1)            # (B, L_k, N)
    A_complex = C[..., :L_ell] / N                          # (B, L_k, L_ell)
    return A_complex, ell_grid, kap_grid


def build_frontend_inputs(
    r: torch.Tensor,
    system: AFDMSystem,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    kappa_max: float = 5.0,
    K: int = 24,
    sigma_w2_block: float | torch.Tensor = 1e-2,
    lambda_ridge: float = 1e-3,
) -> dict:
    """Turn a batch into the tuple consumed by PathSetEstimator.

    Returns dict with keys:
      scalar_feats : (B, K, 5)
      patch_feats  : (B, K, 75)
      valid        : (B, K) bool
      ell_cfar     : (B, K) fractional delay (post-Newton)
      kap_cfar     : (B, K) fractional Doppler
      A_complex    : (B, L_k, L_e) for downstream reuse
      h_ls         : (B, K) initial LS gains (for logging/comparison)
    """
    B, N = r.shape
    device = r.device; dtype = r.dtype
    # Pilot time-domain signal (batch-independent, so build once).
    x_pilot = torch.zeros(N, dtype=dtype, device=device)
    x_pilot[pilot_positions] = pilot_values
    s_pilot = system.idaft(x_pilot.unsqueeze(0))[0]  # (N,)

    # 1. Complex ambiguity map.
    A_complex, ell_grid, kap_grid = complex_ambiguity(r, s_pilot, system, kappa_max)
    A_mag2 = A_complex.abs() ** 2

    # 2. Top-K local maxima (allow adjacency to include weak paths).
    peak_idx, _ = cfar_peaks(A_mag2, K=K, min_separation=1)
    ell_ref, kap_ref = newton_refine(A_mag2, peak_idx, ell_grid, kap_grid, max_iter=2)
    valid = peak_idx[:, :, 0] >= 0

    # 3. Initial LS gains for each candidate (pilot-only regression).
    x_pilot_B = torch.zeros(B, N, dtype=dtype, device=device)
    x_pilot_B[:, pilot_positions] = pilot_values.unsqueeze(0)
    A_reg = build_regression_matrix(system, ell_ref, kap_ref, x_pilot_B)  # (B, N, K)
    AH = A_reg.conj().transpose(-1, -2)
    AhA = AH @ A_reg
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
    ridge = lambda_ridge * torch.eye(K, dtype=dtype, device=device).unsqueeze(0)
    h_ls = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)  # (B, K)

    # 4. Patch features (complex).
    patch_feats, _ = extract_ambiguity_patches(A_complex, peak_idx)   # (B, K, 75)

    # 5. Scalar features: ell, kappa, Re(h_ls), Im(h_ls), sqrt(sigma_w2).
    sig = sigma_w2_block if torch.is_tensor(sigma_w2_block) else torch.tensor(
        float(sigma_w2_block), device=device
    )
    sig_scalar = torch.sqrt(sig).to(torch.float32).expand(B, K)
    scalar_feats = torch.stack([
        ell_ref, kap_ref, h_ls.real, h_ls.imag, sig_scalar,
    ], dim=-1)                                                          # (B, K, 5)
    # Zero invalid rows.
    scalar_feats = scalar_feats * valid.unsqueeze(-1).to(scalar_feats.dtype)

    return {
        "scalar_feats": scalar_feats,
        "patch_feats": patch_feats,
        "valid": valid,
        "ell_cfar": ell_ref,
        "kap_cfar": kap_ref,
        "A_complex": A_complex,
        "h_ls": h_ls,
    }
