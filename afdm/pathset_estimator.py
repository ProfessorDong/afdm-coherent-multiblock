"""Amortized path-set estimator (Option 5, candidate-level variant).

Diagnostics summary that drives this design:
  * DIAG A: even at N_p=32, pilot-only LS gain estimation for P=5 leaves 8% SER
    at 15 dB -> data-aided refinement is essential (handled by data_aided_sbl.py).
  * DIAG B: top-24 CFAR local maxima recover >= 95% of true paths in all tested
    configs -> a candidate-level Set-Transformer is sufficient (no dense-map
    encoder needed).
  * DIAG C: deterministic LM iteration alone cannot exploit overcomplete
    candidates because spurious peaks contaminate LS gain estimation. Existence
    heads must handle candidate suppression.

Architecture:
  Input: overcomplete K CFAR candidates (K=24 default), each described by
    - fractional (ell_cfar, kappa_cfar) after 2-Newton refinement
    - LS gain estimate from pilot-only regression
    - 5x5 local ambiguity patch (Re, Im, log|A|) centered at CFAR peak

  Backbone: SetTransformer (permutation-equivariant).

  Per-candidate outputs (six heads):
    - existence_logit (1)             sigmoid -> probability path is real
    - delta_ell (1)                    refined offset from CFAR peak, in delay samples
    - delta_kappa (1)                  refined offset from CFAR peak, in Doppler units
    - Re_h, Im_h (2)                   complex gain
    - log_var (1)                      predicted log-variance of gain estimate

Note: the network REGRESSES the complex gain directly (not delta from LS), which
addresses the "direction/magnitude decoupling" failure mode identified in the
earlier V-EM plateau memo — every path gets its own complex prediction rather
than sharing one scalar magnitude control.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .set_transformer import SetTransformerBlock


PATCH_HALF = 2         # 5x5 patch: half-width = 2
N_PATCH_CH = 3         # Re(A), Im(A), log|A|
PATCH_FLAT_DIM = (2 * PATCH_HALF + 1) ** 2 * N_PATCH_CH  # 75


def extract_ambiguity_patches(
    A_complex: torch.Tensor,   # (B, L_kappa, L_ell) complex
    peak_idx: torch.Tensor,    # (B, K, 2) long, (kappa_idx, ell_idx), -1 for invalid
    half: int = PATCH_HALF,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract complex patches around each CFAR peak.

    Boundary handling: circular in ell (delay), reflect at kappa boundaries.

    Returns
    -------
    patches : (B, K, (2h+1)*(2h+1)*3) real tensor with Re, Im, log|A| stacked.
    valid   : (B, K) bool, False where peak_idx == -1.
    """
    B, K, _ = peak_idx.shape
    L_k, L_e = A_complex.shape[-2:]
    device = A_complex.device
    valid = peak_idx[:, :, 0] >= 0  # (B, K)

    # Build local index grids (2h+1, 2h+1).
    off = torch.arange(-half, half + 1, device=device)
    dk_grid, de_grid = torch.meshgrid(off, off, indexing="ij")  # (H, H)
    H = 2 * half + 1

    # Broadcast to (B, K, H, H).
    k_center = peak_idx[..., 0].clamp(min=0).unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
    e_center = peak_idx[..., 1].clamp(min=0).unsqueeze(-1).unsqueeze(-1)
    k_idx = (k_center + dk_grid).clamp(min=0, max=L_k - 1)   # reflect at boundary
    e_idx = (e_center + de_grid) % L_e                        # circular in delay
    # Gather.
    b_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(-1, K, H, H)
    A_patch = A_complex[b_idx, k_idx, e_idx]  # (B, K, H, H) complex

    re = A_patch.real
    im = A_patch.imag
    log_mag = torch.log(A_patch.abs().clamp(min=1e-12))

    feats = torch.stack([re, im, log_mag], dim=-1)  # (B, K, H, H, 3)
    feats_flat = feats.reshape(B, K, -1)             # (B, K, H*H*3)
    # Zero out invalid positions.
    feats_flat = feats_flat * valid.unsqueeze(-1).to(feats_flat.dtype)
    return feats_flat, valid


@dataclass
class PathSetEstimatorConfig:
    K: int = 24              # number of overcomplete CFAR candidates
    d_model: int = 128
    n_heads: int = 4
    n_blocks: int = 3
    mlp_mult: int = 4
    dropout: float = 0.0
    max_delta_ell: float = 1.5   # clip refinement offsets (grid spacing)
    max_delta_kap: float = 1.5
    max_h_norm: float = 3.0      # clip predicted gain magnitude


class PathSetEstimator(nn.Module):
    """Permutation-equivariant estimator on K CFAR candidates.

    Input per candidate:
      * 5 scalar features: ell_cfar, kappa_cfar, Re(h_ls), Im(h_ls), sigma_w_hat
      * 75 patch features:  5x5 x (Re, Im, log|A|) of the local ambiguity

    Total input dim = 80.
    """

    N_SCALAR_FEATS = 5

    def __init__(self, cfg: PathSetEstimatorConfig = PathSetEstimatorConfig()):
        super().__init__()
        self.cfg = cfg
        in_dim = self.N_SCALAR_FEATS + PATCH_FLAT_DIM
        self.input_proj = nn.Linear(in_dim, cfg.d_model)
        self.blocks = nn.ModuleList([
            SetTransformerBlock(cfg.d_model, n_heads=cfg.n_heads,
                                mlp_mult=cfg.mlp_mult, dropout=cfg.dropout)
            for _ in range(cfg.n_blocks)
        ])
        # Six heads.
        self.head_exist = nn.Linear(cfg.d_model, 1)
        self.head_dell = nn.Linear(cfg.d_model, 1)
        self.head_dkap = nn.Linear(cfg.d_model, 1)
        self.head_h = nn.Linear(cfg.d_model, 2)      # Re, Im
        self.head_logvar = nn.Linear(cfg.d_model, 1)
        # Zero-init offset heads: at initialization, delta_ell = delta_kap = 0 so
        # positions equal CFAR peaks (already sub-grid via Newton refinement).
        # Training then moves offsets only when it helps.
        nn.init.zeros_(self.head_dell.weight); nn.init.zeros_(self.head_dell.bias)
        nn.init.zeros_(self.head_dkap.weight); nn.init.zeros_(self.head_dkap.bias)

    def forward(
        self,
        scalar_feats: torch.Tensor,   # (B, K, 5)
        patch_feats: torch.Tensor,    # (B, K, 75)
        valid: torch.Tensor,          # (B, K) bool
    ) -> dict:
        """Forward the set-attention network.

        Returns dict with keys:
          existence_logit, delta_ell, delta_kappa, h (complex), log_var, valid.
        Shapes: (B, K) for each scalar output; h is complex (B, K).
        """
        x = torch.cat([scalar_feats, patch_feats], dim=-1)   # (B, K, 80)
        h = self.input_proj(x)                                # (B, K, D)
        key_padding_mask = ~valid                              # True at padding
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        exist = self.head_exist(h).squeeze(-1)                 # (B, K)
        d_ell = self.head_dell(h).squeeze(-1)
        d_kap = self.head_dkap(h).squeeze(-1)
        h_re_im = self.head_h(h)                               # (B, K, 2)
        logvar = self.head_logvar(h).squeeze(-1)
        # Clip refinement offsets to plausible range (bounded regression head).
        d_ell = self.cfg.max_delta_ell * torch.tanh(d_ell)
        d_kap = self.cfg.max_delta_kap * torch.tanh(d_kap)
        # Bounded gain: soft-clip by tanh then rescale.
        h_norm = h_re_im.norm(dim=-1, keepdim=True)
        h_scale = torch.tanh(h_norm / self.cfg.max_h_norm) * self.cfg.max_h_norm
        h_re_im = h_re_im * (h_scale / h_norm.clamp(min=1e-9))
        h_complex = torch.complex(h_re_im[..., 0], h_re_im[..., 1])
        # Zero out invalid positions in outputs.
        vf = valid.to(h_complex.dtype)
        h_complex = h_complex * vf
        rf = valid.to(d_ell.dtype)
        d_ell = d_ell * rf; d_kap = d_kap * rf; exist = exist * rf; logvar = logvar * rf
        return {
            "exist_logit": exist,
            "delta_ell": d_ell,
            "delta_kappa": d_kap,
            "h": h_complex,
            "log_var": logvar,
            "valid": valid,
        }
