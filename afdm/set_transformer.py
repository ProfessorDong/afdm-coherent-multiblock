"""Permutation-equivariant Set-Transformer for path-feature refinement.

This module implements the per-layer neural correction of the proposed receiver
(paper eq. 34-35). It takes an unordered set of path features and returns a
per-node complex correction to the gain estimates. Permutation equivariance is
enforced structurally: multi-head self-attention with no positional encoding is
exactly equivariant under simultaneous permutation of query/key/value tokens.

The uncertainty gate (paper eq. 36) is a scalar function of a cardinality-
normalized posterior-variance summary. Its slope is enforced positive via
softplus reparameterization so that (i) the gate is monotone in uncertainty and
(ii) the gate vanishes as noise -> 0 (required by Theorem 2 in the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetTransformerBlock(nn.Module):
    """One Transformer block: MHSA + residual + LN + FFN + residual + LN.

    Permutation-equivariant because MHSA has no positional encoding.
    """

    def __init__(self, d_model: int, n_heads: int = 4, mlp_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Linear(mlp_mult * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x                : (B, P, d_model) input tokens.
        key_padding_mask : (B, P) bool tensor; True at padding positions.
        """
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class SetTransformer(nn.Module):
    """Permutation-equivariant Set Transformer producing per-node complex corrections.

    Parameters
    ----------
    input_dim  : number of real-valued features per path (default 5:
                 Re(h_hat), Im(h_hat), tau_hat, nu_hat, sqrt(var_h)).
    d_model    : embedding dimension inside the network (default 64).
    n_heads    : number of attention heads (default 4).
    n_blocks   : number of Transformer blocks (default 3).
    output_dim : output real dim per path (default 2: Re/Im of correction).
    max_delta_norm : hard clip on the L2 norm of the output correction, per path
                     (default 5.0), to satisfy the bounded-denoiser assumption.
    """

    def __init__(
        self,
        input_dim: int = 5,
        d_model: int = 64,
        n_heads: int = 4,
        n_blocks: int = 3,
        output_dim: int = 2,
        max_delta_norm: float = 5.0,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [SetTransformerBlock(d_model, n_heads=n_heads) for _ in range(n_blocks)]
        )
        self.output_proj = nn.Linear(d_model, output_dim)
        self.max_delta_norm = float(max_delta_norm)

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce per-node complex corrections.

        Parameters
        ----------
        features : (B, P, input_dim) real-valued path features.
        mask     : (B, P) bool tensor; True at *valid* positions, False at padding.

        Returns
        -------
        delta    : (B, P) complex correction vector. Zeroed at invalid positions.
        """
        key_padding_mask = None if mask is None else (~mask)
        h = self.input_proj(features)
        for block in self.blocks:
            h = block(h, key_padding_mask=key_padding_mask)
        out = self.output_proj(h)  # (B, P, 2)
        # Norm clipping per node, so ||delta_i|| <= max_delta_norm (bounded denoiser).
        norms = out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = torch.minimum(norms, torch.full_like(norms, self.max_delta_norm)) / norms
        out = out * scale
        # Real (B, P, 2) -> complex (B, P)
        delta = torch.complex(out[..., 0], out[..., 1])
        # Zero out padding positions.
        if mask is not None:
            delta = delta * mask.to(delta.dtype)
        return delta


class UncertaintyGate(nn.Module):
    """Scalar gate g_t = sigmoid(a * log(u / u_ref) + b), with a > 0.

    The uncertainty summary u is cardinality-normalized:
        u = (1 / P_hat) * sum_i v_i
    where v_i is the diagonal of the posterior covariance (real, nonneg).

    Two properties by construction:
      * Monotone in u (a > 0 via softplus).
      * Vanishing at high SNR: since v_i = O(sigma_w^2) under bounded regressor
        conditioning, u = O(sigma_w^2), so log(u) -> -infty and g -> 0 as
        sigma_w^2 -> 0.
    """

    def __init__(self, u_ref: float = 1e-2, init_a: float = 1.0, init_b: float = 0.0) -> None:
        super().__init__()
        # Softplus reparameterization: a = softplus(tilde_a) = log(1 + exp(tilde_a)).
        # Init tilde_a so softplus(tilde_a) == init_a.
        # Solving: tilde_a = log(exp(a) - 1). For a=1: tilde_a ≈ 0.5413.
        tilde_a0 = torch.log(torch.expm1(torch.tensor(float(init_a))))
        self.tilde_a = nn.Parameter(tilde_a0)
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.u_ref = float(u_ref)

    def forward(self, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        v    : (B, P) real, nonneg posterior-variance diagonal.
        mask : (B, P) bool; True at valid positions.

        Returns
        -------
        g    : (B,) scalar gate in (0, 1).
        """
        if mask is not None:
            v_masked = v * mask.to(v.dtype)
            count = mask.sum(dim=-1).clamp(min=1).to(v.dtype)
        else:
            v_masked = v
            count = torch.full((v.shape[0],), v.shape[-1], dtype=v.dtype, device=v.device)
        u = v_masked.sum(dim=-1) / count  # (B,) cardinality-normalized mean
        u = u.clamp(min=1e-18)
        a = F.softplus(self.tilde_a)  # positive
        logits = a * torch.log(u / self.u_ref) + self.b
        return torch.sigmoid(logits)
