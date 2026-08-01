"""Deterministic pilot patterns for AFDM."""

from __future__ import annotations

import torch


def uniform_daft_pilots(N: int, N_p: int, device: str | torch.device = "cuda:0") -> torch.Tensor:
    """Return the set of DAFT-domain pilot positions, uniformly spaced.

    Positions are the rounded indices of an arithmetic progression of length N_p
    starting near 0. For example, N=128, N_p=16 -> positions {0, 8, 16, ..., 120}.
    """
    assert N_p <= N
    step = N // N_p
    return torch.arange(0, N_p * step, step, device=device, dtype=torch.long)


def make_pilot_symbols(
    N: int,
    pilot_positions: torch.Tensor,
    constellation: torch.Tensor,
    seed: int = 0,
    device: str | torch.device = "cuda:0",
    dtype: torch.dtype = torch.complex64,
) -> torch.Tensor:
    """Return a deterministic pilot-symbol pattern of shape (N,), with non-pilot positions
    set to zero.

    Pilots are drawn deterministically from the given constellation using the seed.
    """
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    idx = torch.randint(
        0, constellation.numel(), (pilot_positions.shape[0],), device=device, generator=generator
    )
    values = constellation[idx].to(dtype)
    x = torch.zeros(N, device=device, dtype=dtype)
    x[pilot_positions] = values
    return x
