"""Multi-block AFDM primitives for pilot-diverse channel acquisition.

Setup: over a short window of B AFDM blocks,
    r_b = A_b(theta_b, x_b) h_b + w_b,       b = 1, ..., B

Assumptions used here (kept simple for the Day 3-5 oracle experiment):
  * path delays and Dopplers are SHARED across the B blocks (theta_b == theta),
    i.e. blocks are within one channel coherence window;
  * complex gains are SHARED as well (h_b == h) — extending to slowly-drifting
    gains is Day 6-10 work;
  * each block uses a DIFFERENT pilot pattern (positions and/or values) at
    fixed per-block overhead N_p.

The multi-block LS problem stacks the B regression matrices:
    r_stack = [r_1; ...; r_B] = [A_1; ...; A_B] h + w_stack

so the joint LS estimator is
    h_ls = (sum_b A_b^H A_b + lam I)^{-1} (sum_b A_b^H r_b).

Under complementary pilots, sum_b A_b^H A_b is better-conditioned and the
data self-interference biases in different blocks average out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .channels import UniformFractionalChannel
from .classical import build_regression_matrix, cg_solve
from .operators import FastAFDMOperator
from .system import AFDMSystem


# ---------------------------------------------------------------------------
# Pilot design strategies
# ---------------------------------------------------------------------------
def repeated_pilots(N: int, N_p: int, B: int, constellation: torch.Tensor,
                    device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """All B blocks use IDENTICAL pilots (positions + values). Baseline: no diversity."""
    step = N // N_p
    positions = torch.arange(0, N_p * step, step, device=device, dtype=torch.long)
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    idx = torch.randint(0, constellation.numel(), (N_p,), device=device, generator=gen)
    values = constellation[idx]
    # (B, N_p) — same for each block
    pos_B = positions.unsqueeze(0).expand(B, -1).contiguous()
    val_B = values.unsqueeze(0).expand(B, -1).contiguous()
    return pos_B, val_B


def random_hopping_pilots(N: int, N_p: int, B: int, constellation: torch.Tensor,
                          device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Each block picks N_p random positions from {0..N-1} (no replacement)."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    positions = torch.stack([
        torch.randperm(N, device=device, generator=gen)[:N_p].sort().values
        for _ in range(B)
    ])
    idx = torch.randint(0, constellation.numel(), (B, N_p), device=device, generator=gen)
    values = constellation[idx]
    return positions, values


def complementary_pilots(N: int, N_p: int, B: int, constellation: torch.Tensor,
                         device: str, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Complementary offsets: each block shifts pilot positions by k * (N // (N_p * B))
    modulo N. Values still random per block.
    """
    step = N // N_p                                # inter-pilot spacing per block
    micro_step = max(step // B, 1)                 # offset between blocks
    positions = torch.stack([
        (torch.arange(0, N_p * step, step, device=device, dtype=torch.long) + b * micro_step) % N
        for b in range(B)
    ])
    positions, _ = positions.sort(dim=-1)          # canonicalize
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    idx = torch.randint(0, constellation.numel(), (B, N_p), device=device, generator=gen)
    values = constellation[idx]
    return positions, values


PILOT_DESIGNS = {
    "repeated": repeated_pilots,
    "hopping": random_hopping_pilots,
    "complementary": complementary_pilots,
}


# ---------------------------------------------------------------------------
# Multi-block batch sampler
# ---------------------------------------------------------------------------
@dataclass
class MultiBlockBatch:
    r: torch.Tensor           # (B_batch, B_block, N)  time-domain received per block
    y: torch.Tensor           # (B_batch, B_block, N)  DAFT-domain observation
    x_true: torch.Tensor      # (B_batch, B_block, N)  DAFT-domain transmitted (pilots+data)
    labels: torch.Tensor      # (B_batch, B_block, N)  data-symbol constellation indices
    h_true: torch.Tensor      # (B_batch, P)          shared complex gains
    theta_true: torch.Tensor  # (B_batch, P, 2)       shared (ell, kappa)
    pilot_positions: torch.Tensor  # (B_block, N_p)   per-block pilot positions
    pilot_values: torch.Tensor      # (B_block, N_p)  per-block pilot values
    pilot_mask: torch.Tensor       # (B_batch, B_block, N)  True at DATA positions
    sigma_w2_block: float
    snr_db: float


def sample_multiblock(
    system: AFDMSystem,
    channel: UniformFractionalChannel,
    constellation: torch.Tensor,
    pilot_positions: torch.Tensor,   # (B_block, N_p)
    pilot_values: torch.Tensor,      # (B_block, N_p)
    batch_size: int,
    snr_db: float,
    generator: torch.Generator | None = None,
) -> MultiBlockBatch:
    """Sample a batch of B_block-block AFDM frames with SHARED (theta, h) per frame.

    Different frames (batch dim) have different channels; blocks within one
    frame share the physical channel but use different pilot patterns.
    """
    device = system.device; N = system.N; B_block = pilot_positions.shape[0]
    S = constellation.numel()

    d = channel.sample(batch_size, generator=generator)     # shared theta, h
    h_true = d["h"]; ell = d["ell"]; kap = d["kappa"]

    # For each block, sample independent random data (indep symbols per block).
    if generator is None:
        idx = torch.randint(0, S, (batch_size, B_block, N), device=device)
    else:
        idx = torch.randint(0, S, (batch_size, B_block, N), device=device, generator=generator)
    x = constellation[idx]  # (B_batch, B_block, N)

    # Overwrite pilot positions with per-block pilot values.
    for b in range(B_block):
        x[:, b, pilot_positions[b]] = pilot_values[b].unsqueeze(0)
    labels = (x.unsqueeze(-1) - constellation.reshape(1, 1, 1, -1)).abs().argmin(dim=-1)

    # Apply channel per block. Use the SAME (theta, h) for all blocks per batch elem.
    op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h_true)
    y_clean_list = []
    for b in range(B_block):
        y_b = op.matvec(x[:, b, :])   # (B_batch, N)
        y_clean_list.append(y_b)
    y_clean = torch.stack(y_clean_list, dim=1)                          # (B_batch, B_block, N)
    signal_pow = (y_clean.abs() ** 2).mean()
    sigma_w2 = 10 ** (-snr_db / 10)
    noise_std = torch.sqrt(signal_pow * sigma_w2 / 2)
    if generator is None:
        w = torch.randn_like(y_clean) * noise_std
    else:
        w = torch.randn(y_clean.shape, dtype=y_clean.dtype, device=device, generator=generator) * noise_std
    y = y_clean + w
    r = system.idaft(y.reshape(-1, N)).reshape(batch_size, B_block, N)

    # Pilot mask: True at DATA positions.
    pilot_mask = torch.ones(batch_size, B_block, N, dtype=torch.bool, device=device)
    for b in range(B_block):
        pilot_mask[:, b, pilot_positions[b]] = False

    abs_noise = (signal_pow * sigma_w2).item()
    theta_true = torch.stack([ell, kap], dim=-1)
    return MultiBlockBatch(
        r=r, y=y, x_true=x, labels=labels, h_true=h_true, theta_true=theta_true,
        pilot_positions=pilot_positions, pilot_values=pilot_values,
        pilot_mask=pilot_mask, sigma_w2_block=abs_noise, snr_db=snr_db,
    )


# ---------------------------------------------------------------------------
# Multi-block LS estimator (oracle theta, pilot-only stacked LS)
# ---------------------------------------------------------------------------
def multiblock_ls_gains(
    system: AFDMSystem,
    batch: MultiBlockBatch,
    ell: torch.Tensor,        # (B_batch, P)  either true or estimated
    kap: torch.Tensor,        # (B_batch, P)
    use_pilot_only: bool = True,
    lambda_ridge: float = 1e-3,
) -> torch.Tensor:
    """Stack the per-block LS regressions and solve jointly.

    If use_pilot_only=True, non-pilot positions in x_hat are zero (data self-
    interference bias will apply). If False, uses x_true (oracle test, no bias).
    """
    B_batch, B_block, N = batch.r.shape
    device = batch.r.device; dtype = batch.r.dtype
    P = ell.shape[1]
    AhA_sum = torch.zeros(B_batch, P, P, dtype=dtype, device=device)
    Ahr_sum = torch.zeros(B_batch, P, dtype=dtype, device=device)
    for b in range(B_block):
        if use_pilot_only:
            x_hat = torch.zeros(B_batch, N, dtype=dtype, device=device)
            x_hat[:, batch.pilot_positions[b]] = batch.pilot_values[b].unsqueeze(0)
        else:
            x_hat = batch.x_true[:, b, :]
        A = build_regression_matrix(system, ell, kap, x_hat)              # (B_batch, N, P)
        AH = A.conj().transpose(-1, -2)
        AhA_sum += AH @ A
        Ahr_sum += (AH @ batch.r[:, b, :].unsqueeze(-1)).squeeze(-1)
    ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
    h_ls = torch.linalg.solve(AhA_sum + ridge, Ahr_sum.unsqueeze(-1)).squeeze(-1)
    return h_ls


# ---------------------------------------------------------------------------
# Dictionary coherence and Fisher-info minimum eigenvalue diagnostics
# ---------------------------------------------------------------------------
def stacked_atom(
    system: AFDMSystem,
    ell: float, kap: float,
    pilot_positions: torch.Tensor,   # (B_block, N_p)
    pilot_values: torch.Tensor,      # (B_block, N_p)
) -> torch.Tensor:
    """Compute the stacked pilot-only atom a(theta) = [a_1; a_2; ...; a_B] for a
    single-path hypothesis. Shape (B_block * N,), complex.
    """
    device = system.device; dtype = system.dtype; N = system.N
    B_block = pilot_positions.shape[0]
    atoms = []
    for b in range(B_block):
        x_p = torch.zeros(1, N, dtype=dtype, device=device)
        x_p[0, pilot_positions[b]] = pilot_values[b]
        ell_t = torch.tensor([[ell]], dtype=torch.float32, device=device)
        kap_t = torch.tensor([[kap]], dtype=torch.float32, device=device)
        A = build_regression_matrix(system, ell_t, kap_t, x_p)   # (1, N, 1)
        atoms.append(A[0, :, 0])
    return torch.cat(atoms, dim=0)


def dictionary_coherence(
    system: AFDMSystem,
    pilot_positions: torch.Tensor,
    pilot_values: torch.Tensor,
    ell_grid: torch.Tensor,           # (M,) fractional delay candidates
    kap_grid: torch.Tensor,           # (M,) fractional Doppler candidates
) -> float:
    """Compute mu = max_{i != j} |<a_i, a_j>| / (||a_i|| ||a_j||) over the given
    (ell_grid, kap_grid) pairs. Higher mu -> harder to distinguish.
    """
    M = ell_grid.shape[0]
    atoms = torch.stack([
        stacked_atom(system, float(ell_grid[i]), float(kap_grid[i]),
                     pilot_positions, pilot_values)
        for i in range(M)
    ])                                                        # (M, B*N)
    norms = atoms.norm(dim=-1)
    gram = (atoms @ atoms.conj().transpose(-1, -2)).abs()
    gram = gram / (norms.unsqueeze(0) * norms.unsqueeze(1) + 1e-12)
    gram.fill_diagonal_(0)
    return float(gram.max())
