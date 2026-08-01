"""Empirical correlation between pilot-data cross-coherence and bias floor.

Verifies Proposition 1 by sampling K random pilot patterns at (P=3, N_p=32),
computing for each:
  (a) mu_bar = max_{i,j} mu_{P,D}(theta_i, theta_j) at the true channel supports;
  (b) empirical high-SNR (50 dB) pilot-only ridge-LS NMSE floor.

Reports Spearman rank correlation between mu_bar^2 and NMSE floor across the K
patterns. Saves per-pattern (mu_bar^2, NMSE floor) pairs to JSON for figure.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.stats import spearmanr

from afdm.classical import build_regression_matrix
from afdm.experiments import ExperimentConfig
from afdm.training import sample_batch


K_PATTERNS = 50
N_CHANNELS_PER_PATTERN = 32
SNR = 50.0        # high enough to isolate the data-interference / ridge floor
LAMBDA_RIDGE = 1e-3


def sample_pilot_positions(N: int, N_p: int, device: str, seed: int):
    """Sample random N_p pilot positions from {0,...,N-1} without replacement."""
    gen = torch.Generator(device=device); gen.manual_seed(seed)
    perm = torch.randperm(N, device=device, generator=gen)
    return perm[:N_p].sort().values


def build_pilot_data_vecs(N, positions, values, constellation, device, gen):
    """Return x_P, x_D in C^N (data at complementary positions is random constellation)."""
    x_P = torch.zeros(N, dtype=constellation.dtype, device=device)
    x_P[positions] = values
    data_mask = torch.ones(N, dtype=torch.bool, device=device)
    data_mask[positions] = False
    n_D = int(data_mask.sum())
    idx = torch.randint(0, constellation.numel(), (n_D,), device=device, generator=gen)
    x_D = torch.zeros(N, dtype=constellation.dtype, device=device)
    x_D[data_mask] = constellation[idx]
    return x_P, x_D


def compute_mu_bar(system, x_P, x_D, thetas_i, thetas_j):
    """Compute max_{i,j} |<Phi_i x_P, Phi_j x_D>| / (||Phi_i x_P|| ||Phi_j x_D||).

    thetas_i, thetas_j: (P,) tensors of (ell, kappa) pairs — passed as separate
    tensors of shape (1, P).
    """
    # Build atom vectors at each hypothesis
    ell_i, kap_i = thetas_i
    ell_j, kap_j = thetas_j
    P = ell_i.shape[1]
    # A_P columns are Phi(theta_i) x_P
    A_P = build_regression_matrix(system, ell_i, kap_i, x_P.unsqueeze(0))[0]  # (N, P)
    A_D = build_regression_matrix(system, ell_j, kap_j, x_D.unsqueeze(0))[0]  # (N, P)
    # Cross-Gram
    G = A_P.conj().T @ A_D                                                     # (P, P)
    # Normalize by atom norms
    norms_P = torch.linalg.norm(A_P, dim=0)  # (P,)
    norms_D = torch.linalg.norm(A_D, dim=0)  # (P,)
    denom = norms_P.unsqueeze(1) * norms_D.unsqueeze(0)                        # (P, P)
    mu = G.abs() / (denom + 1e-12)
    return float(mu.max())


def measure_floor_for_pattern(cfg, positions, values, snr_db=SNR, n_channels=N_CHANNELS_PER_PATTERN,
                              seed=0):
    """Return empirical NMSE floor of pilot-only ridge LS averaged over channels."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    nmse_acc = 0.0
    mu_bar_acc = 0.0
    for _ in range(n_channels):
        # Sample channel
        d = channel.sample(1, generator=gen)
        h = d["h"][0]; ell = d["ell"]; kap = d["kappa"]

        # Sample data + pilot values
        x_P, x_D = build_pilot_data_vecs(cfg.N, positions, values, const, cfg.device, gen)

        # Compute mu_bar for this channel (using true theta as both i and j)
        mu = compute_mu_bar(system, x_P, x_D, (ell, kap), (ell, kap))
        mu_bar_acc += mu

        # Build received signal: y = A(theta, x_true) h + w
        A_true = build_regression_matrix(system, ell, kap, (x_P + x_D).unsqueeze(0))[0]
        y_clean = A_true @ h
        signal_pow = (y_clean.abs() ** 2).mean()
        sigma_w2 = signal_pow * 10 ** (-snr_db / 10)
        noise_std = torch.sqrt(sigma_w2 / 2)
        w = torch.randn(cfg.N, dtype=y_clean.dtype, device=cfg.device, generator=gen) * noise_std
        y = y_clean + w

        # Pilot-only ridge LS at TRUE theta
        A_P = build_regression_matrix(system, ell, kap, x_P.unsqueeze(0))[0]      # (N, P)
        M = A_P.conj().T @ A_P + LAMBDA_RIDGE * torch.eye(A_P.shape[1], dtype=A_P.dtype, device=cfg.device)
        h_hat = torch.linalg.solve(M, A_P.conj().T @ y)

        nmse = ((h_hat - h).abs() ** 2).sum() / (h.abs() ** 2).sum()
        nmse_acc += float(nmse)

    return nmse_acc / n_channels, mu_bar_acc / n_channels


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)
    system = cfg.system(); const = cfg.constellation()

    print(f"\n{'='*70}\nCOHERENCE vs BIAS-FLOOR CORRELATION\n"
          f"P=3, N_p=32, SNR={SNR} dB (isolated data-interference floor)\n"
          f"K={K_PATTERNS} random pilot patterns, {N_CHANNELS_PER_PATTERN} channels each\n"
          f"{'='*70}")
    print(f"{'k':<4s} {'mu_bar':>10s} {'mu_bar^2':>12s} {'NMSE_floor':>13s}")

    per_pattern = []
    t0 = time.time()
    for k in range(K_PATTERNS):
        positions = sample_pilot_positions(cfg.N, cfg.N_p, cfg.device, seed=k)
        # Fixed pilot values across patterns (isolates position effect)
        vgen = torch.Generator(device=cfg.device); vgen.manual_seed(1234)
        idx = torch.randint(0, const.numel(), (cfg.N_p,), device=cfg.device, generator=vgen)
        values = const[idx]

        nmse, mu = measure_floor_for_pattern(cfg, positions, values, seed=k * 71 + 3)
        per_pattern.append({"k": k, "mu_bar": mu, "mu_bar_sq": mu * mu, "nmse_floor": nmse})
        print(f"{k:<4d} {mu:>10.4f} {mu*mu:>12.4e} {nmse:>13.4e}")

    dt = time.time() - t0
    mus = np.array([p["mu_bar_sq"] for p in per_pattern])
    nmses = np.array([p["nmse_floor"] for p in per_pattern])
    rho, pval = spearmanr(mus, nmses)
    pearson_r = float(np.corrcoef(mus, nmses)[0, 1])

    print(f"\n{'-'*70}")
    print(f"Total wall time: {dt:.0f}s")
    print(f"Spearman correlation between mu_bar^2 and NMSE floor: rho = {rho:.3f}, p = {pval:.3e}")
    print(f"Pearson  correlation between mu_bar^2 and NMSE floor: r   = {pearson_r:.3f}")

    out_path = Path("runs/coherence_floor.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": "P=3, N_p=32, SNR=50dB, K=50",
            "K_patterns": K_PATTERNS,
            "N_channels": N_CHANNELS_PER_PATTERN,
            "lambda_ridge": LAMBDA_RIDGE,
            "spearman_rho": float(rho),
            "spearman_pval": float(pval),
            "pearson_r": pearson_r,
            "per_pattern": per_pattern,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
