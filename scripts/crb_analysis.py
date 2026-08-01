"""Numerical CRB analysis for shared-theta multi-block AFDM channel estimation.

Fisher information for estimating (ell, kappa) from
    r_b = A_b(ell, kappa, x_b) h + w_b,   b = 1, ..., B, w_b ~ CN(0, sigma^2 I)

Assuming (x_b, h) known (oracle CRB, best case), the FIM for the per-path
2-vector (delta_ell_i, delta_kappa_i) is
    J = (2/sigma^2) * Re{ (dmu/dtheta)^H (dmu/dtheta) }
where mu_b = A_b(theta, x_b) h.

For shared-theta multi-block:
    J_multi = sum_b J_b

CRB(theta) = J_multi^{-1}, and RMSE(theta_hat) >= sqrt(diag(CRB)).

We compare our receiver's empirical theta RMSE against this bound.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.optimize import linear_sum_assignment

from afdm.classical import build_regression_matrix
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.operators import FastAFDMOperator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import multiblock_dasbl_receiver


def compute_fim_per_path(system, batch, ell_true, kap_true, h_true, x_ref):
    """Compute (2, 2) Fisher information matrix for each path, using autograd.

    Returns (B_batch, P, 2, 2) real tensor.
    """
    B_batch, B_block, N = batch.r.shape
    device = ell_true.device
    P = ell_true.shape[1]
    sigma_w2 = batch.sigma_w2_block

    # We need d(A(theta, x) h)/d(theta_i) for each path i.
    # Simpler: parameterize each path's theta and compute Jacobian.
    ell = ell_true.detach().clone().requires_grad_(True)
    kap = kap_true.detach().clone().requires_grad_(True)

    fim = torch.zeros(B_batch, P, 2, 2, device=device)

    with torch.enable_grad():
        for b in range(B_block):
            # mu_b = A_b(theta, x_b) h
            A = build_regression_matrix(system, ell, kap, x_ref[:, b, :])   # (B_b, N, P)
            mu = (A @ h_true.unsqueeze(-1)).squeeze(-1)                     # (B_b, N)

            # For each path i, compute d(mu)/d(ell_i) and d(mu)/d(kap_i).
            for i in range(P):
                # d mu / d ell_i and d mu / d kap_i are (B_b, N) complex.
                # Use autograd on scalar surrogate.
                # We compute Re{(d mu/d theta_i)^H (d mu/d theta_j)} for i==j only (diagonal block).
                dmu_dell_real = torch.autograd.grad(
                    mu.real.sum(), ell, retain_graph=True, create_graph=False,
                    allow_unused=True,
                )[0]
                dmu_dell_imag = torch.autograd.grad(
                    mu.imag.sum(), ell, retain_graph=True, create_graph=False,
                    allow_unused=True,
                )[0]
                # This gives d(mu[n,:].sum()) / d(ell_i for each b). We want per-batch,
                # per-i entries. Since ell has shape (B_b, P), grad gives (B_b, P) — the
                # gradient of the sum over batch and n.
                # For per-path FIM we need d(mu[b,n]) / d(ell[b,i]) for each (b, i, n).
                # This is diagonal in the batch, so we can extract from grads directly.
                break
            break
    # Above is too complex. Use a simpler numerical Jacobian instead.
    return None


def numerical_fim(system, batch, ell_true, kap_true, h_true, x_ref, eps=1e-3):
    """Numerical Fisher information via finite differences.

    Per-path 2x2 FIM: J_i = (2/sigma^2) sum_b Re{(d mu_b/d theta_i)^H (d mu_b/d theta_i)}.
    """
    B_batch, B_block, N = batch.r.shape
    device = ell_true.device
    P = ell_true.shape[1]
    sigma_w2 = batch.sigma_w2_block

    def mu_all(ell_c, kap_c):
        """Compute mu_b = A_b(theta, x_b) h for all b. Returns (B_batch, B_block, N)."""
        mu = torch.zeros(B_batch, B_block, N, dtype=torch.complex64, device=device)
        for b in range(B_block):
            A = build_regression_matrix(system, ell_c, kap_c, x_ref[:, b, :])
            mu[:, b, :] = (A @ h_true.unsqueeze(-1)).squeeze(-1)
        return mu

    # Baseline
    mu_0 = mu_all(ell_true, kap_true)

    fim = torch.zeros(B_batch, P, 2, 2, device=device)
    for i in range(P):
        # d mu / d ell_i
        ell_p = ell_true.clone(); ell_p[:, i] += eps
        d_ell = (mu_all(ell_p, kap_true) - mu_0) / eps

        # d mu / d kap_i
        kap_p = kap_true.clone(); kap_p[:, i] += eps
        d_kap = (mu_all(ell_true, kap_p) - mu_0) / eps

        # J_ll = (2/sigma^2) sum_b sum_n Re{|d_ell|^2}
        J_ll = (2.0 / sigma_w2) * (d_ell.abs() ** 2).sum(dim=(1, 2))
        J_kk = (2.0 / sigma_w2) * (d_kap.abs() ** 2).sum(dim=(1, 2))
        J_lk = (2.0 / sigma_w2) * (d_ell.conj() * d_kap).sum(dim=(1, 2)).real
        fim[:, i, 0, 0] = J_ll
        fim[:, i, 1, 1] = J_kk
        fim[:, i, 0, 1] = J_lk
        fim[:, i, 1, 0] = J_lk
    return fim


def crb_from_fim(fim):
    """Invert FIM to get CRB. Returns (B, P, 2) per-path (delta_ell, delta_kap) variances."""
    B, P, _, _ = fim.shape
    crb = torch.zeros(B, P, 2, device=fim.device)
    for b in range(B):
        for p in range(P):
            try:
                inv = torch.linalg.inv(fim[b, p])
                crb[b, p, 0] = inv[0, 0]
                crb[b, p, 1] = inv[1, 1]
            except Exception:
                crb[b, p] = float("nan")
    return crb


def hungarian_match(ell_hat, kap_hat, ell_true, kap_true):
    """Per-batch Hungarian match. Returns (B, P_true) indices into ell_hat."""
    B, P_true = ell_true.shape
    idx = torch.full((B, P_true), -1, dtype=torch.long, device=ell_true.device)
    for b in range(B):
        cost = ((ell_hat[b].unsqueeze(-1) - ell_true[b].unsqueeze(0)) ** 2 +
                (kap_hat[b].unsqueeze(-1) - kap_true[b].unsqueeze(0)) ** 2)
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        for r, c in zip(row, col):
            idx[b, c] = int(r)
    return idx


def measure_theta_mse(cfg, snr_db, B_block, design="hopping",
                     n_batches=8, batch_size=8, seed=42):
    """Return empirical theta MSE per path via Hungarian matching."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS[design](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                   constellation=const, device=cfg.device, seed=seed)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    de_sq = 0.0; dk_sq = 0.0; n = 0
    crb_ell_avg = 0.0; crb_kap_avg = 0.0; n_crb = 0

    for _ in range(n_batches):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=batch_size, snr_db=snr_db, generator=gen)
        # Run receiver
        with torch.no_grad():
            hard, ell_hat, kap_hat, h_hat = multiblock_dasbl_receiver(
                system, batch, const, cfg,
                n_outer=6, n_lm_per_outer=3, rho_min=0.9, use_reacq=True,
            )
        # Match to true.
        match = hungarian_match(ell_hat, kap_hat, batch.theta_true[..., 0],
                                batch.theta_true[..., 1])
        for b in range(match.shape[0]):
            for pi in range(match.shape[1]):
                idx = int(match[b, pi])
                if idx >= 0:
                    de_sq += float((ell_hat[b, idx] - batch.theta_true[b, pi, 0]) ** 2)
                    dk_sq += float((kap_hat[b, idx] - batch.theta_true[b, pi, 1]) ** 2)
                    n += 1

        # CRB (oracle x_true, oracle h)
        fim = numerical_fim(system, batch, batch.theta_true[..., 0], batch.theta_true[..., 1],
                            batch.h_true, batch.x_true)
        crb = crb_from_fim(fim)
        crb_ell_avg += float(crb[..., 0].mean())
        crb_kap_avg += float(crb[..., 1].mean())
        n_crb += 1

    return {
        "empirical_ell_mse": de_sq / max(n, 1),
        "empirical_kap_mse": dk_sq / max(n, 1),
        "crb_ell": crb_ell_avg / max(n_crb, 1),
        "crb_kap": crb_kap_avg / max(n_crb, 1),
        "n_matched": n,
    }


def main():
    print("=" * 90)
    print("NUMERICAL CRB vs RECEIVER MSE (shared-theta multi-block)")
    print("=" * 90)
    for cfg_name, cfg in (
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print(f"\n{cfg_name}")
        print(f"{'B':<4s}  {'SNR':<7s}  {'emp_ell_RMSE':>13s}  {'crb_ell_RMSE':>13s}  {'ratio':>7s}  {'emp_kap_RMSE':>13s}  {'crb_kap_RMSE':>13s}  {'ratio':>7s}")
        for B in (1, 2, 4, 8):
            for snr in (5.0, 15.0, 25.0):
                m = measure_theta_mse(cfg, snr, B_block=B, n_batches=4, batch_size=8)
                ell_rmse = m["empirical_ell_mse"] ** 0.5
                kap_rmse = m["empirical_kap_mse"] ** 0.5
                crb_ell_rmse = m["crb_ell"] ** 0.5
                crb_kap_rmse = m["crb_kap"] ** 0.5
                ratio_ell = ell_rmse / max(crb_ell_rmse, 1e-9)
                ratio_kap = kap_rmse / max(crb_kap_rmse, 1e-9)
                print(f"{B:<4d}  {snr:>5.1f}dB  {ell_rmse:>13.3e}  {crb_ell_rmse:>13.3e}  {ratio_ell:>7.1f}x  "
                      f"{kap_rmse:>13.3e}  {crb_kap_rmse:>13.3e}  {ratio_kap:>7.1f}x")


if __name__ == "__main__":
    main()
