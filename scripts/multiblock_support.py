"""Multi-block support recovery via stacked ambiguity functions.

Hypothesis: with pilot diversity across B blocks, side-lobe patterns of the
ambiguity function are decorrelated. Summing |A_b|^2 across blocks suppresses
side lobes relative to peaks. Test whether this improves CFAR support recall.

Baseline: single-block CFAR (what R4 uses).
Test: sum-magnitude-squared across B blocks with complementary pilots.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.optimize import linear_sum_assignment

from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.support import ambiguity_function, cfar_peaks, newton_refine


TOL_ELL = 0.75; TOL_KAP = 0.75


def match_recall(ell_hat, kap_hat, ell_true, kap_true):
    """Fraction of true paths within tolerance of any detected candidate."""
    B = ell_hat.shape[0]
    n_tp = 0; n_true = 0
    for b in range(B):
        for p in range(ell_true.shape[1]):
            n_true += 1
            de = (ell_hat[b] - ell_true[b, p]).abs()
            dk = (kap_hat[b] - kap_true[b, p]).abs()
            if ((de <= TOL_ELL) & (dk <= TOL_KAP)).any():
                n_tp += 1
    return n_tp / max(n_true, 1)


def run_config(cfg, aggregate_Np, snr_db=15.0, n_batches=8, batch_size=16, K_cfar=6):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    N = cfg.N
    print(f"\n[{cfg.P=}, aggregate_Np={aggregate_Np}, K_cfar={K_cfar}, SNR={snr_db}]")
    print(f"  {'B_block':<8s}  {'N_p':<5s}  {'design':<15s}  {'recall':>7s}  {'d_rmse':>7s}  {'k_rmse':>7s}")

    for B_block in (1, 2, 4):
        N_p = max(aggregate_Np // B_block, 4)
        for design in ("repeated", "hopping", "complementary"):
            if B_block == 1 and design != "repeated":
                continue
            pp, pv = PILOT_DESIGNS[design](N=N, N_p=N_p, B=B_block,
                                           constellation=const, device=cfg.device, seed=42)
            gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
            recall_acc = 0.0; de_acc = 0.0; dk_acc = 0.0
            n_matched = 0
            for _ in range(n_batches):
                batch = sample_multiblock(system, channel, const, pp, pv,
                                          batch_size=batch_size, snr_db=snr_db,
                                          generator=gen)
                # Stacked ambiguity: sum of |A_b|^2 across blocks with COMPATIBLE grids.
                A_sum = None
                ell_grid = None; kap_grid = None
                for b in range(B_block):
                    x_p = torch.zeros(N, dtype=batch.r.dtype, device=batch.r.device)
                    x_p[pp[b]] = pv[b]
                    s_p = system.idaft(x_p.unsqueeze(0))[0]
                    A_b, e_g, k_g = ambiguity_function(
                        batch.r[:, b, :], s_p, N=N, N_cp=int(cfg.ell_max),
                        kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
                        oversample_doppler=2,
                    )
                    if A_sum is None:
                        A_sum = A_b; ell_grid = e_g; kap_grid = k_g
                    else:
                        A_sum = A_sum + A_b   # coherent power combining
                # CFAR on stacked map.
                peak_idx, _ = cfar_peaks(A_sum, K=K_cfar, min_separation=2)
                ell_hat, kap_hat = newton_refine(A_sum, peak_idx, ell_grid, kap_grid, max_iter=2)
                rec = match_recall(ell_hat, kap_hat,
                                   batch.theta_true[..., 0], batch.theta_true[..., 1])
                recall_acc += rec
                # Delay/kappa RMSE on matched paths
                for bi in range(ell_hat.shape[0]):
                    for pi in range(batch.theta_true.shape[1]):
                        de = (ell_hat[bi] - batch.theta_true[bi, pi, 0]).abs()
                        dk = (kap_hat[bi] - batch.theta_true[bi, pi, 1]).abs()
                        mask = (de <= TOL_ELL) & (dk <= TOL_KAP)
                        if mask.any():
                            i = mask.nonzero()[0].item()
                            de_acc += float((ell_hat[bi, i] - batch.theta_true[bi, pi, 0])**2)
                            dk_acc += float((kap_hat[bi, i] - batch.theta_true[bi, pi, 1])**2)
                            n_matched += 1
            recall = recall_acc / n_batches
            de_rmse = (de_acc / max(n_matched, 1)) ** 0.5
            dk_rmse = (dk_acc / max(n_matched, 1)) ** 0.5
            print(f"  {B_block:<8d}  {N_p:<5d}  {design:<15s}  {recall:>7.1%}  {de_rmse:>7.3f}  {dk_rmse:>7.3f}")


def main():
    print("=" * 78)
    print("MULTI-BLOCK SUPPORT RECOVERY (stacked ambiguity functions)")
    print("=" * 78)
    for cfg_name, cfg, agg in (
        ("EASY (P=3)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32), 32),
        ("HARD (P=5)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16), 16),
    ):
        print(f"\n{cfg_name}")
        run_config(cfg, aggregate_Np=agg, snr_db=15.0, K_cfar=max(cfg.P + 3, 6))


if __name__ == "__main__":
    main()
