"""Day-0 Oracle Ladder + top-K recall for the A+ redesign.

Goal: decide, BEFORE writing a network, which failure mode dominates.

Four controlled receivers on the same test channels:
  R1: true support + true gains + CG-MMSE  (should reproduce genie)
  R2: true support + LS gains + CG-MMSE     (gain-estimation penalty only)
  R3: CFAR support + oracle-matched gains + CG-MMSE  (support penalty only)
  R4: CFAR support + LS gains + CG-MMSE     (current classical operating point)

Plus: top-K ambiguity-peak recall at K=5,8,10,12 for each config.

Read from these numbers per the design memo:
  * If R2 SER < 5% at 15 dB: support is fine, gain estimation is the bottleneck.
    -> Amortized path-set inference at high recall will help.
  * If R3 SER < 5% at 15 dB but R4 >> R3: support recovery is the bottleneck.
    -> Focus on overcomplete proposals + Hungarian-matched estimator.
  * If R1 >> genie_target: implementation or channel mismatch. STOP, fix code.
  * If top-K recall poor even at K=12: local-max proposal set is inadequate,
    need dense-map encoder or learned queries.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine
from afdm.training import sample_batch


# ---------------------------------------------------------------------------
# Rung 1: genie (true theta + true h + CG-MMSE)
# ---------------------------------------------------------------------------
@torch.no_grad()
def rung1_genie(batch, system, constellation, K_cg=30):
    op = FastAFDMOperator(system=system,
                          ell=batch["theta_true"][..., 0],
                          kappa=batch["theta_true"][..., 1],
                          h=batch["h_true"])
    sigma_w2 = batch["sigma_w2_block"]
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
    hard = (x_soft.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
    return hard


# ---------------------------------------------------------------------------
# Rung 2: true theta + LS gains from pilot signal
# ---------------------------------------------------------------------------
@torch.no_grad()
def ls_gains_pilots_only(system, ell, kappa, r, pilot_positions, pilot_values,
                         lambda_ridge=1e-3):
    """Compute h_LS with a pilot-only transmit hypothesis.

    x_hat has pilot values at pilot positions and zero elsewhere. build_regression_matrix
    then constructs A using s = IDAFT(x_hat_pilot_only), so h_LS reflects the pilots'
    contribution to r. This is the standard first-iteration LS estimator.
    """
    B, N = r.shape
    dtype = r.dtype
    device = r.device
    x_pilot = torch.zeros(B, N, dtype=dtype, device=device)
    x_pilot[:, pilot_positions] = pilot_values.unsqueeze(0)
    A = build_regression_matrix(system, ell, kappa, x_pilot)   # (B, N, P)
    AH = A.conj().transpose(-1, -2)                            # (B, P, N)
    AhA = AH @ A                                               # (B, P, P)
    Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)                   # (B, P)
    P = ell.shape[1]
    ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
    h_ls = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)
    return h_ls


@torch.no_grad()
def rung2_trueth_lsh(batch, system, constellation, pilot_positions, pilot_values, K_cg=30):
    ell = batch["theta_true"][..., 0]
    kappa = batch["theta_true"][..., 1]
    h_ls = ls_gains_pilots_only(system, ell, kappa, batch["r"],
                                pilot_positions, pilot_values)
    op = FastAFDMOperator(system=system, ell=ell, kappa=kappa, h=h_ls)
    sigma_w2 = batch["sigma_w2_block"]
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
    hard = (x_soft.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
    nmse = ((h_ls - batch["h_true"]).abs()**2).sum() / (batch["h_true"].abs()**2).sum().clamp(min=1e-12)
    return hard, float(nmse.item())


# ---------------------------------------------------------------------------
# Rung 3: CFAR support + oracle-matched gains
# ---------------------------------------------------------------------------
@torch.no_grad()
def hungarian_match(ell_true, kap_true, ell_hat, kap_hat, kap_weight=1.0):
    """Match each true path to nearest CFAR peak (per-batch Hungarian).

    Returns
    -------
    match : (B, P_true) index in P_hat, or -1 for unmatched true paths.
    """
    B, P_true = ell_true.shape
    P_hat = ell_hat.shape[1]
    match = torch.full((B, P_true), -1, dtype=torch.long, device=ell_true.device)
    for b in range(B):
        # Cost matrix: (P_true, P_hat). Skip padded slots in ell_hat (typically none).
        cost = ((ell_true[b].unsqueeze(-1) - ell_hat[b].unsqueeze(0)) ** 2 +
                (kap_weight * (kap_true[b].unsqueeze(-1) - kap_hat[b].unsqueeze(0))) ** 2)
        cost_np = cost.cpu().numpy()
        row, col = linear_sum_assignment(cost_np)
        for r, c in zip(row, col):
            match[b, r] = int(c)
    return match


@torch.no_grad()
def rung3_cfar_oraclematched(batch, system, constellation, support_recovery,
                             pilot_positions, pilot_values, K_cg=30,
                             match_tol_ell=1.5, match_tol_kap=1.5):
    """CFAR support; gains = true h for matched paths, zero elsewhere."""
    B, N = batch["r"].shape
    device = batch["r"].device
    dtype = batch["r"].dtype
    x_pilot = torch.zeros(N, dtype=dtype, device=device)
    x_pilot[pilot_positions] = pilot_values
    s_pilot = system.idaft(x_pilot.unsqueeze(0))[0]
    ell_hat, kappa_hat, p_hat = support_recovery(batch["r"], s_pilot)
    P_hat = ell_hat.shape[1]

    ell_true = batch["theta_true"][..., 0]
    kap_true = batch["theta_true"][..., 1]
    match = hungarian_match(ell_true, kap_true, ell_hat, kappa_hat)

    # Build h_oracle: for each detected CFAR position i, h[b, i] = h_true[b, r] where
    # match[b, r] == i AND within tolerance. Otherwise zero.
    h_oracle = torch.zeros(B, P_hat, dtype=dtype, device=device)
    n_matched = 0; n_true = 0
    for b in range(B):
        for r in range(ell_true.shape[1]):
            n_true += 1
            i = int(match[b, r].item())
            if i < 0:
                continue
            de = float(abs(ell_true[b, r] - ell_hat[b, i]))
            dk = float(abs(kap_true[b, r] - kappa_hat[b, i]))
            if de <= match_tol_ell and dk <= match_tol_kap:
                h_oracle[b, i] = batch["h_true"][b, r]
                n_matched += 1

    op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kappa_hat, h=h_oracle)
    sigma_w2 = batch["sigma_w2_block"]
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
    hard = (x_soft.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
    return hard, {"matched": n_matched, "true_total": n_true,
                  "recall_within_tol": n_matched / max(n_true, 1)}


# ---------------------------------------------------------------------------
# Rung 4: CFAR support + LS gains
# ---------------------------------------------------------------------------
@torch.no_grad()
def rung4_cfar_lsh(batch, system, constellation, support_recovery,
                   pilot_positions, pilot_values, K_cg=30):
    B, N = batch["r"].shape
    device = batch["r"].device
    dtype = batch["r"].dtype
    x_pilot = torch.zeros(N, dtype=dtype, device=device)
    x_pilot[pilot_positions] = pilot_values
    s_pilot = system.idaft(x_pilot.unsqueeze(0))[0]
    ell_hat, kappa_hat, p_hat = support_recovery(batch["r"], s_pilot)
    h_ls = ls_gains_pilots_only(system, ell_hat, kappa_hat, batch["r"],
                                pilot_positions, pilot_values)
    op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kappa_hat, h=h_ls)
    sigma_w2 = batch["sigma_w2_block"]
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    x_soft = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=K_cg)
    hard = (x_soft.unsqueeze(-1) - constellation.reshape(1, 1, -1)).abs().argmin(dim=-1)
    return hard


# ---------------------------------------------------------------------------
# Top-K ambiguity peak recall
# ---------------------------------------------------------------------------
@torch.no_grad()
def topk_recall(batch, system, pilot_positions, pilot_values, Ks=(5, 8, 10, 12),
                match_tol_ell=1.5, match_tol_kap=1.5, min_separation=1):
    """For each K, compute recall = fraction of true paths within tolerance of a top-K local max.

    Uses cfar_peaks (which returns local maxima ordered by amplitude) with min_separation=1
    to allow near neighbors (we want to know if the TRUE peak is among the K strongest local
    maxima, not whether they're well-separated).
    """
    B, N = batch["r"].shape
    device = batch["r"].device
    dtype = batch["r"].dtype
    x_pilot = torch.zeros(N, dtype=dtype, device=device)
    x_pilot[pilot_positions] = pilot_values
    s_pilot = system.idaft(x_pilot.unsqueeze(0))[0]

    A, ell_grid, kappa_grid = ambiguity_function(
        batch["r"], s_pilot, system.N, system.ell_max,
        kappa_max=5.0, ell_max=float(system.ell_max),
        oversample_doppler=2,
    )
    ell_true = batch["theta_true"][..., 0]
    kap_true = batch["theta_true"][..., 1]

    results = {}
    for K in Ks:
        peak_idx, peak_vals = cfar_peaks(A, K=K, min_separation=min_separation)
        # Refine to get fractional (ell, kappa)
        ell_ref, kap_ref = newton_refine(A, peak_idx, ell_grid, kappa_grid, max_iter=2)
        n_matched = 0
        n_true = 0
        for b in range(B):
            for r in range(ell_true.shape[1]):
                n_true += 1
                dists = ((ell_true[b, r] - ell_ref[b]).abs() <= match_tol_ell) & \
                        ((kap_true[b, r] - kap_ref[b]).abs() <= match_tol_kap)
                if dists.any():
                    n_matched += 1
        results[K] = n_matched / max(n_true, 1)
    return results


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_config(name, cfg, snrs, n_batches, batch_size, seed=42):
    print()
    print("=" * 78)
    print(f"CONFIG: {name}  (N={cfg.N}, N_p={cfg.N_p}, P={cfg.P}, P_max={cfg.P_max})")
    print("=" * 78)
    system = cfg.system()
    channel = cfg.channel()
    constellation = cfg.constellation()
    pp, pv = cfg.pilots()
    support = cfg.support_recovery()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    ser = {snr: {r: 0.0 for r in ("R1", "R2", "R3", "R4")} for snr in snrs}
    nmse_R2 = {snr: 0.0 for snr in snrs}
    r3_recall = {snr: 0.0 for snr in snrs}
    r3_recall_n = {snr: 0 for snr in snrs}
    topk = {snr: {K: 0.0 for K in (5, 8, 10, 12)} for snr in snrs}

    for snr in snrs:
        for b_idx in range(n_batches):
            batch = sample_batch(system, channel, constellation, pp, pv,
                                 batch_size=batch_size, snr_db=snr, generator=gen)
            mask = batch["pilot_mask"].float()

            # R1: genie
            hard1 = rung1_genie(batch, system, constellation)
            ser[snr]["R1"] += float(((hard1 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())

            # R2: true theta + LS h
            hard2, nmse = rung2_trueth_lsh(batch, system, constellation, pp, pv)
            ser[snr]["R2"] += float(((hard2 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())
            nmse_R2[snr] += nmse

            # R3: CFAR theta + oracle matched gains
            hard3, info = rung3_cfar_oraclematched(batch, system, constellation, support, pp, pv)
            ser[snr]["R3"] += float(((hard3 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())
            r3_recall[snr] += info["matched"]; r3_recall_n[snr] += info["true_total"]

            # R4: CFAR theta + LS h
            hard4 = rung4_cfar_lsh(batch, system, constellation, support, pp, pv)
            ser[snr]["R4"] += float(((hard4 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())

            # Top-K recall
            tk = topk_recall(batch, system, pp, pv)
            for K, val in tk.items():
                topk[snr][K] += val

    # Normalize
    for snr in snrs:
        for r in ("R1", "R2", "R3", "R4"):
            ser[snr][r] /= n_batches
        nmse_R2[snr] /= n_batches
        for K in topk[snr]:
            topk[snr][K] /= n_batches

    # Print
    print()
    print(f"{'SNR':<6s}  {'R1 genie':>10s}  {'R2 LSh':>10s}  {'R3 orh':>10s}  {'R4 cls':>10s}  {'NMSE(R2)':>10s}  {'CFAR match':>12s}")
    for snr in snrs:
        cfar_recall = r3_recall[snr] / max(r3_recall_n[snr], 1)
        print(f"{snr:>4.1f}dB  {ser[snr]['R1']:>10.3e}  {ser[snr]['R2']:>10.3e}  "
              f"{ser[snr]['R3']:>10.3e}  {ser[snr]['R4']:>10.3e}  "
              f"{nmse_R2[snr]:>10.2e}  {cfar_recall:>11.1%}")

    print()
    print(f"Top-K ambiguity recall (fraction of true paths within a top-K local max, tol 1.5x1.5):")
    print(f"{'SNR':<6s}  {'K=5':>8s}  {'K=8':>8s}  {'K=10':>8s}  {'K=12':>8s}")
    for snr in snrs:
        print(f"{snr:>4.1f}dB  " + "  ".join(f"{topk[snr][K]:>7.1%}" for K in (5, 8, 10, 12)))

    return {"ser": ser, "nmse_R2": nmse_R2, "cfar_recall_R3": {
        snr: r3_recall[snr] / max(r3_recall_n[snr], 1) for snr in snrs
    }, "topk": topk}


def main():
    snrs = [5.0, 15.0, 25.0]
    n_batches = 4; batch_size = 32

    print("=" * 78)
    print("ORACLE LADDER + TOP-K RECALL (Day 0)")
    print(f"  batches per SNR: {n_batches}   batch size: {batch_size}")
    print("=" * 78)

    # Easy config
    cfg_easy = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32,
        T=8, K_cg=10, P_max=6, seed=0,
    )
    t0 = time.time()
    res_easy = run_config("EASY (P=3, N_p=32)", cfg_easy, snrs, n_batches, batch_size)
    print(f"\n[timing] easy config: {time.time()-t0:.1f}s")

    # Hard config
    cfg_hard = ExperimentConfig(
        N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16,
        T=8, K_cg=10, P_max=8, seed=0,
    )
    t0 = time.time()
    res_hard = run_config("HARD (P=5, N_p=16)", cfg_hard, snrs, n_batches, batch_size)
    print(f"\n[timing] hard config: {time.time()-t0:.1f}s")

    print()
    print("=" * 78)
    print("INTERPRETATION GUIDE")
    print("=" * 78)
    print("R1 vs literature: R1 should equal genie MMSE (~1e-3 at 15 dB, 0 at 25 dB).")
    print("R2 - R1        : gain-estimation penalty from using LS instead of true h.")
    print("R3 - R1        : support-recovery penalty (CFAR misses true paths).")
    print("R4 vs R3       : LS-on-CFAR-support penalty vs CFAR-with-oracle-h.")
    print("R4 - min(R2,R3): joint gain+support cost, the true classical bottleneck.")
    print()
    print("Top-K recall < 90% at K=12: local-max candidate set is insufficient.")

    # Persist the ladder so the R1-R4 values quoted in the paper are traceable.
    import json
    from pathlib import Path
    Path("runs/oracle_ladder.json").write_text(json.dumps(
        {"snrs": snrs, "n_batches": n_batches, "batch_size": batch_size,
         "EASY (P=3, N_p=32)": res_easy, "HARD (P=5, N_p=16)": res_hard},
        indent=1, default=str))
    print("Saved: runs/oracle_ladder.json")


if __name__ == "__main__":
    main()
