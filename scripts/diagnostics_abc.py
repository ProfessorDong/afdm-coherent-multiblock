"""Day-0.5 Diagnostics A / B / C.

Before committing to a network architecture (Option 5: dense-map encoder vs
overcomplete-candidate estimator), answer three empirical questions:

  A. Does raising N_p in the hard case (P=5) push R2 (true positions + LS h)
     below the 5% target?
     -> if YES, N_p=24/32 is a viable operating point; both easy and hard
        become solvable and 5% is the right stop criterion for both.
     -> if NO, pilot count alone can't fix it; must include data-aided SBL/EM.

  B. Does top-K recall at K in {12, 16, 24, 32} exceed 95% on the hard case?
     -> if YES, a lighter overcomplete candidate Set-Transformer is enough.
     -> if NO, we need the dense complex ambiguity-map learned-query encoder.

  C. Does iterated safeguarded Levenberg-Marquardt on an overcomplete
     candidate set close a meaningful fraction of the R4 -> R2 gap without
     any learning?
     -> establishes the deterministic-only baseline the network must beat,
        and quantifies how much of the position problem is solvable by
        classical refinement alone.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from scipy.optimize import linear_sum_assignment

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine, SupportRecovery
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step

from oracle_ladder import (
    rung1_genie, ls_gains_pilots_only, rung2_trueth_lsh, hungarian_match,
    topk_recall,
)


# ---------------------------------------------------------------------------
# Diagnostic A: R2 SER vs N_p on hard config
# ---------------------------------------------------------------------------
def diagnostic_A(snrs=(5.0, 15.0, 25.0), n_batches=4, batch_size=32):
    print()
    print("=" * 78)
    print("DIAGNOSTIC A: R2 SER (true positions + LS h) at various N_p")
    print("  P=5 hard config, sweeping N_p in {16, 24, 32}")
    print("=" * 78)
    print()
    print(f"{'N_p':<6s}  {'SNR':>6s}  {'R2 SER':>10s}  {'R1 (genie)':>12s}  {'gap (R2-R1)':>12s}")

    all_results = {}
    for N_p in (16, 24, 32):
        cfg = ExperimentConfig(
            N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=N_p,
            T=8, K_cg=10, P_max=8, seed=0,
        )
        system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
        pp, pv = cfg.pilots()
        gen = torch.Generator(device=cfg.device); gen.manual_seed(42)

        for snr in snrs:
            r1 = 0.0; r2 = 0.0
            for _ in range(n_batches):
                batch = sample_batch(system, channel, const, pp, pv,
                                     batch_size=batch_size, snr_db=snr, generator=gen)
                mask = batch["pilot_mask"].float()
                hard1 = rung1_genie(batch, system, const)
                r1 += float(((hard1 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())
                hard2, _ = rung2_trueth_lsh(batch, system, const, pp, pv)
                r2 += float(((hard2 != batch["labels"]) * mask.bool()).float().sum() / mask.sum())
            r1 /= n_batches; r2 /= n_batches
            all_results[(N_p, snr)] = (r1, r2)
            print(f"{N_p:<6d}  {snr:>4.1f}dB  {r2:>10.3e}  {r1:>12.3e}  {r2 - r1:>+12.3e}")
        print()

    print("Interpretation:")
    print("  If R2 at 15 dB drops below 5% for some N_p, that is a viable hard-case operating point.")
    print("  If R2 stays above 5% at all N_p, pilot-only LS gain estimation is inherent")
    print("  bottleneck; data-aided SBL/EM refinement is essential regardless of N_p.")
    return all_results


# ---------------------------------------------------------------------------
# Diagnostic B: top-K recall for K in {12, 16, 24, 32}
# ---------------------------------------------------------------------------
def diagnostic_B(snrs=(5.0, 15.0, 25.0), n_batches=4, batch_size=32):
    print()
    print("=" * 78)
    print("DIAGNOSTIC B: top-K ambiguity-peak recall")
    print("  Larger K sees more candidates but adds noise; find recall knee.")
    print("=" * 78)

    for name, cfg in (
        ("EASY (P=3, N_p=32)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                                P=3, N_p=32, P_max=6, seed=0)),
        ("HARD (P=5, N_p=16)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                                P=5, N_p=16, P_max=8, seed=0)),
        ("HARD (P=5, N_p=32)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                                P=5, N_p=32, P_max=8, seed=0)),
    ):
        system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
        pp, pv = cfg.pilots()
        gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
        print()
        print(name)
        print(f"  {'SNR':<6s}  " + "  ".join(f"K={K:>3d}" for K in (12, 16, 24, 32)))
        for snr in snrs:
            recall = {K: 0.0 for K in (12, 16, 24, 32)}
            for _ in range(n_batches):
                batch = sample_batch(system, channel, const, pp, pv,
                                     batch_size=batch_size, snr_db=snr, generator=gen)
                r = topk_recall(batch, system, pp, pv, Ks=(12, 16, 24, 32),
                                min_separation=1)
                for K in recall:
                    recall[K] += r[K]
            for K in recall:
                recall[K] /= n_batches
            print(f"  {snr:>4.1f}dB  " + "  ".join(f"{recall[K]:>5.1%}" for K in (12, 16, 24, 32)))


# ---------------------------------------------------------------------------
# Diagnostic C: iterated safeguarded LM on overcomplete candidates
# ---------------------------------------------------------------------------
def _lm_refined_classical(batch, cfg, K_candidates, n_lm_iters, n_alternations,
                          K_cg=30):
    """A classical-style detector with overcomplete initial candidates and
    n_lm_iters safeguarded LM refinements per alternation, alternated
    n_alternations times with LS gain re-estimation.

    Uses the SAME safeguarded_lm_theta_step function as the unrolled VEM
    receiver, but without any learned scalars — pure deterministic ascent
    with a per-batch acceptance test.
    """
    system = cfg.system()
    r = batch["r"]; y = batch["y"]; sigma_w2 = batch["sigma_w2_block"]
    device = r.device; dtype = r.dtype
    B, N = r.shape
    pp, pv = cfg.pilots()

    # 1. Overcomplete CFAR candidates.
    x_pilot_1d = torch.zeros(N, dtype=dtype, device=device)
    x_pilot_1d[pp] = pv
    s_pilot = system.idaft(x_pilot_1d.unsqueeze(0))[0]
    A_amb, ell_grid, kap_grid = ambiguity_function(
        r, s_pilot, N=system.N, N_cp=system.ell_max,
        kappa_max=5.0, ell_max=float(system.ell_max),
        oversample_doppler=2,
    )
    peak_idx, _ = cfar_peaks(A_amb, K=K_candidates, min_separation=1)
    ell_hat, kap_hat = newton_refine(A_amb, peak_idx, ell_grid, kap_grid, max_iter=2)

    # 2. Initial LS gains (with only pilots as x_hat).
    x_pilot_B = torch.zeros(B, N, dtype=dtype, device=device)
    x_pilot_B[:, pp] = pv.unsqueeze(0)
    h_hat = ls_gains_pilots_only(system, ell_hat, kap_hat, r, pp, pv)

    # 3. Alternation: LM refine (n_lm_iters) then re-estimate LS h.
    x_hat = x_pilot_B
    for alt in range(n_alternations):
        for _ in range(n_lm_iters):
            ell_hat, kap_hat, _acc = safeguarded_lm_theta_step(
                system, r, h_hat, x_hat, ell_hat, kap_hat,
                sigma_w2=sigma_w2, v_h=None, gamma_lr=0.5,
                max_step=0.2, slack=1e-4, max_backtracks=4,
            )
        h_hat = ls_gains_pilots_only(system, ell_hat, kap_hat, r, pp, pv)
        # Update x_hat via one CG-MMSE + hard demap.
        op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_hat)
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        z = cg_solve(mv, op.rmatvec(y), max_iter=K_cg)
        const = cfg.constellation()
        dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs()
        x_hat = const[dists.argmin(dim=-1)]
        x_hat[:, pp] = pv.unsqueeze(0)

    # 4. Final CG-MMSE.
    op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_hat)
    def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
    z = cg_solve(mv, op.rmatvec(y), max_iter=K_cg)
    const = cfg.constellation()
    hard = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
    return hard, ell_hat, kap_hat, h_hat


def diagnostic_C(snrs=(5.0, 15.0, 25.0), n_batches=4, batch_size=32):
    print()
    print("=" * 78)
    print("DIAGNOSTIC C: iterated safeguarded LM on overcomplete candidates")
    print("  K=12 initial candidates, n_lm=[0,3,6] LM per alternation, 3 alternations")
    print("=" * 78)

    for name, cfg in (
        ("EASY (P=3, N_p=32)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                                P=3, N_p=32, P_max=12, seed=0)),
        ("HARD (P=5, N_p=16)", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0,
                                                P=5, N_p=16, P_max=12, seed=0)),
    ):
        print()
        print(name)
        system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
        pp, pv = cfg.pilots()

        print(f"  {'SNR':<6s}  {'n_lm=0':>10s}  {'n_lm=3':>10s}  {'n_lm=6':>10s}")
        for snr in snrs:
            gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
            sers = {n: 0.0 for n in (0, 3, 6)}
            for _ in range(n_batches):
                batch = sample_batch(system, channel, const, pp, pv,
                                     batch_size=batch_size, snr_db=snr, generator=gen)
                mask = batch["pilot_mask"].float()
                for n_lm in (0, 3, 6):
                    hard, _, _, _ = _lm_refined_classical(
                        batch, cfg, K_candidates=12,
                        n_lm_iters=n_lm, n_alternations=3,
                    )
                    ser = float(((hard != batch["labels"]) * mask.bool()).float().sum() / mask.sum())
                    sers[n_lm] += ser
            for n in sers: sers[n] /= n_batches
            print(f"  {snr:>4.1f}dB  {sers[0]:>10.3e}  {sers[3]:>10.3e}  {sers[6]:>10.3e}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("DIAGNOSTICS A / B / C — decides architecture choice for A+")
    print("=" * 78)
    t0 = time.time()
    diagnostic_A()
    t1 = time.time(); print(f"\n[timing] Diag A: {t1 - t0:.1f}s")
    diagnostic_B()
    t2 = time.time(); print(f"\n[timing] Diag B: {t2 - t1:.1f}s")
    diagnostic_C()
    t3 = time.time(); print(f"\n[timing] Diag C: {t3 - t2:.1f}s")
    print(f"\n[timing] TOTAL: {t3 - t0:.1f}s")


if __name__ == "__main__":
    main()
