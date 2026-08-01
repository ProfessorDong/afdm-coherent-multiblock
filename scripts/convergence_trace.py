"""Convergence traces: SER vs iteration for our receiver.

Show that multi-block DASBL converges within 4-6 outer iterations, and that
each iteration is monotone or near-monotone. This provides the algorithmic
behavior figure for the paper.
"""

from __future__ import annotations

import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine


def trace_receiver(cfg, snr_db, B_block=4, n_outer=10, n_lm_per_outer=3,
                   rho_min=0.9, n_batches=6, batch_size=16, seed=42):
    """Return SER at each outer iteration."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                       constellation=const, device=cfg.device, seed=seed)

    ser_per_iter = [[] for _ in range(n_outer + 1)]  # [0] is initial

    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    for _ in range(n_batches):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=batch_size, snr_db=snr_db, generator=gen)
        r = batch.r; y = batch.y; sigma_w2 = batch.sigma_w2_block
        B_batch, B, N = r.shape
        dtype = r.dtype; device = r.device
        pp_ = batch.pilot_positions; pv_ = batch.pilot_values

        # ---- Multi-block CFAR ----
        x_pilot = torch.zeros(B_batch, B, N, dtype=dtype, device=device)
        for b in range(B):
            x_pilot[:, b, pp_[b]] = pv_[b].unsqueeze(0)

        def multi_amb(x_hats_2d):
            A_sum = None; e_g = None; k_g = None
            for b in range(B):
                s_b = system.idaft(x_hats_2d[:, b, :])
                A_b, e_g, k_g = ambiguity_function(
                    r[:, b, :], s_b, N=N, N_cp=int(cfg.ell_max),
                    kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
                    oversample_doppler=2,
                )
                A_sum = A_b if A_sum is None else A_sum + A_b
            return A_sum, e_g, k_g

        A_sum, e_g, k_g = multi_amb(x_pilot)
        peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
        ell_hat, kap_hat = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)

        def solve_h(x_hats):
            P = ell_hat.shape[1]
            AhA_sum = torch.zeros(B_batch, P, P, dtype=dtype, device=device)
            Ahr_sum = torch.zeros(B_batch, P, dtype=dtype, device=device)
            for b in range(B):
                A = build_regression_matrix(system, ell_hat, kap_hat, x_hats[:, b, :])
                AH = A.conj().transpose(-1, -2)
                AhA_sum += AH @ A
                Ahr_sum += (AH @ r[:, b, :].unsqueeze(-1)).squeeze(-1)
            ridge = 1e-3 * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
            return torch.linalg.solve(AhA_sum + ridge, Ahr_sum.unsqueeze(-1)).squeeze(-1)

        h_hat = solve_h(x_pilot)
        omega = 1.0 / max(sigma_w2, 1e-6)

        def detect():
            p_ms_all = torch.zeros(B_batch, B, N, const.numel(), dtype=torch.float32, device=device)
            hard_all = torch.zeros(B_batch, B, N, dtype=torch.long, device=device)
            for b in range(B):
                op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_hat)
                def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
                z = cg_solve(mv, op.rmatvec(y[:, b, :]), max_iter=30)
                dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
                p_ms_all[:, b, :] = F.softmax(-omega * dists, dim=-1)
                hard_all[:, b, :] = p_ms_all[:, b, :].argmax(dim=-1)
            return p_ms_all, hard_all

        p_ms, hard = detect()
        mask = batch.pilot_mask
        ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
        ser_per_iter[0].append(ser)

        for it in range(n_outer):
            x_hats = torch.zeros(B_batch, B, N, dtype=dtype, device=device)
            for b in range(B):
                rho_b = p_ms[:, b, :].max(dim=-1).values
                reliable_b = rho_b >= rho_min
                x_hats[:, b, :][reliable_b] = const[hard[:, b, :][reliable_b]]
                x_hats[:, b, pp_[b]] = pv_[b].unsqueeze(0)
            # Reacquire
            A_sum, e_g, k_g = multi_amb(x_hats)
            peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
            ell_new, kap_new = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
            h_new = solve_h(x_hats)
            # Simplified: always accept for trace
            ell_hat = ell_new; kap_hat = kap_new; h_hat = h_new
            h_hat = solve_h(x_hats)
            # LM (simple)
            from afdm.vem import safeguarded_lm_theta_step
            for _ in range(n_lm_per_outer):
                # Use block 0 as representative
                ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                    system, r[:, 0, :], h_hat, x_hats[:, 0, :], ell_hat, kap_hat,
                    sigma_w2=sigma_w2, v_h=None,
                    gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
                )
            h_hat = solve_h(x_hats)
            p_ms, hard = detect()
            ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
            ser_per_iter[it + 1].append(ser)

    return [sum(x)/len(x) for x in ser_per_iter]


def main():
    results = {}
    for cfg_name, cfg in (
        ("Easy (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("Hard (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ):
        print(f"\n{cfg_name} at 15 dB, B=4:")
        traj = trace_receiver(cfg, snr_db=15.0, B_block=4, n_outer=10)
        results[cfg_name] = traj
        for i, s in enumerate(traj):
            tag = "init" if i == 0 else f"iter {i}"
            print(f"  {tag:>8s}: SER = {s:.3e}")

    with open("runs/convergence_trace.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: runs/convergence_trace.json")


if __name__ == "__main__":
    main()
