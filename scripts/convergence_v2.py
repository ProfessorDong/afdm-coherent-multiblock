"""Regenerate Fig 10 convergence traces (SER per outer iteration).

Runs the MB-IDAR receiver at B=4, 15 dB, records SER after each outer iteration
for t = 0, 1, ..., T_MAX. Both EASY and HARD, K seeds x n_batches x batch_size,
multi-seed averaged.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from multiblock_dasbl import (multiblock_ls_gains_data_aided, multiblock_lm_theta,
                              aperture_synthesis_kappa_refine)
from afdm.multi_block import block_doppler_phase


SNR = 15.0
B_BLOCK = 4
T_MAX = 10
N_SEEDS = 3
N_BATCHES = 8
BATCH_SIZE = 32


def receiver_with_trace(system, batch, const, cfg, T_max=T_MAX,
                        n_lm_per_outer=3, rho_min=0.5, use_reacq=True,
                        lambda_ridge=1e-3):
    """Same as multiblock_dasbl_receiver but returns SER trace over iterations."""
    r = batch.r; y = batch.y
    B_batch, B_block, N = r.shape
    dtype = r.dtype; device = r.device
    pp = batch.pilot_positions; pv = batch.pilot_values
    sigma_w2 = batch.sigma_w2_block

    def multi_block_ambiguity(x_hats_2d):
        A_sum = None; e_g = None; k_g = None
        for b in range(B_block):
            s_b = system.idaft(x_hats_2d[:, b, :])
            A_b, e_g, k_g = ambiguity_function(
                r[:, b, :], s_b, N=N, N_cp=int(cfg.ell_max),
                kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
                oversample_doppler=2,
            )
            A_sum = A_b if A_sum is None else A_sum + A_b
        return A_sum, e_g, k_g

    x_pilot = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
    for b in range(B_block):
        x_pilot[:, b, pp[b]] = pv[b].unsqueeze(0)

    N_cp_int = int(cfg.ell_max)
    beta = (N + N_cp_int) / N

    A_sum, e_g, k_g = multi_block_ambiguity(x_pilot)
    peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
    ell_hat, kap_hat = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
    if B_block > 1:
        kap_hat = aperture_synthesis_kappa_refine(
            system, r, x_pilot, ell_hat, kap_hat, N_cp=N_cp_int,
            kappa_window=0.30, kappa_step=0.003, beta=beta,
        )

    h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_pilot,
                                           lambda_ridge=lambda_ridge)

    omega = 1.0 / max(sigma_w2, 1e-6)
    def detect_all_blocks():
        p_ms = torch.zeros(B_batch, B_block, N, const.numel(), dtype=torch.float32, device=device)
        hard = torch.zeros(B_batch, B_block, N, dtype=torch.long, device=device)
        for b in range(B_block):
            phase_b = block_doppler_phase(kap_hat, b, N, N_cp_int)
            h_b = h_hat * phase_b
            op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_b)
            def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
            z = cg_solve(mv, op.rmatvec(y[:, b, :]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms[:, b, :] = F.softmax(-omega * dists, dim=-1)
            hard[:, b, :] = p_ms[:, b, :].argmax(dim=-1)
        return p_ms, hard

    def compute_ser(hard_now):
        mask = batch.pilot_mask
        return float(((hard_now != batch.labels) * mask).float().sum() / mask.float().sum())

    p_ms, hard = detect_all_blocks()
    ser_trace = [compute_ser(hard)]   # t=0

    for it in range(T_max):
        x_hats = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
        for b in range(B_block):
            rho_b = p_ms[:, b, :].max(dim=-1).values
            reliable_b = rho_b >= rho_min
            x_hats[:, b, :][reliable_b] = const[hard[:, b, :][reliable_b]]
            x_hats[:, b, pp[b]] = pv[b].unsqueeze(0)

        if use_reacq:
            A_sum, e_g, k_g = multi_block_ambiguity(x_hats)
            peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
            ell_new, kap_new = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
            if B_block > 1:
                kap_new = aperture_synthesis_kappa_refine(
                    system, r, x_hats, ell_new, kap_new, N_cp=N_cp_int,
                    kappa_window=0.30, kappa_step=0.003, beta=beta,
                )
            def stacked_residual(ell_t, kap_t):
                h_t = multiblock_ls_gains_data_aided(system, batch, ell_t, kap_t, x_hats,
                                                     lambda_ridge=lambda_ridge)
                res = torch.zeros(B_batch, device=device)
                for b in range(B_block):
                    A = build_regression_matrix(system, ell_t, kap_t, x_hats[:, b, :])
                    phase_b = block_doppler_phase(kap_t, b, N, N_cp_int)
                    A_b = A * phase_b.unsqueeze(1)
                    r_hat = (A_b @ h_t.unsqueeze(-1)).squeeze(-1)
                    res = res + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1)
                return res, h_t
            res_new, _ = stacked_residual(ell_new, kap_new)
            res_old, _ = stacked_residual(ell_hat, kap_hat)
            accept = res_new < res_old
            for b_idx in range(B_batch):
                if accept[b_idx]:
                    ell_hat[b_idx] = ell_new[b_idx]
                    kap_hat[b_idx] = kap_new[b_idx]

        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat = multiblock_lm_theta(
                system, batch, ell_hat, kap_hat, h_hat, x_hats, sigma_w2,
                gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
            )
        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)
        p_ms, hard = detect_all_blocks()
        ser_trace.append(compute_ser(hard))

    return ser_trace


def eval_seed(cfg, seed):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](N=cfg.N, N_p=cfg.N_p, B=B_BLOCK,
                                      constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    traces = []
    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=BATCH_SIZE, snr_db=SNR, generator=gen)
        with torch.no_grad():
            tr = receiver_with_trace(system, batch, const, cfg)
        traces.append(tr)
    return np.array(traces).mean(axis=0)   # (T_MAX + 1,)


def main():
    configs = [
        ("Easy (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("Hard (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
    ]

    print(f"\n{'='*70}\nMB-IDAR CONVERGENCE TRACE at B={B_BLOCK}, {SNR} dB\n"
          f"K={N_SEEDS} seeds x {N_BATCHES} batches x {BATCH_SIZE} realizations\n{'='*70}")

    all_results = {}
    for name, cfg in configs:
        print(f"\n[{name}]")
        seed_traces = []
        t0 = time.time()
        for k in range(N_SEEDS):
            tr = eval_seed(cfg, seed=k * 137 + 42)
            seed_traces.append(tr)
        arr = np.stack(seed_traces, axis=0)      # (K, T_MAX+1)
        mean = arr.mean(axis=0); std = arr.std(axis=0)
        dt = time.time() - t0
        print(f"  {'t':>3s}  {'mean SER':>12s}  {'std':>10s}")
        for t, (m, s) in enumerate(zip(mean, std)):
            print(f"  {t:>3d}  {m:>12.4e}  {s:>10.4e}")
        print(f"  wall time: {dt:.0f}s")
        all_results[name] = {"mean": mean.tolist(), "std": std.tolist()}

    out_path = Path("runs/convergence_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "B": B_BLOCK, "T_max": T_MAX,
                   "N_seeds": N_SEEDS, "N_batches": N_BATCHES, "batch_size": BATCH_SIZE,
                   "results": all_results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
