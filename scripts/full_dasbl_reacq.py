"""Full iterative DASBL with data-aided re-acquisition each outer iteration.

Pipeline:
  0. CFAR on pilot-only ambiguity -> theta_0
  1. LS h with pilot-only  -> h_0
  2. For outer iter t = 1..T:
     a. Detect symbols -> (p_ms, hard)
     b. reliable = (rho >= rho_min)
     c. x_hat = pilots + reliable-hard
     d. RE-ACQUIRE: compute ambiguity with s = idaft(x_hat), CFAR -> new theta_t
        (this replaces theta if the new theta yields lower LS residual)
     e. LS h given (theta_t, x_hat)
     f. LM refinement of theta_t (small step)
     g. Detect symbols again

This is the receiver at the identifiability boundary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step


def dasbl_reacq(system, batch, const, pp, pv, cfg,
                n_outer=6, n_lm_per_outer=2, rho_min=0.9,
                reacq_from_iter=1, use_reacq=True,
                lambda_ridge=1e-3):
    r = batch["r"]; y = batch["y"]; sigma_w2 = batch["sigma_w2_block"]
    B, N = r.shape
    dtype = r.dtype; device = r.device

    def acquire(s_signal, K):
        A_amb, e_g, k_g = ambiguity_function(
            r, s_signal, N=N, N_cp=int(cfg.ell_max),
            kappa_max=cfg.kappa_max, ell_max=float(cfg.ell_max),
        )
        peak_idx, _ = cfar_peaks(A_amb, K=K, min_separation=2)
        return newton_refine(A_amb, peak_idx, e_g, k_g, max_iter=2)

    def solve_h(x_ref, ell_c, kap_c):
        A = build_regression_matrix(system, ell_c, kap_c, x_ref)
        AH = A.conj().transpose(-1, -2)
        AhA = AH @ A
        Ahr = (AH @ r.unsqueeze(-1)).squeeze(-1)
        P = ell_c.shape[1]
        ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
        return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

    omega = 1.0 / max(sigma_w2, 1e-6)
    def detect(h, ell, kap):
        op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
        def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
        z = cg_solve(mv, op.rmatvec(y), max_iter=30)
        dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
        p = F.softmax(-omega * dists, dim=-1)
        return p, p.argmax(dim=-1)

    def residual_norm(x_ref, ell_c, kap_c, h_c):
        op = FastAFDMOperator(system=system, ell=ell_c, kappa=kap_c, h=h_c)
        s = system.idaft(x_ref)
        r_hat = op.matvec(x_ref)
        r_hat_time = system.idaft(r_hat)   # in time domain, y is DAFT-domain
        return (system.daft(r) - r_hat).abs().pow(2).sum(dim=-1).sqrt()

    # ---- init ----
    x_pilot = torch.zeros(B, N, dtype=dtype, device=device)
    x_pilot[:, pp] = pv.unsqueeze(0)
    s_pilot = system.idaft(x_pilot)
    ell_hat, kap_hat = acquire(s_pilot, K=cfg.P_max)
    h_hat = solve_h(x_pilot, ell_hat, kap_hat)
    p_ms, hard = detect(h_hat, ell_hat, kap_hat)

    for it in range(n_outer):
        rho = p_ms.max(dim=-1).values
        reliable = rho >= rho_min
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[reliable] = const[hard[reliable]]
        x_hat[:, pp] = pv.unsqueeze(0)

        # Data-aided re-acquisition (if enabled and past the warmup iter).
        if use_reacq and it >= reacq_from_iter:
            s_data_aided = system.idaft(x_hat)
            ell_new, kap_new = acquire(s_data_aided, K=cfg.P_max)
            # Accept the new theta if it yields lower LS residual.
            h_new = solve_h(x_hat, ell_new, kap_new)
            res_new = residual_norm(x_hat, ell_new, kap_new, h_new)
            res_old = residual_norm(x_hat, ell_hat, kap_hat, h_hat)
            accept = res_new < res_old
            for bi in range(B):
                if accept[bi]:
                    ell_hat[bi] = ell_new[bi]
                    kap_hat[bi] = kap_new[bi]
                    h_hat[bi] = h_new[bi]

        # h update (with current best theta and x_hat)
        h_hat = solve_h(x_hat, ell_hat, kap_hat)

        # LM refinement
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                system, r, h_hat, x_hat, ell_hat, kap_hat,
                sigma_w2=sigma_w2, v_h=None,
                gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
            )
        h_hat = solve_h(x_hat, ell_hat, kap_hat)

        p_ms, hard = detect(h_hat, ell_hat, kap_hat)

    return hard, ell_hat, kap_hat, h_hat


def eval_config(cfg, snr_db, use_reacq=True, n_batches=8, batch_size=32):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(42)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        hard, _, _, _ = dasbl_reacq(system, batch, const, pp, pv, cfg,
                                    n_outer=6, n_lm_per_outer=2, rho_min=0.9,
                                    use_reacq=use_reacq)
        mask = batch["pilot_mask"]
        ser = float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    print("=" * 78)
    print("DASBL WITH DATA-AIDED RE-ACQUISITION")
    print("=" * 78)
    for name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    ):
        print()
        print(f"{name}")
        print(f"  {'SNR':<6s}  {'no reacq':>10s}  {'w/ reacq':>10s}")
        for snr in (5.0, 15.0, 25.0):
            ser_no = eval_config(cfg, snr, use_reacq=False)
            ser_yes = eval_config(cfg, snr, use_reacq=True)
            print(f"  {snr:>4.1f}dB  {ser_no:>10.3e}  {ser_yes:>10.3e}")


if __name__ == "__main__":
    main()
