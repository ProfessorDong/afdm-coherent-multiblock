"""Multi-block iterative DASBL with SHARED theta across B blocks.

Key idea: same physical channel over a coherence window, different pilot masks
per block. Multi-block observations reduce theta estimator variance by ~1/B.

Pipeline (given B blocks):
  0. Initial theta from multi-block CFAR (sum |A_b|^2 across blocks).
  1. Multi-block pilot-only LS h.
  2. For outer iter t = 1..T:
     a. For each block, detect symbols with shared (theta, h).
     b. reliable_b = (rho_b >= rho_min)
     c. x_hat_b = pilots_b + reliable-hard_b
     d. Multi-block LS h (using x_hat_b per block).
     e. LM refinement of shared theta using stacked objective:
        L = sum_b |r_b - A_b(theta, x_hat_b) h|^2
     f. Detect again.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from afdm.classical import build_regression_matrix, cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock
from afdm.operators import FastAFDMOperator
from afdm.support import ambiguity_function, cfar_peaks, newton_refine


def multiblock_ls_gains_data_aided(system, batch, ell, kap, x_hats, lambda_ridge=1e-3):
    """Solve multi-block LS with SHARED (ell, kap) and per-block x_hats."""
    B_batch, B_block, N = batch.r.shape
    dtype = batch.r.dtype; device = batch.r.device
    P = ell.shape[1]
    AhA_sum = torch.zeros(B_batch, P, P, dtype=dtype, device=device)
    Ahr_sum = torch.zeros(B_batch, P, dtype=dtype, device=device)
    for b in range(B_block):
        A = build_regression_matrix(system, ell, kap, x_hats[:, b, :])
        AH = A.conj().transpose(-1, -2)
        AhA_sum += AH @ A
        Ahr_sum += (AH @ batch.r[:, b, :].unsqueeze(-1)).squeeze(-1)
    ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
    return torch.linalg.solve(AhA_sum + ridge, Ahr_sum.unsqueeze(-1)).squeeze(-1)


def multiblock_lm_theta(system, batch, ell, kap, h, x_hats, sigma_w2,
                         gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4):
    """Safeguarded LM step for SHARED theta using stacked B-block objective.

    L(theta) = -(1/N) sum_b |r_b - A_b(theta, x_b) h|^2

    Backtracks per-batch as in single-block version.
    """
    B_batch, B_block, N = batch.r.shape
    device = ell.device
    ell_v = ell.detach().clone().requires_grad_(True)
    kap_v = kap.detach().clone().requires_grad_(True)

    def obj():
        # Sum residual squared over blocks.
        loss_per_b = torch.zeros(B_batch, device=device)
        for b in range(B_block):
            A = build_regression_matrix(system, ell_v, kap_v, x_hats[:, b, :])
            r_hat = (A @ h.unsqueeze(-1)).squeeze(-1)
            res = batch.r[:, b, :] - r_hat
            loss_per_b = loss_per_b + (res.abs() ** 2).sum(dim=-1) / N
        return -loss_per_b   # negative so higher = better

    with torch.enable_grad():
        Q_old = obj()
        Q_sum = Q_old.sum()
    ge, gk = torch.autograd.grad(Q_sum, [ell_v, kap_v])
    Q_old = Q_old.detach()
    step_e = torch.clamp(gamma_lr * ge.detach(), min=-max_step, max=max_step)
    step_k = torch.clamp(gamma_lr * gk.detach(), min=-max_step, max=max_step)
    ell_best = ell.detach().clone(); kap_best = kap.detach().clone()
    accepted = torch.zeros(B_batch, dtype=torch.bool, device=device)
    for i in range(max_backtracks + 1):
        scale = 0.5 ** i
        ce = (ell.detach() + scale * step_e).clamp(min=0.0, max=system.ell_max)
        ck = (kap.detach() + scale * step_k).clamp(min=-system.kappa_max, max=system.kappa_max)
        with torch.no_grad():
            # Recompute obj at candidate
            Q_c = torch.zeros(B_batch, device=device)
            for b in range(B_block):
                A = build_regression_matrix(system, ce, ck, x_hats[:, b, :])
                r_hat = (A @ h.unsqueeze(-1)).squeeze(-1)
                Q_c = Q_c + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1) / N
            Q_c = -Q_c
        improved = (Q_c >= Q_old - slack) & (~accepted)
        if improved.any():
            for b_idx in torch.where(improved)[0].tolist():
                ell_best[b_idx] = ce[b_idx]
                kap_best[b_idx] = ck[b_idx]
            accepted = accepted | improved
        if accepted.all():
            break
    return ell_best.detach(), kap_best.detach()


def multiblock_dasbl_receiver(system, batch, const, cfg,
                              n_outer=6, n_lm_per_outer=3, rho_min=0.9,
                              use_reacq=True, lambda_ridge=1e-3):
    """Multi-block iterative DASBL with data-aided reacquisition and shared theta."""
    r = batch.r; y = batch.y
    B_batch, B_block, N = r.shape
    dtype = r.dtype; device = r.device
    pp = batch.pilot_positions; pv = batch.pilot_values
    sigma_w2 = batch.sigma_w2_block

    # 1. Initial multi-block CFAR: sum |A_b|^2 across blocks, then peak-find.
    def multi_block_ambiguity(x_hats_2d):
        """x_hats_2d: (B_batch, B_block, N). Returns stacked |A|^2 (B_batch, L_k, L_e)."""
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

    # Initial: pilot-only signals.
    x_pilot = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
    for b in range(B_block):
        x_pilot[:, b, pp[b]] = pv[b].unsqueeze(0)

    A_sum, e_g, k_g = multi_block_ambiguity(x_pilot)
    peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
    ell_hat, kap_hat = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)

    # 2. Multi-block LS h.
    h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_pilot,
                                           lambda_ridge=lambda_ridge)

    # 3. Detect symbols per block with shared (theta, h).
    omega = 1.0 / max(sigma_w2, 1e-6)
    def detect_all_blocks():
        p_ms = torch.zeros(B_batch, B_block, N, const.numel(), dtype=torch.float32, device=device)
        hard = torch.zeros(B_batch, B_block, N, dtype=torch.long, device=device)
        for b in range(B_block):
            op = FastAFDMOperator(system=system, ell=ell_hat, kappa=kap_hat, h=h_hat)
            def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
            z = cg_solve(mv, op.rmatvec(y[:, b, :]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms[:, b, :] = F.softmax(-omega * dists, dim=-1)
            hard[:, b, :] = p_ms[:, b, :].argmax(dim=-1)
        return p_ms, hard

    p_ms, hard = detect_all_blocks()

    # 4. Outer loop.
    for it in range(n_outer):
        # Build data-aided x_hats per block.
        x_hats = torch.zeros(B_batch, B_block, N, dtype=dtype, device=device)
        for b in range(B_block):
            rho_b = p_ms[:, b, :].max(dim=-1).values
            reliable_b = rho_b >= rho_min
            x_hats[:, b, :][reliable_b] = const[hard[:, b, :][reliable_b]]
            x_hats[:, b, pp[b]] = pv[b].unsqueeze(0)

        # Reacquire theta from data-aided multi-block ambiguity.
        if use_reacq:
            A_sum, e_g, k_g = multi_block_ambiguity(x_hats)
            peak_idx, _ = cfar_peaks(A_sum, K=cfg.P_max, min_separation=2)
            ell_new, kap_new = newton_refine(A_sum, peak_idx, e_g, k_g, max_iter=2)
            # Accept if lower residual.
            def stacked_residual(ell_t, kap_t):
                h_t = multiblock_ls_gains_data_aided(system, batch, ell_t, kap_t, x_hats,
                                                     lambda_ridge=lambda_ridge)
                res = torch.zeros(B_batch, device=device)
                for b in range(B_block):
                    A = build_regression_matrix(system, ell_t, kap_t, x_hats[:, b, :])
                    r_hat = (A @ h_t.unsqueeze(-1)).squeeze(-1)
                    res = res + (batch.r[:, b, :] - r_hat).abs().pow(2).sum(dim=-1)
                return res, h_t
            res_new, h_new = stacked_residual(ell_new, kap_new)
            res_old, _ = stacked_residual(ell_hat, kap_hat)
            accept = res_new < res_old
            for b_idx in range(B_batch):
                if accept[b_idx]:
                    ell_hat[b_idx] = ell_new[b_idx]
                    kap_hat[b_idx] = kap_new[b_idx]

        # LS h with current theta and x_hats.
        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)

        # LM refinement of shared theta.
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat = multiblock_lm_theta(
                system, batch, ell_hat, kap_hat, h_hat, x_hats, sigma_w2,
                gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
            )

        # Refit h.
        h_hat = multiblock_ls_gains_data_aided(system, batch, ell_hat, kap_hat, x_hats,
                                               lambda_ridge=lambda_ridge)

        # Detect all blocks.
        p_ms, hard = detect_all_blocks()

    return hard, ell_hat, kap_hat, h_hat


def eval_multiblock(cfg, snr_db, B_block, design="hopping", n_batches=6, batch_size=16, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS[design](N=cfg.N, N_p=cfg.N_p, B=B_block,
                                   constellation=const, device=cfg.device, seed=42)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=batch_size, snr_db=snr_db, generator=gen)
        hard, _, _, _ = multiblock_dasbl_receiver(system, batch, const, cfg,
                                                   n_outer=6, n_lm_per_outer=3,
                                                   rho_min=0.9, use_reacq=True)
        mask = batch.pilot_mask
        ser = float(((hard != batch.labels) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", default="hopping",
                    choices=("repeated", "hopping", "complementary"))
    ap.add_argument("--Bs", type=str, default="1,2,4")
    ap.add_argument("--configs", type=str, default="all")
    args = ap.parse_args()

    Bs = [int(b) for b in args.Bs.split(",")]

    all_cfgs = (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    )
    cfgs = all_cfgs if args.configs == "all" else [c for c in all_cfgs if args.configs in c[0]]

    print("=" * 78)
    print(f"MULTI-BLOCK ITERATIVE DASBL (shared theta) — design={args.design}, Bs={Bs}")
    print("=" * 78)
    for name, cfg in cfgs:
        print(f"\n{name}")
        print(f"  {'SNR':<6s}  " + "  ".join(f"B={b}" for b in Bs))
        for snr in (5.0, 15.0, 25.0):
            row = []
            for B in Bs:
                ser = eval_multiblock(cfg, snr, B_block=B, design=args.design)
                row.append(ser)
            print(f"  {snr:>4.1f}dB  " + "  ".join(f"{v:.3e}" for v in row))


if __name__ == "__main__":
    main()
