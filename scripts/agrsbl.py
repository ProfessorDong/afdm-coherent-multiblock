"""Adaptive Grid-Refinement SBL (AGR-SBL).

Combines:
  * overcomplete K CFAR candidates (K >> P);
  * iterative data-aided regression with reliable pseudo-pilots;
  * SBL/ARD-style variance updates and pruning of weak candidates;
  * safeguarded LM refinement of surviving candidates.

Goal: bridge the gap between the previous iterative-DASBL result at oracle
theta (~0.9% SER on hard) and at CFAR theta (~54% SER). If AGR-SBL closes
even half this gap, we have a viable receiver.
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


def agrsbl_receiver(
    system, batch, const, pp, pv,
    K_init: int = 24,
    n_outer: int = 6,
    n_lm_per_outer: int = 3,
    rho_min: float = 0.9,
    prune_ratio: float = 0.05,       # prune candidates with |h_k|^2 < prune_ratio * max |h|^2
    lambda_ridge: float = 1e-3,
    kappa_max: float = 5.0,
):
    r = batch["r"]; y = batch["y"]; sigma_w2 = batch["sigma_w2_block"]
    B, N = r.shape
    device = r.device; dtype = r.dtype

    # ---- 1. Initial overcomplete candidates ----
    x_p = torch.zeros(N, dtype=dtype, device=device); x_p[pp] = pv
    s_pilot = system.idaft(x_p.unsqueeze(0))[0]
    A_amb, e_g, k_g = ambiguity_function(
        r, s_pilot, N=N, N_cp=int(system.ell_max),
        kappa_max=kappa_max, ell_max=float(system.ell_max),
        oversample_doppler=2,
    )
    peak_idx, _ = cfar_peaks(A_amb, K=K_init, min_separation=1)
    ell_hat, kap_hat = newton_refine(A_amb, peak_idx, e_g, k_g, max_iter=2)
    valid = peak_idx[:, :, 0] >= 0    # (B, K)
    # We keep a per-batch active mask; start with all valid slots active.
    active = valid.clone()

    # ---- 2. Initial pilot-only h on all candidates ----
    def solve_h_on(ell_c, kap_c, active_c, x_ref):
        """Solve LS restricted to the ACTIVE candidates. Returns (B, K) with
        zeros in inactive slots."""
        K = ell_c.shape[1]
        h_out = torch.zeros(B, K, dtype=dtype, device=device)
        for bi in range(B):
            act = active_c[bi]
            n_act = int(act.sum())
            if n_act == 0:
                continue
            ell_a = ell_c[bi:bi+1, act]
            kap_a = kap_c[bi:bi+1, act]
            A = build_regression_matrix(system, ell_a, kap_a, x_ref[bi:bi+1])  # (1, N, n_act)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ r[bi:bi+1].unsqueeze(-1)).squeeze(-1)
            ridge = lambda_ridge * torch.eye(n_act, dtype=dtype, device=device).unsqueeze(0)
            h_a = torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)  # (1, n_act)
            h_out[bi, act] = h_a[0]
        return h_out

    x_hat = torch.zeros(B, N, dtype=dtype, device=device)
    x_hat[:, pp] = pv.unsqueeze(0)
    h_hat = solve_h_on(ell_hat, kap_hat, active, x_hat)

    # ---- 3. Detect symbols ----
    omega = 1.0 / max(sigma_w2, 1e-6)

    def detect(ell_c, kap_c, h_c, active_c):
        """CG-MMSE on active-only candidates."""
        p_ms_out = torch.zeros(B, N, const.numel(), dtype=torch.float32, device=device)
        hard_out = torch.zeros(B, N, dtype=torch.long, device=device)
        for bi in range(B):
            act = active_c[bi]
            n_act = int(act.sum())
            if n_act == 0:
                # No active paths: uniform posterior
                p_ms_out[bi] = 1.0 / const.numel()
                hard_out[bi] = 0
                continue
            ell_a = ell_c[bi:bi+1, act]
            kap_a = kap_c[bi:bi+1, act]
            h_a = h_c[bi:bi+1, act]
            op = FastAFDMOperator(system=system, ell=ell_a, kappa=kap_a, h=h_a)
            def mv(v): return op.rmatvec(op.matvec(v)) + sigma_w2 * v
            z = cg_solve(mv, op.rmatvec(y[bi:bi+1]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms_out[bi] = F.softmax(-omega * dists, dim=-1)[0]
            hard_out[bi] = p_ms_out[bi].argmax(dim=-1)
        return p_ms_out, hard_out

    p_ms, hard = detect(ell_hat, kap_hat, h_hat, active)

    # ---- 4. Outer loop ----
    for it in range(n_outer):
        # Reliable pseudo-pilots
        rho = p_ms.max(dim=-1).values
        reliable = rho >= rho_min
        x_hat_it = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat_it[reliable] = const[hard[reliable]]
        x_hat_it[:, pp] = pv.unsqueeze(0)

        # h update
        h_hat = solve_h_on(ell_hat, kap_hat, active, x_hat_it)

        # LM theta refinement on active candidates only.
        # Note: safeguarded_lm_theta_step operates on the FULL (B, K) tensors,
        # so we pass the full arrays; inactive slots have h=0 so contribute
        # nothing to the LM objective.
        for _ in range(n_lm_per_outer):
            ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                system, r, h_hat, x_hat_it, ell_hat, kap_hat,
                sigma_w2=sigma_w2, v_h=None,
                gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
            )

        # Refit h with refined theta
        h_hat = solve_h_on(ell_hat, kap_hat, active, x_hat_it)

        # Prune: any candidate with |h_k|^2 < prune_ratio * max_active |h|^2 goes off.
        h_pow = (h_hat.abs() ** 2)
        # Per-batch prune threshold
        max_active_pow = torch.where(active, h_pow, torch.zeros_like(h_pow)).max(dim=-1, keepdim=True).values
        thr = prune_ratio * max_active_pow
        prune = (h_pow < thr) & active
        active = active & ~prune

        # Detect again
        p_ms, hard = detect(ell_hat, kap_hat, h_hat, active)

    return hard, p_ms, ell_hat, kap_hat, h_hat, active


def eval_agrsbl(cfg, snr_db, n_batches=8, batch_size=32, seed=42, **kwargs):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        hard, _, _, _, _, _ = agrsbl_receiver(system, batch, const, pp, pv,
                                              kappa_max=cfg.kappa_max, **kwargs)
        mask = batch["pilot_mask"]
        ser = float(((hard != batch["labels"]) * mask).float().sum() / mask.float().sum())
        ser_acc += ser
    return ser_acc / n_batches


def main():
    for cfg_name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32, P_max=6)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32, P_max=8)),
    ):
        print()
        print("=" * 78)
        print(f"CONFIG: {cfg_name}")
        print("=" * 78)
        for snr in (5.0, 15.0, 25.0):
            ser = eval_agrsbl(cfg, snr, K_init=24, n_outer=6, n_lm_per_outer=3,
                              rho_min=0.9, prune_ratio=0.05)
            print(f"  SNR {snr}dB: AGR-SBL SER = {ser:.3e}")


if __name__ == "__main__":
    main()
