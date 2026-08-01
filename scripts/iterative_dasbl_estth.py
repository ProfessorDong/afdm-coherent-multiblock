"""Iterative data-aided SBL with ESTIMATED theta (CFAR + iterated LM refinement).

This closes the loop: initial CFAR support -> iterative (x_hat, h, theta) updates.
The question is how much SER degrades compared to oracle-theta DASBL.

Pipeline:
  theta_0 = CFAR + 2-Newton refinement
  h_0     = pilot-only LS given theta_0
  for it = 1..I:
      z = CG-MMSE(y, H(theta_{it-1}, h_{it-1}))
      x_hat = pilots + reliable-hard symbols
      h_it  = LS given (theta_{it-1}, x_hat)
      theta_it = safeguarded LM step given (h_it, x_hat)
  final CG-MMSE detection
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
from afdm.support import SupportRecovery
from afdm.training import sample_batch
from afdm.vem import safeguarded_lm_theta_step


def run(cfg, snr_db, n_iters=6, rho_min=0.9, n_lm_per_iter=2,
        n_batches=8, batch_size=32, lambda_ridge=1e-3, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    support = SupportRecovery(N=cfg.N, N_cp=int(cfg.ell_max),
                              kappa_max=cfg.kappa_max, ell_max=cfg.ell_max,
                              P_max=cfg.P_max)
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    def solve_h(x_ref, ell, kap):
        A = build_regression_matrix(system, ell, kap, x_ref)
        AH = A.conj().transpose(-1, -2)
        AhA = AH @ A
        Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
        P = ell.shape[1]
        ridge = lambda_ridge * torch.eye(P, dtype=A.dtype, device=A.device).unsqueeze(0)
        return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

    ser_traj = [[] for _ in range(n_iters + 1)]

    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        B, N = batch["r"].shape
        dtype = batch["r"].dtype; device = batch["r"].device

        # ---- CFAR + Newton initial theta ----
        x_pilot_1d = torch.zeros(N, dtype=dtype, device=device)
        x_pilot_1d[pp] = pv
        s_pilot = system.idaft(x_pilot_1d.unsqueeze(0))[0]
        ell_hat, kap_hat, p_hat = support(batch["r"], s_pilot)
        P_hat = ell_hat.shape[1]

        # ---- initial pilot-only h ----
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, pp] = pv.unsqueeze(0)
        h_hat = solve_h(x_hat, ell_hat, kap_hat)

        # ---- initial detection ----
        omega = 1.0 / max(batch["sigma_w2_block"], 1e-6)
        def detect(h, ell, kap):
            op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
            def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms = F.softmax(-omega * dists, dim=-1)
            return p_ms, p_ms.argmax(dim=-1)

        p_ms, hard = detect(h_hat, ell_hat, kap_hat)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_traj[0].append(ser)

        for it in range(1, n_iters + 1):
            # Reliable-symbol pseudo-pilots.
            rho = p_ms.max(dim=-1).values
            reliable = rho >= rho_min
            x_hat_it = torch.zeros(B, N, dtype=dtype, device=device)
            x_hat_it[reliable] = const[hard[reliable]]
            x_hat_it[:, pp] = pv.unsqueeze(0)

            # h update.
            h_hat = solve_h(x_hat_it, ell_hat, kap_hat)

            # theta LM refinement (small step).
            for _ in range(n_lm_per_iter):
                ell_hat, kap_hat, _ = safeguarded_lm_theta_step(
                    system, batch["r"], h_hat, x_hat_it, ell_hat, kap_hat,
                    sigma_w2=batch["sigma_w2_block"], v_h=None,
                    gamma_lr=0.5, max_step=0.15, slack=1e-4, max_backtracks=4,
                )
            # h update again with refined theta.
            h_hat = solve_h(x_hat_it, ell_hat, kap_hat)

            # Detect.
            p_ms, hard = detect(h_hat, ell_hat, kap_hat)
            ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
            ser_traj[it].append(ser)

    for it in range(n_iters + 1):
        avg = sum(ser_traj[it]) / len(ser_traj[it])
        tag = "cfar+pilot" if it == 0 else f"iter {it}"
        print(f"  {tag:<12s}: SER = {avg:.3e}")


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
        print(f"CONFIG: {cfg_name}   (estimated theta via CFAR + iterated LM)")
        print("=" * 78)
        for snr in (5.0, 15.0, 25.0):
            print(f"\nSNR {snr} dB:")
            run(cfg, snr, n_iters=6, rho_min=0.9, n_lm_per_iter=2)


if __name__ == "__main__":
    main()
