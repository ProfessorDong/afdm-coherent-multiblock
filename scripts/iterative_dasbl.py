"""Day 6-10 initial test: iterative data-aided SBL.

Question: can iterative data-aided regression (using DECODED reliable symbols as
pseudo-pilots) close the gap between pilot-only LS (20% SER on hard) and
data-aided oracle upper bound (0.5% SER)?

Algorithm per batch, given oracle theta:
  h_0 = pilot-only LS
  for it = 1..I:
      z = CG-MMSE(y, H(theta, h_{it-1}), sigma_w2)
      compute soft posterior p_ms
      x_hat = pilots + soft-symbol mean (masked by reliability rho >= rho_min)
      h_it = LS(r, A(theta, x_hat)) with ridge

Report: SER after each iteration + final h NMSE.
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
from afdm.training import sample_batch


def iterative_dasbl(cfg, snr_db, n_iters=5, rho_min=0.9, n_batches=8,
                     batch_size=32, lambda_ridge=1e-3, seed=42):
    """Single-block iterative data-aided SBL (theta = oracle)."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    ser_traj = [[] for _ in range(n_iters + 1)]
    nmse_traj = [[] for _ in range(n_iters + 1)]

    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        ell = batch["theta_true"][..., 0]; kap = batch["theta_true"][..., 1]
        B, N = batch["r"].shape
        dtype = batch["r"].dtype; device = batch["r"].device

        # --- iter 0: pilot-only LS ---
        x_hat = torch.zeros(B, N, dtype=dtype, device=device)
        x_hat[:, pp] = pv.unsqueeze(0)
        def solve_h(x_ref):
            A = build_regression_matrix(system, ell, kap, x_ref)
            AH = A.conj().transpose(-1, -2)
            AhA = AH @ A
            Ahr = (AH @ batch["r"].unsqueeze(-1)).squeeze(-1)
            P = ell.shape[1]
            ridge = lambda_ridge * torch.eye(P, dtype=dtype, device=device).unsqueeze(0)
            return torch.linalg.solve(AhA + ridge, Ahr.unsqueeze(-1)).squeeze(-1)

        h = solve_h(x_hat)
        nmse = float(((h - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum().clamp(min=1e-12))
        nmse_traj[0].append(nmse)

        # detect with h, get SER
        op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
        def mv(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
        z = cg_solve(mv, op.rmatvec(batch["y"]), max_iter=30)
        dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
        omega = 1.0 / max(batch["sigma_w2_block"], 1e-6)
        p_ms = F.softmax(-omega * dists, dim=-1)
        hard = p_ms.argmax(dim=-1)
        ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        ser_traj[0].append(ser)

        # --- iterative data-aided ---
        for it in range(1, n_iters + 1):
            # Reliability weights.
            rho = p_ms.max(dim=-1).values                      # (B, N)
            reliable = rho >= rho_min
            # x_hat = pilots (hard) + reliable soft mean
            x_soft_hat = (p_ms * const.reshape(1, 1, -1)).sum(dim=-1)   # (B, N)
            # Use reliable hard-decoded symbols in addition to pilots
            x_hat_it = torch.zeros(B, N, dtype=dtype, device=device)
            x_hat_it[reliable] = const[hard[reliable]]
            x_hat_it[:, pp] = pv.unsqueeze(0)

            h = solve_h(x_hat_it)
            nmse = float(((h - batch["h_true"]).abs() ** 2).sum() / (batch["h_true"].abs() ** 2).sum().clamp(min=1e-12))
            nmse_traj[it].append(nmse)

            # Redetect
            op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h)
            def mv2(v): return op.rmatvec(op.matvec(v)) + batch["sigma_w2_block"] * v
            z = cg_solve(mv2, op.rmatvec(batch["y"]), max_iter=30)
            dists = (z.unsqueeze(-1) - const.reshape(1, 1, -1)).abs() ** 2
            p_ms = F.softmax(-omega * dists, dim=-1)
            hard = p_ms.argmax(dim=-1)
            ser = float(((hard != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
            ser_traj[it].append(ser)

    for it in range(n_iters + 1):
        avg_ser = sum(ser_traj[it]) / len(ser_traj[it])
        avg_nmse = sum(nmse_traj[it]) / len(nmse_traj[it])
        tag = "pilot-only" if it == 0 else f"iter {it}"
        print(f"  {tag:<12s}: SER = {avg_ser:.3e}, NMSE(h) = {avg_nmse:.3e}")


def main():
    for cfg_name, cfg in (
        ("EASY (P=3, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32)),
        ("HARD (P=5, N_p=16)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16)),
        ("HARD (P=5, N_p=32)",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32)),
    ):
        print()
        print("=" * 78)
        print(f"CONFIG: {cfg_name}   (oracle theta, iterative data-aided)")
        print("=" * 78)
        for snr in (5.0, 15.0, 25.0):
            print(f"\nSNR {snr} dB:")
            iterative_dasbl(cfg, snr, n_iters=5, rho_min=0.9)


if __name__ == "__main__":
    main()
