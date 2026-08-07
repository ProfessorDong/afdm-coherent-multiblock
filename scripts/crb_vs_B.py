"""Direct CRB vs B verification for Theorem 2.

For HARD (P=5, N_p=16) at 15 dB and B in {1, 2, 4, 8}, compute:
  (a) numerical Fisher matrix J_MB per B (per-path 2x2 blocks, oracle x_b and h)
      and derive per-path CRB variance (diagonal)
  (b) empirical per-path RMSE from multi-seed MB-IDAR runs (Hungarian matched)

Save JSON with per-B numbers so the TikZ figure in the paper is reproducible.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from afdm.experiments import ExperimentConfig
from afdm.multi_block import PILOT_DESIGNS, sample_multiblock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from crb_analysis import numerical_fim, crb_from_fim, hungarian_match
from multiblock_dasbl import multiblock_dasbl_receiver


SNR = 15.0
BS = [1, 2, 4, 8]
N_SEEDS = 6
N_BATCHES = 4
BATCH_SIZE = 8


def measure_one(cfg, B_block, seed):
    """Return {crb_ell_mse, crb_kap_mse, emp_ell_mse, emp_kap_mse, n_matched}."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS["hopping"](
        N=cfg.N, N_p=cfg.N_p, B=B_block, constellation=const,
        device=cfg.device, seed=42,
    )
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)

    crb_ell = 0.0; crb_kap = 0.0; n_crb = 0
    de_sq = 0.0; dk_sq = 0.0; n = 0

    for _ in range(N_BATCHES):
        batch = sample_multiblock(system, channel, const, pp, pv,
                                  batch_size=BATCH_SIZE, snr_db=SNR, generator=gen)

        # Numerical Fisher matrix (oracle x_b, oracle h) and per-path CRB.
        fim = numerical_fim(system, batch,
                            batch.theta_true[..., 0], batch.theta_true[..., 1],
                            batch.h_true, batch.x_true)
        crb = crb_from_fim(fim)   # (B_batch, P, 2) variances
        crb_ell += float(crb[..., 0].mean()); crb_kap += float(crb[..., 1].mean())
        n_crb += 1

        # Empirical MSE from full MB-IDAR receiver.
        with torch.no_grad():
            hard, ell_hat, kap_hat, h_hat = multiblock_dasbl_receiver(
                system, batch, const, cfg,
                n_outer=6, n_lm_per_outer=3, rho_min=0.5, use_reacq=True,
            )
        match = hungarian_match(ell_hat, kap_hat,
                                batch.theta_true[..., 0], batch.theta_true[..., 1])
        for b in range(match.shape[0]):
            for pi in range(match.shape[1]):
                idx = int(match[b, pi])
                if idx >= 0:
                    de_sq += float((ell_hat[b, idx] - batch.theta_true[b, pi, 0]) ** 2)
                    dk_sq += float((kap_hat[b, idx] - batch.theta_true[b, pi, 1]) ** 2)
                    n += 1

    return {
        "crb_ell_mse": crb_ell / n_crb,
        "crb_kap_mse": crb_kap / n_crb,
        "emp_ell_mse": de_sq / max(n, 1),
        "emp_kap_mse": dk_sq / max(n, 1),
        "n_matched": n,
    }


def main():
    cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16, P_max=8)
    print(f"\n{'='*72}\nCRB vs B at HARD (P=5, N_p=16), {SNR} dB, K={N_SEEDS} seeds\n{'='*72}")
    print(f"{'B':<3s}  {'CRB(ell)':>10s}  {'RMSE(ell)':>10s}  {'ratio_ell':>9s}  "
          f"{'CRB(kap)':>10s}  {'RMSE(kap)':>10s}  {'ratio_kap':>9s}")
    results = {}
    for B in BS:
        # Aggregate over seeds
        crb_ell_seeds = []; crb_kap_seeds = []
        emp_ell_seeds = []; emp_kap_seeds = []
        t0 = time.time()
        for k in range(N_SEEDS):
            m = measure_one(cfg, B, seed=k * 137 + 42)
            crb_ell_seeds.append(m["crb_ell_mse"]); crb_kap_seeds.append(m["crb_kap_mse"])
            emp_ell_seeds.append(m["emp_ell_mse"]); emp_kap_seeds.append(m["emp_kap_mse"])
        crb_ell_mse = float(np.mean(crb_ell_seeds)); crb_kap_mse = float(np.mean(crb_kap_seeds))
        emp_ell_mse = float(np.mean(emp_ell_seeds)); emp_kap_mse = float(np.mean(emp_kap_seeds))
        emp_ell_std = float(np.std(emp_ell_seeds)); emp_kap_std = float(np.std(emp_kap_seeds))

        crb_ell_rmse = crb_ell_mse ** 0.5
        crb_kap_rmse = crb_kap_mse ** 0.5
        emp_ell_rmse = emp_ell_mse ** 0.5
        emp_kap_rmse = emp_kap_mse ** 0.5

        r_ell = emp_ell_rmse / max(crb_ell_rmse, 1e-9)
        r_kap = emp_kap_rmse / max(crb_kap_rmse, 1e-9)

        results[str(B)] = {
            "crb_ell_mse": crb_ell_mse, "crb_kap_mse": crb_kap_mse,
            "emp_ell_mse": emp_ell_mse, "emp_kap_mse": emp_kap_mse,
            "emp_ell_std": emp_ell_std, "emp_kap_std": emp_kap_std,
        }
        dt = time.time() - t0
        print(f"{B:<3d}  {crb_ell_mse:>10.3e}  {emp_ell_rmse:>10.3e}  {r_ell:>7.1f}x  "
              f"{crb_kap_mse:>10.3e}  {emp_kap_rmse:>10.3e}  {r_kap:>7.1f}x  ({dt:.0f}s)")

    out_path = Path("runs/crb_vs_B.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"snr_db": SNR, "config": "HARD (P=5, N_p=16)", "N_seeds": N_SEEDS,
                   "results": results}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
