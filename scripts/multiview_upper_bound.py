"""Multi-view upper bound: use TRUE x (data-aided oracle) for LS gain estimation.

This is the ceiling that data-aided SBL / sequential Bayesian tracking could
reach if the reliably-decoded pseudo-pilots were 100% accurate.

If this ceiling is close to the pilot-only multi-view result, pilot diversity is
enough. If it's much better, data-aided is the missing piece.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.classical import cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import (
    PILOT_DESIGNS, sample_multiblock, multiblock_ls_gains,
)
from afdm.operators import FastAFDMOperator


def eval_multiblock_dataaided(cfg, B_block, N_p_per_block, design,
                              snrs=(5.0, 15.0, 25.0), n_batches=8, batch_size=16, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = PILOT_DESIGNS[design](N=cfg.N, N_p=N_p_per_block, B=B_block,
                                   constellation=const, device=cfg.device, seed=seed)

    results = {}
    for snr in snrs:
        gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
        ser_acc = 0.0; nmse_acc = 0.0
        for _ in range(n_batches):
            batch = sample_multiblock(system, channel, const, pp, pv,
                                      batch_size=batch_size, snr_db=snr, generator=gen)
            ell = batch.theta_true[..., 0]; kap = batch.theta_true[..., 1]
            # ORACLE data-aided: use x_true, not pilot-only.
            h_ls = multiblock_ls_gains(system, batch, ell, kap,
                                       use_pilot_only=False, lambda_ridge=1e-6)
            nmse = ((h_ls - batch.h_true).abs() ** 2).sum() / (batch.h_true.abs() ** 2).sum().clamp(min=1e-12)
            nmse_acc += float(nmse)
            ser_sum = 0.0
            for b in range(B_block):
                op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h_ls)
                def mv(v): return op.rmatvec(op.matvec(v)) + batch.sigma_w2_block * v
                x_soft = cg_solve(mv, op.rmatvec(batch.y[:, b, :]), max_iter=30)
                hard = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
                mask_b = batch.pilot_mask[:, b, :]
                ser_sum += float(((hard != batch.labels[:, b, :]) * mask_b).float().sum() / mask_b.float().sum())
            ser_acc += ser_sum / B_block
        results[snr] = {"nmse_h": nmse_acc / n_batches, "ser": ser_acc / n_batches}
    return results


def main():
    print("=" * 80)
    print("MULTI-VIEW DATA-AIDED UPPER BOUND (with true x)")
    print("=" * 80)
    for name, cfg, agg in (
        ("EASY  P=3, aggregate=32", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32), 32),
        ("HARD  P=5, aggregate=16", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16), 16),
        ("HARD  P=5, aggregate=32", ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32), 32),
    ):
        print()
        print(name)
        print(f"{'B_block':<8s}  {'N_p':<5s}  {'design':<15s}  {'5dB':>16s}  {'15dB':>16s}  {'25dB':>16s}")
        for B_block in (1, 2, 4):
            N_p = max(agg // B_block, 4)
            for design in ("repeated", "hopping", "complementary"):
                if B_block == 1 and design != "repeated":
                    continue
                res = eval_multiblock_dataaided(cfg, B_block, N_p, design)
                line = f"{B_block:<8d}  {N_p:<5d}  {design:<15s}  "
                for snr in (5.0, 15.0, 25.0):
                    line += f"{res[snr]['nmse_h']:.1e}/{res[snr]['ser']:.1e}  "
                print(line)


if __name__ == "__main__":
    main()
