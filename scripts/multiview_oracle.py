"""Day 3-5: Multi-view oracle experiment.

Question: does adding COMPLEMENTARY pilot views (not merely more pilot energy)
break the pilot-only LS gain bias that Test 2b of the audit revealed?

Setup: B_block AFDM blocks with SHARED (theta, h) but different pilot patterns.
At fixed AGGREGATE pilot budget B_block * N_p:

  * B=1: baseline. This is the standard single-block operating point.
  * B=2: half the per-block pilots, but complementary positions.
  * B=4: quarter per-block pilots, complementary.
  * Design variants:
      - repeated (same pilots per block): no diversity gain
      - hopping (random per block): stochastic diversity
      - complementary (designed offsets): structured diversity

Metrics:
  * h NMSE with true theta (isolates gain estimation)
  * Combined SER (all data symbols across all blocks)
  * Fisher information ratio and dictionary coherence (theoretical support)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.channels import UniformFractionalChannel
from afdm.classical import cg_solve
from afdm.experiments import ExperimentConfig
from afdm.multi_block import (
    PILOT_DESIGNS, sample_multiblock, multiblock_ls_gains, dictionary_coherence,
)
from afdm.operators import FastAFDMOperator
from afdm.system import AFDMSystem


def eval_multiblock(cfg: ExperimentConfig, B_block: int, N_p_per_block: int,
                    design: str, snrs=(5.0, 15.0, 25.0), n_batches=8,
                    batch_size=16, seed=42):
    """Run the multi-block oracle-theta experiment at (B_block, N_p_per_block, design)."""
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    N = cfg.N
    pp, pv = PILOT_DESIGNS[design](N=N, N_p=N_p_per_block, B=B_block,
                                   constellation=const, device=cfg.device, seed=seed)

    results = {}
    for snr in snrs:
        gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
        ser_acc = 0.0; nmse_acc = 0.0
        for _ in range(n_batches):
            batch = sample_multiblock(system, channel, const, pp, pv,
                                      batch_size=batch_size, snr_db=snr,
                                      generator=gen)
            # Oracle theta from batch.
            ell = batch.theta_true[..., 0]; kap = batch.theta_true[..., 1]
            # Pilot-only stacked LS.
            h_ls = multiblock_ls_gains(system, batch, ell, kap,
                                       use_pilot_only=True, lambda_ridge=1e-3)
            nmse = ((h_ls - batch.h_true).abs() ** 2).sum() / (batch.h_true.abs() ** 2).sum().clamp(min=1e-12)
            nmse_acc += float(nmse)
            # CG-MMSE per block using the shared h_ls.
            ser_block_sum = 0.0
            for b in range(B_block):
                op = FastAFDMOperator(system=system, ell=ell, kappa=kap, h=h_ls)
                def mv(v): return op.rmatvec(op.matvec(v)) + batch.sigma_w2_block * v
                x_soft = cg_solve(mv, op.rmatvec(batch.y[:, b, :]), max_iter=30)
                hard = (x_soft.unsqueeze(-1) - const.reshape(1, 1, -1)).abs().argmin(dim=-1)
                mask_b = batch.pilot_mask[:, b, :]
                ser_b = float(((hard != batch.labels[:, b, :]) * mask_b).float().sum() / mask_b.float().sum())
                ser_block_sum += ser_b
            ser_acc += ser_block_sum / B_block
        results[snr] = {
            "nmse_h": nmse_acc / n_batches,
            "ser": ser_acc / n_batches,
        }
    return results


def main():
    print("=" * 80)
    print("MULTI-VIEW ORACLE (Day 3-5)")
    print("=" * 80)
    print("Fix aggregate pilot budget B * N_p = 32 (matches easy) or 20 (matches hard scaling).")
    print()

    for name, base_cfg, aggregate_Np in (
        ("EASY (P=3), aggregate=32",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=3, N_p=32), 32),
        ("HARD (P=5), aggregate=16",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=16), 16),
        ("HARD (P=5), aggregate=32",
         ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=5, N_p=32), 32),
    ):
        print()
        print("=" * 80)
        print(f"CONFIG: {name}")
        print("=" * 80)
        print(f"{'B_block':<8s}  {'N_p':<5s}  {'design':<15s}  " +
              "  ".join(f"{snr:>4.0f}dB h_NMSE / SER" for snr in (5.0, 15.0, 25.0)))

        for B_block in (1, 2, 4):
            N_p_per_block = max(aggregate_Np // B_block, 4)  # keep per-block sane
            for design in ("repeated", "hopping", "complementary"):
                if B_block == 1 and design != "repeated":
                    continue  # for B=1, all designs are equivalent
                t0 = time.time()
                res = eval_multiblock(base_cfg, B_block, N_p_per_block, design)
                dt = time.time() - t0
                line = f"{B_block:<8d}  {N_p_per_block:<5d}  {design:<15s}  "
                for snr in (5.0, 15.0, 25.0):
                    line += f"{res[snr]['nmse_h']:.2e}/{res[snr]['ser']:.2e}  "
                line += f"  ({dt:.0f}s)"
                print(line)


if __name__ == "__main__":
    main()
