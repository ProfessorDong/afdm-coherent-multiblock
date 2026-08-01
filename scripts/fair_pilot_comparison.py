"""Fair comparison: same AGGREGATE pilot budget across B blocks vs single block.

Reviewer will ask: "MB-DASBL uses B×N_p total pilots; of course it beats
single-block using only N_p pilots."

This experiment tests: at the SAME aggregate pilot count, does MB-DASBL still
beat single-block methods? If yes, the diversity gain is REAL (not just extra
pilot energy).

Setup:
  Aggregate pilot budget = 64.
  Single-block: 1 block × 64 pilots per block = 64 total.
  MB B=2:      2 blocks × 32 pilots per block = 64 total.
  MB B=4:      4 blocks × 16 pilots per block = 64 total.
  MB B=8:      8 blocks × 8  pilots per block = 64 total.

Compare receiver SER at 15 dB across these configurations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from afdm.classical import ClassicalCGDetector
from afdm.experiments import ExperimentConfig
from afdm.jpnce_sbl import JPNCESBLDetector
from afdm.training import sample_batch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase_diagram import genie_ser, receiver_ser
from multiblock_dasbl import eval_multiblock


def baseline_ser(det, cfg, snr_db, n_batches, batch_size, seed=42):
    system = cfg.system(); channel = cfg.channel(); const = cfg.constellation()
    pp, pv = cfg.pilots()
    gen = torch.Generator(device=cfg.device); gen.manual_seed(seed)
    ser_acc = 0.0
    for _ in range(n_batches):
        batch = sample_batch(system, channel, const, pp, pv,
                             batch_size=batch_size, snr_db=snr_db, generator=gen)
        try:
            out = det.detect(batch["r"], sigma_w2=batch["sigma_w2_block"])
            ser = float(((out["hard_x"] != batch["labels"]) * batch["pilot_mask"]).float().sum() / batch["pilot_mask"].float().sum())
        except Exception as e:
            print(f"    {type(det).__name__} failed: {e}")
            ser = float("nan")
        ser_acc += ser
    return ser_acc / n_batches


def main():
    print("=" * 90)
    print("FAIR PILOT-BUDGET COMPARISON  (aggregate = 64 pilots)")
    print("=" * 90)

    for P in (3, 5):
        print(f"\n--- P = {P} paths (channel) ---")
        print(f"{'Method':<45s}  {'total pilots':>12s}  {'SER at 15 dB':>13s}")

        # Single-block baselines at increasing N_p per block.
        for N_p in (8, 16, 32, 64):
            cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=P, N_p=N_p, P_max=P+3)
            g = genie_ser(cfg, 15.0)
            system = cfg.system(); const = cfg.constellation()
            pp, pv = cfg.pilots()
            classical = ClassicalCGDetector(
                system=system, support_recovery=cfg.support_recovery(),
                constellation=const, pilot_positions=pp, pilot_values=pv,
                T=8, K_cg=10, alpha=1.0, lambda_ridge=1e-3,
            )
            c = baseline_ser(classical, cfg, 15.0, 4, 16)
            jpnce = JPNCESBLDetector(system=system, constellation=const,
                                     pilot_positions=pp, pilot_values=pv,
                                     support_recovery=cfg.support_recovery(),
                                     T_em=15, T_grid=3, grid_lr=0.05, K_cg=15)
            j = baseline_ser(jpnce, cfg, 15.0, 4, 16)
            sb = receiver_ser(cfg, 15.0, use_reacq=True, n_batches=4, batch_size=16)
            print(f"1-block classical CG  (N_p={N_p})              {N_p:>12d}  {c:>13.3e}")
            print(f"1-block JPNCE-SBL     (N_p={N_p})              {N_p:>12d}  {j:>13.3e}")
            print(f"1-block SB-DASBL     (N_p={N_p})               {N_p:>12d}  {sb:>13.3e}")
            print(f"    (genie MMSE @ N_p={N_p}: {g:.3e})")

        # MB-DASBL at fixed aggregate pilots = 64.
        print()
        for B, N_p_per_b in [(2, 32), (4, 16), (8, 8)]:
            cfg = ExperimentConfig(N=128, kappa_max=5.0, ell_max=10.0, P=P, N_p=N_p_per_b, P_max=P+3)
            ser = eval_multiblock(cfg, 15.0, B_block=B, design="hopping",
                                  n_batches=4, batch_size=16)
            total = B * N_p_per_b
            print(f"MB-DASBL B={B}, N_p_per_block={N_p_per_b} (ours)      {total:>12d}  {ser:>13.3e}")


if __name__ == "__main__":
    main()
